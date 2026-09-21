"""Single-resolution coefficient scan for the local Roman profile."""

from __future__ import annotations

import sys
from typing import Callable, Sequence
import time

import numpy as np

from hammlet._core.jax_backend import JAXConsistentGeometryBatchScanner
from hammlet._core.multires import MultiResolutionScanResult
from hammlet._core.trajectory import ConsistentEventKernel

from .atlas import LocalMapAtlas, _append_point_lens_tail


MapFilter = Callable[[np.ndarray], np.ndarray]
_SCAN_BATCH_QUANTUM = 128


def scan_single_resolution(
    atlas: LocalMapAtlas,
    kernel_factory: Callable[
        [np.ndarray, int], Sequence[Sequence[ConsistentEventKernel]]
    ],
    *,
    m_max: int = 128,
    n_alpha: int = 540,
    batch_size: int = 1024,
    map_filter: MapFilter | None = None,
    compute_dtype: str = "float64",
    timing: dict[str, object] | None = None,
    progress_every: int = 0,
) -> MultiResolutionScanResult:
    """Scan every selected map at one Fourier resolution.

    This is intentionally the no-certificate path: it calls only
    :meth:`JAXConsistentGeometryBatchScanner.scan_minima`, never loads an error
    envelope, and returns ``chi2_lower`` / ``chi2_upper`` as ``None``.  A map
    filter leaves excluded rows in the result with ``chi2=inf`` and
    ``scanned=False`` so the coverage is explicit in the saved artifact.
    """
    m_max = int(m_max)
    n_alpha = int(n_alpha)
    batch_size = int(batch_size)
    if m_max < 0 or m_max > atlas.m_max:
        raise ValueError("m_max must be within the stored atlas mode range")
    if n_alpha < 4 or n_alpha % 2 or n_alpha < 4 * m_max:
        raise ValueError("n_alpha must be even and at least 4*m_max")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    progress_every = int(progress_every)
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")

    map_ids_parts: list[np.ndarray] = []
    parameter_parts: list[np.ndarray] = []
    chi2_parts: list[np.ndarray] = []
    geometry_parts: list[np.ndarray] = []
    alpha_parts: list[np.ndarray] = []
    scanned_parts: list[np.ndarray] = []
    scanner_by_shape: dict[tuple[int, int, int, int], JAXConsistentGeometryBatchScanner] = {}
    started = time.perf_counter()
    bucket_timings: list[dict[str, object]] = []
    total_scan_calls = 0
    total_padded_rows = 0

    # Whole batches are kept on a stable multiple of the padding quantum.  The
    # last batch of a shard can still have a smaller JAX shape, but it is padded
    # by repeating a real coefficient row rather than inventing a zero map.
    batch_step = max(
        _SCAN_BATCH_QUANTUM,
        int(np.ceil(batch_size / _SCAN_BATCH_QUANTUM)) * _SCAN_BATCH_QUANTUM,
    )

    for bucket_index, bucket in enumerate(atlas.iter_buckets(), start=1):
        bucket_started = time.perf_counter()
        bucket_timing: dict[str, object] = {
            "name": bucket.name,
            "maps": int(len(bucket.map_ids)),
            "radial_nodes": int(len(bucket.radial_nodes)),
        }
        radial_nodes = np.asarray(bucket.radial_nodes)
        kernel_started = time.perf_counter()
        kernels = kernel_factory(radial_nodes, m_max)
        bucket_timing["kernel_factory_seconds"] = (
            time.perf_counter() - kernel_started
        )
        if not kernels or not kernels[0]:
            raise ValueError("kernel_factory must return a non-empty geometry/dataset batch")
        shape_key = (
            int(radial_nodes.size),
            len(kernels),
            len(kernels[0]),
            m_max,
        )
        scanner_started = time.perf_counter()
        scanner = scanner_by_shape.get(shape_key)
        if scanner is None:
            scanner = JAXConsistentGeometryBatchScanner(
                kernels,
                n_alpha=n_alpha,
                compute_dtype=compute_dtype,
            )
            scanner_by_shape[shape_key] = scanner
            bucket_timing["scanner_init_seconds"] = time.perf_counter() - scanner_started
            bucket_timing["scanner_set_kernels_seconds"] = 0.0
        else:
            scanner.set_kernels(kernels)
            bucket_timing["scanner_init_seconds"] = 0.0
            bucket_timing["scanner_set_kernels_seconds"] = (
                time.perf_counter() - scanner_started
            )

        bucket_map_ids = np.asarray(bucket.map_ids, dtype=np.int64)
        bucket_parameters = np.asarray(bucket.parameters, dtype=np.float64)
        bucket_keep = _resolve_map_filter(map_filter, bucket_parameters)
        bucket_chi2 = np.full(len(bucket_map_ids), np.inf, dtype=np.float64)
        bucket_geometry = np.zeros(len(bucket_map_ids), dtype=np.int64)
        bucket_alpha = np.zeros(len(bucket_map_ids), dtype=np.int64)
        bucket_row = {
            int(map_id): index for index, map_id in enumerate(bucket_map_ids)
        }
        pending_rows = np.empty(batch_step, dtype=np.int64)
        pending_coefficients: np.ndarray | None = None
        pending_count = 0
        bucket_scan_calls = 0
        bucket_padded_rows = 0
        bucket_scan_seconds = 0.0
        bucket_copy_seconds = 0.0
        bucket_tail_seconds = 0.0

        def flush_pending() -> None:
            nonlocal pending_count, bucket_scan_calls
            nonlocal bucket_padded_rows, bucket_scan_seconds
            nonlocal bucket_tail_seconds
            if pending_count == 0:
                return
            if pending_coefficients is None:
                raise RuntimeError("coefficient batch was not initialized")
            batch = pending_coefficients[:pending_count]
            actual_count = len(batch)
            rows = pending_rows[:actual_count]
            if bucket.point_lens_tail_nodes is not None:
                tail_started = time.perf_counter()
                batch = _append_point_lens_tail(
                    batch,
                    bucket.point_lens_tail_nodes,
                    bucket_parameters[rows],
                )
                bucket_tail_seconds += time.perf_counter() - tail_started
            padded_batch = _pad_coefficients(batch)
            scan_started = time.perf_counter()
            minima, geometries, alphas = scanner.scan_minima(
                padded_batch
            )
            bucket_scan_seconds += time.perf_counter() - scan_started
            bucket_scan_calls += 1
            bucket_padded_rows += len(padded_batch) - actual_count
            bucket_chi2[rows] = minima[:actual_count]
            bucket_geometry[rows] = geometries[:actual_count]
            bucket_alpha[rows] = alphas[:actual_count]
            pending_count = 0

        shard_timing: dict[str, float | int] = {}
        for shard in bucket.iter_shards(
            m_max,
            timing=shard_timing,
            append_tail=False,
        ):
            local_ids = np.asarray(shard.map_ids, dtype=np.int64)
            if pending_coefficients is None:
                pending_coefficients = np.empty(
                    (batch_step,) + tuple(shard.x_coeff.shape[1:]),
                    dtype=shard.x_coeff.dtype,
                )
            local_rows = np.asarray(
                [bucket_row[int(map_id)] for map_id in local_ids], dtype=np.int64
            )
            selected_rows = local_rows[bucket_keep[local_rows]]
            if not len(selected_rows):
                continue
            selected_local_rows = np.flatnonzero(bucket_keep[local_rows])
            source_offset = 0
            while source_offset < len(selected_rows):
                capacity = batch_step - pending_count
                take = min(capacity, len(selected_rows) - source_offset)
                destination = slice(pending_count, pending_count + take)
                source = selected_local_rows[source_offset : source_offset + take]
                copy_started = time.perf_counter()
                pending_coefficients[destination] = np.asarray(
                    shard.x_coeff[source]
                )
                bucket_copy_seconds += time.perf_counter() - copy_started
                pending_rows[destination] = selected_rows[
                    source_offset : source_offset + take
                ]
                pending_count += take
                source_offset += take
                if pending_count == batch_step:
                    flush_pending()

        flush_pending()
        bucket_timing.update(
            {
                "selected_maps": int(np.count_nonzero(bucket_keep)),
                "shards_yielded": int(shard_timing.get("shards_yielded", 0)),
                "shards_filtered_empty": int(
                    shard_timing.get("shards_filtered_empty", 0)
                ),
                "coefficient_load_seconds": float(
                    shard_timing.get("coefficient_load_seconds", 0.0)
                ),
                "point_lens_tail_seconds": float(
                    bucket_tail_seconds
                    + float(shard_timing.get("point_lens_tail_seconds", 0.0))
                ),
                "shard_seconds": float(shard_timing.get("shard_seconds", 0.0)),
                "copy_seconds": float(bucket_copy_seconds),
                "scan_minima_seconds": float(bucket_scan_seconds),
                "scan_calls": int(bucket_scan_calls),
                "padded_rows": int(bucket_padded_rows),
                "total_seconds": float(time.perf_counter() - bucket_started),
            }
        )
        bucket_timings.append(bucket_timing)
        total_scan_calls += bucket_scan_calls
        total_padded_rows += bucket_padded_rows
        if progress_every and (
            bucket_index % progress_every == 0
            or bucket_index == len(atlas.buckets)
        ):
            print(
                "[roman-local] "
                f"bucket {bucket_index}/{len(atlas.buckets)} "
                f"maps={sum(len(part) for part in map_ids_parts)} "
                f"shards={sum(int(item['shards_yielded']) for item in bucket_timings)} "
                f"scan={sum(float(item['scan_minima_seconds']) for item in bucket_timings):.1f}s "
                f"tail={sum(float(item['point_lens_tail_seconds']) for item in bucket_timings):.1f}s "
                f"elapsed={time.perf_counter() - started:.1f}s",
                file=sys.stderr,
                flush=True,
            )
        map_ids_parts.append(bucket_map_ids)
        parameter_parts.append(bucket_parameters)
        chi2_parts.append(bucket_chi2)
        geometry_parts.append(bucket_geometry)
        alpha_parts.append(bucket_alpha)
        scanned_parts.append(bucket_keep)

    if not map_ids_parts:
        raise ValueError("atlas contains no maps after reading its shards")
    map_ids = np.concatenate(map_ids_parts)
    parameters = np.concatenate(parameter_parts)
    scanned = np.concatenate(scanned_parts)
    if timing is not None:
        ordered_buckets = sorted(
            bucket_timings,
            key=lambda item: float(item["total_seconds"]),
            reverse=True,
        )
        timing.clear()
        timing.update(
            {
                "bucket_count": len(bucket_timings),
                "shard_count": int(
                    sum(int(item["shards_yielded"]) for item in bucket_timings)
                ),
                "scan_call_count": int(total_scan_calls),
                "padded_rows": int(total_padded_rows),
                "scan_wall_seconds": float(time.perf_counter() - started),
                "kernel_factory_seconds": float(
                    sum(float(item["kernel_factory_seconds"]) for item in bucket_timings)
                ),
                "scanner_init_seconds": float(
                    sum(float(item["scanner_init_seconds"]) for item in bucket_timings)
                ),
                "scanner_set_kernels_seconds": float(
                    sum(
                        float(item["scanner_set_kernels_seconds"])
                        for item in bucket_timings
                    )
                ),
                "coefficient_load_seconds": float(
                    sum(
                        float(item["coefficient_load_seconds"])
                        for item in bucket_timings
                    )
                ),
                "point_lens_tail_seconds": float(
                    sum(
                        float(item["point_lens_tail_seconds"])
                        for item in bucket_timings
                    )
                ),
                "shard_seconds": float(
                    sum(float(item["shard_seconds"]) for item in bucket_timings)
                ),
                "copy_seconds": float(
                    sum(float(item["copy_seconds"]) for item in bucket_timings)
                ),
                "scan_minima_seconds": float(
                    sum(float(item["scan_minima_seconds"]) for item in bucket_timings)
                ),
                "top_buckets_by_total_seconds": ordered_buckets[:20],
            }
        )
    return MultiResolutionScanResult(
        map_ids=map_ids,
        parameters=parameters,
        chi2=np.concatenate(chi2_parts),
        geometry_index=np.concatenate(geometry_parts),
        alpha_index=np.concatenate(alpha_parts),
        spectral_risk=np.zeros(len(map_ids), dtype=np.float64),
        rescanned=scanned.copy(),
        base_seconds=0.0,
        full_seconds=time.perf_counter() - started,
        chi2_lower=None,
        chi2_upper=None,
        anomaly_score=None,
        anomaly_rescued=None,
        anomaly_masked_chi2=None,
        scanned=scanned,
    )


def _resolve_map_filter(
    map_filter: MapFilter | None, parameters: np.ndarray
) -> np.ndarray:
    if map_filter is None:
        return np.ones(len(parameters), dtype=bool)
    keep = np.asarray(map_filter(parameters), dtype=bool)
    if keep.shape != (len(parameters),):
        raise ValueError("map_filter must return one boolean per map")
    return keep


def _pad_coefficients(coefficients: np.ndarray) -> np.ndarray:
    count = len(coefficients)
    if count == 0:
        raise ValueError("cannot scan an empty coefficient batch")
    remainder = count % _SCAN_BATCH_QUANTUM
    if not remainder:
        return coefficients
    padding = _SCAN_BATCH_QUANTUM - remainder
    return np.concatenate(
        (coefficients, np.repeat(coefficients[-1:], padding, axis=0)), axis=0
    )
