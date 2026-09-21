"""Process-level parallel scan of a local packed Roman atlas.

The production map tree is never used as a work queue here.  Each worker opens
the same read-only packed snapshot and owns a disjoint set of radial buckets;
the parent only collects small minima arrays and writes local result files.
JAX is imported inside the spawned worker initializer so no process forks after
the JAX runtime has been initialized.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np

from hammlet._core.multires import MultiResolutionScanResult

from .packed import PackedAtlas, open_packed_atlas


_SCAN_BATCH_QUANTUM = 128
_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

_WORKER_STATE: dict[str, Any] | None = None


def configure_parallel_execution(
    cores: int,
    workers: int | None = None,
    threads_per_worker: int = 1,
    *,
    bucket_count: int | None = None,
) -> dict[str, Any]:
    """Pin the parent to a CPU budget and return the worker assignment plan."""
    cores = int(cores)
    if cores < 1:
        raise ValueError("cores must be positive")
    threads_per_worker = int(threads_per_worker)
    if threads_per_worker < 1:
        raise ValueError("threads_per_worker must be positive")
    requested_workers = cores if workers is None else int(workers)
    if requested_workers < 1:
        raise ValueError("workers must be positive")
    if bucket_count is not None:
        if int(bucket_count) < 1:
            raise ValueError("packed atlas contains no buckets")
        requested_workers = min(requested_workers, int(bucket_count))
    if requested_workers * threads_per_worker > cores:
        raise ValueError(
            "workers * threads_per_worker must not exceed the requested core budget"
        )

    available: list[int] = []
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        available = sorted(os.sched_getaffinity(0))
        if cores > len(available):
            raise ValueError(
                f"cores={cores} exceeds the {len(available)} CPUs available to this process"
            )
        selected = available[:cores]
        os.sched_setaffinity(0, selected)
    else:
        selected = []

    for name in _THREAD_ENV:
        os.environ[name] = str(threads_per_worker)
    worker_affinities = [
        selected[index * threads_per_worker : (index + 1) * threads_per_worker]
        for index in range(requested_workers)
    ]
    return {
        "cores_requested": cores,
        "workers_requested": int(workers if workers is not None else cores),
        "workers_used": requested_workers,
        "threads_per_worker": threads_per_worker,
        "cpu_affinity": selected,
        "worker_affinity_plan": worker_affinities,
        "thread_env": {name: os.environ[name] for name in _THREAD_ENV},
        "affinity_supported": bool(selected),
    }


def scan_packed_atlas(
    atlas: PackedAtlas,
    datasets: Sequence[Any],
    geometries: Sequence[Any],
    *,
    m_max: int = 128,
    n_alpha: int = 540,
    batch_size: int = 1024,
    compute_dtype: str = "float64",
    radial_order: int = 1,
    cores: int = 4,
    workers: int | None = None,
    threads_per_worker: int = 1,
    max_maps: int | None = None,
    required_radius: float | None = None,
    map_filter_logq_max: float | None = None,
    bucket_result_dir: str | Path | None = None,
    resume: bool = False,
    progress_every: int = 10,
) -> tuple[MultiResolutionScanResult, dict[str, Any], dict[str, Any]]:
    """Scan packed buckets in spawned processes and combine their minima.

    ``atlas`` is the parent-side metadata view.  Workers reopen the cache by
    path, which keeps the large coefficient payload out of task pickles and
    lets the operating system share clean read-only pages between processes.
    """
    m_max = int(m_max)
    n_alpha = int(n_alpha)
    batch_size = int(batch_size)
    radial_order = int(radial_order)
    if m_max < 0 or m_max > atlas.m_max:
        raise ValueError("m_max must be within the packed atlas mode range")
    if n_alpha < 4 or n_alpha % 2 or n_alpha < 4 * m_max:
        raise ValueError("n_alpha must be even and at least 4*m_max")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")
    if map_filter_logq_max is not None and not np.isfinite(map_filter_logq_max):
        raise ValueError("map_filter_logq_max must be finite")

    execution = configure_parallel_execution(
        cores,
        workers,
        threads_per_worker,
        bucket_count=len(atlas.buckets),
    )
    result_directory = (
        None
        if bucket_result_dir is None
        else Path(bucket_result_dir).expanduser().resolve()
    )
    if result_directory is not None:
        result_directory.mkdir(parents=True, exist_ok=True)

    bucket_results: dict[int, dict[str, Any]] = {}
    resumed_count = 0
    if resume and result_directory is not None:
        for bucket_index, bucket in enumerate(atlas.buckets):
            loaded = _load_bucket_result(
                result_directory / _bucket_filename(bucket_index), bucket
            )
            if loaded is not None:
                bucket_results[bucket_index] = loaded
                resumed_count += 1

    missing_indices = [
        index for index in range(len(atlas.buckets)) if index not in bucket_results
    ]
    progress_every = int(progress_every)
    started = time.perf_counter()
    monitor = _CpuMonitor(execution["cpu_affinity"])
    if missing_indices:
        monitor.start()
        try:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=int(execution["workers_used"]),
                mp_context=context,
                initializer=_worker_initialize,
                initargs=(
                    str(atlas.path),
                    max_maps,
                    required_radius,
                    radial_order,
                    tuple(datasets),
                    tuple(geometries),
                    m_max,
                    n_alpha,
                    batch_size,
                    compute_dtype,
                    map_filter_logq_max,
                    int(execution["workers_used"]),
                    tuple(execution["cpu_affinity"]),
                    int(threads_per_worker),
                ),
            ) as pool:
                future_to_index = {
                    pool.submit(_scan_bucket, bucket_index): bucket_index
                    for bucket_index in missing_indices
                }
                completed = resumed_count
                total = len(atlas.buckets)
                for future in as_completed(future_to_index):
                    bucket_index = future_to_index[future]
                    payload = future.result()
                    bucket_results[bucket_index] = payload
                    if result_directory is not None:
                        _save_bucket_result(
                            result_directory / _bucket_filename(bucket_index),
                            payload,
                        )
                    completed += 1
                    if progress_every and (
                        completed % progress_every == 0 or completed == total
                    ):
                        elapsed = time.perf_counter() - started
                        print(
                            "[roman-local-parallel] "
                            f"bucket {completed}/{total} "
                            f"maps={sum(len(atlas.buckets[i].map_ids) for i in bucket_results)} "
                            f"elapsed={elapsed:.1f}s",
                            file=sys.stderr,
                            flush=True,
                        )
        finally:
            monitor.stop()

    wall_seconds = time.perf_counter() - started
    result, timing = _combine_results(
        atlas,
        bucket_results,
        wall_seconds=wall_seconds,
        resumed_count=resumed_count,
        monitor=monitor,
        result_directory=result_directory,
    )
    return result, timing, execution


def _worker_initialize(
    cache_path: str,
    max_maps: int | None,
    required_radius: float | None,
    radial_order: int,
    datasets: tuple[Any, ...],
    geometries: tuple[Any, ...],
    m_max: int,
    n_alpha: int,
    batch_size: int,
    compute_dtype: str,
    map_filter_logq_max: float | None,
    worker_count: int,
    cpu_ids: tuple[int, ...],
    threads_per_worker: int,
) -> None:
    """Initialize one JAX runtime after spawn and bind it to its CPU slice."""
    global _WORKER_STATE
    identity = getattr(mp.current_process(), "_identity", ())
    worker_slot = (int(identity[0]) - 1) % int(worker_count) if identity else 0
    target = list(
        cpu_ids[
            worker_slot * threads_per_worker : (worker_slot + 1) * threads_per_worker
        ]
    )
    for name in _THREAD_ENV:
        os.environ[name] = str(threads_per_worker)
    if target and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, target)

    # These imports intentionally happen only in the spawned child.
    from hammlet._core.jax_backend import JAXConsistentGeometryBatchScanner
    from hammlet._core.trajectory import ConsistentKernelFactory

    atlas = open_packed_atlas(cache_path, max_maps=max_maps)
    if required_radius is not None:
        atlas = atlas.with_point_lens_tail(
            float(required_radius), radial_order=int(radial_order)
        )
    factory_started = time.perf_counter()
    factory = ConsistentKernelFactory(
        datasets,
        geometries,
        int(m_max),
        radial_order=int(radial_order),
    )
    factory_wall_seconds = time.perf_counter() - factory_started
    _WORKER_STATE = {
        "atlas": atlas,
        "factory": factory,
        "scanner_type": JAXConsistentGeometryBatchScanner,
        "m_max": int(m_max),
        "n_alpha": int(n_alpha),
        "batch_size": int(batch_size),
        "compute_dtype": compute_dtype,
        "map_filter_logq_max": map_filter_logq_max,
        "scanners": {},
        "factory_wall_seconds": factory_wall_seconds,
        "factory_basis_seconds": float(factory.basis_seconds),
        "factory_reported": False,
        "worker_slot": worker_slot,
        "pid": os.getpid(),
        "affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else target,
    }


def _scan_bucket(bucket_index: int) -> dict[str, Any]:
    state = _WORKER_STATE
    if state is None:
        raise RuntimeError("parallel worker was not initialized")
    atlas: PackedAtlas = state["atlas"]
    bucket = atlas.buckets[int(bucket_index)]
    m_max = int(state["m_max"])
    started = time.perf_counter()
    process_started = time.process_time()

    kernel_started = time.perf_counter()
    kernels = state["factory"](np.asarray(bucket.radial_nodes), m_max)
    kernel_factory_seconds = time.perf_counter() - kernel_started
    if not kernels or not kernels[0]:
        raise ValueError("kernel factory returned no kernels")
    shape_key = (
        int(len(bucket.radial_nodes)),
        len(kernels),
        len(kernels[0]),
        m_max,
    )
    scanner_started = time.perf_counter()
    scanner = state["scanners"].get(shape_key)
    if scanner is None:
        scanner = state["scanner_type"](
            kernels,
            n_alpha=int(state["n_alpha"]),
            compute_dtype=state["compute_dtype"],
        )
        state["scanners"][shape_key] = scanner
        scanner_init_seconds = time.perf_counter() - scanner_started
        scanner_set_kernels_seconds = 0.0
    else:
        scanner.set_kernels(kernels)
        scanner_init_seconds = 0.0
        scanner_set_kernels_seconds = time.perf_counter() - scanner_started

    load_started = time.perf_counter()
    coefficients = bucket.load_coefficients(m_max)
    coefficient_load_seconds = time.perf_counter() - load_started
    map_ids = np.asarray(bucket.map_ids, dtype=np.int64)
    parameters = np.asarray(bucket.parameters, dtype=np.float64)
    if state["map_filter_logq_max"] is None:
        keep = np.ones(len(map_ids), dtype=bool)
    else:
        keep = parameters[:, 1] <= float(state["map_filter_logq_max"]) + 1.0e-12

    chi2 = np.full(len(map_ids), np.inf, dtype=np.float64)
    geometry_index = np.zeros(len(map_ids), dtype=np.int64)
    alpha_index = np.zeros(len(map_ids), dtype=np.int64)
    selected = np.flatnonzero(keep)
    batch_step = max(
        _SCAN_BATCH_QUANTUM,
        int(np.ceil(int(state["batch_size"]) / _SCAN_BATCH_QUANTUM))
        * _SCAN_BATCH_QUANTUM,
    )
    scan_seconds = 0.0
    scan_calls = 0
    padded_rows = 0
    for offset in range(0, len(selected), batch_step):
        rows = selected[offset : offset + batch_step]
        batch = np.asarray(coefficients[rows])
        actual_count = len(batch)
        padded = _pad_coefficients(batch)
        scan_started = time.perf_counter()
        minima, geometries, alphas = scanner.scan_minima(padded)
        scan_seconds += time.perf_counter() - scan_started
        scan_calls += 1
        padded_rows += len(padded) - actual_count
        chi2[rows] = np.asarray(minima[:actual_count], dtype=np.float64)
        geometry_index[rows] = np.asarray(geometries[:actual_count], dtype=np.int64)
        alpha_index[rows] = np.asarray(alphas[:actual_count], dtype=np.int64)
    del coefficients

    factory_basis_seconds = 0.0
    if not state["factory_reported"]:
        factory_basis_seconds = float(state["factory_basis_seconds"])
        state["factory_reported"] = True
    return {
        "bucket_index": int(bucket_index),
        "name": bucket.name,
        "map_ids": map_ids,
        "parameters": parameters,
        "chi2": chi2,
        "geometry_index": geometry_index,
        "alpha_index": alpha_index,
        "scanned": keep,
        "timing": {
            "maps": int(len(map_ids)),
            "selected_maps": int(np.count_nonzero(keep)),
            "radial_nodes": int(len(bucket.radial_nodes)),
            "kernel_factory_seconds": float(kernel_factory_seconds),
            "scanner_init_seconds": float(scanner_init_seconds),
            "scanner_set_kernels_seconds": float(scanner_set_kernels_seconds),
            "coefficient_load_seconds": float(coefficient_load_seconds),
            "point_lens_tail_enabled": bool(bucket.point_lens_tail_nodes is not None),
            "scan_minima_seconds": float(scan_seconds),
            "scan_calls": int(scan_calls),
            "padded_rows": int(padded_rows),
            "total_seconds": float(time.perf_counter() - started),
            "process_seconds": float(time.process_time() - process_started),
        },
        "worker": {
            "pid": int(state["pid"]),
            "slot": int(state["worker_slot"]),
            "affinity": list(state["affinity"]),
            "factory_basis_seconds": float(factory_basis_seconds),
            "factory_wall_seconds": float(state["factory_wall_seconds"])
            if factory_basis_seconds
            else 0.0,
        },
    }


def _pad_coefficients(coefficients: np.ndarray) -> np.ndarray:
    count = len(coefficients)
    if count == 0:
        raise ValueError("cannot scan an empty coefficient batch")
    remainder = count % _SCAN_BATCH_QUANTUM
    if not remainder:
        return coefficients
    return np.concatenate(
        (
            coefficients,
            np.repeat(
                coefficients[-1:], _SCAN_BATCH_QUANTUM - remainder, axis=0
            ),
        ),
        axis=0,
    )


def _bucket_filename(bucket_index: int) -> str:
    return f"bucket-{int(bucket_index):05d}"


def _save_bucket_result(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.partial-{time.time_ns()}"
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            map_ids=payload["map_ids"],
            parameters=payload["parameters"],
            chi2=payload["chi2"],
            geometry_index=payload["geometry_index"],
            alpha_index=payload["alpha_index"],
            scanned=payload["scanned"],
        )
    temporary.replace(path.with_suffix(".npz"))
    timing_path = path.with_suffix(".json")
    timing_temporary = timing_path.parent / f".{timing_path.name}.partial-{time.time_ns()}"
    timing_temporary.write_text(
        json.dumps(
            {
                "bucket_index": int(payload["bucket_index"]),
                "name": payload["name"],
                "timing": payload["timing"],
                "worker": payload["worker"],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    timing_temporary.replace(timing_path)


def _load_bucket_result(path: Path, bucket: Any) -> dict[str, Any] | None:
    npz_path = path.with_suffix(".npz")
    if not npz_path.is_file():
        return None
    try:
        with np.load(npz_path, allow_pickle=False) as arrays:
            map_ids = np.asarray(arrays["map_ids"], dtype=np.int64)
            parameters = np.asarray(arrays["parameters"], dtype=np.float64)
            chi2 = np.asarray(arrays["chi2"], dtype=np.float64)
            geometry_index = np.asarray(arrays["geometry_index"], dtype=np.int64)
            alpha_index = np.asarray(arrays["alpha_index"], dtype=np.int64)
            scanned = np.asarray(arrays["scanned"], dtype=bool)
        if not np.array_equal(map_ids, bucket.map_ids):
            return None
        if parameters.shape != bucket.parameters.shape or not np.array_equal(
            parameters, bucket.parameters
        ):
            return None
        if any(
            array.shape != (len(bucket.map_ids),)
            for array in (chi2, geometry_index, alpha_index, scanned)
        ):
            return None
    except (OSError, ValueError, KeyError):
        return None
    metadata: dict[str, Any] = {}
    json_path = path.with_suffix(".json")
    if json_path.is_file():
        try:
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
    return {
        "bucket_index": int(metadata.get("bucket_index", bucket.index)),
        "name": str(metadata.get("name", bucket.name)),
        "map_ids": map_ids,
        "parameters": parameters,
        "chi2": chi2,
        "geometry_index": geometry_index,
        "alpha_index": alpha_index,
        "scanned": scanned,
        "timing": dict(metadata.get("timing", {})),
        "worker": dict(metadata.get("worker", {"pid": None, "slot": None})),
        "resumed": True,
    }


def _combine_results(
    atlas: PackedAtlas,
    bucket_results: Mapping[int, Mapping[str, Any]],
    *,
    wall_seconds: float,
    resumed_count: int,
    monitor: "_CpuMonitor",
    result_directory: Path | None,
) -> tuple[MultiResolutionScanResult, dict[str, Any]]:
    if len(bucket_results) != len(atlas.buckets):
        missing = sorted(set(range(len(atlas.buckets))) - set(bucket_results))
        raise RuntimeError(f"parallel scan is missing bucket results: {missing[:5]}")
    map_ids_parts: list[np.ndarray] = []
    parameter_parts: list[np.ndarray] = []
    chi2_parts: list[np.ndarray] = []
    geometry_parts: list[np.ndarray] = []
    alpha_parts: list[np.ndarray] = []
    scanned_parts: list[np.ndarray] = []
    bucket_timings: list[dict[str, Any]] = []
    worker_rows: dict[str, dict[str, Any]] = {}
    for bucket_index, bucket in enumerate(atlas.buckets):
        payload = bucket_results[bucket_index]
        if not np.array_equal(payload["map_ids"], bucket.map_ids):
            raise RuntimeError(f"bucket {bucket_index} returned unexpected map IDs")
        map_ids_parts.append(np.asarray(payload["map_ids"], dtype=np.int64))
        parameter_parts.append(np.asarray(payload["parameters"], dtype=np.float64))
        chi2_parts.append(np.asarray(payload["chi2"], dtype=np.float64))
        geometry_parts.append(np.asarray(payload["geometry_index"], dtype=np.int64))
        alpha_parts.append(np.asarray(payload["alpha_index"], dtype=np.int64))
        scanned_parts.append(np.asarray(payload["scanned"], dtype=bool))
        timing = dict(payload.get("timing", {}))
        timing.update(
            {
                "bucket_index": int(bucket_index),
                "name": bucket.name,
                "resumed": bool(payload.get("resumed", False)),
            }
        )
        bucket_timings.append(timing)
        worker = dict(payload.get("worker", {}))
        pid = worker.get("pid")
        if pid is not None:
            key = str(pid)
            aggregate = worker_rows.setdefault(
                key,
                {
                    "pid": int(pid),
                    "slot": worker.get("slot"),
                    "affinity": worker.get("affinity", []),
                    "bucket_count": 0,
                    "process_seconds": 0.0,
                    "wall_seconds": 0.0,
                    "factory_basis_seconds": 0.0,
                },
            )
            aggregate["bucket_count"] += 1
            aggregate["process_seconds"] += float(
                timing.get("process_seconds", 0.0)
            )
            aggregate["wall_seconds"] += float(timing.get("total_seconds", 0.0))
            aggregate["factory_basis_seconds"] += float(
                worker.get("factory_basis_seconds", 0.0)
            )

    map_ids = np.concatenate(map_ids_parts)
    parameters = np.concatenate(parameter_parts, axis=0)
    scanned = np.concatenate(scanned_parts)
    result = MultiResolutionScanResult(
        map_ids=map_ids,
        parameters=parameters,
        chi2=np.concatenate(chi2_parts),
        geometry_index=np.concatenate(geometry_parts),
        alpha_index=np.concatenate(alpha_parts),
        spectral_risk=np.zeros(len(map_ids), dtype=np.float64),
        rescanned=scanned.copy(),
        base_seconds=0.0,
        full_seconds=float(wall_seconds),
        chi2_lower=None,
        chi2_upper=None,
        anomaly_score=None,
        anomaly_rescued=None,
        anomaly_masked_chi2=None,
        scanned=scanned,
    )
    timing_sums = {
        key: float(
            sum(float(item.get(key, 0.0)) for item in bucket_timings)
        )
        for key in (
            "kernel_factory_seconds",
            "scanner_init_seconds",
            "scanner_set_kernels_seconds",
            "coefficient_load_seconds",
            "scan_minima_seconds",
            "total_seconds",
            "process_seconds",
        )
    }
    timing: dict[str, Any] = {
        "bucket_count": len(bucket_timings),
        "completed_bucket_count": len(bucket_timings),
        "resumed_bucket_count": int(resumed_count),
        "new_bucket_count": len(bucket_timings) - int(resumed_count),
        "map_count": int(len(map_ids)),
        "scanned_map_count": int(np.count_nonzero(scanned)),
        "scan_call_count": int(
            sum(int(item.get("scan_calls", 0)) for item in bucket_timings)
        ),
        "padded_rows": int(
            sum(int(item.get("padded_rows", 0)) for item in bucket_timings)
        ),
        "parallel_wall_seconds": float(wall_seconds),
        "scan_wall_seconds": float(wall_seconds),
        "bucket_wall_sum_seconds": timing_sums["total_seconds"],
        "worker_process_seconds_sum": timing_sums["process_seconds"],
        "kernel_factory_basis_seconds": float(
            sum(float(row.get("factory_basis_seconds", 0.0)) for row in worker_rows.values())
        ),
        "kernel_factory_binding_seconds": timing_sums["kernel_factory_seconds"],
        "scanner_init_seconds": timing_sums["scanner_init_seconds"],
        "scanner_set_kernels_seconds": timing_sums["scanner_set_kernels_seconds"],
        "coefficient_load_seconds": timing_sums["coefficient_load_seconds"],
        "scan_minima_seconds": timing_sums["scan_minima_seconds"],
        "top_buckets_by_total_seconds": sorted(
            bucket_timings,
            key=lambda item: float(item.get("total_seconds", 0.0)),
            reverse=True,
        )[:20],
        "workers": sorted(worker_rows.values(), key=lambda item: int(item["pid"])),
        "cpu_utilization": monitor.summary(),
        "bucket_result_directory": (
            None if result_directory is None else str(result_directory)
        ),
    }
    return result, timing


class _CpuMonitor:
    """Sample aggregate utilization of the selected CPUs from /proc/stat."""

    def __init__(self, cpu_ids: Sequence[int], interval: float = 0.5) -> None:
        self.cpu_ids = tuple(int(value) for value in cpu_ids)
        self.interval = float(interval)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[tuple[float, dict[int, tuple[int, int]]]] = []

    def start(self) -> None:
        if not self.cpu_ids or not Path("/proc/stat").is_file():
            return
        self._samples.append((time.perf_counter(), _read_cpu_counters(self.cpu_ids)))
        self._thread = threading.Thread(target=self._run, name="cpu-monitor", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._samples.append(
                (time.perf_counter(), _read_cpu_counters(self.cpu_ids))
            )

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval * 4.0))
        self._samples.append((time.perf_counter(), _read_cpu_counters(self.cpu_ids)))

    def summary(self) -> dict[str, Any]:
        utilizations: list[float] = []
        for (_, previous), (_, current) in zip(self._samples, self._samples[1:]):
            busy = 0
            total = 0
            for cpu_id in self.cpu_ids:
                old_busy, old_total = previous.get(cpu_id, (0, 0))
                new_busy, new_total = current.get(cpu_id, (0, 0))
                busy += max(new_busy - old_busy, 0)
                total += max(new_total - old_total, 0)
            if total:
                utilizations.append(100.0 * busy / total)
        if not utilizations:
            return {
                "available": False,
                "cpu_ids": list(self.cpu_ids),
                "sample_count": len(self._samples),
            }
        values = np.asarray(utilizations, dtype=np.float64)
        return {
            "available": True,
            "cpu_ids": list(self.cpu_ids),
            "sample_count": len(self._samples),
            "interval_seconds": self.interval,
            "mean_percent_of_selected_capacity": float(np.mean(values)),
            "p50_percent_of_selected_capacity": float(np.percentile(values, 50)),
            "p95_percent_of_selected_capacity": float(np.percentile(values, 95)),
            "max_percent_of_selected_capacity": float(np.max(values)),
            "mean_core_equivalents": float(
                np.mean(values) * len(self.cpu_ids) / 100.0
            ),
            "note": (
                "Aggregate system busy time on the selected CPUs; this is not a "
                "per-process CPU counter.  100 percent means all selected CPUs "
                "were busy."
            ),
        }


def _read_cpu_counters(cpu_ids: Sequence[int]) -> dict[int, tuple[int, int]]:
    wanted = {int(value) for value in cpu_ids}
    counters: dict[int, tuple[int, int]] = {}
    try:
        lines = Path("/proc/stat").read_text(encoding="ascii").splitlines()
    except OSError:
        return counters
    for line in lines:
        fields = line.split()
        if not fields or not fields[0].startswith("cpu") or fields[0] == "cpu":
            continue
        try:
            cpu_id = int(fields[0][3:])
        except ValueError:
            continue
        if cpu_id not in wanted or len(fields) < 5:
            continue
        values = [int(value) for value in fields[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        total = sum(values)
        counters[cpu_id] = (total - idle, total)
    return counters
