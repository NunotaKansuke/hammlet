from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Sequence

import numpy as np

from .adaptive_grid import (
    interpolated_trajectory_error,
    select_adaptive_rescan_indices,
    select_certified_rescan_indices,
    total_angular_error_envelope,
)
from .atlas import AtlasShard, PolarAtlas
from .bucketed_atlas import BucketedPolarAtlas
from .jax_backend import JAXConsistentGeometryBatchScanner
from .trajectory import ConsistentEventKernel


@dataclass(frozen=True)
class MultiResolutionScanResult:
    map_ids: np.ndarray
    parameters: np.ndarray
    chi2: np.ndarray
    geometry_index: np.ndarray
    alpha_index: np.ndarray
    spectral_risk: np.ndarray
    rescanned: np.ndarray
    base_seconds: float
    full_seconds: float
    chi2_lower: np.ndarray | None = None
    chi2_upper: np.ndarray | None = None
    anomaly_score: np.ndarray | None = None
    anomaly_rescued: np.ndarray | None = None
    anomaly_masked_chi2: np.ndarray | None = None
    scanned: np.ndarray | None = None

    def candidates(
        self,
        *,
        n_alpha: int,
        count: int | None = None,
        rescanned_only: bool = True,
    ) -> list["MultiResolutionCandidate"]:
        if n_alpha < 2:
            raise ValueError("n_alpha must be at least two")
        rows = (
            np.flatnonzero(self.rescanned)
            if rescanned_only
            else np.arange(len(self.map_ids))
        )
        rows = rows[np.argsort(self.chi2[rows], kind="stable")]
        if count is not None:
            if count < 1:
                raise ValueError("count must be positive")
            rows = rows[:count]
        return [
            MultiResolutionCandidate(
                map_id=int(self.map_ids[row]),
                logs=float(self.parameters[row, 0]),
                logq=float(self.parameters[row, 1]),
                logrho=float(self.parameters[row, 2]),
                geometry_index=int(self.geometry_index[row]),
                alpha_index=int(self.alpha_index[row]),
                alpha=float(2.0 * np.pi * self.alpha_index[row] / n_alpha),
                chi2=float(self.chi2[row]),
                chi2_lower=(
                    float(self.chi2[row])
                    if self.chi2_lower is None
                    else float(self.chi2_lower[row])
                ),
                chi2_upper=(
                    float(self.chi2[row])
                    if self.chi2_upper is None
                    else float(self.chi2_upper[row])
                ),
                spectral_risk=float(self.spectral_risk[row]),
                anomaly_score=(
                    None
                    if self.anomaly_score is None
                    else float(self.anomaly_score[row])
                ),
                anomaly_rescued=(
                    False
                    if self.anomaly_rescued is None
                    else bool(self.anomaly_rescued[row])
                ),
            )
            for row in rows
        ]


@dataclass(frozen=True)
class MultiResolutionCandidate:
    """Backend-neutral handoff record for direct map or VBM evaluation."""

    map_id: int
    logs: float
    logq: float
    logrho: float
    geometry_index: int
    alpha_index: int
    alpha: float
    chi2: float
    spectral_risk: float
    chi2_lower: float | None = None
    chi2_upper: float | None = None
    anomaly_score: float | None = None
    anomaly_rescued: bool = False


def _grouped(iterable, count: int):
    pending = []
    for item in iterable:
        pending.append(item)
        if len(pending) == count:
            yield pending
            pending = []
    if pending:
        yield pending


AnomalyScorer = Callable[[np.ndarray, AtlasShard], np.ndarray]

# ``(radial_nodes, coefficients, n_alpha) -> (map, geometry, alpha)`` booleans.
AnomalyMaskBuilder = Callable[[np.ndarray, np.ndarray, int], np.ndarray]

ANOMALY_MASK_MODES = ("rank", "full")

# ``(map, parameter) -> (map,)`` booleans, deciding which maps are scanned at all.
MapFilter = Callable[[np.ndarray], np.ndarray]

# Filtered batches are padded up to a multiple of this before being handed to
# JAX.  Without it every batch would present a different leading dimension and
# the scan would recompile far more often than the pruning saves.
SCAN_BATCH_QUANTUM = 128


def _certificate_or_reconstruction_error(shard: AtlasShard) -> np.ndarray:
    """Prefer a deterministic atlas certificate, with legacy fallback."""
    value = (
        shard.reconstruction_error
        if shard.certified_error is None
        else shard.certified_error
    )
    return np.asarray(value)


def _padded_scan_batch(coefficients: np.ndarray) -> tuple[np.ndarray, int]:
    """Round a batch up to ``SCAN_BATCH_QUANTUM`` rows by repeating the last one.

    Repetition rather than zero padding: an all-zero spectrum is a constant
    magnification, whose flux design matrix is rank deficient, so the padding
    would decide whether the scan returns a number or a NaN.
    """
    count = len(coefficients)
    if count == 0:
        raise ValueError("cannot pad an empty batch")
    remainder = count % SCAN_BATCH_QUANTUM
    if not remainder:
        return coefficients, count
    padding = SCAN_BATCH_QUANTUM - remainder
    filler = np.repeat(coefficients[-1:], padding, axis=0)
    return np.concatenate((coefficients, filler)), count


def _pad_error_envelope(error: np.ndarray, padded_count: int) -> np.ndarray:
    """Apply the coefficient batch's repeat-last padding to its error rows."""
    error = np.asarray(error)
    if len(error) == padded_count:
        return error
    if len(error) == 0 or len(error) > padded_count:
        raise ValueError("error envelope cannot match the padded coefficient batch")
    filler = np.repeat(error[-1:], padded_count - len(error), axis=0)
    return np.concatenate((error, filler))


def _resolve_map_filter(
    map_filter: "MapFilter | None", parameters: np.ndarray
) -> np.ndarray:
    if map_filter is None:
        return np.ones(len(parameters), dtype=bool)
    keep = np.asarray(map_filter(parameters), dtype=bool)
    if keep.shape != (len(parameters),):
        raise ValueError("map_filter must return one boolean per map")
    return keep


def _rescan_union(
    chi2: np.ndarray,
    spectral_risk: np.ndarray,
    anomaly_score: np.ndarray,
    *,
    top_count: int,
    risk_count: int,
    anomaly_count: int,
) -> np.ndarray:
    selected = select_adaptive_rescan_indices(
        chi2,
        spectral_risk,
        top_count=top_count,
        risk_count=risk_count,
    )
    if anomaly_count <= 0:
        return selected
    finite = np.flatnonzero(np.isfinite(anomaly_score) & (anomaly_score > 0.0))
    if finite.size:
        # Scores >= 1 mean the conservative angular envelope can reach every
        # requested anomaly amplitude.  Among those compatible maps, retain
        # the best low-mode light curves; a much larger envelope is not
        # intrinsically a better match.  If no map reaches the requested
        # amplitude, fall back to the closest envelopes for diagnostics.
        compatible = finite[anomaly_score[finite] >= 1.0]
        if compatible.size:
            hint_rows = compatible[
                np.argsort(chi2[compatible], kind="stable")[:anomaly_count]
            ]
        else:
            hint_rows = finite[
                np.argsort(-anomaly_score[finite], kind="stable")[:anomaly_count]
            ]
        selected = np.unique(np.concatenate((selected, hint_rows)))
    return selected


def scan_atlas_multiresolution(
    atlas: PolarAtlas,
    base_scanner: JAXConsistentGeometryBatchScanner,
    full_scanner: JAXConsistentGeometryBatchScanner,
    *,
    base_m_max: int,
    full_m_max: int,
    top_count: int = 500,
    risk_count: int = 500,
    shards_per_call: int = 4,
    trajectory_radii: np.ndarray | None = None,
    full_batch_size: int = 1024,
    anomaly_scorer: AnomalyScorer | None = None,
    anomaly_count: int = 0,
    certified_selection: bool = False,
) -> MultiResolutionScanResult:
    """Fast low-mode pass followed by a targeted high-mode rescue pass."""
    if not 0 <= base_m_max < full_m_max <= atlas.m_max:
        raise ValueError("require 0 <= base_m_max < full_m_max <= atlas.m_max")
    if base_scanner.n_mode != base_m_max + 1:
        raise ValueError("base scanner mode count does not match base_m_max")
    if full_scanner.n_mode != full_m_max + 1:
        raise ValueError("full scanner mode count does not match full_m_max")
    if full_scanner.n_alpha % base_scanner.n_alpha:
        raise ValueError("full n_alpha must be an integer multiple of base n_alpha")
    if shards_per_call < 1 or full_batch_size < 1:
        raise ValueError("batch sizes must be positive")

    map_ids_parts = []
    parameter_parts = []
    chi2_parts = []
    lower_parts = []
    upper_parts = []
    geometry_parts = []
    alpha_parts = []
    risk_parts = []
    anomaly_parts = []
    base_started = time.perf_counter()
    for shards in _grouped(
        atlas.iter_shards(m_max=full_m_max), shards_per_call
    ):
        coefficients = np.concatenate([shard.x_coeff for shard in shards])
        reconstruction_error = np.concatenate(
            [_certificate_or_reconstruction_error(shard) for shard in shards]
        )
        total_error = total_angular_error_envelope(
            coefficients, base_m_max, reconstruction_error
        )
        (
            local_chi2,
            local_geometry,
            local_alpha,
            local_lower,
            local_upper,
        ) = base_scanner.scan_minima_with_error(
            coefficients[..., : base_m_max + 1], total_error
        )
        if trajectory_radii is None:
            local_risk = np.max(total_error, axis=1)
        else:
            local_risk = interpolated_trajectory_error(
                total_error, atlas.radial_nodes, trajectory_radii
            )
        map_ids_parts.append(
            np.concatenate([np.asarray(shard.map_ids) for shard in shards])
        )
        parameter_parts.append(
            np.concatenate([np.asarray(shard.parameters) for shard in shards])
        )
        chi2_parts.append(local_chi2)
        lower_parts.append(local_lower)
        upper_parts.append(local_upper)
        geometry_parts.append(local_geometry)
        alpha_parts.append(local_alpha)
        risk_parts.append(local_risk)
        if anomaly_scorer is None:
            anomaly_parts.append(np.zeros(len(coefficients), dtype=np.float64))
        else:
            local_scores = [
                np.asarray(
                    anomaly_scorer(np.asarray(atlas.radial_nodes), shard),
                    dtype=np.float64,
                )
                for shard in shards
            ]
            if any(score.shape != (len(shard.map_ids),) for score, shard in zip(local_scores, shards)):
                raise ValueError("anomaly_scorer must return one score per map")
            anomaly_parts.append(np.concatenate(local_scores))

    map_ids = np.concatenate(map_ids_parts)
    parameters = np.concatenate(parameter_parts)
    chi2 = np.concatenate(chi2_parts)
    chi2_lower = np.concatenate(lower_parts)
    chi2_upper = np.concatenate(upper_parts)
    geometry_index = np.concatenate(geometry_parts)
    alpha_index = np.concatenate(alpha_parts)
    spectral_risk = np.concatenate(risk_parts)
    anomaly_score = np.concatenate(anomaly_parts)
    alpha_index *= full_scanner.n_alpha // base_scanner.n_alpha
    base_seconds = time.perf_counter() - base_started

    ordinary_rows = (
        select_certified_rescan_indices(
            chi2_lower,
            chi2_upper,
            spectral_risk,
            target_count=top_count,
            risk_count=risk_count,
        )
        if certified_selection
        else select_adaptive_rescan_indices(
            chi2,
            spectral_risk,
            top_count=top_count,
            risk_count=risk_count,
        )
    )
    selected_rows = _rescan_union(
        chi2,
        spectral_risk,
        top_count=top_count,
        risk_count=risk_count,
        anomaly_score=anomaly_score,
        anomaly_count=anomaly_count,
    )
    if certified_selection:
        selected_rows = np.unique(np.concatenate((selected_rows, ordinary_rows)))
    selected_ids = set(map(int, map_ids[selected_rows]))
    row_for_id = {int(map_id): row for row, map_id in enumerate(map_ids)}
    coefficient_parts = []
    error_parts = []
    row_parts = []
    for shard in atlas.iter_shards(m_max=full_m_max):
        local_mask = np.fromiter(
            (int(map_id) in selected_ids for map_id in shard.map_ids),
            dtype=bool,
            count=len(shard.map_ids),
        )
        if np.any(local_mask):
            coefficient_parts.append(np.asarray(shard.x_coeff[local_mask]))
            error_parts.append(
                _certificate_or_reconstruction_error(shard)[local_mask]
            )
            row_parts.append(
                np.asarray(
                    [row_for_id[int(map_id)] for map_id in shard.map_ids[local_mask]],
                    dtype=np.int64,
                )
            )
    selected_coefficients = np.concatenate(coefficient_parts)
    selected_errors = np.concatenate(error_parts)
    selected_global_rows = np.concatenate(row_parts)

    full_started = time.perf_counter()
    for start in range(0, len(selected_coefficients), full_batch_size):
        stop = min(start + full_batch_size, len(selected_coefficients))
        (
            local_chi2,
            local_geometry,
            local_alpha,
            local_lower,
            local_upper,
        ) = full_scanner.scan_minima_with_error(
            selected_coefficients[start:stop], selected_errors[start:stop]
        )
        rows = selected_global_rows[start:stop]
        chi2[rows] = local_chi2
        chi2_lower[rows] = local_lower
        chi2_upper[rows] = local_upper
        geometry_index[rows] = local_geometry
        alpha_index[rows] = local_alpha
    full_seconds = time.perf_counter() - full_started
    rescanned = np.zeros(len(map_ids), dtype=bool)
    rescanned[selected_global_rows] = True
    anomaly_rescued = np.zeros(len(map_ids), dtype=bool)
    anomaly_rescued[np.setdiff1d(selected_rows, ordinary_rows)] = True

    return MultiResolutionScanResult(
        map_ids=map_ids,
        parameters=parameters,
        chi2=chi2,
        geometry_index=geometry_index,
        alpha_index=alpha_index,
        spectral_risk=spectral_risk,
        rescanned=rescanned,
        base_seconds=base_seconds,
        full_seconds=full_seconds,
        chi2_lower=chi2_lower,
        chi2_upper=chi2_upper,
        anomaly_score=anomaly_score,
        anomaly_rescued=anomaly_rescued,
    )


def scan_bucketed_atlas_multiresolution(
    atlas: BucketedPolarAtlas,
    kernel_factory: Callable[
        [np.ndarray, int], Sequence[Sequence[ConsistentEventKernel]]
    ],
    *,
    base_m_max: int,
    full_m_max: int,
    base_n_alpha: int,
    full_n_alpha: int,
    top_count: int = 500,
    risk_count: int = 500,
    shards_per_call: int = 4,
    trajectory_radii: np.ndarray | None = None,
    full_batch_size: int = 1024,
    anomaly_scorer: AnomalyScorer | None = None,
    anomaly_count: int = 0,
    anomaly_mask_builder: AnomalyMaskBuilder | None = None,
    anomaly_mask_mode: str = "rank",
    map_filter: MapFilter | None = None,
    base_compute_dtype: str = "float64",
    base_radial_order: int | None = None,
    certified_selection: bool = False,
) -> MultiResolutionScanResult:
    """Globally rank a bucketed atlas while reusing two compiled JAX scanners.

    Every bucket owns different radial nodes, so its event kernels differ.
    Their shapes are identical; only dynamic JAX arguments are replaced after
    the first bucket.  The low-mode ranking and rescue selection remain global,
    rather than allocating ``top_count`` independently to every bucket.

    ``base_compute_dtype`` and ``base_radial_order`` only affect the ranking
    pass, which selects the maps to rescan; the reported chi-square of every
    rescanned map still comes from the full-mode pass.  A cheaper ranking pass
    is worthwhile because a higher radial order costs one extra product
    contraction per stencil node, while the selection it feeds only needs the
    maps in roughly the right order.

    ``anomaly_mask_builder`` turns the observed anomaly into a hard
    ``(map, geometry, alpha)`` feasibility cube, so the search stops spending
    its rescan budget on maps whose best-fitting trajectory angle cannot produce
    the observed planetary signal at all.  Under ``anomaly_mask_mode="rank"``
    the mask only decides *which* maps are rescanned and every reported
    chi-square keeps its unrestricted meaning; under ``"full"`` the reported
    minimum is itself restricted to anomaly-compatible angles.  The
    ``risk_count`` spectral-risk selection stays unmasked either way, which
    keeps a rescan path open for maps the low-mode spectrum describes poorly.

    ``map_filter`` decides from the grid parameters alone which maps are worth
    computing at all.  Unlike the anomaly mask, which reshapes a chi-square cube
    that has already been built, this removes maps before their coefficients are
    read, so it is the only hook here that makes the base pass cheaper rather
    than better targeted.  Skipped maps stay in every returned array with an
    infinite chi-square, a zero spectral risk, and ``scanned=False``, so callers
    keep a complete map list and cannot mistake an unscanned map for a bad fit.
    """
    if not 0 <= base_m_max < full_m_max <= atlas.m_max:
        raise ValueError("require 0 <= base_m_max < full_m_max <= atlas.m_max")
    if full_n_alpha % base_n_alpha:
        raise ValueError("full n_alpha must be an integer multiple of base n_alpha")
    if shards_per_call < 1 or full_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if anomaly_mask_mode not in ANOMALY_MASK_MODES:
        raise ValueError(f"anomaly_mask_mode must be one of {ANOMALY_MASK_MODES}")

    map_ids_parts: list[np.ndarray] = []
    parameter_parts: list[np.ndarray] = []
    chi2_parts: list[np.ndarray] = []
    lower_parts: list[np.ndarray] = []
    upper_parts: list[np.ndarray] = []
    geometry_parts: list[np.ndarray] = []
    alpha_parts: list[np.ndarray] = []
    risk_parts: list[np.ndarray] = []
    anomaly_parts: list[np.ndarray] = []
    masked_chi2_parts: list[np.ndarray] = []
    scanned_parts: list[np.ndarray] = []
    base_scanner: JAXConsistentGeometryBatchScanner | None = None
    full_scanner: JAXConsistentGeometryBatchScanner | None = None
    base_started = time.perf_counter()

    def base_kernel_batches(radial_nodes: np.ndarray):
        if base_radial_order is None:
            return kernel_factory(radial_nodes, base_m_max)
        return kernel_factory(radial_nodes, base_m_max, base_radial_order)

    for bucket in atlas.iter_buckets():
        radial_nodes = np.asarray(bucket.atlas.radial_nodes)
        base_kernels = base_kernel_batches(radial_nodes)
        if base_scanner is None:
            full_kernels = kernel_factory(radial_nodes, full_m_max)
            base_scanner = JAXConsistentGeometryBatchScanner(
                base_kernels,
                n_alpha=base_n_alpha,
                compute_dtype=base_compute_dtype,
            )
            full_scanner = JAXConsistentGeometryBatchScanner(
                full_kernels, n_alpha=full_n_alpha
            )
        else:
            base_scanner.set_kernels(base_kernels)

        for shards in _grouped(
            bucket.atlas.iter_shards(m_max=full_m_max), shards_per_call
        ):
            local_ids = np.concatenate(
                [np.asarray(shard.map_ids) for shard in shards]
            )
            local_parameters = np.concatenate(
                [np.asarray(shard.parameters) for shard in shards]
            )
            keep = _resolve_map_filter(map_filter, local_parameters)
            count = len(local_ids)
            local_chi2 = np.full(count, np.inf, dtype=np.float64)
            local_lower = np.full(count, np.inf, dtype=np.float64)
            local_upper = np.full(count, np.inf, dtype=np.float64)
            local_masked_chi2 = np.full(count, np.inf, dtype=np.float64)
            local_geometry = np.zeros(count, dtype=np.int64)
            local_alpha = np.zeros(count, dtype=np.int64)
            # A skipped map must not reach the rescan set through the
            # spectral-risk or anomaly-rescue valves either: it was ruled out on
            # physical grounds, not because its low-mode spectrum was unclear.
            local_risk = np.zeros(count, dtype=np.float64)
            local_anomaly = np.zeros(count, dtype=np.float64)

            if keep.any():
                if keep.all():
                    coefficients = np.concatenate(
                        [shard.x_coeff for shard in shards]
                    )
                    reconstruction_error = np.concatenate(
                        [
                            _certificate_or_reconstruction_error(shard)
                            for shard in shards
                        ]
                    )
                else:
                    # Index each memory-mapped shard separately, so a pruned rho
                    # slice is never read off disk in the first place.
                    bounds = np.cumsum([0] + [len(s.map_ids) for s in shards])
                    coefficients = np.concatenate(
                        [
                            np.asarray(shard.x_coeff[keep[start:stop]])
                            for shard, start, stop in zip(
                                shards, bounds[:-1], bounds[1:]
                            )
                        ]
                    )
                    reconstruction_error = np.concatenate(
                        [
                            _certificate_or_reconstruction_error(shard)[
                                keep[start:stop]
                            ]
                            for shard, start, stop in zip(
                                shards, bounds[:-1], bounds[1:]
                            )
                        ]
                    )
                total_error = total_angular_error_envelope(
                    coefficients, base_m_max, reconstruction_error
                )
                padded, kept = _padded_scan_batch(coefficients)
                padded_error = _pad_error_envelope(total_error, len(padded))
                if anomaly_mask_builder is None:
                    (
                        kept_chi2,
                        kept_geometry,
                        kept_alpha,
                        kept_lower,
                        kept_upper,
                    ) = base_scanner.scan_minima_with_error(
                        padded[..., : base_m_max + 1], padded_error
                    )
                    kept_masked_chi2 = kept_chi2
                else:
                    (
                        kept_chi2,
                        kept_geometry,
                        kept_alpha,
                        kept_masked_chi2,
                        masked_geometry,
                        masked_alpha,
                        unrestricted_lower,
                        unrestricted_upper,
                        masked_lower,
                        masked_upper,
                    ) = base_scanner.scan_minima_with_error(
                        padded[..., : base_m_max + 1],
                        padded_error,
                        anomaly_mask_builder(radial_nodes, padded, base_n_alpha),
                    )
                    kept_lower = unrestricted_lower
                    kept_upper = unrestricted_upper
                    if anomaly_mask_mode == "full":
                        kept_chi2 = kept_masked_chi2
                        kept_geometry = masked_geometry
                        kept_alpha = masked_alpha
                        kept_lower = masked_lower
                        kept_upper = masked_upper
                local_chi2[keep] = kept_chi2[:kept]
                local_lower[keep] = kept_lower[:kept]
                local_upper[keep] = kept_upper[:kept]
                local_masked_chi2[keep] = kept_masked_chi2[:kept]
                local_geometry[keep] = kept_geometry[:kept]
                local_alpha[keep] = kept_alpha[:kept]
                if trajectory_radii is None:
                    local_risk[keep] = np.max(total_error, axis=1)
                else:
                    local_risk[keep] = interpolated_trajectory_error(
                        total_error, radial_nodes, trajectory_radii
                    )
                if anomaly_scorer is not None:
                    local_scores = [
                        np.asarray(
                            anomaly_scorer(radial_nodes, shard), dtype=np.float64
                        )
                        for shard in shards
                    ]
                    if any(
                        score.shape != (len(shard.map_ids),)
                        for score, shard in zip(local_scores, shards)
                    ):
                        raise ValueError(
                            "anomaly_scorer must return one score per map"
                        )
                    local_anomaly = np.where(
                        keep, np.concatenate(local_scores), 0.0
                    )

            map_ids_parts.append(local_ids)
            parameter_parts.append(local_parameters)
            chi2_parts.append(local_chi2)
            lower_parts.append(local_lower)
            upper_parts.append(local_upper)
            geometry_parts.append(local_geometry)
            alpha_parts.append(local_alpha)
            risk_parts.append(local_risk)
            masked_chi2_parts.append(local_masked_chi2)
            anomaly_parts.append(local_anomaly)
            scanned_parts.append(keep)

    assert base_scanner is not None and full_scanner is not None
    map_ids = np.concatenate(map_ids_parts)
    parameters = np.concatenate(parameter_parts)
    chi2 = np.concatenate(chi2_parts)
    chi2_lower = np.concatenate(lower_parts)
    chi2_upper = np.concatenate(upper_parts)
    geometry_index = np.concatenate(geometry_parts)
    alpha_index = np.concatenate(alpha_parts)
    spectral_risk = np.concatenate(risk_parts)
    anomaly_score = np.concatenate(anomaly_parts)
    masked_chi2 = np.concatenate(masked_chi2_parts)
    scanned = np.concatenate(scanned_parts)
    alpha_index *= full_n_alpha // base_n_alpha
    base_seconds = time.perf_counter() - base_started

    ordinary_rows = (
        select_certified_rescan_indices(
            chi2_lower,
            chi2_upper,
            spectral_risk,
            target_count=top_count,
            risk_count=risk_count,
        )
        if certified_selection
        else select_adaptive_rescan_indices(
            chi2,
            spectral_risk,
            top_count=top_count,
            risk_count=risk_count,
        )
    )
    # Ranking on the anomaly-compatible minimum is the whole point of the mask:
    # a map that only fits well at an angle incompatible with the observed
    # signal drops out of the top list instead of consuming a rescan slot.
    selected_rows = _rescan_union(
        masked_chi2,
        spectral_risk,
        top_count=top_count,
        risk_count=risk_count,
        anomaly_score=anomaly_score,
        anomaly_count=anomaly_count,
    )
    if certified_selection:
        selected_rows = np.unique(np.concatenate((selected_rows, ordinary_rows)))
    if not scanned.all():
        # Both selections rank a fixed number of rows, so a scan small enough to
        # run out of real candidates would otherwise promote a skipped map and
        # hand it a full-resolution chi-square it never earned.
        ordinary_rows = ordinary_rows[scanned[ordinary_rows]]
        selected_rows = selected_rows[scanned[selected_rows]]
    selected_ids = set(map(int, map_ids[selected_rows]))
    row_for_id = {int(map_id): row for row, map_id in enumerate(map_ids)}
    full_started = time.perf_counter()
    selected_global_rows_parts: list[np.ndarray] = []

    for bucket in atlas.iter_buckets():
        bucket_ids = set(map(int, bucket.map_ids))
        local_selected = selected_ids.intersection(bucket_ids)
        if not local_selected:
            continue
        radial_nodes = np.asarray(bucket.atlas.radial_nodes)
        full_scanner.set_kernels(kernel_factory(radial_nodes, full_m_max))
        coefficient_parts: list[np.ndarray] = []
        error_parts: list[np.ndarray] = []
        row_parts: list[np.ndarray] = []
        for shard in bucket.atlas.iter_shards(m_max=full_m_max):
            local_mask = np.fromiter(
                (int(map_id) in local_selected for map_id in shard.map_ids),
                dtype=bool,
                count=len(shard.map_ids),
            )
            if np.any(local_mask):
                coefficient_parts.append(np.asarray(shard.x_coeff[local_mask]))
                error_parts.append(
                    _certificate_or_reconstruction_error(shard)[local_mask]
                )
                row_parts.append(
                    np.asarray(
                        [
                            row_for_id[int(map_id)]
                            for map_id in shard.map_ids[local_mask]
                        ],
                        dtype=np.int64,
                    )
                )
        selected_coefficients = np.concatenate(coefficient_parts)
        selected_errors = np.concatenate(error_parts)
        selected_global_rows = np.concatenate(row_parts)
        selected_global_rows_parts.append(selected_global_rows)
        restrict_full = (
            anomaly_mask_builder is not None and anomaly_mask_mode == "full"
        )
        for start in range(0, len(selected_coefficients), full_batch_size):
            stop = min(start + full_batch_size, len(selected_coefficients))
            batch = selected_coefficients[start:stop]
            batch_error = selected_errors[start:stop]
            if restrict_full:
                # Rebuilt at the full alpha resolution: the rescan refines the
                # angle, so a mask sampled on the coarse grid would not line up.
                result = full_scanner.scan_minima_with_error(
                    batch,
                    batch_error,
                    anomaly_mask_builder(radial_nodes, batch, full_n_alpha),
                )
                local_chi2, local_geometry, local_alpha = result[3:6]
                local_lower, local_upper = result[8:10]
            else:
                (
                    local_chi2,
                    local_geometry,
                    local_alpha,
                    local_lower,
                    local_upper,
                ) = full_scanner.scan_minima_with_error(
                    batch, batch_error
                )
            rows = selected_global_rows[start:stop]
            chi2[rows] = local_chi2
            chi2_lower[rows] = local_lower
            chi2_upper[rows] = local_upper
            geometry_index[rows] = local_geometry
            alpha_index[rows] = local_alpha

    full_seconds = time.perf_counter() - full_started
    rescanned = np.zeros(len(map_ids), dtype=bool)
    if selected_global_rows_parts:
        rescanned[np.concatenate(selected_global_rows_parts)] = True
    anomaly_rescued = np.zeros(len(map_ids), dtype=bool)
    anomaly_rescued[np.setdiff1d(selected_rows, ordinary_rows)] = True
    return MultiResolutionScanResult(
        map_ids=map_ids,
        parameters=parameters,
        chi2=chi2,
        geometry_index=geometry_index,
        alpha_index=alpha_index,
        spectral_risk=spectral_risk,
        rescanned=rescanned,
        base_seconds=base_seconds,
        full_seconds=full_seconds,
        chi2_lower=chi2_lower,
        chi2_upper=chi2_upper,
        anomaly_score=anomaly_score,
        anomaly_rescued=anomaly_rescued,
        anomaly_masked_chi2=(
            None if anomaly_mask_builder is None else masked_chi2
        ),
        scanned=(None if map_filter is None else scanned),
    )
