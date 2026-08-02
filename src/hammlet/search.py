"""Short public API for multi-resolution alpha-FFT seed search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .maps import Maps
from .config import SearchConfig
from ._core.config import PSPLGeometry
from ._core.jax_backend import JAXConsistentGeometryBatchScanner
from ._core.jax_seed_refine import (
    JAXFixedAlphaBatchEvaluator,
    batched_pairwise_refine,
    batched_pattern_refine,
    evaluate_in_fixed_batches,
)
from ._core.multires import (
    scan_atlas_multiresolution,
    scan_bucketed_atlas_multiresolution,
)
from ._core.trajectory import (
    ConsistentKernelFactory,
    PhotometricDataset,
    build_consistent_event_kernel,
)


Dataset = PhotometricDataset
Geometry = PSPLGeometry


@dataclass(frozen=True)
class Candidate:
    start_map_id: int
    map_id: int
    s: float
    q: float
    rho: float
    alpha: float
    geometry_index: int
    chi2: float
    chi2_lower: float
    chi2_upper: float
    spectral_risk: float
    t0: float
    u0: float
    tE: float
    fft_chi2: float
    refined: bool
    map_changed: bool


@dataclass(frozen=True)
class SearchResult:
    candidates: tuple[Candidate, ...]
    base_seconds: float
    full_seconds: float
    maps_scanned: int
    maps_rescanned: int
    refinement_seconds: float = 0.0


def _kernels(datasets, geometries, nodes, m_max, radial_order):
    return [
        [
            build_consistent_event_kernel(
                dataset, geometry, nodes, m_max, radial_order=radial_order
            )
            for dataset in datasets
        ]
        for geometry in geometries
    ]


def _coefficient_rows(maps: Maps, map_ids, m_max):
    if maps._bucketed:
        return maps._core.coefficient_rows(map_ids, m_max=m_max)
    loaded = maps._core.coefficient_rows(map_ids, m_max=m_max)
    nodes = np.repeat(
        np.asarray(maps._core.radial_nodes)[None, :], len(map_ids), axis=0
    )
    return (
        nodes,
        np.stack([loaded[int(map_id)][0] for map_id in map_ids]),
        np.stack([loaded[int(map_id)][1] for map_id in map_ids]),
    )


def _refinement_bounds(records, geometries, config):
    geometry_values = np.asarray(
        [(item.t0, item.u0, np.log(item.tE)) for item in geometries], dtype=float
    )
    fallback = np.asarray(
        (
            config.refine_t0_half_width,
            config.refine_u0_half_width,
            config.refine_log_tE_half_width,
        )
    )
    initial, bounds = [], []
    alpha_width = config.cluster_alpha_bins * 2.0 * np.pi / config.full_n_alpha
    for record in records:
        geometry = geometries[int(record.geometry_index)]
        center = np.asarray((geometry.t0, geometry.u0, np.log(geometry.tE)), float)
        widths = fallback.copy()
        for axis in range(3):
            distance = np.abs(geometry_values[:, axis] - center[axis])
            distance = distance[distance > 64.0 * np.finfo(float).eps]
            if len(distance):
                widths[axis] = np.min(distance)
        row = np.asarray((center[0], center[1], center[2], record.alpha))
        half_width = np.concatenate((widths, [alpha_width]))
        initial.append(row)
        bounds.append(np.stack((row - half_width, row + half_width), axis=-1))
    return np.asarray(initial), np.asarray(bounds)


def _refine_candidates(maps, datasets, geometries, records, config):
    import time

    if not records or not config.refine:
        return None, 0.0
    started = time.perf_counter()
    mode_budget = min(config.handoff_m_max, maps.m_max)
    map_ids = np.asarray([item.map_id for item in records], dtype=np.int64)
    nodes, map_parameters, coefficients = _coefficient_rows(
        maps, map_ids, mode_budget
    )
    initial, bounds = _refinement_bounds(records, geometries, config)
    evaluator = JAXFixedAlphaBatchEvaluator(
        datasets,
        m_max=mode_budget,
        radial_order=config.full_radial_order,
    )
    shallow = batched_pattern_refine(
        evaluator,
        coefficients,
        nodes,
        initial,
        bounds,
        levels=config.refine_shallow_levels,
        batch_size=config.refine_batch_size,
    )
    final_parameters = shallow.parameters.copy()
    final_chi2 = shallow.chi2.copy()
    final_ids = map_ids.copy()
    final_grid = map_parameters.copy()
    deep_count = min(config.refine_deep_count, len(records))
    deep_indices = np.argsort(final_chi2, kind="stable")[:deep_count]

    if deep_count:
        trial_ids, parents = [], []
        for parent in deep_indices:
            local = [(int(final_ids[parent]), final_grid[parent])]
            if config.refine_neighbor_limit:
                local.extend(
                    maps._core.neighboring_maps(
                        int(final_ids[parent]), config.refine_neighbor_limit
                    )
                )
            for map_id, _ in local:
                trial_ids.append(map_id)
                parents.append(int(parent))
        # At M=512 the complete 32 x 27 neighbourhood would occupy about
        # 0.9 GiB before JAX device buffers. Stream fixed-size batches and keep
        # only the winning map ID for each parent candidate.
        best_trial = {
            int(parent): (np.inf, int(final_ids[parent])) for parent in deep_indices
        }
        for start in range(0, len(trial_ids), config.refine_batch_size):
            stop = min(start + config.refine_batch_size, len(trial_ids))
            local_ids = trial_ids[start:stop]
            local_parents = parents[start:stop]
            local_nodes, _, local_coefficients = _coefficient_rows(
                maps, local_ids, mode_budget
            )
            local_parameters = np.asarray(
                [final_parameters[parent] for parent in local_parents]
            )
            local_chi2 = evaluate_in_fixed_batches(
                evaluator,
                local_coefficients,
                local_nodes,
                local_parameters,
                batch_size=config.refine_batch_size,
            )
            for map_id, parent, chi2 in zip(
                local_ids, local_parents, local_chi2, strict=True
            ):
                if chi2 < best_trial[parent][0]:
                    best_trial[parent] = (float(chi2), int(map_id))
        chosen_ids = np.asarray(
            [best_trial[int(parent)][1] for parent in deep_indices], dtype=np.int64
        )
        deep_nodes, deep_grid, deep_coefficients = _coefficient_rows(
            maps, chosen_ids, mode_budget
        )
        deep_initial = final_parameters[deep_indices]
        deep_bounds = bounds[deep_indices]
        if config.refine_pairwise_levels:
            pairwise = batched_pairwise_refine(
                evaluator,
                deep_coefficients,
                deep_nodes,
                deep_initial,
                deep_bounds,
                levels=config.refine_pairwise_levels,
                batch_size=config.refine_batch_size,
            )
            deep_initial = pairwise.parameters
        deep = batched_pattern_refine(
            evaluator,
            deep_coefficients,
            deep_nodes,
            deep_initial,
            deep_bounds,
            levels=config.refine_deep_levels,
            batch_size=config.refine_batch_size,
        )
        final_parameters[deep_indices] = deep.parameters
        final_chi2[deep_indices] = deep.chi2
        final_ids[deep_indices] = chosen_ids
        final_grid[deep_indices] = deep_grid
    return (
        final_ids,
        final_grid,
        final_parameters,
        final_chi2,
        shallow.initial_chi2,
    ), time.perf_counter() - started


def search(
    maps: Maps,
    datasets: Sequence[Dataset],
    geometries: Sequence[Geometry],
    *,
    config: SearchConfig | None = None,
) -> SearchResult:
    """Search all maps and alpha, returning independent downstream seeds."""
    config = config or SearchConfig()
    datasets, geometries = tuple(datasets), tuple(geometries)
    if not datasets or not geometries:
        raise ValueError("at least one dataset and one geometry are required")
    if config.full_m_max > maps.m_max:
        raise ValueError("full_m_max exceeds the modes stored in these maps")
    if maps._bucketed:
        factory = ConsistentKernelFactory(
            datasets,
            geometries,
            config.full_m_max,
            radial_order=config.full_radial_order,
        )
        raw = scan_bucketed_atlas_multiresolution(
            maps._core,
            factory,
            base_m_max=config.base_m_max,
            full_m_max=config.full_m_max,
            base_n_alpha=config.base_n_alpha,
            full_n_alpha=config.full_n_alpha,
            top_count=config.top_count,
            risk_count=config.risk_count,
            shards_per_call=config.shards_per_call,
            base_radial_order=config.base_radial_order,
            certified_selection=config.certified_selection,
        )
    else:
        nodes = maps.radial_nodes
        base = JAXConsistentGeometryBatchScanner(
            _kernels(
                datasets,
                geometries,
                nodes,
                config.base_m_max,
                config.base_radial_order,
            ),
            n_alpha=config.base_n_alpha,
        )
        full = JAXConsistentGeometryBatchScanner(
            _kernels(
                datasets,
                geometries,
                nodes,
                config.full_m_max,
                config.full_radial_order,
            ),
            n_alpha=config.full_n_alpha,
        )
        raw = scan_atlas_multiresolution(
            maps._core,
            base,
            full,
            base_m_max=config.base_m_max,
            full_m_max=config.full_m_max,
            top_count=config.top_count,
            risk_count=config.risk_count,
            shards_per_call=config.shards_per_call,
            certified_selection=config.certified_selection,
        )
    records = raw.candidates(
        n_alpha=config.full_n_alpha,
        count=None,
        rescanned_only=True,
    )
    clustered = []
    alpha_tolerance = config.cluster_alpha_bins * 2.0 * np.pi / config.full_n_alpha
    for item in records:
        duplicate = False
        for kept in clustered:
            angle_delta = abs(item.alpha - kept.alpha)
            angle_delta = min(angle_delta, 2.0 * np.pi - angle_delta)
            if (
                abs(item.logs - kept.logs) <= config.cluster_log_s
                and abs(item.logq - kept.logq) <= config.cluster_log_q
                and abs(item.logrho - kept.logrho) <= config.cluster_log_rho
                and angle_delta <= alpha_tolerance
            ):
                duplicate = True
                break
        if not duplicate:
            clustered.append(item)
            if len(clustered) == config.candidate_count:
                break
    records = clustered
    refinement, refinement_seconds = _refine_candidates(
        maps, datasets, geometries, records, config
    )
    candidates = []
    for row, item in enumerate(records):
        geometry = geometries[item.geometry_index]
        if refinement is None:
            map_id = item.map_id
            grid_parameters = np.asarray((item.logs, item.logq, item.logrho))
            fit_parameters = np.asarray(
                (geometry.t0, geometry.u0, np.log(geometry.tE), item.alpha)
            )
            chi2 = item.chi2
        else:
            map_id = int(refinement[0][row])
            grid_parameters = refinement[1][row]
            fit_parameters = refinement[2][row]
            chi2 = float(refinement[3][row])
        candidates.append(
            Candidate(
                start_map_id=item.map_id,
                map_id=map_id,
                s=10.0 ** float(grid_parameters[0]),
                q=10.0 ** float(grid_parameters[1]),
                rho=10.0 ** float(grid_parameters[2]),
                alpha=float(fit_parameters[3] % (2.0 * np.pi)),
                geometry_index=item.geometry_index,
                chi2=chi2,
                chi2_lower=float(item.chi2_lower),
                chi2_upper=float(item.chi2_upper),
                spectral_risk=item.spectral_risk,
                t0=float(fit_parameters[0]),
                u0=float(fit_parameters[1]),
                tE=float(np.exp(fit_parameters[2])),
                fft_chi2=item.chi2,
                refined=refinement is not None,
                map_changed=map_id != item.map_id,
            )
        )
    independent = []
    for candidate in sorted(candidates, key=lambda item: item.chi2):
        duplicate = False
        for kept in independent:
            angle_delta = abs(candidate.alpha - kept.alpha)
            angle_delta = min(angle_delta, 2.0 * np.pi - angle_delta)
            if (
                abs(np.log10(candidate.s / kept.s)) <= config.cluster_log_s
                and abs(np.log10(candidate.q / kept.q)) <= config.cluster_log_q
                and abs(np.log10(candidate.rho / kept.rho))
                <= config.cluster_log_rho
                and angle_delta <= alpha_tolerance
            ):
                duplicate = True
                break
        if not duplicate:
            independent.append(candidate)
    candidates = tuple(independent)
    return SearchResult(
        candidates=candidates,
        base_seconds=raw.base_seconds,
        full_seconds=raw.full_seconds,
        maps_scanned=len(raw.map_ids),
        maps_rescanned=int(np.sum(raw.rescanned)),
        refinement_seconds=refinement_seconds,
    )
