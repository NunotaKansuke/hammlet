"""Deterministic candidate selection for the Roman FFT fallback.

The fallback intentionally uses only observed-data scores and atlas
coordinates.  Injected parameters never enter candidate selection.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def candidate_key(candidate: dict) -> tuple[int, int, int]:
    """Return the discrete identity of an FFT trajectory seed."""
    return (
        int(candidate["map_id"]),
        int(candidate["geometry_index"]),
        int(candidate["alpha_index"]),
    )


def _grid_steps(candidates: list[dict]) -> np.ndarray:
    steps = []
    for field, fallback in (("logs", 0.05), ("logq", 0.1), ("logrho", 0.3)):
        values = np.unique([float(candidate[field]) for candidate in candidates])
        differences = np.diff(values)
        positive = differences[differences > 1.0e-12]
        steps.append(float(np.min(positive)) if len(positive) else fallback)
    return np.asarray(steps, dtype=np.float64)


def _grid_vector(candidate: dict, steps: np.ndarray) -> np.ndarray:
    return np.asarray(
        [candidate["logs"], candidate["logq"], candidate["logrho"]],
        dtype=np.float64,
    ) / steps


def select_fft_seeds(
    candidates: Iterable[dict],
    *,
    elite_per_geometry: int = 12,
    diverse_per_geometry: int = 4,
    minimum_grid_distance: float = 2.0,
) -> list[dict]:
    """Keep low-FFT seeds plus spatial coverage independently per geometry.

    The elite quota protects narrow minima.  The greedy non-maximum-
    suppression quota protects distinct broad valleys without requiring a
    fragile watershed or a prescribed number of clusters.
    """
    rows = [dict(candidate) for candidate in candidates]
    if elite_per_geometry < 0 or diverse_per_geometry < 0:
        raise ValueError("seed quotas must be non-negative")
    if elite_per_geometry + diverse_per_geometry < 1:
        raise ValueError("at least one seed per geometry must be requested")
    if minimum_grid_distance < 0.0:
        raise ValueError("minimum_grid_distance must be non-negative")
    if not rows:
        return []

    selected: list[dict] = []
    for geometry_index in sorted({int(row["geometry_index"]) for row in rows}):
        group = [
            row for row in rows if int(row["geometry_index"]) == geometry_index
        ]
        group.sort(
            key=lambda row: (
                float(row["fft_chi2"]),
                int(row["map_id"]),
                int(row["alpha_index"]),
            )
        )
        steps = _grid_steps(group)
        elite = group[: min(int(elite_per_geometry), len(group))]
        accepted = list(elite)
        accepted_keys = {candidate_key(row) for row in accepted}
        accepted_vectors = [_grid_vector(row, steps) for row in accepted]

        diverse_added = 0
        for row in group:
            if diverse_added >= int(diverse_per_geometry):
                break
            if candidate_key(row) in accepted_keys:
                continue
            vector = _grid_vector(row, steps)
            if accepted_vectors:
                distance = min(
                    float(np.linalg.norm(vector - other))
                    for other in accepted_vectors
                )
                if distance < float(minimum_grid_distance):
                    continue
            accepted.append(row)
            accepted_keys.add(candidate_key(row))
            accepted_vectors.append(vector)
            diverse_added += 1

        # Sparse or compact FFT regions may not satisfy the requested
        # separation.  Fill deterministically so every geometry has the same
        # maximum budget and no branch disappears accidentally.
        target = min(
            int(elite_per_geometry) + int(diverse_per_geometry), len(group)
        )
        for row in group:
            if len(accepted) >= target:
                break
            if candidate_key(row) not in accepted_keys:
                accepted.append(row)
                accepted_keys.add(candidate_key(row))
        selected.extend(accepted)

    selected.sort(
        key=lambda row: (
            int(row["geometry_index"]),
            float(row["fft_chi2"]),
            int(row["map_id"]),
            int(row["alpha_index"]),
        )
    )
    for rank, row in enumerate(selected, start=1):
        row["rank_fallback_fft"] = int(rank)
    return selected


def log_improvement(candidate: dict) -> float:
    """Return a scale-free LM improvement score."""
    start = float(candidate["start_direct_vbm_chi2"])
    final = float(candidate["direct_vbm_chi2"])
    if not (math.isfinite(start) and math.isfinite(final)) or start <= 0.0:
        return -math.inf
    return math.log(max(start, np.finfo(float).tiny) / max(final, np.finfo(float).tiny))


def select_continuations(
    short_results: Iterable[dict],
    *,
    best_count: int = 8,
    improvement_count: int = 8,
    one_improvement_per_geometry: bool = True,
) -> list[dict]:
    """Select LM continuations by fit quality and basin-entry evidence."""
    rows = [dict(candidate) for candidate in short_results]
    if best_count < 0 or improvement_count < 0:
        raise ValueError("continuation counts must be non-negative")
    if best_count + improvement_count < 1 and not one_improvement_per_geometry:
        raise ValueError("at least one continuation must be requested")
    if not rows:
        return []

    by_final = sorted(
        rows,
        key=lambda row: (
            float(row["direct_vbm_chi2"]),
            candidate_key(row),
        ),
    )
    by_gain = sorted(
        rows,
        key=lambda row: (
            -log_improvement(row),
            float(row["direct_vbm_chi2"]),
            candidate_key(row),
        ),
    )
    selected: list[dict] = []
    selected_keys: set[tuple[int, int, int]] = set()

    def add(row: dict, reason: str) -> None:
        key = candidate_key(row)
        if key in selected_keys:
            for existing in selected:
                if candidate_key(existing) == key:
                    reasons = existing.setdefault("continuation_reasons", [])
                    if reason not in reasons:
                        reasons.append(reason)
                    break
            return
        result = dict(row)
        result["short_log_improvement"] = float(log_improvement(row))
        result["continuation_reasons"] = [reason]
        selected.append(result)
        selected_keys.add(key)

    for row in by_final[: int(best_count)]:
        add(row, "best-final-chi2")
    for row in by_gain[: int(improvement_count)]:
        add(row, "largest-log-improvement")
    if one_improvement_per_geometry:
        for geometry_index in sorted(
            {int(row["geometry_index"]) for row in rows}
        ):
            geometry_rows = [
                row
                for row in by_gain
                if int(row["geometry_index"]) == geometry_index
            ]
            if geometry_rows:
                add(geometry_rows[0], "best-improvement-in-geometry")

    selected.sort(
        key=lambda row: (
            float(row["direct_vbm_chi2"]),
            candidate_key(row),
        )
    )
    for rank, row in enumerate(selected, start=1):
        row["rank_continuation"] = int(rank)
    return selected
