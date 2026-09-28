from __future__ import annotations

from tools.roman_local.fft_fallback import (
    candidate_key,
    log_improvement,
    select_continuations,
    select_fft_seeds,
)


def _candidate(map_id, geometry, fft, logs, logq, logrho):
    return {
        "map_id": map_id,
        "geometry_index": geometry,
        "alpha_index": 100 + map_id,
        "fft_chi2": fft,
        "logs": logs,
        "logq": logq,
        "logrho": logrho,
    }


def test_fft_selection_keeps_elite_and_separate_valleys_per_geometry():
    candidates = [
        _candidate(1, 0, 1.0, 0.0, 0.0, 0.0),
        _candidate(2, 0, 2.0, 0.0, 0.1, 0.0),
        _candidate(3, 0, 3.0, 1.0, 1.0, 1.0),
        _candidate(4, 1, 0.5, -1.0, -1.0, -1.0),
        _candidate(5, 1, 0.6, 1.0, 1.0, 1.0),
    ]

    selected = select_fft_seeds(
        candidates,
        elite_per_geometry=1,
        diverse_per_geometry=1,
        minimum_grid_distance=2.0,
    )

    assert {candidate_key(row) for row in selected} == {
        candidate_key(candidates[index]) for index in (0, 2, 3, 4)
    }


def test_fft_selection_does_not_use_direct_or_truth_scores():
    candidates = [
        {
            **_candidate(1, 0, 1.0, 0.0, 0.0, 0.0),
            "direct_vbm_chi2": 1.0e9,
            "truth_distance": 1.0e9,
        },
        {
            **_candidate(2, 0, 2.0, 0.1, 0.1, 0.1),
            "direct_vbm_chi2": 1.0,
            "truth_distance": 0.0,
        },
    ]

    selected = select_fft_seeds(
        candidates,
        elite_per_geometry=1,
        diverse_per_geometry=0,
    )

    assert [row["map_id"] for row in selected] == [1]


def test_continuation_union_keeps_best_fit_and_improving_geometry_branch():
    rows = []
    for map_id, geometry, start, final in (
        (1, 0, 100.0, 10.0),
        (2, 0, 1000.0, 20.0),
        (3, 1, 100.0, 30.0),
        (4, 1, 1000.0, 400.0),
    ):
        rows.append(
            {
                **_candidate(map_id, geometry, float(map_id), 0.0, 0.0, 0.0),
                "start_direct_vbm_chi2": start,
                "direct_vbm_chi2": final,
            }
        )

    selected = select_continuations(
        rows,
        best_count=1,
        improvement_count=1,
        one_improvement_per_geometry=True,
    )

    assert {row["map_id"] for row in selected} == {1, 2, 3}
    assert log_improvement(rows[1]) > log_improvement(rows[0])
    map_two = next(row for row in selected if row["map_id"] == 2)
    assert "largest-log-improvement" in map_two["continuation_reasons"]
    assert "best-improvement-in-geometry" in map_two["continuation_reasons"]


def test_continuation_selection_handles_an_early_stopped_partial_bank():
    rows = [
        {
            **_candidate(1, 3, 1.0, 0.0, 0.0, 0.0),
            "start_direct_vbm_chi2": 100.0,
            "direct_vbm_chi2": 1.0,
        },
        {
            **_candidate(2, 7, 2.0, 1.0, 1.0, 1.0),
            "start_direct_vbm_chi2": 100.0,
            "direct_vbm_chi2": 50.0,
        },
    ]

    selected = select_continuations(
        rows,
        best_count=8,
        improvement_count=8,
        one_improvement_per_geometry=True,
    )

    assert {row["geometry_index"] for row in selected} == {3, 7}
