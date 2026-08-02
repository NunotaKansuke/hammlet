from __future__ import annotations

import numpy as np
import pytest

from hammlet._core.adaptive_grid import (
    angular_tail_envelope,
    equidistribute_radial_nodes,
    interpolated_trajectory_error,
    select_adaptive_rescan_indices,
    select_certified_rescan_indices,
    select_interval_rescan_indices,
    sharp_qs_pilot_indices,
    spectral_radial_difficulty,
    stratified_pilot_indices,
    total_angular_error_envelope,
)


class RadialFeatureMap:
    def __init__(self, center: float, width: float = 0.12) -> None:
        self.center = center
        self.width = width

    def magnification(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        radius = np.hypot(x, y)
        feature = np.exp(-0.5 * ((radius - self.center) / self.width) ** 2)
        return 1.0 + feature * (1.0 + 0.25 * np.cos(np.arctan2(y, x)))


def test_equidistribution_moves_fixed_budget_to_outer_feature() -> None:
    pilot = np.linspace(0.0, 8.0, 801)
    difficulty = np.exp(-0.5 * ((pilot - 6.0) / 0.3) ** 2)
    nodes = equidistribute_radial_nodes(pilot, difficulty, 81)
    near_feature = np.count_nonzero((nodes > 5.5) & (nodes < 6.5))
    uniform = np.linspace(0.0, 8.0, 81)
    uniform_near_feature = np.count_nonzero((uniform > 5.5) & (uniform < 6.5))
    assert near_feature > 3 * uniform_near_feature
    assert nodes[0] == 0.0
    assert nodes[-1] == 8.0


def test_spectral_difficulty_detects_outer_radial_structure() -> None:
    radii = np.linspace(0.0, 8.0, 401)
    difficulty = spectral_radial_difficulty(
        [RadialFeatureMap(6.0)],
        radii,
        n_phi=32,
        m_max=4,
    )
    peak_radius = radii[np.argmax(difficulty)]
    assert peak_radius == pytest.approx(6.0, abs=0.25)


def test_interpolated_trajectory_error_returns_per_map_maximum() -> None:
    nodes = np.asarray([0.0, 1.0, 3.0])
    errors = np.asarray([[0.0, 2.0, 0.0], [1.0, 1.0, 5.0]])
    actual = interpolated_trajectory_error(errors, nodes, [0.5, 2.0])
    np.testing.assert_allclose(actual, [1.0, 3.0])


def test_tail_envelope_and_adaptive_selection_rescue_risky_map() -> None:
    coefficients = np.zeros((3, 2, 5), dtype=np.complex128)
    coefficients[2, :, 3] = 4.0
    envelope = angular_tail_envelope(coefficients, base_m_max=2)
    np.testing.assert_allclose(envelope[:, 0], [0.0, 0.0, 8.0])
    selected = select_adaptive_rescan_indices(
        np.asarray([1.0, 2.0, 100.0]),
        np.max(envelope, axis=1),
        top_count=1,
        risk_count=1,
    )
    np.testing.assert_array_equal(selected, [0, 2])


def test_total_angular_error_includes_unstored_atlas_residual() -> None:
    coefficients = np.zeros((2, 3, 4), dtype=np.complex128)
    coefficients[0, :, 2] = 0.5
    coefficients[1, :, 3] = 0.25j
    residual = np.asarray(
        [[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]], dtype=np.float64
    )

    actual = total_angular_error_envelope(coefficients, 1, residual)

    np.testing.assert_allclose(actual[0], [1.1, 1.2, 1.3])
    np.testing.assert_allclose(actual[1], [1.5, 2.5, 3.5])


def test_total_angular_error_rejects_invalid_residuals() -> None:
    coefficients = np.zeros((1, 2, 3), dtype=np.complex128)
    with pytest.raises(ValueError, match="match"):
        total_angular_error_envelope(coefficients, 1, np.zeros((1, 3)))
    with pytest.raises(ValueError, match="non-negative"):
        total_angular_error_envelope(coefficients, 1, -np.ones((1, 2)))


def test_interval_selection_retains_every_overlapping_candidate() -> None:
    lower = np.asarray([0.0, 2.0, 7.0, 50.0])
    upper = np.asarray([1.0, 10.0, 8.0, 60.0])

    selected = select_interval_rescan_indices(lower, upper, target_count=2)

    np.testing.assert_array_equal(selected, np.asarray([0, 1, 2]))


def test_certified_selection_reports_m_dependent_extra_check_cost() -> None:
    central = np.asarray([1.0, 2.0, 3.0, 4.0, 20.0])
    risk = np.zeros_like(central)
    wide = select_certified_rescan_indices(
        np.maximum(central - 5.0, 0.0),
        central + 5.0,
        risk,
        target_count=2,
        risk_count=0,
    )
    narrow = select_certified_rescan_indices(
        np.maximum(central - 0.1, 0.0),
        central + 0.1,
        risk,
        target_count=2,
        risk_count=0,
    )

    # A low-M certificate needs two ambiguity checks beyond the desired top 2;
    # once M narrows the intervals, only those two target candidates remain.
    np.testing.assert_array_equal(wide, np.asarray([0, 1, 2, 3]))
    np.testing.assert_array_equal(narrow, np.asarray([0, 1]))


def test_stratified_pilots_cover_parameter_cube_extremes() -> None:
    coordinates = np.asarray(
        np.meshgrid([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    ).reshape(3, -1).T
    parameters = np.vstack((np.full((1, 3), 0.5), coordinates))
    selected = stratified_pilot_indices(parameters, 9)
    assert selected[0] == 0
    assert set(selected[1:]) == set(range(1, 9))


def test_sharp_qs_pilots_choose_minimum_rho_per_selected_cell() -> None:
    parameters = np.asarray(
        [
            [0.0, -3.0, -4.0],
            [0.0, -3.0, -2.0],
            [0.5, -2.0, -3.7],
            [0.5, -2.0, -1.6],
        ]
    )
    selected = sharp_qs_pilot_indices(parameters, 2)
    assert set(selected) == {0, 2}
