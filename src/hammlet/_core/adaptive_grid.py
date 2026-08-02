from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .map_adapter import MagnificationEvaluator


def stratified_pilot_indices(
    parameters: np.ndarray,
    count: int,
) -> np.ndarray:
    """Deterministic farthest-point sample of the (logs, logq, logrho) bank."""
    parameters = np.asarray(parameters, dtype=np.float64)
    if parameters.ndim != 2 or parameters.shape[1] != 3 or not np.all(
        np.isfinite(parameters)
    ):
        raise ValueError("parameters must be a finite (map, 3) array")
    if not 1 <= count <= len(parameters):
        raise ValueError("count must be between one and the number of maps")
    lower = np.min(parameters, axis=0)
    span = np.ptp(parameters, axis=0)
    normalized = (parameters - lower) / np.where(span > 0, span, 1.0)
    center = np.full(3, 0.5)
    first = int(np.argmin(np.sum((normalized - center) ** 2, axis=1)))
    selected = [first]
    minimum_distance = np.sum((normalized - normalized[first]) ** 2, axis=1)
    minimum_distance[first] = -np.inf
    while len(selected) < count:
        next_index = int(np.argmax(minimum_distance))
        selected.append(next_index)
        distance = np.sum(
            (normalized - normalized[next_index]) ** 2, axis=1
        )
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[selected] = -np.inf
    return np.asarray(selected, dtype=np.int64)


def sharp_qs_pilot_indices(parameters: np.ndarray, count: int) -> np.ndarray:
    """Sample Q-S space using the smallest-rho (sharpest) map in each cell."""
    parameters = np.asarray(parameters, dtype=np.float64)
    if parameters.ndim != 2 or parameters.shape[1] != 3:
        raise ValueError("parameters must have shape (map, 3)")
    pairs, inverse = np.unique(parameters[:, :2], axis=0, return_inverse=True)
    sharp_rows = np.asarray(
        [
            (rows := np.flatnonzero(inverse == pair_index))[
                np.argmin(parameters[rows, 2])
            ]
            for pair_index in range(len(pairs))
        ],
        dtype=np.int64,
    )
    selected = stratified_pilot_indices(
        np.column_stack((pairs, np.zeros(len(pairs)))),
        min(count, len(pairs)),
    )
    return sharp_rows[selected]


def equidistribute_radial_nodes(
    candidate_radii: np.ndarray,
    difficulty: np.ndarray,
    n_nodes: int,
    *,
    floor_fraction: float = 0.05,
    adaptive_fraction: float = 0.5,
    uniform_baseline_fraction: float = 0.5,
) -> np.ndarray:
    """Redistribute a fixed node budget according to radial difficulty.

    A positive floor keeps smooth regions represented.  Nodes are continuous
    quantiles of the integrated difficulty rather than a subset of the pilot
    grid, so a fine pilot grid does not constrain the final locations.
    """
    radii = np.asarray(candidate_radii, dtype=np.float64)
    difficulty = np.asarray(difficulty, dtype=np.float64)
    if radii.ndim != 1 or radii.size < 3 or np.any(np.diff(radii) <= 0):
        raise ValueError("candidate_radii must be a strictly increasing 1-D array")
    if difficulty.shape != radii.shape:
        raise ValueError("difficulty must have the same shape as candidate_radii")
    if not np.all(np.isfinite(difficulty)) or np.any(difficulty < 0):
        raise ValueError("difficulty must be finite and nonnegative")
    if not 2 <= n_nodes <= radii.size:
        raise ValueError("n_nodes must be between 2 and the pilot-grid size")
    if not 0.0 < floor_fraction <= 1.0:
        raise ValueError("floor_fraction must be in (0, 1]")
    if not 0.0 <= adaptive_fraction < 1.0:
        raise ValueError("adaptive_fraction must be in [0, 1)")
    if not 0.0 <= uniform_baseline_fraction <= 1.0:
        raise ValueError("uniform_baseline_fraction must be in [0, 1]")

    span = radii[-1] - radii[0]
    positive = difficulty[difficulty > 0]
    reference = float(np.median(positive)) if positive.size else 1.0
    signal = difficulty + floor_fraction * reference
    signal_integral = float(np.trapezoid(signal, radii))
    signal_density = signal / signal_integral if signal_integral > 0 else np.full_like(
        radii, 1.0 / span
    )
    uniform_density = np.full_like(radii, 1.0 / span)
    # Preserve the useful central coverage of the historical r=t**2 grid,
    # while mixing in uniform-r mass so no outer interval can consume most of
    # the domain.  The epsilon is its first nonzero quadratic node scale.
    central_scale = span / max((n_nodes - 1) ** 2, 1)
    quadratic_density = 1.0 / np.sqrt(
        np.maximum(radii - radii[0], central_scale)
    )
    quadratic_density /= np.trapezoid(quadratic_density, radii)
    baseline_density = (
        uniform_baseline_fraction * uniform_density
        + (1.0 - uniform_baseline_fraction) * quadratic_density
    )
    density = (
        adaptive_fraction * signal_density
        + (1.0 - adaptive_fraction) * baseline_density
    )
    interval_mass = 0.5 * (density[:-1] + density[1:]) * np.diff(radii)
    cumulative = np.concatenate(([0.0], np.cumsum(interval_mass)))
    targets = np.linspace(0.0, cumulative[-1], n_nodes)
    nodes = np.interp(targets, cumulative, radii)
    nodes[0] = radii[0]
    nodes[-1] = radii[-1]
    if np.any(np.diff(nodes) <= 0):
        raise RuntimeError("adaptive-node construction produced duplicate nodes")
    return nodes


def spectral_radial_difficulty(
    evaluators: Sequence[MagnificationEvaluator],
    candidate_radii: np.ndarray,
    *,
    n_phi: int = 512,
    m_max: int = 96,
    map_quantile: float = 0.9,
) -> np.ndarray:
    """Estimate radial interpolation difficulty from pilot-map spectra.

    The score is an angular-series L1 bound on the error made by replacing an
    interior pilot spectrum with linear interpolation of its two neighbours.
    A quantile across pilot maps prevents one pathological map from consuming
    the complete shared radial-node budget.
    """
    radii = np.asarray(candidate_radii, dtype=np.float64)
    if radii.ndim != 1 or radii.size < 3 or np.any(np.diff(radii) <= 0):
        raise ValueError("candidate_radii must be a strictly increasing 1-D array")
    if not evaluators:
        raise ValueError("at least one pilot evaluator is required")
    if n_phi < 2 * (m_max + 1):
        raise ValueError("n_phi is too small for m_max")
    if not 0.0 <= map_quantile <= 1.0:
        raise ValueError("map_quantile must be in [0, 1]")

    phi = 2.0 * np.pi * np.arange(n_phi, dtype=np.float64) / n_phi
    x = radii[:, None] * np.cos(phi)[None, :]
    y = radii[:, None] * np.sin(phi)[None, :]
    mode_weights = np.ones(m_max + 1, dtype=np.float64)
    mode_weights[1:] = 2.0
    per_map = []
    left_width = radii[1:-1] - radii[:-2]
    span = radii[2:] - radii[:-2]
    fraction = left_width / span

    for evaluator in evaluators:
        magnification = np.asarray(evaluator.magnification(x, y), dtype=np.float64)
        if magnification.shape != x.shape or not np.all(np.isfinite(magnification)):
            raise ValueError("pilot evaluator returned invalid magnifications")
        excess = magnification - 1.0
        coefficients = np.fft.rfft(excess, axis=-1)[..., : m_max + 1] / n_phi
        interpolated = coefficients[:-2] + fraction[:, None] * (
            coefficients[2:] - coefficients[:-2]
        )
        residual = np.abs(coefficients[1:-1] - interpolated)
        # Normalize only by the broad map scale.  Sharp local structure still
        # receives more nodes, while maps with arbitrary amplitude do not
        # dominate solely because of that amplitude.
        scale = max(float(np.quantile(np.abs(excess), 0.9)), 1.0)
        score = np.sum(residual * mode_weights[None, :], axis=-1) / scale
        padded = np.empty_like(radii)
        padded[1:-1] = score
        padded[0] = score[0]
        padded[-1] = score[-1]
        per_map.append(padded)
    return np.quantile(np.stack(per_map), map_quantile, axis=0)


def build_adaptive_radial_nodes(
    evaluators: Sequence[MagnificationEvaluator],
    *,
    max_radius: float,
    n_nodes: int,
    pilot_bins: int = 1025,
    n_phi: int = 512,
    m_max: int = 96,
    map_quantile: float = 0.9,
    floor_fraction: float = 0.05,
    adaptive_fraction: float = 0.5,
    uniform_baseline_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return adaptive nodes plus the pilot radii and measured difficulty."""
    if not np.isfinite(max_radius) or max_radius <= 0:
        raise ValueError("max_radius must be finite and positive")
    if pilot_bins < n_nodes:
        raise ValueError("pilot_bins must be at least n_nodes")
    # A dense central pilot is essential for small central caustics; a uniform
    # pilot can step completely over them before adaptation gets a chance to
    # measure their curvature.
    pilot_radii = np.linspace(0.0, np.sqrt(max_radius), pilot_bins) ** 2
    difficulty = spectral_radial_difficulty(
        evaluators,
        pilot_radii,
        n_phi=n_phi,
        m_max=m_max,
        map_quantile=map_quantile,
    )
    nodes = equidistribute_radial_nodes(
        pilot_radii,
        difficulty,
        n_nodes,
        floor_fraction=floor_fraction,
        adaptive_fraction=adaptive_fraction,
        uniform_baseline_fraction=uniform_baseline_fraction,
    )
    return nodes, pilot_radii, difficulty


def interpolated_trajectory_error(
    reconstruction_error: np.ndarray,
    radial_nodes: np.ndarray,
    trajectory_radii: np.ndarray,
) -> np.ndarray:
    """Conservative per-map error along one or more trajectory radii."""
    errors = np.asarray(reconstruction_error, dtype=np.float64)
    nodes = np.asarray(radial_nodes, dtype=np.float64)
    radii = np.asarray(trajectory_radii, dtype=np.float64).ravel()
    if errors.ndim != 2 or errors.shape[1] != nodes.size:
        raise ValueError("reconstruction_error must have shape (map, radial_node)")
    if radii.size == 0:
        raise ValueError("trajectory_radii cannot be empty")
    if np.any(radii < nodes[0]) or np.any(radii > nodes[-1]):
        raise ValueError("trajectory radius lies outside radial_nodes")
    upper = np.searchsorted(nodes, radii, side="right")
    upper = np.clip(upper, 1, nodes.size - 1)
    lower = upper - 1
    fraction = (radii - nodes[lower]) / (nodes[upper] - nodes[lower])
    values = (
        errors[:, lower] * (1.0 - fraction)[None, :]
        + errors[:, upper] * fraction[None, :]
    )
    return np.max(values, axis=1)


def angular_tail_envelope(
    coefficients: np.ndarray,
    base_m_max: int,
) -> np.ndarray:
    """Upper-bound omitted angular-series amplitude at every map/radius."""
    coefficients = np.asarray(coefficients)
    if coefficients.ndim != 3:
        raise ValueError("coefficients must have shape (map, radius, mode)")
    full_m_max = coefficients.shape[-1] - 1
    if not 0 <= base_m_max < full_m_max:
        raise ValueError("base_m_max must be below the stored full mode")
    # Every positive-frequency Fourier-series coefficient contributes twice.
    return 2.0 * np.sum(
        np.abs(coefficients[..., base_m_max + 1 :]), axis=-1
    )


def total_angular_error_envelope(
    coefficients: np.ndarray,
    base_m_max: int,
    reconstruction_error: np.ndarray,
) -> np.ndarray:
    """Bound base-series error through stored modes and the atlas residual.

    The ordinary tail stops at the largest stored Fourier mode.  Direct-VBM
    atlases also measure the residual beyond that ceiling.  Adding both
    non-negative envelopes prevents a narrow caustic with little resolved tail
    from looking artificially safe to the adaptive rescan selector.
    """
    tail = angular_tail_envelope(coefficients, base_m_max)
    residual = np.asarray(reconstruction_error, dtype=np.float64)
    if residual.shape != tail.shape:
        raise ValueError(
            "reconstruction_error must match the map/radius coefficient axes"
        )
    if np.any(~np.isfinite(residual)) or np.any(residual < 0.0):
        raise ValueError("reconstruction_error must be finite and non-negative")
    return tail + residual


def select_interval_rescan_indices(
    chi2_lower: np.ndarray,
    chi2_upper: np.ndarray,
    *,
    target_count: int,
) -> np.ndarray:
    """Retain every row that can still belong to the true top candidates.

    The target-th smallest upper bound supplies a conservative cutoff.  Any
    candidate whose lower bound exceeds it cannot beat all of those target
    candidates, while overlapping intervals are deliberately retained.
    """
    lower = np.asarray(chi2_lower, dtype=np.float64)
    upper = np.asarray(chi2_upper, dtype=np.float64)
    if lower.ndim != 1 or upper.shape != lower.shape:
        raise ValueError("chi-square bounds must be matching one-dimensional arrays")
    if target_count < 1:
        raise ValueError("target_count must be positive")
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
        raise ValueError("chi-square bounds cannot contain NaN")
    if np.any(lower < 0.0) or np.any(lower > upper):
        raise ValueError("require 0 <= chi2_lower <= chi2_upper")
    finite_upper = upper[np.isfinite(upper)]
    if finite_upper.size == 0:
        return np.empty(0, dtype=np.int64)
    rank = min(target_count, finite_upper.size) - 1
    cutoff = np.partition(finite_upper, rank)[rank]
    return np.flatnonzero(lower <= cutoff)


def select_certified_rescan_indices(
    chi2_lower: np.ndarray,
    chi2_upper: np.ndarray,
    spectral_risk: np.ndarray,
    *,
    target_count: int,
    risk_count: int,
) -> np.ndarray:
    """Union interval-safe top-K candidates with the spectral rescue valve."""
    lower = np.asarray(chi2_lower, dtype=np.float64)
    risk = np.asarray(spectral_risk, dtype=np.float64)
    if risk.shape != lower.shape or not np.all(np.isfinite(risk)):
        raise ValueError("spectral_risk must be finite and match the bounds")
    if risk_count < 0:
        raise ValueError("risk_count must be non-negative")
    interval = select_interval_rescan_indices(
        lower, chi2_upper, target_count=target_count
    )
    count = min(risk_count, len(risk))
    risky = (
        np.argsort(risk, kind="stable")[-count:]
        if count
        else np.empty(0, dtype=np.int64)
    )
    return np.unique(np.concatenate((interval, risky)))


def select_adaptive_rescan_indices(
    base_chi2: np.ndarray,
    spectral_risk: np.ndarray,
    *,
    top_count: int,
    risk_count: int,
) -> np.ndarray:
    """Union promising fits with maps most at risk of low-mode truncation."""
    chi2 = np.asarray(base_chi2, dtype=np.float64)
    risk = np.asarray(spectral_risk, dtype=np.float64)
    if chi2.ndim != 1 or risk.shape != chi2.shape:
        raise ValueError("base_chi2 and spectral_risk must be same-length 1-D arrays")
    if np.any(np.isnan(chi2)) or not np.all(np.isfinite(risk)):
        raise ValueError("scores cannot contain NaN and risk must be finite")
    if top_count < 1 or risk_count < 0:
        raise ValueError("top_count must be positive and risk_count nonnegative")
    count = len(chi2)
    top_count = min(top_count, count)
    risk_count = min(risk_count, count)
    promising = np.argsort(chi2, kind="stable")[:top_count]
    risky = (
        np.argsort(risk, kind="stable")[-risk_count:]
        if risk_count
        else np.empty(0, dtype=np.int64)
    )
    return np.unique(np.concatenate((promising, risky)))
