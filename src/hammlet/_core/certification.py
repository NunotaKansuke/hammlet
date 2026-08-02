"""Deterministic error certificates for sampled periodic VBM rings.

The certified reference is the periodic piecewise-linear interpolant through
the VBM verification samples. This does not claim a proof about the continuous
VBM implementation between unevaluated points, but it covers every point of
the declared reference interpolant instead of only sampled holdouts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .radial import validate_radial_order


def fourier_series_tail_bound(
    coefficients: np.ndarray, retained_m_max: int
) -> float:
    """Return an L-infinity bound for the omitted positive-frequency tail."""
    coefficients = np.asarray(coefficients)
    if coefficients.ndim != 1:
        raise ValueError("coefficients must be one-dimensional")
    if not 0 <= retained_m_max < len(coefficients):
        raise ValueError("retained_m_max must be inside the coefficient array")
    return 2.0 * float(np.sum(np.abs(coefficients[retained_m_max + 1 :])))


def periodic_piecewise_linear_residual_bound(
    coefficients: np.ndarray,
    angles: np.ndarray,
    values: np.ndarray,
) -> float:
    """Bound a Fourier series against the periodic linear sample interpolant.

    On each interval the reference is linear, hence the residual's second
    derivative is minus that of the trigonometric polynomial. The standard
    linear-interpolation remainder gives an interval-wide, not node-only,
    bound.
    """
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    angles = np.asarray(angles, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if coefficients.ndim != 1 or coefficients.size == 0:
        raise ValueError("coefficients must be a non-empty one-dimensional array")
    if angles.ndim != 1 or values.shape != angles.shape or len(angles) < 3:
        raise ValueError("angles and values must be equal one-dimensional arrays")
    if not np.all(np.isfinite(angles)) or not np.all(np.isfinite(values)):
        raise ValueError("angles and values must be finite")
    normalized = np.mod(angles, 2.0 * np.pi)
    order = np.argsort(normalized, kind="stable")
    normalized = normalized[order]
    values = values[order]
    if np.any(np.diff(normalized) <= 0.0):
        raise ValueError("angles must be unique modulo 2*pi")

    modes = np.arange(len(coefficients), dtype=np.float64)
    widths = np.diff(
        np.concatenate((normalized, np.asarray([normalized[0] + 2.0 * np.pi])))
    )
    uniform = np.allclose(
        widths, 2.0 * np.pi / len(normalized), rtol=1.0e-10, atol=1.0e-13
    )
    if uniform and abs(normalized[0]) <= 1.0e-13:
        spectrum = np.zeros(len(normalized) // 2 + 1, dtype=np.complex128)
        copied = min(len(spectrum), len(coefficients))
        spectrum[:copied] = coefficients[:copied] * len(normalized)
        reconstructed = np.fft.irfft(spectrum, n=len(normalized))
    else:
        reconstructed = np.empty(len(normalized), dtype=np.float64)
        # Local caustic refinement makes the grid non-uniform. Blocking avoids
        # an O(n_angle*n_mode) complex temporary while retaining exact series
        # evaluation at every verification node.
        for start in range(0, len(normalized), 256):
            stop = min(start + 256, len(normalized))
            phase = np.exp(
                1j * normalized[start:stop, None] * modes[None, 1:]
            )
            reconstructed[start:stop] = np.real(
                coefficients[0]
                + 2.0 * np.sum(coefficients[None, 1:] * phase, axis=1)
            )
    endpoint_error = np.abs(values - reconstructed)
    second_derivative_bound = 2.0 * float(
        np.sum((modes[1:] ** 2) * np.abs(coefficients[1:]))
    )
    interval_endpoint_error = np.maximum(endpoint_error, np.roll(endpoint_error, -1))
    return float(
        np.max(
            interval_endpoint_error
            + 0.125 * widths * widths * second_derivative_bound
        )
    )


@dataclass(frozen=True)
class NestedFourierCertificate:
    """Mode-dependent certificate relative to one sampled VBM reference."""

    diagnostic_coefficients: np.ndarray
    unresolved_bound: float
    coefficient_change_bound: float = 0.0

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.diagnostic_coefficients)
        if coefficients.ndim != 1 or coefficients.size < 2:
            raise ValueError("diagnostic_coefficients must contain at least two modes")
        for name in ("unresolved_bound", "coefficient_change_bound"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def diagnostic_m_max(self) -> int:
        return len(self.diagnostic_coefficients) - 1

    def error_bound(self, retained_m_max: int) -> float:
        """Bound the reference error after retaining modes through ``M``."""
        if not 0 <= retained_m_max <= self.diagnostic_m_max:
            raise ValueError("retained_m_max is outside the diagnostic spectrum")
        tail = (
            0.0
            if retained_m_max == self.diagnostic_m_max
            else fourier_series_tail_bound(
                self.diagnostic_coefficients, retained_m_max
            )
        )
        return tail + self.unresolved_bound + self.coefficient_change_bound

    def frontier(self, mode_limits: np.ndarray) -> np.ndarray:
        """Return the monotone accuracy frontier for candidate M values."""
        modes = np.asarray(mode_limits)
        if modes.ndim != 1:
            raise ValueError("mode_limits must be one-dimensional")
        return np.asarray([self.error_bound(int(mode)) for mode in modes])


def certify_sampled_ring(
    diagnostic_coefficients: np.ndarray,
    angles: np.ndarray,
    values: np.ndarray,
    *,
    coefficient_change_bound: float = 0.0,
) -> NestedFourierCertificate:
    """Build a mode hierarchy certificate from one sampled VBM ring."""
    unresolved = periodic_piecewise_linear_residual_bound(
        diagnostic_coefficients, angles, values
    )
    return NestedFourierCertificate(
        diagnostic_coefficients=np.asarray(diagnostic_coefficients),
        unresolved_bound=unresolved,
        coefficient_change_bound=float(coefficient_change_bound),
    )


def _radial_stencil_indices(cell: int, n_node: int, order: int) -> np.ndarray:
    """Return the fixed Lagrange stencil used inside one radial cell."""
    order = validate_radial_order(order)
    if not 0 <= cell < n_node - 1:
        raise ValueError("cell is outside the radial-node range")
    if order == 1:
        return np.asarray((cell, cell + 1), dtype=np.int64)
    start = int(np.clip(cell - 1, 0, n_node - 4))
    return start + np.arange(4, dtype=np.int64)


def _lagrange_weights(nodes: np.ndarray, x: float) -> np.ndarray:
    weights = np.ones(len(nodes), dtype=np.float64)
    for target in range(len(nodes)):
        for other in range(len(nodes)):
            if other != target:
                weights[target] *= (x - nodes[other]) / (
                    nodes[target] - nodes[other]
                )
    return weights


def _lagrange_second_derivative_weights(
    nodes: np.ndarray, x: float
) -> np.ndarray:
    """Evaluate the second derivative of every Lagrange basis polynomial."""
    output = np.zeros(len(nodes), dtype=np.float64)
    if len(nodes) <= 2:
        return output
    for target in range(len(nodes)):
        other = np.delete(nodes, target)
        denominator = float(np.prod(nodes[target] - other))
        polynomial = np.poly(other) / denominator
        output[target] = float(np.polyval(np.polyder(polynomial, 2), x))
    return output


def radial_piecewise_linear_residual_bound(
    stored_nodes: np.ndarray,
    stored_coefficients: np.ndarray,
    reference_nodes: np.ndarray,
    reference_coefficients: np.ndarray,
    reference_error: np.ndarray,
    *,
    order: int,
) -> np.ndarray:
    """Bound radial interpolation against a sampled piecewise-linear reference.

    ``reference_nodes`` must include every stored node and may add nested direct
    VBM holdout rings. Between consecutive reference rings the declared
    coefficient reference is linear. The runtime Lagrange interpolant is a
    polynomial, so on each subinterval its residual is bounded by its endpoint
    residual plus ``h**2/8`` times a computable second-derivative bound. The
    returned value is one angular-series L-infinity envelope per stored cell.

    ``reference_error`` is the angular certificate at each reference ring. It
    is interpolated convexly between rings and therefore enters through the
    larger endpoint value.
    """
    nodes = np.asarray(stored_nodes, dtype=np.float64)
    stored = np.asarray(stored_coefficients, dtype=np.complex128)
    radii = np.asarray(reference_nodes, dtype=np.float64)
    reference = np.asarray(reference_coefficients, dtype=np.complex128)
    angular_error = np.asarray(reference_error, dtype=np.float64)
    order = validate_radial_order(order)
    if nodes.ndim != 1 or len(nodes) < order + 1 or np.any(np.diff(nodes) <= 0.0):
        raise ValueError("stored_nodes must be strictly increasing and wide enough")
    if stored.ndim != 2 or stored.shape[0] != len(nodes):
        raise ValueError("stored_coefficients must have shape (stored_node, mode)")
    if (
        radii.ndim != 1
        or len(radii) < len(nodes)
        or np.any(np.diff(radii) <= 0.0)
        or reference.shape != (len(radii), stored.shape[1])
        or angular_error.shape != (len(radii),)
    ):
        raise ValueError("reference arrays have incompatible shapes")
    if (
        not np.all(np.isfinite(radii))
        or not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(angular_error))
        or np.any(angular_error < 0.0)
    ):
        raise ValueError("reference arrays must be finite with non-negative errors")
    scale = max(1.0, abs(float(nodes[0])), abs(float(nodes[-1])))
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    if abs(radii[0] - nodes[0]) > tolerance or abs(radii[-1] - nodes[-1]) > tolerance:
        raise ValueError("reference_nodes must share the stored radial endpoints")
    stored_positions = np.searchsorted(radii, nodes)
    if np.any(stored_positions >= len(radii)) or not np.allclose(
        radii[stored_positions], nodes, rtol=0.0, atol=tolerance
    ):
        raise ValueError("reference_nodes must include every stored node")

    mode_weight = np.full(stored.shape[1], 2.0, dtype=np.float64)
    mode_weight[0] = 1.0
    output = np.zeros(len(nodes) - 1, dtype=np.float64)
    for cell in range(len(output)):
        left_position = int(stored_positions[cell])
        right_position = int(stored_positions[cell + 1])
        if right_position <= left_position:
            raise RuntimeError("stored radial cell has no reference segment")
        stencil_indices = _radial_stencil_indices(cell, len(nodes), order)
        stencil_nodes = nodes[stencil_indices]
        stencil_coefficients = stored[stencil_indices]
        bound = 0.0
        for segment in range(left_position, right_position):
            left = float(radii[segment])
            right = float(radii[segment + 1])
            left_value = _lagrange_weights(stencil_nodes, left) @ stencil_coefficients
            right_value = (
                _lagrange_weights(stencil_nodes, right) @ stencil_coefficients
            )
            endpoint = np.maximum(
                np.abs(reference[segment] - left_value),
                np.abs(reference[segment + 1] - right_value),
            )
            if order == 3:
                second_left = (
                    _lagrange_second_derivative_weights(stencil_nodes, left)
                    @ stencil_coefficients
                )
                second_right = (
                    _lagrange_second_derivative_weights(stencil_nodes, right)
                    @ stencil_coefficients
                )
                second = np.maximum(np.abs(second_left), np.abs(second_right))
                endpoint = endpoint + (right - left) ** 2 * second / 8.0
            coefficient_bound = float(np.dot(mode_weight, endpoint))
            local = coefficient_bound + max(
                float(angular_error[segment]),
                float(angular_error[segment + 1]),
            )
            bound = max(bound, local)
        output[cell] = bound
    return output


def radial_interval_to_node_envelope(
    interval_bound: np.ndarray, *, order: int
) -> np.ndarray:
    """Encode interval bounds as node errors for the existing fast propagator.

    Every node read by an interval's runtime stencil receives at least that
    interval's bound. Since Lagrange weights sum to one, the absolute-weight
    propagation is then no smaller than the interval bound, including for
    cubic stencils with negative weights.
    """
    intervals = np.asarray(interval_bound, dtype=np.float64)
    order = validate_radial_order(order)
    if (
        intervals.ndim != 1
        or intervals.size < order
        or not np.all(np.isfinite(intervals))
        or np.any(intervals < 0.0)
    ):
        raise ValueError("interval_bound must be a finite non-negative vector")
    n_node = len(intervals) + 1
    output = np.zeros(n_node, dtype=np.float64)
    for cell, value in enumerate(intervals):
        indices = _radial_stencil_indices(cell, n_node, order)
        output[indices] = np.maximum(output[indices], value)
    return output


__all__ = [
    "NestedFourierCertificate",
    "certify_sampled_ring",
    "fourier_series_tail_bound",
    "periodic_piecewise_linear_residual_bound",
    "radial_interval_to_node_envelope",
    "radial_piecewise_linear_residual_bound",
]
