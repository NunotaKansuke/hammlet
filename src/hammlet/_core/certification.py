"""Deterministic error certificates for sampled periodic VBM rings.

The certified reference is the periodic piecewise-linear interpolant through
the VBM verification samples. This does not claim a proof about the continuous
VBM implementation between unevaluated points, but it covers every point of
the declared reference interpolant instead of only sampled holdouts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


__all__ = [
    "NestedFourierCertificate",
    "certify_sampled_ring",
    "fourier_series_tail_bound",
    "periodic_piecewise_linear_residual_bound",
]
