from __future__ import annotations

from typing import Sequence

import numpy as np

from .trajectory import ConsistentEventKernel, EventKernel


def _batched_radial_contraction(
    coefficients: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Contract radius while retaining map, geometry, and angular mode."""
    if weights.shape[0] >= 16:
        # For larger geometry batches, one matrix multiplication per Fourier
        # mode is faster than NumPy's general einsum path.
        return np.matmul(
            coefficients.transpose(2, 0, 1),
            weights.transpose(2, 1, 0),
        ).transpose(1, 2, 0)
    return np.einsum("nrm,krm->nkm", coefficients, weights, optimize=True)


def fourier_series_values(coefficients: np.ndarray, n_alpha: int) -> np.ndarray:
    """Evaluate positive-frequency Fourier-series coefficients on a uniform grid."""
    if coefficients.shape[-1] - 1 > n_alpha // 2:
        raise ValueError("coefficient array contains modes beyond the output Nyquist limit")
    spectrum = np.zeros(
        coefficients.shape[:-1] + (n_alpha // 2 + 1,), dtype=np.complex128
    )
    spectrum[..., : coefficients.shape[-1]] = coefficients * n_alpha
    return np.fft.irfft(spectrum, n=n_alpha, axis=-1)


def _is_smooth(size: int) -> bool:
    for factor in (2, 3, 5):
        while size % factor == 0:
            size //= factor
    return size == 1


def product_grid_size(m_max: int) -> int:
    """Smallest 5-smooth grid that samples ``x**2`` without angular aliasing.

    ``x`` carries modes up to ``m_max``, so ``x**2`` reaches ``2 * m_max`` and
    needs at least ``4 * m_max + 2`` samples.  Rounding up to a power of two
    overshoots by up to a factor of two; pocketfft (behind both NumPy and the
    JAX CPU backend) is equally happy with any 2/3/5-smooth length, and the
    product stage dominates the low-mode ranking pass.
    """
    m_max = int(m_max)
    if m_max < 0:
        raise ValueError("m_max must be nonnegative")
    size = max(4 * m_max + 2, 2)
    size += size % 2  # rfft of an even length keeps the Nyquist bin real.
    while not _is_smooth(size):
        size += 2
    return size


def consistent_product_coefficients(
    x_coeff: np.ndarray, n_lag: int = 2
) -> np.ndarray:
    """Derive the lag-``0 .. n_lag-1`` node-product spectra from ``x``.

    Returns shape ``(n_lag, map, radius, 2 * m_max + 1)``.  Lag ``d`` holds the
    angular spectrum of ``x_r * x_{r+d}``; its trailing ``d`` radius rows have no
    partner node and are zero, matching the padded kernel layout.
    """
    x_coeff = np.asarray(x_coeff)
    if x_coeff.ndim != 3:
        raise ValueError("x_coeff must have shape (map, radius, mode)")
    n_lag = int(n_lag)
    n_radius = x_coeff.shape[1]
    if not 1 <= n_lag <= n_radius:
        raise ValueError("n_lag must be between one and the radial-node count")
    m_max = x_coeff.shape[-1] - 1
    grid_size = product_grid_size(m_max)
    x_values = fourier_series_values(x_coeff, grid_size)
    products = np.zeros(
        (n_lag,) + x_coeff.shape[:2] + (2 * m_max + 1,), dtype=np.complex128
    )
    for lag in range(n_lag):
        stop = n_radius - lag
        products[lag, :, :stop] = (
            np.fft.rfft(x_values[:, :stop] * x_values[:, lag:], axis=-1) / grid_size
        )[..., : 2 * m_max + 1]
    return products


def profile_chi2(
    q_x: np.ndarray,
    q_xx: np.ndarray,
    q_xy: np.ndarray,
    kernel: EventKernel | ConsistentEventKernel,
    singular_rtol: float = 1e-12,
) -> np.ndarray:
    """Profile a source amplitude and intercept independently for one dataset."""
    s_xx = q_xx - q_x * q_x / kernel.weight_sum
    s_xy = q_xy - q_x * kernel.flux_sum / kernel.weight_sum
    s_yy = kernel.flux2_sum - kernel.flux_sum**2 / kernel.weight_sum
    scale = np.maximum(np.abs(q_xx), np.abs(q_x * q_x / kernel.weight_sum))
    valid = s_xx > singular_rtol * np.maximum(scale, 1.0)
    cauchy_limit = s_xx * s_yy
    cauchy_tolerance = singular_rtol * np.maximum(np.abs(cauchy_limit), 1.0)
    valid &= s_xy * s_xy <= cauchy_limit + cauchy_tolerance
    chi2 = np.full(np.broadcast_shapes(s_xx.shape, s_xy.shape), np.inf, dtype=np.float64)
    raw = s_yy - s_xy[valid] ** 2 / s_xx[valid]
    # Tiny roundoff at the Cauchy boundary is clipped.
    raw[raw < -singular_rtol * max(abs(s_yy), 1.0)] = np.inf
    chi2[valid] = np.maximum(raw, 0.0)
    return chi2


def scan_coefficients(
    x_coeff: np.ndarray,
    x2_coeff: np.ndarray,
    kernels: Sequence[EventKernel],
    *,
    n_alpha: int,
    singular_rtol: float = 1e-12,
) -> np.ndarray:
    """Return ``(n_map, n_alpha)`` profile chi-square values."""
    x_coeff = np.asarray(x_coeff)
    x2_coeff = np.asarray(x2_coeff)
    if x_coeff.shape != x2_coeff.shape or x_coeff.ndim != 3:
        raise ValueError("coefficient arrays must have shape (map, radius, mode)")
    total = np.zeros((x_coeff.shape[0], n_alpha), dtype=np.float64)
    for kernel in kernels:
        c_x = np.einsum("nrm,rm->nm", x_coeff, kernel.weighted, optimize=True)
        c_xx = np.einsum("nrm,rm->nm", x2_coeff, kernel.weighted, optimize=True)
        c_xy = np.einsum("nrm,rm->nm", x_coeff, kernel.weighted_flux, optimize=True)
        q_x = fourier_series_values(c_x, n_alpha)
        q_xx = fourier_series_values(c_xx, n_alpha)
        q_xy = fourier_series_values(c_xy, n_alpha)
        total += profile_chi2(q_x, q_xx, q_xy, kernel, singular_rtol)
    return total


def scan_coefficients_consistent(
    x_coeff: np.ndarray,
    kernels: Sequence[ConsistentEventKernel],
    *,
    n_alpha: int,
    singular_rtol: float = 1e-12,
) -> np.ndarray:
    """Scan alpha with all moments derived from one low-pass light curve."""
    x_coeff = np.asarray(x_coeff)
    if x_coeff.ndim != 3:
        raise ValueError("x_coeff must have shape (map, radius, mode)")
    m_max = x_coeff.shape[-1] - 1
    if n_alpha // 2 < 2 * m_max:
        raise ValueError("n_alpha must be at least 4*m_max")
    kernels = list(kernels)
    if not kernels:
        raise ValueError("at least one event kernel is required")
    n_lag = kernels[0].weighted_square_lags.shape[0]
    if any(kernel.weighted_square_lags.shape[0] != n_lag for kernel in kernels):
        raise ValueError("every kernel must use the same radial order")
    products = consistent_product_coefficients(x_coeff, n_lag)

    total = np.zeros((x_coeff.shape[0], n_alpha), dtype=np.float64)
    for kernel in kernels:
        if kernel.weighted.shape != x_coeff.shape[1:]:
            raise ValueError("kernel and coefficient radius/mode shapes do not match")
        if kernel.weighted_square_lags.shape[1:] != (
            x_coeff.shape[1],
            2 * m_max + 1,
        ):
            raise ValueError("square-moment kernel shape does not match x_coeff")

        c_x = np.einsum("nrm,rm->nm", x_coeff, kernel.weighted, optimize=True)
        c_xy = np.einsum(
            "nrm,rm->nm", x_coeff, kernel.weighted_flux, optimize=True
        )
        c_xx = np.einsum(
            "lnrm,lrm->nm",
            products,
            kernel.weighted_square_lags,
            optimize=True,
        )
        total += profile_chi2(
            fourier_series_values(c_x, n_alpha),
            fourier_series_values(c_xx, n_alpha),
            fourier_series_values(c_xy, n_alpha),
            kernel,
            singular_rtol,
        )
    return total


def scan_coefficients_batched(
    x_coeff: np.ndarray,
    x2_coeff: np.ndarray,
    kernel_batches: Sequence[Sequence[EventKernel]],
    *,
    n_alpha: int,
    singular_rtol: float = 1e-12,
) -> np.ndarray:
    """Scan several geometries while reading each coefficient shard once.

    ``kernel_batches[g][d]`` is the event kernel for geometry ``g`` and
    dataset ``d``.  The result has shape ``(n_map, n_geometry, n_alpha)``.
    """
    x_coeff = np.asarray(x_coeff)
    x2_coeff = np.asarray(x2_coeff)
    if x_coeff.shape != x2_coeff.shape or x_coeff.ndim != 3:
        raise ValueError("coefficient arrays must have shape (map, radius, mode)")
    if not kernel_batches:
        raise ValueError("kernel_batches cannot be empty")
    dataset_count = len(kernel_batches[0])
    if dataset_count == 0 or any(
        len(kernels) != dataset_count for kernels in kernel_batches
    ):
        raise ValueError("every geometry must provide the same nonzero datasets")

    n_map = x_coeff.shape[0]
    n_geometry = len(kernel_batches)
    total = np.zeros((n_map, n_geometry, n_alpha), dtype=np.float64)
    use_mode_major = n_geometry >= 16
    if use_mode_major:
        # The atlas is map-major for efficient single-geometry scans.  A
        # geometry batch instead performs one GEMM per Fourier mode; making
        # that dimension leading and each matrix contiguous is substantially
        # faster even after paying this once-per-shard transpose.
        x_mode_major = np.ascontiguousarray(x_coeff.transpose(2, 0, 1))
        x2_mode_major = np.ascontiguousarray(x2_coeff.transpose(2, 0, 1))

    for dataset_index in range(dataset_count):
        kernels = [
            geometry_kernels[dataset_index] for geometry_kernels in kernel_batches
        ]
        weighted = np.stack([kernel.weighted for kernel in kernels])
        weighted_flux = np.stack([kernel.weighted_flux for kernel in kernels])
        if weighted.shape[1:] != x_coeff.shape[1:]:
            raise ValueError("kernel and coefficient radius/mode shapes do not match")

        if use_mode_major:
            weighted_mode_major = np.ascontiguousarray(
                weighted.transpose(2, 1, 0)
            )
            weighted_flux_mode_major = np.ascontiguousarray(
                weighted_flux.transpose(2, 1, 0)
            )
            c_x = np.matmul(
                x_mode_major, weighted_mode_major
            ).transpose(1, 2, 0)
            c_xx = np.matmul(
                x2_mode_major, weighted_mode_major
            ).transpose(1, 2, 0)
            c_xy = np.matmul(
                x_mode_major, weighted_flux_mode_major
            ).transpose(1, 2, 0)
        else:
            c_x = _batched_radial_contraction(x_coeff, weighted)
            c_xx = _batched_radial_contraction(x2_coeff, weighted)
            c_xy = _batched_radial_contraction(x_coeff, weighted_flux)
        q_x = fourier_series_values(c_x, n_alpha)
        q_xx = fourier_series_values(c_xx, n_alpha)
        q_xy = fourier_series_values(c_xy, n_alpha)
        for geometry_index, kernel in enumerate(kernels):
            total[:, geometry_index] += profile_chi2(
                q_x[:, geometry_index],
                q_xx[:, geometry_index],
                q_xy[:, geometry_index],
                kernel,
                singular_rtol,
            )
    return total
