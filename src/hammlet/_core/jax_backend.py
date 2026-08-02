from __future__ import annotations

from typing import Sequence

import numpy as np

from .alpha_fft import product_grid_size
from .trajectory import ConsistentEventKernel, EventKernel


class JAXGeometryBatchScanner:
    """JIT-compiled CPU scanner for a fixed batch of event geometries."""

    def __init__(
        self,
        kernel_batches: Sequence[Sequence[EventKernel]],
        *,
        n_alpha: int,
        singular_rtol: float = 1e-12,
    ) -> None:
        try:
            import jax

            jax.config.update("jax_enable_x64", True)
            import jax.numpy as jnp
        except ImportError as error:
            raise ImportError(
                "JAXGeometryBatchScanner requires the optional jax dependency"
            ) from error

        if not kernel_batches:
            raise ValueError("kernel_batches cannot be empty")
        dataset_count = len(kernel_batches[0])
        if dataset_count == 0 or any(
            len(kernels) != dataset_count for kernels in kernel_batches
        ):
            raise ValueError(
                "every geometry must provide the same nonzero datasets"
            )
        if n_alpha < 4 or n_alpha % 2:
            raise ValueError("n_alpha must be an even integer >= 4")

        weights = np.stack(
            [
                [
                    kernel_batches[geometry][dataset].weighted
                    for geometry in range(len(kernel_batches))
                ]
                for dataset in range(dataset_count)
            ]
        )
        weighted_flux = np.stack(
            [
                [
                    kernel_batches[geometry][dataset].weighted_flux
                    for geometry in range(len(kernel_batches))
                ]
                for dataset in range(dataset_count)
            ]
        )
        weight_sum = np.asarray(
            [
                [
                    kernel_batches[geometry][dataset].weight_sum
                    for geometry in range(len(kernel_batches))
                ]
                for dataset in range(dataset_count)
            ],
            dtype=np.float64,
        )
        flux_sum = np.asarray(
            [
                [
                    kernel_batches[geometry][dataset].flux_sum
                    for geometry in range(len(kernel_batches))
                ]
                for dataset in range(dataset_count)
            ],
            dtype=np.float64,
        )
        flux2_sum = np.asarray(
            [
                [
                    kernel_batches[geometry][dataset].flux2_sum
                    for geometry in range(len(kernel_batches))
                ]
                for dataset in range(dataset_count)
            ],
            dtype=np.float64,
        )

        self._jax = jax
        self._jnp = jnp
        self._weights = jax.device_put(weights)
        self._weighted_flux = jax.device_put(weighted_flux)
        self._weight_sum = jax.device_put(weight_sum)
        self._flux_sum = jax.device_put(flux_sum)
        self._flux2_sum = jax.device_put(flux2_sum)
        self.n_alpha = int(n_alpha)
        self.n_geometry = len(kernel_batches)
        self.n_radius = int(weights.shape[-2])
        self.n_mode = int(weights.shape[-1])
        self.singular_rtol = float(singular_rtol)
        if self.n_mode > self.n_alpha // 2:
            raise ValueError("kernel modes exceed the alpha-grid Nyquist limit")
        self._compiled_scan = jax.jit(self._scan_impl)

    def _series_values(self, coefficients):
        spectrum_size = self.n_alpha // 2 + 1
        padding = spectrum_size - coefficients.shape[-1]
        spectrum = self._jnp.pad(
            coefficients,
            ((0, 0), (0, 0), (0, padding)),
        )
        return self._jnp.fft.irfft(
            spectrum * self.n_alpha,
            n=self.n_alpha,
            axis=-1,
        )

    def _scan_impl(
        self,
        x_coeff,
        x2_coeff,
        weights,
        weighted_flux,
        weight_sum,
        flux_sum,
        flux2_sum,
    ):
        jnp = self._jnp
        total = jnp.zeros(
            (x_coeff.shape[0], weights.shape[1], self.n_alpha),
            dtype=jnp.float64,
        )
        for dataset in range(weights.shape[0]):
            c_x = jnp.einsum(
                "nrm,krm->nkm", x_coeff, weights[dataset]
            )
            c_xx = jnp.einsum(
                "nrm,krm->nkm", x2_coeff, weights[dataset]
            )
            c_xy = jnp.einsum(
                "nrm,krm->nkm", x_coeff, weighted_flux[dataset]
            )
            q_x = self._series_values(c_x)
            q_xx = self._series_values(c_xx)
            q_xy = self._series_values(c_xy)
            local_weight = weight_sum[dataset][None, :, None]
            local_flux = flux_sum[dataset][None, :, None]
            s_xx = q_xx - q_x * q_x / local_weight
            s_xy = q_xy - q_x * local_flux / local_weight
            s_yy = (
                flux2_sum[dataset]
                - flux_sum[dataset] ** 2 / weight_sum[dataset]
            )
            scale = jnp.maximum(
                jnp.abs(q_xx),
                jnp.abs(q_x * q_x / local_weight),
            )
            valid = s_xx > self.singular_rtol * jnp.maximum(scale, 1.0)
            raw = s_yy[None, :, None] - s_xy * s_xy / s_xx
            raw = jnp.where(
                raw
                < -self.singular_rtol
                * jnp.maximum(jnp.abs(s_yy)[None, :, None], 1.0),
                jnp.inf,
                raw,
            )
            total += jnp.where(valid, jnp.maximum(raw, 0.0), jnp.inf)
        return total

    def scan(self, x_coeff: np.ndarray, x2_coeff: np.ndarray) -> np.ndarray:
        x_coeff = np.asarray(x_coeff)
        x2_coeff = np.asarray(x2_coeff)
        expected = (self.n_radius, self.n_mode)
        if (
            x_coeff.shape != x2_coeff.shape
            or x_coeff.ndim != 3
            or x_coeff.shape[1:] != expected
        ):
            raise ValueError(
                "coefficient arrays do not match the batched kernel shape"
            )
        result = self._compiled_scan(
            self._jnp.asarray(x_coeff),
            self._jnp.asarray(x2_coeff),
            self._weights,
            self._weighted_flux,
            self._weight_sum,
            self._flux_sum,
            self._flux2_sum,
        )
        return np.asarray(result)


class JAXConsistentGeometryBatchScanner:
    """JIT scanner whose fitted moments come from one approximate light curve."""

    def __init__(
        self,
        kernel_batches: Sequence[Sequence[ConsistentEventKernel]],
        *,
        n_alpha: int,
        singular_rtol: float = 1e-12,
        compute_dtype: str = "float64",
    ) -> None:
        try:
            import jax

            jax.config.update("jax_enable_x64", True)
            import jax.numpy as jnp
        except ImportError as error:
            raise ImportError(
                "JAXConsistentGeometryBatchScanner requires the optional jax dependency"
            ) from error

        if not kernel_batches:
            raise ValueError("kernel_batches cannot be empty")
        dataset_count = len(kernel_batches[0])
        if dataset_count == 0 or any(
            len(kernels) != dataset_count for kernels in kernel_batches
        ):
            raise ValueError("every geometry must provide the same nonzero datasets")
        if n_alpha < 4 or n_alpha % 2:
            raise ValueError("n_alpha must be an even integer >= 4")
        if compute_dtype not in ("float64", "float32"):
            raise ValueError("compute_dtype must be 'float64' or 'float32'")

        arrays = self._stack_kernel_batches(kernel_batches)
        weights = arrays["weighted"]

        self._jax = jax
        self._jnp = jnp
        # The atlas stores complex64 coefficients, so the contraction and the
        # product spectra carry no extra information in double precision.  Only
        # the profiled chi-square combination, which subtracts comparably sized
        # moments, is always accumulated in float64.
        self.compute_dtype = compute_dtype
        self._real_dtype = (
            np.float64 if compute_dtype == "float64" else np.float32
        )
        self._complex_dtype = (
            np.complex128 if compute_dtype == "float64" else np.complex64
        )
        self._kernel_shape_signature = {
            name: value.shape for name, value in arrays.items()
        }
        self._put_kernel_arrays(arrays)
        self.n_alpha = int(n_alpha)
        self.n_geometry = len(kernel_batches)
        self.n_radius = int(weights.shape[-2])
        self.n_mode = int(weights.shape[-1])
        self.m_max = self.n_mode - 1
        self.n_lag = int(arrays["square_lags"].shape[2])
        self.radial_order = self.n_lag - 1
        self.singular_rtol = float(singular_rtol)
        if self.n_alpha // 2 < 2 * self.m_max:
            raise ValueError("n_alpha must be at least 4*m_max")
        self.product_grid_size = product_grid_size(self.m_max)
        self._compiled_scan = jax.jit(self._scan_impl)
        self._compiled_scan_minima = jax.jit(self._scan_minima_impl)
        self._compiled_scan_minima_with_error = jax.jit(
            self._scan_minima_with_error_impl
        )
        # A separate program, so an unmasked scan never carries the mask
        # argument and never pays for the extra reduction.
        self._compiled_scan_minima_masked = jax.jit(self._scan_minima_masked_impl)
        self._compiled_scan_minima_masked_with_error = jax.jit(
            self._scan_minima_masked_with_error_impl
        )

    @staticmethod
    def _stack_kernel_batches(
        kernel_batches: Sequence[Sequence[ConsistentEventKernel]],
    ) -> dict[str, np.ndarray]:
        if not kernel_batches or not kernel_batches[0]:
            raise ValueError("kernel_batches cannot be empty")
        dataset_count = len(kernel_batches[0])
        if any(len(kernels) != dataset_count for kernels in kernel_batches):
            raise ValueError("every geometry must provide the same nonzero datasets")

        def stack_field(name: str) -> np.ndarray:
            return np.stack(
                [
                    [
                        getattr(kernel_batches[geometry][dataset], name)
                        for geometry in range(len(kernel_batches))
                    ]
                    for dataset in range(dataset_count)
                ]
            )

        def stack_scalar(name: str) -> np.ndarray:
            return np.asarray(
                [
                    [
                        getattr(kernel_batches[geometry][dataset], name)
                        for geometry in range(len(kernel_batches))
                    ]
                    for dataset in range(dataset_count)
                ],
                dtype=np.float64,
            )

        return {
            "weighted": stack_field("weighted"),
            "weighted_flux": stack_field("weighted_flux"),
            "square_lags": stack_field("weighted_square_lags"),
            "error_abs_flux": stack_field(
                "error_weighted_abs_centered_flux"
            ),
            "error_square_lags": stack_field("error_weighted_square_lags"),
            "weight_sum": stack_scalar("weight_sum"),
            "flux_sum": stack_scalar("flux_sum"),
            "flux2_sum": stack_scalar("flux2_sum"),
        }

    def _put_kernel_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        complex_dtype = self._complex_dtype
        self._weights = self._jax.device_put(
            arrays["weighted"].astype(complex_dtype)
        )
        self._weighted_flux = self._jax.device_put(
            arrays["weighted_flux"].astype(complex_dtype)
        )
        self._square_lags = self._jax.device_put(
            arrays["square_lags"].astype(complex_dtype)
        )
        self._error_abs_flux = self._jax.device_put(
            arrays["error_abs_flux"].astype(self._real_dtype)
        )
        self._error_square_lags = self._jax.device_put(
            arrays["error_square_lags"].astype(self._real_dtype)
        )
        self._weight_sum = self._jax.device_put(arrays["weight_sum"])
        self._flux_sum = self._jax.device_put(arrays["flux_sum"])
        self._flux2_sum = self._jax.device_put(arrays["flux2_sum"])

    def set_kernels(
        self,
        kernel_batches: Sequence[Sequence[ConsistentEventKernel]],
    ) -> None:
        """Replace equal-shaped kernels without recompiling the JAX program.

        Bucketed atlases share their array dimensions but use different radial
        nodes.  Keeping the compiled callable and swapping only its dynamic
        arguments avoids one JAX compilation per Q-S bucket.
        """
        arrays = self._stack_kernel_batches(kernel_batches)
        signature = {name: value.shape for name, value in arrays.items()}
        if signature != self._kernel_shape_signature:
            raise ValueError(
                "replacement kernels must have the scanner's existing shapes"
            )
        self._put_kernel_arrays(arrays)

    def _series_values(self, coefficients, n_output: int):
        spectrum_size = n_output // 2 + 1
        padding = spectrum_size - coefficients.shape[-1]
        spectrum = self._jnp.pad(
            coefficients,
            ((0, 0), (0, 0), (0, padding)),
        )
        values = self._jnp.fft.irfft(spectrum * n_output, n=n_output, axis=-1)
        return values.astype(self._jnp.float64)

    def _scan_impl(
        self,
        x_coeff,
        weights,
        weighted_flux,
        square_lags,
        weight_sum,
        flux_sum,
        flux2_sum,
        error_envelope=None,
        error_abs_flux=None,
        error_square_lags=None,
    ):
        jnp = self._jnp
        x_coeff = x_coeff.astype(self._complex_dtype)
        product_spectrum_size = self.product_grid_size // 2 + 1
        product_padding = product_spectrum_size - x_coeff.shape[-1]
        product_spectrum = jnp.pad(
            x_coeff, ((0, 0), (0, 0), (0, product_padding))
        )
        x_values = jnp.fft.irfft(
            product_spectrum * self.product_grid_size,
            n=self.product_grid_size,
            axis=-1,
        ).astype(self._real_dtype)
        product_modes = 2 * self.m_max + 1
        n_radius = x_coeff.shape[1]
        # Lag ``d`` pairs radial node ``r`` with ``r + d``; the trailing ``d``
        # rows have no partner and are padded with zeros so every lag shares one
        # shape with the kernel array.
        products = [
            (
                jnp.pad(
                    jnp.fft.rfft(
                        x_values[:, : n_radius - lag] * x_values[:, lag:],
                        axis=-1,
                    )[..., :product_modes]
                    / self.product_grid_size,
                    ((0, 0), (0, lag), (0, 0)),
                )
            ).astype(self._complex_dtype)
            for lag in range(self.n_lag)
        ]

        total = jnp.zeros(
            (x_coeff.shape[0], weights.shape[1], self.n_alpha),
            dtype=jnp.float64,
        )
        with_error = error_envelope is not None
        if with_error:
            lower_total = jnp.zeros_like(total)
            upper_total = jnp.zeros_like(total)
        for dataset in range(weights.shape[0]):
            c_x = jnp.einsum("nrm,krm->nkm", x_coeff, weights[dataset])
            c_xy = jnp.einsum(
                "nrm,krm->nkm", x_coeff, weighted_flux[dataset]
            )
            c_xx = jnp.einsum(
                "nrm,krm->nkm", products[0], square_lags[dataset][:, 0]
            )
            for lag in range(1, self.n_lag):
                c_xx += jnp.einsum(
                    "nrm,krm->nkm", products[lag], square_lags[dataset][:, lag]
                )
            q_x = self._series_values(c_x, self.n_alpha)
            q_xx = self._series_values(c_xx, self.n_alpha)
            q_xy = self._series_values(c_xy, self.n_alpha)
            local_weight = weight_sum[dataset][None, :, None]
            local_flux = flux_sum[dataset][None, :, None]
            s_xx = q_xx - q_x * q_x / local_weight
            s_xy = q_xy - q_x * local_flux / local_weight
            s_yy = flux2_sum[dataset] - flux_sum[dataset] ** 2 / weight_sum[dataset]
            scale = jnp.maximum(
                jnp.abs(q_xx), jnp.abs(q_x * q_x / local_weight)
            )
            valid = s_xx > self.singular_rtol * jnp.maximum(scale, 1.0)
            cauchy_limit = s_xx * s_yy[None, :, None]
            violation = s_xy * s_xy - cauchy_limit
            cauchy_tolerance = self.singular_rtol * jnp.maximum(
                jnp.abs(cauchy_limit), 1.0
            )
            valid &= violation <= cauchy_tolerance
            raw = s_yy[None, :, None] - s_xy * s_xy / s_xx
            local_chi2 = jnp.where(valid, jnp.maximum(raw, 0.0), jnp.inf)
            total += local_chi2

            if with_error:
                # For |delta A(t)| <= e(t), the centered cross moment changes
                # by at most sum w |y-ybar| e.  The centered model norm changes
                # by at most ||e||_W.  Triangle inequalities then bound the
                # profiled improvement s_xy**2 / s_xx without evaluating any
                # observation-by-map error matrix.
                cross_error = jnp.einsum(
                    "nr,kr->nk", error_envelope, error_abs_flux[dataset]
                )
                error2 = jnp.einsum(
                    "nr,kr->nk",
                    error_envelope * error_envelope,
                    error_square_lags[dataset][:, 0],
                )
                for lag in range(1, self.n_lag):
                    error_product = jnp.pad(
                        error_envelope[:, : n_radius - lag]
                        * error_envelope[:, lag:],
                        ((0, 0), (0, lag)),
                    )
                    error2 += jnp.einsum(
                        "nr,kr->nk",
                        error_product,
                        error_square_lags[dataset][:, lag],
                    )
                error_norm = jnp.sqrt(jnp.maximum(error2, 0.0))
                model_norm = jnp.sqrt(jnp.maximum(s_xx, 0.0))
                denominator_low = jnp.maximum(
                    model_norm - error_norm[:, :, None], 0.0
                ) ** 2
                denominator_high = (model_norm + error_norm[:, :, None]) ** 2
                cross = jnp.abs(s_xy)
                numerator_low = jnp.maximum(
                    cross - cross_error[:, :, None], 0.0
                ) ** 2
                numerator_high = (cross + cross_error[:, :, None]) ** 2
                improvement_low = jnp.where(
                    denominator_high > 0.0,
                    numerator_low / denominator_high,
                    0.0,
                )
                improvement_high = jnp.where(
                    denominator_low > 0.0,
                    numerator_high / denominator_low,
                    s_yy[None, :, None],
                )
                # Cauchy bounds the physical improvement by s_yy.  Clipping
                # also absorbs harmless roundoff in very strong signals.
                improvement_low = jnp.clip(
                    improvement_low, 0.0, s_yy[None, :, None]
                )
                improvement_high = jnp.clip(
                    improvement_high, 0.0, s_yy[None, :, None]
                )
                lower_total += jnp.maximum(
                    s_yy[None, :, None] - improvement_high, 0.0
                )
                upper_total += jnp.maximum(
                    s_yy[None, :, None] - improvement_low, 0.0
                )
        if with_error:
            return total, lower_total, upper_total
        return total

    def _scan_minima_impl(
        self,
        x_coeff,
        weights,
        weighted_flux,
        square_lags,
        weight_sum,
        flux_sum,
        flux2_sum,
    ):
        """Reduce the geometry/alpha axes before results leave JAX."""
        chi2 = self._scan_impl(
            x_coeff,
            weights,
            weighted_flux,
            square_lags,
            weight_sum,
            flux_sum,
            flux2_sum,
        )
        return self._reduce_minima(chi2)

    def _scan_minima_with_error_impl(
        self,
        x_coeff,
        error_envelope,
        weights,
        weighted_flux,
        square_lags,
        error_abs_flux,
        error_square_lags,
        weight_sum,
        flux_sum,
        flux2_sum,
    ):
        """Reduce central chi-square and a conservative minimized interval."""
        chi2, lower, upper = self._scan_impl(
            x_coeff,
            weights,
            weighted_flux,
            square_lags,
            weight_sum,
            flux_sum,
            flux2_sum,
            error_envelope,
            error_abs_flux,
            error_square_lags,
        )
        central = self._reduce_minima(chi2)
        return central + (
            self._jnp.min(lower, axis=(1, 2)),
            self._jnp.min(upper, axis=(1, 2)),
        )

    def _reduce_minima(self, chi2):
        flat_chi2 = chi2.reshape((chi2.shape[0], -1))
        flat_index = self._jnp.argmin(flat_chi2, axis=1)
        minimum = self._jnp.take_along_axis(
            flat_chi2, flat_index[:, None], axis=1
        )[:, 0]
        return (
            minimum,
            flat_index // self.n_alpha,
            flat_index % self.n_alpha,
        )

    def _scan_minima_masked_impl(
        self,
        x_coeff,
        weights,
        weighted_flux,
        square_lags,
        weight_sum,
        flux_sum,
        flux2_sum,
        feasible,
    ):
        """Reduce the same chi-square cube twice: unrestricted and anomaly-only.

        The expensive part is the cube itself, so the constrained minimum is
        almost free.  Returning both lets a caller rank on the anomaly-compatible
        minimum while still reporting the unrestricted one.
        """
        chi2 = self._scan_impl(
            x_coeff,
            weights,
            weighted_flux,
            square_lags,
            weight_sum,
            flux_sum,
            flux2_sum,
        )
        masked = self._jnp.where(feasible, chi2, self._jnp.inf)
        return self._reduce_minima(chi2) + self._reduce_minima(masked)

    def _scan_minima_masked_with_error_impl(
        self,
        x_coeff,
        error_envelope,
        weights,
        weighted_flux,
        square_lags,
        error_abs_flux,
        error_square_lags,
        weight_sum,
        flux_sum,
        flux2_sum,
        feasible,
    ):
        """Reduce central values and intervals with and without a mask."""
        chi2, lower, upper = self._scan_impl(
            x_coeff,
            weights,
            weighted_flux,
            square_lags,
            weight_sum,
            flux_sum,
            flux2_sum,
            error_envelope,
            error_abs_flux,
            error_square_lags,
        )
        masked_chi2 = self._jnp.where(feasible, chi2, self._jnp.inf)
        masked_lower = self._jnp.where(feasible, lower, self._jnp.inf)
        masked_upper = self._jnp.where(feasible, upper, self._jnp.inf)
        return (
            self._reduce_minima(chi2)
            + self._reduce_minima(masked_chi2)
            + (
                self._jnp.min(lower, axis=(1, 2)),
                self._jnp.min(upper, axis=(1, 2)),
                self._jnp.min(masked_lower, axis=(1, 2)),
                self._jnp.min(masked_upper, axis=(1, 2)),
            )
        )

    def _validate_coefficients(self, x_coeff: np.ndarray) -> np.ndarray:
        x_coeff = np.asarray(x_coeff)
        if x_coeff.ndim != 3 or x_coeff.shape[1:] != (
            self.n_radius,
            self.n_mode,
        ):
            raise ValueError("coefficient array does not match the batched kernel shape")
        return x_coeff

    def scan(self, x_coeff: np.ndarray) -> np.ndarray:
        x_coeff = self._validate_coefficients(x_coeff)
        result = self._compiled_scan(
            self._jnp.asarray(x_coeff),
            self._weights,
            self._weighted_flux,
            self._square_lags,
            self._weight_sum,
            self._flux_sum,
            self._flux2_sum,
        )
        return np.asarray(result)

    def scan_minima(
        self, x_coeff: np.ndarray, feasible: np.ndarray | None = None
    ) -> tuple[np.ndarray, ...]:
        """Return each map's minimum chi-square and its geometry/alpha indices.

        The geometry and alpha reductions happen in the compiled JAX program, so
        this transfers only three length-``n_map`` arrays to NumPy.  Ties follow
        :func:`jax.numpy.argmin` / NumPy row-major convention: the first
        ``(geometry_index, alpha_index)`` is selected.

        ``feasible`` is an optional ``(n_map, n_geometry, n_alpha)`` boolean cube
        restricting which trajectory angles are allowed, as produced by
        :func:`~adamgrid.binary_fft.anomaly_mask.anomaly_feasibility_mask`.  With
        it, six arrays are returned: the unrestricted triple followed by the
        restricted one.  A map with no feasible angle reports ``inf``.
        """
        x_coeff = self._validate_coefficients(x_coeff)
        if feasible is None:
            result = self._compiled_scan_minima(
                self._jnp.asarray(x_coeff),
                self._weights,
                self._weighted_flux,
                self._square_lags,
                self._weight_sum,
                self._flux_sum,
                self._flux2_sum,
            )
            return tuple(np.asarray(value) for value in result)
        feasible = np.asarray(feasible, dtype=bool)
        if feasible.shape != (len(x_coeff), self.n_geometry, self.n_alpha):
            raise ValueError(
                "feasible must have shape (map, geometry, alpha) matching the scan"
            )
        result = self._compiled_scan_minima_masked(
            self._jnp.asarray(x_coeff),
            self._weights,
            self._weighted_flux,
            self._square_lags,
            self._weight_sum,
            self._flux_sum,
            self._flux2_sum,
            self._jnp.asarray(feasible),
        )
        return tuple(np.asarray(value) for value in result)

    def scan_minima_with_error(
        self,
        x_coeff: np.ndarray,
        error_envelope: np.ndarray,
        feasible: np.ndarray | None = None,
    ) -> tuple[np.ndarray, ...]:
        """Return minima plus lower/upper bounds from a node error envelope.

        The final two arrays bound the minimum over every geometry and alpha,
        rather than only the central winner.  Consequently a candidate whose
        lower bound exceeds a competing top-K upper bound can be discarded
        without relying on the approximate chi-square ordering alone.
        """
        x_coeff = self._validate_coefficients(x_coeff)
        error_envelope = np.asarray(error_envelope, dtype=self._real_dtype)
        if error_envelope.shape != x_coeff.shape[:2]:
            raise ValueError(
                "error_envelope must have shape (map, radius) matching coefficients"
            )
        if np.any(~np.isfinite(error_envelope)) or np.any(error_envelope < 0.0):
            raise ValueError("error_envelope must be finite and non-negative")
        arguments = (
            self._jnp.asarray(x_coeff),
            self._jnp.asarray(error_envelope),
            self._weights,
            self._weighted_flux,
            self._square_lags,
            self._error_abs_flux,
            self._error_square_lags,
            self._weight_sum,
            self._flux_sum,
            self._flux2_sum,
        )
        if feasible is None:
            result = self._compiled_scan_minima_with_error(*arguments)
            return tuple(np.asarray(value) for value in result)
        feasible = np.asarray(feasible, dtype=bool)
        if feasible.shape != (len(x_coeff), self.n_geometry, self.n_alpha):
            raise ValueError(
                "feasible must have shape (map, geometry, alpha) matching the scan"
            )
        result = self._compiled_scan_minima_masked_with_error(
            *arguments, self._jnp.asarray(feasible)
        )
        return tuple(np.asarray(value) for value in result)
