"""Direct, error-controlled Fourier spectra from binary-lens evaluators.

The fast path uses nested uniform angular grids and FFTs.  Rings intersecting
or approaching a point-source caustic receive additional non-uniform samples
around the caustic angles.  The stored low modes are therefore the object of
the adaptive error test; no intermediate Cartesian magnification map is
required.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from numbers import Integral
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .atlas_builder import PolarAtlasBuilder
from .caustics import adamgrid_map_origin_shift, caustics_to_adamgrid_map_frame
from .certification import certify_sampled_ring
from .map_adapter import point_lens_magnification


class RingMagnificationEvaluator(Protocol):
    """Minimum interface required by the direct spectrum sampler."""

    source_radius: float
    caustic_components: tuple[np.ndarray, ...]

    def magnification(self, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...


def _power_of_two(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 2:
        raise ValueError(f"{name} must be an integer >= 2")
    value = int(value)
    if value & (value - 1):
        raise ValueError(f"{name} must be a power of two")
    return value


@dataclass(frozen=True)
class DirectSpectrumConfig:
    """Accuracy and work limits for one direct-VBM angular spectrum."""

    m_max: int = 384
    core_m_max: int = 96
    min_n_phi: int = 512
    caustic_base_n_phi: int = 2048
    max_n_phi: int = 8192
    coefficient_rtol: float = 2.0e-4
    coefficient_atol: float = 2.0e-7
    caustic_guard_rho: float = 4.0
    caustic_local_levels: int = 8
    caustic_points_per_side: int = 4
    diagnostic_m_max: int = 768

    def __post_init__(self) -> None:
        for name in ("m_max", "core_m_max", "diagnostic_m_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
        if self.m_max < 1 or not 0 <= self.core_m_max <= self.m_max:
            raise ValueError("require m_max >= 1 and 0 <= core_m_max <= m_max")
        if self.diagnostic_m_max < self.m_max:
            raise ValueError("diagnostic_m_max must be at least m_max")
        minimum = _power_of_two(self.min_n_phi, "min_n_phi")
        caustic_base = _power_of_two(self.caustic_base_n_phi, "caustic_base_n_phi")
        maximum = _power_of_two(self.max_n_phi, "max_n_phi")
        if (
            minimum < 2 * (self.core_m_max + 1)
            or caustic_base < 2 * (self.m_max + 1)
            or not minimum <= caustic_base <= maximum
        ):
            raise ValueError(
                "angular grids must satisfy max_n_phi >= caustic_base_n_phi >= "
                "2*(m_max+1) and min_n_phi >= 2*(core_m_max+1)"
            )
        for name in ("coefficient_rtol", "coefficient_atol", "caustic_guard_rho"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("caustic_local_levels", "caustic_points_per_side"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class DirectRingSpectrum:
    x_coeff: np.ndarray
    x2_coeff: np.ndarray
    reconstruction_error: float
    reconstruction_error_core: float
    certified_error: float
    certified_error_core: float
    certificate_unresolved_error: float
    coefficient_error: float
    normalized_coefficient_error: float
    x_normalized_coefficient_error: float
    x2_normalized_coefficient_error: float
    deviation_envelope: float
    evaluations: int
    uniform_n_phi: int
    caustic_angles: tuple[float, ...]
    caustic_guarded: bool
    local_refined: bool
    global_converged: bool
    local_converged: bool
    converged: bool


def _circle_segment_angles(
    components: Sequence[np.ndarray], radius: float, guard_distance: float
) -> tuple[float, ...]:
    """Return circle intersections and near-tangent caustic directions."""
    if radius < 0.0 or guard_distance < 0.0:
        raise ValueError("radius and guard_distance must be non-negative")
    angles: list[float] = []
    for raw_component in components:
        component = np.asarray(raw_component, dtype=np.float64)
        if component.ndim != 2 or component.shape[1] != 2 or len(component) < 2:
            raise ValueError("caustic components must have shape (point, 2)")
        if not np.all(np.isfinite(component)):
            raise ValueError("caustic components must be finite")
        for left, right in zip(component[:-1], component[1:]):
            delta = right - left
            a = float(np.dot(delta, delta))
            if a == 0.0:
                continue
            b = 2.0 * float(np.dot(left, delta))
            c = float(np.dot(left, left) - radius * radius)
            discriminant = b * b - 4.0 * a * c
            if discriminant >= 0.0:
                root = np.sqrt(discriminant)
                for fraction in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)):
                    if 0.0 <= fraction <= 1.0:
                        point = left + fraction * delta
                        angles.append(
                            float(np.arctan2(point[1], point[0]) % (2 * np.pi))
                        )
            candidates = (0.0, 1.0, float(np.clip(-np.dot(left, delta) / a, 0, 1)))
            closest = min(
                (left + fraction * delta for fraction in candidates),
                key=lambda point: abs(float(np.hypot(point[0], point[1])) - radius),
            )
            if abs(float(np.hypot(closest[0], closest[1])) - radius) <= guard_distance:
                angles.append(float(np.arctan2(closest[1], closest[0]) % (2 * np.pi)))
    if not angles:
        return ()
    ordered = np.sort(np.asarray(angles))
    angular_tolerance = max(
        guard_distance / max(radius, guard_distance, 1.0e-15), 1.0e-8
    )
    clusters = [[float(ordered[0])]]
    for angle in ordered[1:]:
        if float(angle) - clusters[-1][-1] <= angular_tolerance:
            clusters[-1].append(float(angle))
        else:
            clusters.append([float(angle)])
    if len(clusters) > 1 and (
        clusters[0][0] + 2.0 * np.pi - clusters[-1][-1] <= angular_tolerance
    ):
        clusters[0] = [angle + 2.0 * np.pi for angle in clusters[0]] + clusters.pop()
    return tuple(float(np.mean(cluster) % (2.0 * np.pi)) for cluster in clusters)


def _series_bound(coefficients: np.ndarray) -> float:
    if coefficients.size == 0:
        return 0.0
    return float(abs(coefficients[0]) + 2.0 * np.sum(np.abs(coefficients[1:])))


def _upper_float32(values: np.ndarray) -> np.ndarray:
    """Round non-negative bounds outward when storing them as float32."""
    values = np.asarray(values, dtype=np.float64)
    stored = values.astype(np.float32)
    return np.where(
        stored.astype(np.float64) < values,
        np.nextafter(stored, np.float32(np.inf)),
        stored,
    )


def _coefficient_change(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    m_max: int,
) -> tuple[float, float]:
    delta = _series_bound(current[: m_max + 1] - previous[: m_max + 1])
    scale = max(_series_bound(current[: m_max + 1]), 1.0)
    return delta, scale


def _nonuniform_coefficients(
    angles: np.ndarray, values: np.ndarray, m_max: int
) -> np.ndarray:
    """Fourier integral of the periodic piecewise-linear interpolant.

    Integrating the phase analytically avoids the mode leakage that an
    ordinary trapezoid rule introduces when locally refined and coarse
    intervals coexist.
    """
    order = np.argsort(angles, kind="stable")
    angle = np.asarray(angles, dtype=np.float64)[order]
    value = np.asarray(values, dtype=np.float64)[order]
    if angle.ndim != 1 or value.shape != angle.shape or len(angle) < 3:
        raise ValueError("angles and values must be equal one-dimensional arrays")
    if np.any(np.diff(angle) <= 0.0):
        raise ValueError("non-uniform angular nodes must be unique")
    width = np.diff(np.concatenate((angle, [angle[0] + 2.0 * np.pi])))
    right_value = np.roll(value, -1)
    output = np.empty(m_max + 1, dtype=np.complex128)
    output[0] = np.sum(0.5 * width * (value + right_value)) / (2.0 * np.pi)
    if m_max:
        modes = np.arange(1, m_max + 1, dtype=np.float64)
        mh = width[:, None] * modes[None, :]
        exponential = np.exp(-1j * mh)
        j0 = (1.0 - exponential) / (1j * modes[None, :])
        j1 = -width[:, None] * exponential / (1j * modes[None, :]) + j0 / (
            1j * modes[None, :]
        )
        slope = (right_value - value) / width
        phase = np.exp(-1j * angle[:, None] * modes[None, :])
        output[1:] = np.sum(
            phase * (value[:, None] * j0 + slope[:, None] * j1),
            axis=0,
        ) / (2.0 * np.pi)
    return output


def _sample_angles(
    evaluator: RingMagnificationEvaluator,
    radius: float,
    angles: np.ndarray,
) -> np.ndarray:
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    magnification = np.asarray(evaluator.magnification(x, y), dtype=np.float64)
    if magnification.shape != angles.shape or not np.all(np.isfinite(magnification)):
        raise ValueError("direct magnification evaluator returned invalid values")
    return magnification - 1.0


def _reconstruction_error(
    coefficients: np.ndarray, angles: np.ndarray, values: np.ndarray
) -> float:
    modes = np.arange(len(coefficients), dtype=np.float64)
    reconstructed = np.real(
        coefficients[0]
        + 2.0
        * np.sum(
            coefficients[None, 1:] * np.exp(1j * angles[:, None] * modes[None, 1:]),
            axis=1,
        )
    )
    return float(np.max(np.abs(values - reconstructed)))


def adaptive_ring_spectrum(
    evaluator: RingMagnificationEvaluator,
    radius: float,
    config: DirectSpectrumConfig,
    *,
    radial_guard_distance: float = 0.0,
) -> DirectRingSpectrum:
    """Compute one auditable low-pass ring spectrum directly from an evaluator."""
    radius = float(radius)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("radius must be finite and non-negative")
    radial_guard_distance = float(radial_guard_distance)
    if not np.isfinite(radial_guard_distance) or radial_guard_distance < 0.0:
        raise ValueError("radial_guard_distance must be finite and non-negative")
    source_radius = float(evaluator.source_radius)
    if not np.isfinite(source_radius) or source_radius <= 0.0:
        raise ValueError("evaluator source_radius must be finite and positive")
    if radius == 0.0:
        excess = float(
            _sample_angles(evaluator, radius, np.asarray([0.0], dtype=np.float64))[0]
        )
        x_coeff = np.zeros(config.m_max + 1, dtype=np.complex128)
        x2_coeff = np.zeros(config.m_max + 1, dtype=np.complex128)
        x_coeff[0] = excess
        x2_coeff[0] = excess * excess
        return DirectRingSpectrum(
            x_coeff=x_coeff,
            x2_coeff=x2_coeff,
            reconstruction_error=0.0,
            reconstruction_error_core=0.0,
            certified_error=0.0,
            certified_error_core=0.0,
            certificate_unresolved_error=0.0,
            coefficient_error=0.0,
            normalized_coefficient_error=0.0,
            x_normalized_coefficient_error=0.0,
            x2_normalized_coefficient_error=0.0,
            deviation_envelope=np.inf,
            evaluations=1,
            uniform_n_phi=1,
            caustic_angles=(),
            caustic_guarded=False,
            local_refined=False,
            global_converged=True,
            local_converged=True,
            converged=True,
        )
    guard_distance = max(
        config.caustic_guard_rho * source_radius, radial_guard_distance
    )
    caustic_angles = _circle_segment_angles(
        evaluator.caustic_components, radius, guard_distance
    )
    local_refined = bool(caustic_angles) and radius > source_radius
    diagnostic_m_max = config.diagnostic_m_max
    cache: dict[float, float] = {}

    def values_for(angles: np.ndarray) -> np.ndarray:
        normalized = np.mod(np.asarray(angles, dtype=np.float64), 2.0 * np.pi)
        keys = [round(float(angle), 14) for angle in normalized]
        missing_keys = []
        missing_angles = []
        for key, angle in zip(keys, normalized):
            if key not in cache:
                missing_keys.append(key)
                missing_angles.append(angle)
        if missing_angles:
            sampled = _sample_angles(
                evaluator, radius, np.asarray(missing_angles, dtype=np.float64)
            )
            cache.update(zip(missing_keys, map(float, sampled)))
        return np.asarray([cache[key] for key in keys], dtype=np.float64)

    previous_x = None
    previous_x2 = None
    uniform_n_phi = config.min_n_phi
    coefficient_error = np.inf
    normalized_coefficient_error = np.inf
    x_normalized_coefficient_error = np.inf
    x2_normalized_coefficient_error = np.inf
    global_converged = False
    final_angles = np.empty(0)
    final_values = np.empty(0)
    final_x = np.empty(0, dtype=np.complex128)
    final_x2 = np.empty(0, dtype=np.complex128)

    while True:
        angles = 2.0 * np.pi * np.arange(uniform_n_phi) / uniform_n_phi
        values = values_for(angles)
        modes = min(diagnostic_m_max, uniform_n_phi // 2)
        x_coeff = np.fft.rfft(values)[: modes + 1] / uniform_n_phi
        x2_coeff = np.fft.rfft(values * values)[: modes + 1] / uniform_n_phi
        if previous_x is not None:
            common = min(len(previous_x), len(x_coeff)) - 1
            x_error, x_scale = _coefficient_change(
                previous_x, x_coeff, m_max=min(config.m_max, common)
            )
            x2_error, x2_scale = _coefficient_change(
                previous_x2, x2_coeff, m_max=min(config.m_max, common)
            )
            coefficient_error = max(x_error, x2_error)
            x_normalized_coefficient_error = x_error / (
                config.coefficient_atol + config.coefficient_rtol * x_scale
            )
            x2_normalized_coefficient_error = x2_error / (
                config.coefficient_atol + config.coefficient_rtol * x2_scale
            )
            normalized_coefficient_error = max(
                x_normalized_coefficient_error,
                x2_normalized_coefficient_error,
            )
            global_converged = (
                x_error <= config.coefficient_atol + config.coefficient_rtol * x_scale
                and x2_error
                <= config.coefficient_atol + config.coefficient_rtol * x2_scale
            )
            if common < config.m_max and len(x_coeff) > common + 1:
                stored_stop = min(config.m_max + 1, len(x_coeff))
                new_x = 2.0 * float(np.sum(np.abs(x_coeff[common + 1 : stored_stop])))
                new_x2 = 2.0 * float(np.sum(np.abs(x2_coeff[common + 1 : stored_stop])))
                global_converged = global_converged and (
                    new_x <= config.coefficient_atol + config.coefficient_rtol * x_scale
                    and new_x2
                    <= config.coefficient_atol + config.coefficient_rtol * x2_scale
                )
        final_angles, final_values = angles, values
        final_x, final_x2 = x_coeff, x2_coeff
        has_stored_modes = len(x_coeff) >= config.m_max + 1
        if (
            (global_converged and has_stored_modes)
            or uniform_n_phi == config.max_n_phi
            or (local_refined and uniform_n_phi >= config.caustic_base_n_phi)
        ):
            break
        previous_x, previous_x2 = x_coeff, x2_coeff
        uniform_n_phi *= 2

    local_converged = not local_refined
    if local_refined:
        half_width = max(
            guard_distance / max(radius, source_radius),
            2.0 * np.pi / uniform_n_phi,
        )
        previous_x, previous_x2 = final_x, final_x2
        base_angles = final_angles
        for level in range(config.caustic_local_levels):
            points_per_side = config.caustic_points_per_side * (2**level)
            local = [base_angles]
            for center in caustic_angles:
                local.append(
                    center
                    + np.linspace(-half_width, half_width, 2 * points_per_side + 1)
                )
            angles = np.unique(np.round(np.mod(np.concatenate(local), 2 * np.pi), 14))
            values = values_for(angles)
            modes = min(diagnostic_m_max, max(config.m_max, uniform_n_phi // 2))
            x_coeff = _nonuniform_coefficients(angles, values, modes)
            x2_coeff = _nonuniform_coefficients(angles, values * values, modes)
            x_error, x_scale = _coefficient_change(
                previous_x, x_coeff, m_max=config.m_max
            )
            x2_error, x2_scale = _coefficient_change(
                previous_x2, x2_coeff, m_max=config.m_max
            )
            coefficient_error = max(x_error, x2_error)
            x_normalized_coefficient_error = x_error / (
                config.coefficient_atol + config.coefficient_rtol * x_scale
            )
            x2_normalized_coefficient_error = x2_error / (
                config.coefficient_atol + config.coefficient_rtol * x2_scale
            )
            normalized_coefficient_error = max(
                x_normalized_coefficient_error,
                x2_normalized_coefficient_error,
            )
            local_converged = (
                x_error <= config.coefficient_atol + config.coefficient_rtol * x_scale
                and x2_error
                <= config.coefficient_atol + config.coefficient_rtol * x2_scale
            )
            final_angles, final_values = angles, values
            final_x, final_x2 = x_coeff, x2_coeff
            previous_x, previous_x2 = x_coeff, x2_coeff
            if local_converged and level >= 1:
                break

    stored_x = final_x[: config.m_max + 1]
    stored_x2 = final_x2[: config.m_max + 1]
    core_x = final_x[: config.core_m_max + 1]
    reconstruction_error = _reconstruction_error(
        stored_x, final_angles, final_values
    ) + float(coefficient_error)
    reconstruction_error_core = _reconstruction_error(
        core_x, final_angles, final_values
    ) + float(coefficient_error)
    certificate = certify_sampled_ring(
        final_x,
        final_angles,
        final_values,
        coefficient_change_bound=(
            float(coefficient_error) if np.isfinite(coefficient_error) else 0.0
        ),
    )
    pspl = float(point_lens_magnification(np.asarray(radius * radius)))
    return DirectRingSpectrum(
        x_coeff=stored_x,
        x2_coeff=stored_x2,
        reconstruction_error=reconstruction_error,
        reconstruction_error_core=reconstruction_error_core,
        certified_error=certificate.error_bound(config.m_max),
        certified_error_core=certificate.error_bound(config.core_m_max),
        certificate_unresolved_error=certificate.unresolved_bound,
        coefficient_error=float(coefficient_error),
        normalized_coefficient_error=float(normalized_coefficient_error),
        x_normalized_coefficient_error=float(x_normalized_coefficient_error),
        x2_normalized_coefficient_error=float(x2_normalized_coefficient_error),
        deviation_envelope=float(
            np.max(np.abs(final_values + 1.0 - pspl)) + coefficient_error
        ),
        evaluations=len(cache),
        uniform_n_phi=uniform_n_phi,
        caustic_angles=caustic_angles,
        caustic_guarded=bool(caustic_angles),
        local_refined=local_refined,
        global_converged=bool(global_converged),
        local_converged=bool(local_converged),
        converged=bool(local_converged if local_refined else global_converged),
    )


class VBMBinaryLensEvaluator:
    """Vector-shaped adapter around scalar ``VBMicrolensing.BinaryMag2``."""

    def __init__(
        self,
        separation: float,
        mass_ratio: float,
        source_radius: float,
        *,
        tolerance: float = 1.0e-3,
        relative_tolerance: float = 1.0e-4,
    ) -> None:
        try:
            import VBMicrolensing
        except ImportError as error:
            raise ImportError(
                "VBMBinaryLensEvaluator requires the optional VBMicrolensing package"
            ) from error
        for name, value in (
            ("separation", separation),
            ("mass_ratio", mass_ratio),
            ("source_radius", source_radius),
            ("tolerance", tolerance),
            ("relative_tolerance", relative_tolerance),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        self.separation = float(separation)
        self.mass_ratio = float(mass_ratio)
        self.source_radius = float(source_radius)
        self._shift_x = adamgrid_map_origin_shift(separation, mass_ratio)
        self._vbm = VBMicrolensing.VBMicrolensing()
        self._vbm.Tol = float(tolerance)
        self._vbm.RelTol = float(relative_tolerance)
        self._vbm.a1 = 0.0
        native_components = []
        for component in self._vbm.Caustics(self.separation, self.mass_ratio):
            array = np.asarray(component, dtype=np.float64)
            if array.ndim == 2 and array.shape[0] == 2:
                array = array.T
            if array.ndim != 2 or array.shape[1] != 2:
                raise ValueError("VBMicrolensing returned an invalid caustic component")
            if not np.allclose(array[0], array[-1]):
                array = np.vstack((array, array[0]))
            native_components.append(array)
        self.caustic_components = caustics_to_adamgrid_map_frame(
            native_components,
            separation=self.separation,
            mass_ratio=self.mass_ratio,
        )

    def magnification(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x, y = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        )
        shape = x.shape
        output = np.fromiter(
            (
                self._vbm.BinaryMag2(
                    self.separation,
                    self.mass_ratio,
                    float(local_x + self._shift_x),
                    float(local_y),
                    self.source_radius,
                )
                for local_x, local_y in zip(x.ravel(), y.ravel())
            ),
            dtype=np.float64,
            count=x.size,
        )
        return output.reshape(shape)


class DirectVBMPolarAtlasBuilder(PolarAtlasBuilder):
    """Build the existing atlas format without an intermediate spatial map."""

    def __init__(
        self,
        radial_nodes: np.ndarray,
        *,
        spectrum_config: DirectSpectrumConfig,
        shard_size: int = 256,
        coefficient_dtype: str = "complex64",
    ) -> None:
        super().__init__(
            radial_nodes,
            n_phi_build=spectrum_config.max_n_phi,
            m_max=spectrum_config.m_max,
            core_m_max=spectrum_config.core_m_max,
            shard_size=shard_size,
            coefficient_dtype=coefficient_dtype,
        )
        self.spectrum_config = spectrum_config
        self._map_diagnostics: list[dict[str, object]] = []

    def sample(self, evaluator: RingMagnificationEvaluator) -> dict[str, np.ndarray]:
        radial_guard = np.empty_like(self.radial_nodes)
        radial_guard[0] = 0.5 * (self.radial_nodes[1] - self.radial_nodes[0])
        radial_guard[-1] = 0.5 * (self.radial_nodes[-1] - self.radial_nodes[-2])
        radial_guard[1:-1] = 0.25 * (self.radial_nodes[2:] - self.radial_nodes[:-2])
        rings = [
            adaptive_ring_spectrum(
                evaluator,
                radius,
                self.spectrum_config,
                radial_guard_distance=float(radial_guard[index]),
            )
            for index, radius in enumerate(self.radial_nodes)
        ]
        exact_x = np.stack([item.x_coeff for item in rings])
        stored_x = exact_x.astype(self.coefficient_dtype)
        rounding_delta = stored_x.astype(np.complex128) - exact_x
        rounding_error = np.abs(rounding_delta[:, 0]) + 2.0 * np.sum(
            np.abs(rounding_delta[:, 1:]), axis=1
        )
        core_delta = rounding_delta[:, : self.spectrum_config.core_m_max + 1]
        core_rounding_error = np.abs(core_delta[:, 0]) + 2.0 * np.sum(
            np.abs(core_delta[:, 1:]), axis=1
        )
        self._map_diagnostics.append(
            {
                "evaluations": np.asarray(
                    [item.evaluations for item in rings], dtype=np.int32
                ),
                "uniform_n_phi": np.asarray(
                    [item.uniform_n_phi for item in rings], dtype=np.int32
                ),
                "caustic_guarded": np.asarray(
                    [item.caustic_guarded for item in rings], dtype=bool
                ),
                "local_refined": np.asarray(
                    [item.local_refined for item in rings], dtype=bool
                ),
                "converged": np.asarray([item.converged for item in rings], dtype=bool),
                "coefficient_error": np.asarray(
                    [item.coefficient_error for item in rings], dtype=np.float64
                ),
                "normalized_coefficient_error": np.asarray(
                    [item.normalized_coefficient_error for item in rings],
                    dtype=np.float64,
                ),
                "x_normalized_coefficient_error": np.asarray(
                    [item.x_normalized_coefficient_error for item in rings],
                    dtype=np.float64,
                ),
                "x2_normalized_coefficient_error": np.asarray(
                    [item.x2_normalized_coefficient_error for item in rings],
                    dtype=np.float64,
                ),
                "certificate_unresolved_error": np.asarray(
                    [item.certificate_unresolved_error for item in rings],
                    dtype=np.float64,
                ),
            }
        )
        return {
            "x_coeff": stored_x,
            "x2_coeff": np.stack([item.x2_coeff for item in rings]).astype(
                self.coefficient_dtype
            ),
            "reconstruction_error": np.asarray(
                [item.reconstruction_error for item in rings], dtype=np.float32
            ),
            "reconstruction_error_core": np.asarray(
                [item.reconstruction_error_core for item in rings], dtype=np.float32
            ),
            "certified_error": _upper_float32(
                np.asarray([item.certified_error for item in rings])
                + rounding_error
            ),
            "certified_error_core": _upper_float32(
                np.asarray([item.certified_error_core for item in rings])
                + core_rounding_error
            ),
            "deviation_envelope": np.asarray(
                [item.deviation_envelope for item in rings],
                dtype=np.float32,
            ),
        }

    def build(self, *args, **kwargs) -> Path:
        output = super().build(*args, **kwargs)
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        diagnostics = {
            name: np.stack([np.asarray(item[name]) for item in self._map_diagnostics])
            for name in self._map_diagnostics[0]
        }
        np.savez_compressed(output / "direct_diagnostics.npz", **diagnostics)
        manifest["builder"] = "direct-vbm-adaptive-fourier"
        manifest["direct_spectrum_config"] = asdict(self.spectrum_config)
        manifest["direct_diagnostics"] = {
            "total_vbm_evaluations": int(np.sum(diagnostics["evaluations"])),
            "unconverged_rings": int(np.sum(~diagnostics["converged"])),
            "caustic_guarded_rings": int(np.sum(diagnostics["caustic_guarded"])),
            "max_normalized_coefficient_error": float(
                np.max(diagnostics["normalized_coefficient_error"])
            ),
            "per_ring_diagnostics": "direct_diagnostics.npz",
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return output


__all__ = [
    "DirectRingSpectrum",
    "DirectSpectrumConfig",
    "DirectVBMPolarAtlasBuilder",
    "RingMagnificationEvaluator",
    "VBMBinaryLensEvaluator",
    "adaptive_ring_spectrum",
]
