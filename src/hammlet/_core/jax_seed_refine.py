"""JIT-batched coefficient objective and bounded seed pattern search."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np

from .trajectory import PhotometricDataset


@dataclass(frozen=True)
class JAXSeedBatchConfig:
    """Controls the breadth/depth racing used for large seed handoffs."""

    m_max: int = 96
    batch_size: int = 32
    shallow_levels: int = 2
    deep_levels: int = 10
    pairwise_levels: int = 0
    deep_count: int = 32
    neighbor_limit: int = 26

    def __post_init__(self) -> None:
        for name in (
            "m_max",
            "batch_size",
            "shallow_levels",
            "deep_levels",
            "pairwise_levels",
            "deep_count",
            "neighbor_limit",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
        if self.m_max < 1 or self.batch_size < 1 or self.deep_count < 1:
            raise ValueError("m_max, batch_size, and deep_count must be positive")
        if self.shallow_levels < 0 or self.deep_levels < 0 or self.pairwise_levels < 0:
            raise ValueError("refinement levels must be non-negative")
        if self.neighbor_limit < 0:
            raise ValueError("neighbor_limit must be non-negative")


@dataclass(frozen=True)
class BatchedPatternResult:
    parameters: np.ndarray
    chi2: np.ndarray
    initial_chi2: np.ndarray
    evaluations_per_candidate: int
    levels: int


class JAXFixedAlphaBatchEvaluator:
    """Evaluate many map/trajectory pairs without materializing mode cubes.

    The angular series is accumulated mode by mode using a complex phase
    recurrence. Peak memory therefore scales as ``candidate * observation``
    rather than ``candidate * observation * angular_mode``.
    """

    def __init__(
        self,
        datasets: Sequence[PhotometricDataset],
        *,
        m_max: int,
        radial_order: int = 3,
        singular_rtol: float = 1.0e-12,
    ) -> None:
        try:
            import jax

            jax.config.update("jax_enable_x64", True)
            import jax.numpy as jnp
        except ImportError as error:
            raise ImportError(
                "JAXFixedAlphaBatchEvaluator requires the optional jax dependency"
            ) from error
        if not datasets:
            raise ValueError("at least one photometric dataset is required")
        if not isinstance(m_max, Integral) or isinstance(m_max, bool) or m_max < 1:
            raise ValueError("m_max must be a positive integer")
        if not np.isfinite(singular_rtol) or singular_rtol <= 0.0:
            raise ValueError("singular_rtol must be finite and positive")

        self._jax = jax
        self._jnp = jnp
        self.m_max = int(m_max)
        if int(radial_order) not in (1, 3):
            raise ValueError("radial_order must be 1 or 3")
        self.radial_order = int(radial_order)
        self.singular_rtol = float(singular_rtol)
        self._time = jnp.asarray(np.concatenate([item.time for item in datasets]))
        self._flux = jnp.asarray(np.concatenate([item.flux for item in datasets]))
        self._weight = jnp.asarray(
            np.concatenate([1.0 / item.error**2 for item in datasets])
        )
        offsets = np.cumsum([0] + [len(item.time) for item in datasets])
        self._dataset_slices = tuple(
            (int(offsets[index]), int(offsets[index + 1]))
            for index in range(len(datasets))
        )
        self._compiled = jax.jit(self._evaluate_impl)

    @property
    def backend(self) -> str:
        """Active JAX backend name for provenance reporting."""
        return str(self._jax.default_backend())

    @property
    def devices(self) -> tuple[str, ...]:
        """JAX devices visible after any caller-applied affinity limit."""
        return tuple(str(device) for device in self._jax.devices())

    def _evaluate_impl(self, coefficients, radial_nodes, parameters):
        jnp = self._jnp
        time = self._time
        tau = (time[None, :] - parameters[:, 0, None]) / jnp.exp(parameters[:, 2, None])
        u0 = parameters[:, 1, None]
        radius = jnp.hypot(tau, u0)
        angle = jnp.arctan2(-u0, -tau) + parameters[:, 3, None]
        upper = self._jax.vmap(
            lambda nodes, values: jnp.searchsorted(nodes, values, side="right")
        )(radial_nodes, radius)
        upper = jnp.clip(upper, 1, radial_nodes.shape[1] - 1)
        lower = upper - 1
        if self.radial_order == 1:
            lower_radius = jnp.take_along_axis(radial_nodes, lower, axis=1)
            upper_radius = jnp.take_along_axis(radial_nodes, upper, axis=1)
            fraction = (radius - lower_radius) / (upper_radius - lower_radius)

            def interpolated_mode(mode):
                local = coefficients[:, :, mode]
                left = jnp.take_along_axis(local, lower, axis=1)
                right = jnp.take_along_axis(local, upper, axis=1)
                return (1.0 - fraction) * left + fraction * right

        else:
            start = jnp.clip(lower - 1, 0, radial_nodes.shape[1] - 4)
            indices = start[:, :, None] + jnp.arange(4)[None, None, :]
            node_rows = jnp.take_along_axis(
                radial_nodes[:, :, None], indices, axis=1
            )
            weights = jnp.ones_like(node_rows)
            for target in range(4):
                for other in range(4):
                    if target != other:
                        weights = weights.at[:, :, target].multiply(
                            (radius - node_rows[:, :, other])
                            / (node_rows[:, :, target] - node_rows[:, :, other])
                        )

            def interpolated_mode(mode):
                local = coefficients[:, :, mode]
                values = jnp.take_along_axis(local[:, :, None], indices, axis=1)
                return jnp.sum(values * weights, axis=2)

        excess = jnp.real(interpolated_mode(0))
        rotation = jnp.exp(1j * angle)

        def add_mode(mode, state):
            values, harmonic = state
            values = values + 2.0 * jnp.real(interpolated_mode(mode) * harmonic)
            return values, harmonic * rotation

        excess, _ = self._jax.lax.fori_loop(
            1,
            coefficients.shape[2],
            add_mode,
            (excess, rotation),
        )

        total = jnp.zeros(coefficients.shape[0], dtype=jnp.float64)
        for start, stop in self._dataset_slices:
            weight = self._weight[start:stop]
            flux = self._flux[start:stop]
            design = excess[:, start:stop]
            weight_sum = jnp.sum(weight)
            flux_sum = jnp.sum(weight * flux)
            flux2_sum = jnp.sum(weight * flux * flux)
            x_sum = jnp.sum(design * weight, axis=1)
            xx_sum = jnp.sum(design * design * weight, axis=1)
            xy_sum = jnp.sum(design * (weight * flux)[None, :], axis=1)
            s_xx = xx_sum - x_sum * x_sum / weight_sum
            s_xy = xy_sum - x_sum * flux_sum / weight_sum
            s_yy = flux2_sum - flux_sum * flux_sum / weight_sum
            valid = s_xx > self.singular_rtol * jnp.maximum(jnp.abs(xx_sum), 1.0)
            local = s_yy - s_xy * s_xy / s_xx
            total += jnp.where(valid, jnp.maximum(local, 0.0), jnp.inf)

        outside = jnp.any(
            (radius < radial_nodes[:, :1]) | (radius > radial_nodes[:, -1:]),
            axis=1,
        )
        return jnp.where(outside, jnp.inf, total)

    def evaluate(
        self,
        coefficients: np.ndarray,
        radial_nodes: np.ndarray,
        parameters: np.ndarray,
    ) -> np.ndarray:
        """Return one profiled chi-square per candidate row."""
        coefficients = np.asarray(coefficients)
        radial_nodes = np.asarray(radial_nodes, dtype=np.float64)
        parameters = np.asarray(parameters, dtype=np.float64)
        if coefficients.ndim != 3:
            raise ValueError("coefficients must have shape (candidate, radius, mode)")
        expected = (coefficients.shape[0], coefficients.shape[1])
        if radial_nodes.shape != expected:
            raise ValueError("radial_nodes must have shape (candidate, radius)")
        if parameters.shape != (coefficients.shape[0], 4):
            raise ValueError("parameters must contain (t0, u0, log(tE), alpha)")
        if coefficients.shape[2] != self.m_max + 1:
            raise ValueError("coefficient mode count does not match m_max")
        if np.any(np.diff(radial_nodes, axis=1) <= 0.0):
            raise ValueError("every radial-node row must be strictly increasing")
        if not np.all(np.isfinite(parameters)):
            raise ValueError("parameters must be finite")
        return np.asarray(
            self._compiled(
                self._jnp.asarray(coefficients),
                self._jnp.asarray(radial_nodes),
                self._jnp.asarray(parameters),
            )
        )


def _padded_batch(
    arrays: Sequence[np.ndarray], start: int, stop: int, batch_size: int
) -> tuple[list[np.ndarray], int]:
    count = stop - start
    output = [np.asarray(value[start:stop]) for value in arrays]
    if count < batch_size:
        padding = batch_size - count
        output = [
            np.concatenate((value, np.repeat(value[-1:], padding, axis=0)), axis=0)
            for value in output
        ]
    return output, count


def evaluate_in_fixed_batches(
    evaluator: JAXFixedAlphaBatchEvaluator,
    coefficients: np.ndarray,
    radial_nodes: np.ndarray,
    parameters: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    """Use one static JIT shape, padding only the final batch."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output = []
    for start in range(0, len(parameters), batch_size):
        stop = min(start + batch_size, len(parameters))
        (local_coefficients, local_nodes, local_parameters), count = _padded_batch(
            (coefficients, radial_nodes, parameters), start, stop, batch_size
        )
        output.append(
            evaluator.evaluate(local_coefficients, local_nodes, local_parameters)[
                :count
            ]
        )
    return np.concatenate(output) if output else np.empty(0, dtype=np.float64)


def batched_pattern_refine(
    evaluator: JAXFixedAlphaBatchEvaluator,
    coefficients: np.ndarray,
    radial_nodes: np.ndarray,
    initial: np.ndarray,
    bounds: np.ndarray,
    *,
    levels: int,
    batch_size: int,
) -> BatchedPatternResult:
    """Run a projected ``center +/- axis`` search for every candidate.

    Each level needs eight trial evaluations instead of the 80 new points in
    the four-dimensional tensor stencil. The two directions of one axis are
    compared against the same incumbent, then that axis is committed before
    moving to the next one. This coordinate-pattern sweep can improve several
    parameters per level while remaining deterministic.
    """
    initial = np.asarray(initial, dtype=np.float64)
    bounds = np.asarray(bounds, dtype=np.float64)
    if initial.ndim != 2 or initial.shape[1] != 4:
        raise ValueError("initial must have shape (candidate, 4)")
    if bounds.shape != (len(initial), 4, 2):
        raise ValueError("bounds must have shape (candidate, 4, 2)")
    if np.any(bounds[:, :, 0] >= bounds[:, :, 1]):
        raise ValueError("every refinement bound must be increasing")
    if levels < 0:
        raise ValueError("levels must be non-negative")
    current = initial.copy()
    widths = 0.5 * (bounds[:, :, 1] - bounds[:, :, 0])
    initial_chi2 = evaluate_in_fixed_batches(
        evaluator,
        coefficients,
        radial_nodes,
        current,
        batch_size=batch_size,
    )
    best = initial_chi2.copy()
    for _ in range(levels):
        for axis in range(4):
            axis_parameters = current.copy()
            axis_chi2 = best.copy()
            for sign in (-1.0, 1.0):
                trial = current.copy()
                trial[:, axis] = np.clip(
                    current[:, axis] + sign * widths[:, axis],
                    bounds[:, axis, 0],
                    bounds[:, axis, 1],
                )
                values = evaluate_in_fixed_batches(
                    evaluator,
                    coefficients,
                    radial_nodes,
                    trial,
                    batch_size=batch_size,
                )
                improved = values < axis_chi2
                axis_chi2[improved] = values[improved]
                axis_parameters[improved] = trial[improved]
            current = axis_parameters
            best = axis_chi2
        widths *= 0.5
    return BatchedPatternResult(
        parameters=current,
        chi2=best,
        initial_chi2=initial_chi2,
        evaluations_per_candidate=1 + 8 * levels,
        levels=int(levels),
    )


def batched_pairwise_refine(
    evaluator: JAXFixedAlphaBatchEvaluator,
    coefficients: np.ndarray,
    radial_nodes: np.ndarray,
    initial: np.ndarray,
    bounds: np.ndarray,
    *,
    levels: int,
    batch_size: int,
) -> BatchedPatternResult:
    """Search correlated two-parameter directions around each incumbent.

    Each level sweeps the six axis pairs.  Four sign combinations are compared
    against the same incumbent for one pair, then the best point is committed.
    This is reserved for the small deep-racing cohort: it captures diagonal
    valleys without paying for the full 4-D tensor stencil on every seed.
    """
    initial = np.asarray(initial, dtype=np.float64)
    bounds = np.asarray(bounds, dtype=np.float64)
    if initial.ndim != 2 or initial.shape[1] != 4:
        raise ValueError("initial must have shape (candidate, 4)")
    if bounds.shape != (len(initial), 4, 2):
        raise ValueError("bounds must have shape (candidate, 4, 2)")
    if np.any(bounds[:, :, 0] >= bounds[:, :, 1]):
        raise ValueError("every refinement bound must be increasing")
    if levels < 0:
        raise ValueError("levels must be non-negative")

    current = initial.copy()
    widths = 0.5 * (bounds[:, :, 1] - bounds[:, :, 0])
    initial_chi2 = evaluate_in_fixed_batches(
        evaluator,
        coefficients,
        radial_nodes,
        current,
        batch_size=batch_size,
    )
    best = initial_chi2.copy()
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    for _ in range(levels):
        for first, second in pairs:
            pair_parameters = current.copy()
            pair_chi2 = best.copy()
            for first_sign, second_sign in (
                (-1.0, -1.0),
                (-1.0, 1.0),
                (1.0, -1.0),
                (1.0, 1.0),
            ):
                trial = current.copy()
                for axis, sign in (
                    (first, first_sign),
                    (second, second_sign),
                ):
                    trial[:, axis] = np.clip(
                        current[:, axis] + sign * widths[:, axis],
                        bounds[:, axis, 0],
                        bounds[:, axis, 1],
                    )
                values = evaluate_in_fixed_batches(
                    evaluator,
                    coefficients,
                    radial_nodes,
                    trial,
                    batch_size=batch_size,
                )
                improved = values < pair_chi2
                pair_chi2[improved] = values[improved]
                pair_parameters[improved] = trial[improved]
            current = pair_parameters
            best = pair_chi2
        widths *= 0.5
    return BatchedPatternResult(
        parameters=current,
        chi2=best,
        initial_chi2=initial_chi2,
        evaluations_per_candidate=1 + 24 * levels,
        levels=int(levels),
    )


__all__ = [
    "BatchedPatternResult",
    "JAXFixedAlphaBatchEvaluator",
    "JAXSeedBatchConfig",
    "batched_pattern_refine",
    "batched_pairwise_refine",
    "evaluate_in_fixed_batches",
]
