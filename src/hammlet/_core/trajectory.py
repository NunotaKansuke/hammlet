from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import PSPLGeometry
from .radial import (
    DEFAULT_RADIAL_ORDER,
    radial_stencil,
    segmented_angular_sums,
    validate_radial_order,
)


@dataclass(frozen=True)
class PhotometricDataset:
    time: np.ndarray
    flux: np.ndarray
    error: np.ndarray
    name: str = ""

    def __post_init__(self) -> None:
        time = np.asarray(self.time, dtype=np.float64)
        flux = np.asarray(self.flux, dtype=np.float64)
        error = np.asarray(self.error, dtype=np.float64)
        if time.ndim != 1 or not (time.shape == flux.shape == error.shape):
            raise ValueError("time, flux, and error must be same-length 1-D arrays")
        if time.size < 3:
            raise ValueError("a dataset needs at least three observations")
        if not np.all(np.isfinite(time)) or not np.all(np.isfinite(flux)):
            raise ValueError("time and flux must be finite")
        if not np.all(np.isfinite(error)) or np.any(error <= 0):
            raise ValueError("errors must be finite and positive")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "error", error)


@dataclass(frozen=True)
class EventKernel:
    weighted: np.ndarray
    weighted_flux: np.ndarray
    weight_sum: float
    flux_sum: float
    flux2_sum: float


@dataclass(frozen=True)
class ConsistentEventKernel:
    """Weights for moments of one consistently interpolated light curve.

    ``weighted_square_lags`` has shape ``(radial_order + 1, radial_node, mode)``.
    Entry ``d`` carries the weights that multiply the product of the coefficients
    at radial nodes ``r`` and ``r + d``, so a width-``w`` radial stencil needs
    lags ``0 .. w - 1``.  Rows without a partner node are zero, which keeps one
    dense shape for the batched scanners.

    The two ``error_*`` arrays propagate a non-negative magnification-error
    envelope without expanding it to every observation.  Absolute radial
    stencil weights make this valid for cubic interpolation, whose ordinary
    Lagrange weights can be negative.
    """

    weighted: np.ndarray
    weighted_flux: np.ndarray
    weighted_square_lags: np.ndarray
    error_weighted_abs_centered_flux: np.ndarray
    error_weighted_square_lags: np.ndarray
    weight_sum: float
    flux_sum: float
    flux2_sum: float

    @property
    def radial_order(self) -> int:
        return int(self.weighted_square_lags.shape[0]) - 1

    @property
    def weighted_square_node(self) -> np.ndarray:
        """Lag-zero weights, the only square term a linear stencil produces."""
        return self.weighted_square_lags[0]

    @property
    def weighted_square_edge(self) -> np.ndarray:
        """Lag-one weights, trimmed to the edges they belong to."""
        return self.weighted_square_lags[1][:-1]


@dataclass(frozen=True)
class TrajectoryBasis:
    """Radius, phase and angular basis for one dataset and one geometry.

    None of these depend on the atlas radial nodes, so a bucketed atlas can
    build them once and reuse them for every bucket and every mode budget.  The
    angular matrix is the dominant construction cost, and rebuilding it per
    bucket was the largest single component of the measured search time.

    Observations are stored in ascending radius.  Every quantity a kernel builds
    from them is a sum, so the order is unobservable, but it makes the radial
    stencil indices of each column monotonic and lets the scatter skip its
    reordering gather.
    """

    radius: np.ndarray
    phase: np.ndarray
    weight: np.ndarray
    weighted_flux_values: np.ndarray
    angular: np.ndarray
    weight_sum: float
    flux_sum: float
    flux2_sum: float

    @property
    def max_mode(self) -> int:
        return int(self.angular.shape[1]) - 1


def radial_coordinates(time: np.ndarray, geometry: PSPLGeometry) -> tuple[np.ndarray, np.ndarray]:
    geometry.validate()
    tau = (np.asarray(time, dtype=np.float64) - geometry.t0) / geometry.tE
    radius = np.hypot(tau, geometry.u0)
    # This phase matches x=-tau, y=-u0 at alpha=0 in the released C scorer.
    phase = np.arctan2(-geometry.u0, -tau)
    return radius, phase


def _angular_basis(phase: np.ndarray, m_max: int) -> np.ndarray:
    """Return ``exp(1j * phase * m)`` for ``m = 0 .. m_max``.

    A cumulative product of one complex exponential replaces ``m_max`` complex
    exponentials per observation.  The accumulated rounding error grows like
    ``m_max * eps`` and stays far below the atlas coefficient precision.
    """
    phase = np.asarray(phase, dtype=np.float64)
    angular = np.empty((phase.size, int(m_max) + 1), dtype=np.complex128)
    angular[:, 0] = 1.0
    if m_max >= 1:
        step = np.exp(1j * phase)
        angular[:, 1] = step
        for mode in range(2, int(m_max) + 1):
            angular[:, mode] = angular[:, mode - 1] * step
    return angular


def build_trajectory_basis(
    dataset: PhotometricDataset,
    geometry: PSPLGeometry,
    m_max: int,
    *,
    radial_order: int = DEFAULT_RADIAL_ORDER,
) -> TrajectoryBasis:
    """Precompute everything a kernel needs that is independent of radial nodes.

    ``m_max`` is the *linear* mode budget; the basis stores ``2 * m_max + 1``
    modes because the square moment reaches twice as far.
    """
    validate_radial_order(radial_order)
    if m_max < 0:
        raise ValueError("m_max must be nonnegative")
    radius, phase = radial_coordinates(dataset.time, geometry)
    weight = 1.0 / dataset.error**2
    ascending = np.argsort(radius, kind="stable")
    radius = radius[ascending]
    phase = phase[ascending]
    weight = weight[ascending]
    flux = dataset.flux[ascending]
    return TrajectoryBasis(
        radius=radius,
        phase=phase,
        weight=weight,
        weighted_flux_values=weight * flux,
        angular=_angular_basis(phase, 2 * int(m_max)),
        weight_sum=float(np.sum(weight)),
        flux_sum=float(np.dot(weight, flux)),
        flux2_sum=float(np.dot(weight, flux**2)),
    )


def build_event_kernel(
    dataset: PhotometricDataset,
    geometry: PSPLGeometry,
    radial_nodes: np.ndarray,
    m_max: int,
    *,
    radial_order: int = DEFAULT_RADIAL_ORDER,
) -> EventKernel:
    radius, phase = radial_coordinates(dataset.time, geometry)
    # Ascending radius makes every stencil column's indices monotonic.
    ascending = np.argsort(radius, kind="stable")
    radius = radius[ascending]
    phase = phase[ascending]
    weight = 1.0 / dataset.error[ascending] ** 2
    flux = dataset.flux[ascending]
    indices, basis = radial_stencil(radius, radial_nodes, radial_order)
    angular = _angular_basis(phase, m_max)
    n_node = len(radial_nodes)
    width = indices.shape[1]
    scale = weight[:, None] * basis
    terms = segmented_angular_sums(
        n_node,
        np.hstack((indices, indices)),
        np.hstack((scale, scale * flux[:, None])),
        angular,
    )
    weighted = terms[:width].sum(axis=0)
    weighted_flux = terms[width:].sum(axis=0)
    return EventKernel(
        weighted=weighted,
        weighted_flux=weighted_flux,
        weight_sum=float(np.sum(weight)),
        flux_sum=float(np.dot(weight, flux)),
        flux2_sum=float(np.dot(weight, flux**2)),
    )


def consistent_kernel_from_basis(
    basis: TrajectoryBasis,
    radial_nodes: np.ndarray,
    m_max: int,
    *,
    radial_order: int = DEFAULT_RADIAL_ORDER,
) -> ConsistentEventKernel:
    """Bind a node-independent trajectory basis to one atlas radial grid."""
    radial_order = validate_radial_order(radial_order)
    m_max = int(m_max)
    if m_max < 0:
        raise ValueError("m_max must be nonnegative")
    if 2 * m_max > basis.max_mode:
        raise ValueError(
            "trajectory basis was built for a smaller m_max than requested"
        )
    nodes = np.asarray(radial_nodes, dtype=np.float64)
    n_node = nodes.size
    indices, stencil = radial_stencil(basis.radius, nodes, radial_order)
    width = indices.shape[1]

    linear_angular = basis.angular[:, : m_max + 1]
    square_angular = basis.angular[:, : 2 * m_max + 1]

    # Both linear moments read the same stencil, so they go through one sparse
    # operator with the stencil columns listed twice.
    scale = basis.weight[:, None] * stencil
    linear_terms = segmented_angular_sums(
        n_node,
        np.hstack((indices, indices)),
        np.hstack((scale, basis.weighted_flux_values[:, None] * stencil)),
        linear_angular,
    )
    weighted = linear_terms[:width].sum(axis=0)
    weighted_flux = linear_terms[width:].sum(axis=0)

    # x(t)**2 expands into products of stencil nodes.  Grouping the (column,
    # column + lag) pairs by lag keeps one dense array per lag, so the scanner
    # contracts exactly ``radial_order + 1`` product spectra.
    pairs = [
        (lag, column)
        for lag in range(radial_order + 1)
        for column in range(width - lag)
    ]
    square_indices = np.empty((basis.radius.size, len(pairs)), dtype=indices.dtype)
    square_weights = np.empty((basis.radius.size, len(pairs)), dtype=np.float64)
    for term, (lag, column) in enumerate(pairs):
        partner = column + lag
        if indices[0, partner] - indices[0, column] != lag:
            raise RuntimeError("radial stencil indices are not consecutive")
        square_indices[:, term] = indices[:, column]
        pair_weight = basis.weight * stencil[:, column] * stencil[:, partner]
        # (a, b) and (b, a) are the same product coefficient.
        square_weights[:, term] = 2.0 * pair_weight if lag else pair_weight
    square_terms = segmented_angular_sums(
        n_node, square_indices, square_weights, square_angular
    )

    lags = np.zeros(
        (radial_order + 1, n_node, 2 * m_max + 1), dtype=np.complex128
    )
    offset = 0
    for lag in range(radial_order + 1):
        count = width - lag
        lags[lag] = square_terms[offset : offset + count].sum(axis=0)
        offset += count
        if lag:
            # Node ``n_node - lag`` and beyond have no partner to pair with.
            lags[lag, n_node - lag :] = 0.0

    # If a stored node has magnification uncertainty e_r, the interpolated
    # uncertainty is at most sum_r |a_r| e_r.  These moments let the scanner
    # evaluate ||e||_W and |<e, y-ybar>_W| with one real contraction per lag,
    # independent of the angular grid size.
    absolute_stencil = np.abs(stencil)
    flux = basis.weighted_flux_values / basis.weight
    centered_flux = np.abs(flux - basis.flux_sum / basis.weight_sum)
    error_weighted_abs_centered_flux = np.zeros(n_node, dtype=np.float64)
    for column in range(width):
        np.add.at(
            error_weighted_abs_centered_flux,
            indices[:, column],
            basis.weight
            * centered_flux
            * absolute_stencil[:, column],
        )

    error_weighted_square_lags = np.zeros(
        (radial_order + 1, n_node), dtype=np.float64
    )
    for lag in range(radial_order + 1):
        for column in range(width - lag):
            partner = column + lag
            pair_weight = (
                basis.weight
                * absolute_stencil[:, column]
                * absolute_stencil[:, partner]
            )
            if lag:
                pair_weight *= 2.0
            np.add.at(
                error_weighted_square_lags[lag],
                indices[:, column],
                pair_weight,
            )

    return ConsistentEventKernel(
        weighted=weighted,
        weighted_flux=weighted_flux,
        weighted_square_lags=lags,
        error_weighted_abs_centered_flux=error_weighted_abs_centered_flux,
        error_weighted_square_lags=error_weighted_square_lags,
        weight_sum=basis.weight_sum,
        flux_sum=basis.flux_sum,
        flux2_sum=basis.flux2_sum,
    )


def build_consistent_event_kernel(
    dataset: PhotometricDataset,
    geometry: PSPLGeometry,
    radial_nodes: np.ndarray,
    m_max: int,
    *,
    radial_order: int = DEFAULT_RADIAL_ORDER,
) -> ConsistentEventKernel:
    """Build weights for squaring the same radially interpolated low-pass map."""
    basis = build_trajectory_basis(
        dataset, geometry, m_max, radial_order=radial_order
    )
    return consistent_kernel_from_basis(
        basis, radial_nodes, m_max, radial_order=radial_order
    )


class ConsistentKernelFactory:
    """Produce kernel batches for a bucketed atlas from one set of bases.

    ``scan_bucketed_atlas_multiresolution`` asks for a fresh kernel batch per
    bucket and per mode budget, because every bucket owns its own radial nodes.
    Everything except the radial stencil is shared across those calls, so the
    trajectory bases are built once at the largest mode budget and re-bound to
    each radial grid.  The call signature matches the ``kernel_factory``
    protocol expected by the multi-resolution scan.
    """

    def __init__(
        self,
        datasets: Sequence[PhotometricDataset],
        geometries: Sequence[PSPLGeometry],
        max_m_max: int,
        *,
        radial_order: int = DEFAULT_RADIAL_ORDER,
    ) -> None:
        datasets = list(datasets)
        geometries = list(geometries)
        if not datasets or not geometries:
            raise ValueError("at least one dataset and one geometry are required")
        self.radial_order = validate_radial_order(radial_order)
        self.max_m_max = int(max_m_max)
        self.call_count = 0
        self.build_seconds = 0.0
        started = time.perf_counter()
        self._bases = [
            [
                build_trajectory_basis(
                    dataset, geometry, self.max_m_max, radial_order=self.radial_order
                )
                for dataset in datasets
            ]
            for geometry in geometries
        ]
        self.basis_seconds = time.perf_counter() - started

    def __call__(
        self,
        radial_nodes: np.ndarray,
        m_max: int,
        radial_order: int | None = None,
    ) -> list[list[ConsistentEventKernel]]:
        if int(m_max) > self.max_m_max:
            raise ValueError(
                "kernel factory was built for a smaller maximum mode budget"
            )
        order = (
            self.radial_order
            if radial_order is None
            else validate_radial_order(radial_order)
        )
        started = time.perf_counter()
        batches = [
            [
                consistent_kernel_from_basis(
                    basis, radial_nodes, m_max, radial_order=order
                )
                for basis in geometry_bases
            ]
            for geometry_bases in self._bases
        ]
        self.build_seconds += time.perf_counter() - started
        self.call_count += 1
        return batches
