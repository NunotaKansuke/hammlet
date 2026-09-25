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


DEFAULT_MOMENT_GROUP_SIZE = 32


@dataclass(frozen=True)
class GroupedMoments:
    """Radial moments of consecutive observation groups, for linear stencils.

    With ``radial_order=1`` every stencil weight is affine in the radius inside
    one radial cell, so the contribution of a group of observations that all
    fall in the same cell depends on them only through ``sum w x**k A`` for
    ``k = 0, 1, 2``, with ``x`` the radius offset from the group centre.  Those
    moments do not depend on the atlas nodes and are built once per basis.
    Binding a bucket then touches one moment row per interior group plus the raw
    rows of the few groups that straddle a cell boundary, instead of every
    observation.

    Moments are taken about each group's own centre rather than as global prefix
    sums: atlas cells are up to thousands of times narrower than their radius,
    and differencing cumulative sums would cancel most of the significant
    digits.  Here every recombination coefficient is of order one.
    """

    starts: np.ndarray
    ends: np.ndarray
    centers: np.ndarray
    # (3 * n_group, 2 * M + 1): sum w A, sum w x A, sum w x**2 A, group-major blocks.
    square: np.ndarray
    # (4 * n_group, M + 1): sum w A, sum w x A, sum wf A, sum wf x A.
    linear: np.ndarray

    @property
    def n_group(self) -> int:
        return int(self.starts.size)

    @property
    def max_mode(self) -> int:
        return int(self.square.shape[1]) - 1


def build_grouped_moments(
    basis: TrajectoryBasis, group_size: int = DEFAULT_MOMENT_GROUP_SIZE
) -> GroupedMoments:
    from scipy import sparse

    group_size = int(group_size)
    if group_size < 1:
        raise ValueError("group_size must be positive")
    n_observation = basis.radius.size
    starts = np.arange(0, n_observation, group_size, dtype=np.int64)
    ends = np.minimum(starts + group_size, n_observation)
    n_group = starts.size
    centers = 0.5 * (basis.radius[starts] + basis.radius[ends - 1])
    group = np.repeat(np.arange(n_group, dtype=np.int64), ends - starts)
    offset = basis.radius - centers[group]
    columns = np.arange(n_observation, dtype=np.int64)

    def moments(values: list[np.ndarray], angular: np.ndarray) -> np.ndarray:
        rows = np.concatenate([group + k * n_group for k in range(len(values))])
        operator = sparse.csr_matrix(
            (
                np.concatenate(values),
                (rows, np.tile(columns, len(values))),
            ),
            shape=(len(values) * n_group, n_observation),
        )
        real = np.ascontiguousarray(angular).view(np.float64)
        return np.ascontiguousarray(
            np.asarray(operator @ real)
        ).view(np.complex128)

    weight = basis.weight
    weighted_flux = basis.weighted_flux_values
    square = moments(
        [weight, weight * offset, weight * offset**2], basis.angular
    )
    linear = moments(
        [weight, weight * offset, weighted_flux, weighted_flux * offset],
        basis.angular[:, : basis.max_mode // 2 + 1],
    )
    return GroupedMoments(
        starts=starts, ends=ends, centers=centers, square=square, linear=linear
    )


def _grouped_angular_terms(
    basis: TrajectoryBasis,
    moments: GroupedMoments,
    nodes: np.ndarray,
    indices: np.ndarray,
    stencil: np.ndarray,
    m_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy import sparse

    n_node = nodes.size
    n_group = moments.n_group
    lower = indices[:, 0]
    first = moments.starts
    last = moments.ends - 1
    # Clipped radii are not affine in the raw radius, so a group touching the
    # clipped range is also handled observation by observation.
    interior = (
        (lower[first] == lower[last])
        & (basis.radius[first] >= nodes[0])
        & (basis.radius[last] <= nodes[-1])
    )
    raw = np.repeat(~interior, moments.ends - moments.starts)
    weighted, weighted_flux, lags = _stencil_angular_terms(
        n_node,
        indices[raw],
        stencil[raw],
        basis.weight[raw],
        basis.weighted_flux_values[raw],
        basis.angular[raw, : 2 * m_max + 1],
        m_max,
        1,
    )

    group = np.flatnonzero(interior)
    if group.size == 0:
        return weighted, weighted_flux, lags
    cell = lower[first[group]]
    inverse_width = 1.0 / (nodes[cell + 1] - nodes[cell])
    a = (moments.centers[group] - nodes[cell]) * inverse_width
    b = inverse_width
    upper = cell + 1

    def apply(
        n_row: int,
        entries: list[tuple[np.ndarray, int, np.ndarray]],
        dense: np.ndarray,
        n_mode: int,
    ) -> np.ndarray:
        rows = np.concatenate([row for row, _, _ in entries])
        cols = np.concatenate(
            [np.full(group.size, block * n_group) + group for _, block, _ in entries]
        )
        data = np.concatenate([value for _, _, value in entries])
        operator = sparse.csr_matrix(
            (data, (rows, cols)), shape=(n_row, dense.shape[0])
        )
        if dense.shape[1] != n_mode:
            dense = np.ascontiguousarray(dense[:, :n_mode])
        product = operator @ dense.view(np.float64)
        return np.asarray(product).view(np.complex128)

    # Linear weights: 1 - f at the lower node and f at the upper node, with
    # f = a + b x.  Rows [0, n_node) are weighted, [n_node, 2 n_node) are flux.
    linear = apply(
        2 * n_node,
        [
            (cell, 0, 1.0 - a),
            (cell, 1, -b),
            (upper, 0, a),
            (upper, 1, b),
            (cell + n_node, 2, 1.0 - a),
            (cell + n_node, 3, -b),
            (upper + n_node, 2, a),
            (upper + n_node, 3, b),
        ],
        moments.linear,
        m_max + 1,
    )
    weighted += linear[:n_node]
    weighted_flux += linear[n_node:]

    # Square weights: (1 - f)**2 and f**2 on lag 0, 2 f (1 - f) on lag 1.
    one_minus = 1.0 - a
    square = apply(
        2 * n_node,
        [
            (cell, 0, one_minus * one_minus),
            (cell, 1, -2.0 * one_minus * b),
            (cell, 2, b * b),
            (upper, 0, a * a),
            (upper, 1, 2.0 * a * b),
            (upper, 2, b * b),
            (cell + n_node, 0, 2.0 * a * one_minus),
            (cell + n_node, 1, 2.0 * b * (one_minus - a)),
            (cell + n_node, 2, -2.0 * b * b),
        ],
        moments.square,
        2 * m_max + 1,
    )
    lags[0] += square[:n_node]
    lags[1] += square[n_node:]
    return weighted, weighted_flux, lags


def _stencil_angular_terms(
    n_node: int,
    indices: np.ndarray,
    stencil: np.ndarray,
    weight: np.ndarray,
    weighted_flux_values: np.ndarray,
    angular: np.ndarray,
    m_max: int,
    radial_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate the linear and square angular moments observation by observation."""
    width = indices.shape[1]
    n_observation = indices.shape[0]
    linear_angular = angular[:, : m_max + 1]
    square_angular = angular[:, : 2 * m_max + 1]
    if n_observation == 0:
        return (
            np.zeros((n_node, m_max + 1), dtype=np.complex128),
            np.zeros((n_node, m_max + 1), dtype=np.complex128),
            np.zeros((radial_order + 1, n_node, 2 * m_max + 1), dtype=np.complex128),
        )

    # Both linear moments read the same stencil, so they go through one sparse
    # operator with the stencil columns listed twice.
    scale = weight[:, None] * stencil
    linear_terms = segmented_angular_sums(
        n_node,
        np.hstack((indices, indices)),
        np.hstack((scale, weighted_flux_values[:, None] * stencil)),
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
    square_indices = np.empty((n_observation, len(pairs)), dtype=indices.dtype)
    square_weights = np.empty((n_observation, len(pairs)), dtype=np.float64)
    for term, (lag, column) in enumerate(pairs):
        partner = column + lag
        if indices[0, partner] - indices[0, column] != lag:
            raise RuntimeError("radial stencil indices are not consecutive")
        square_indices[:, term] = indices[:, column]
        pair_weight = weight * stencil[:, column] * stencil[:, partner]
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
    return weighted, weighted_flux, lags


def consistent_kernel_from_basis(
    basis: TrajectoryBasis,
    radial_nodes: np.ndarray,
    m_max: int,
    *,
    radial_order: int = DEFAULT_RADIAL_ORDER,
    moments: "GroupedMoments | None" = None,
) -> ConsistentEventKernel:
    """Bind a node-independent trajectory basis to one atlas radial grid.

    ``moments`` (linear radial order only) lets whole groups of observations
    that fall inside one radial cell be reduced from precomputed moments; the
    result is the same sum, evaluated with far less memory traffic.
    """
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

    if moments is not None and radial_order == 1:
        if moments.max_mode < 2 * m_max:
            raise ValueError("grouped moments were built for a smaller m_max")
        weighted, weighted_flux, lags = _grouped_angular_terms(
            basis, moments, nodes, indices, stencil, m_max
        )
    else:
        weighted, weighted_flux, lags = _stencil_angular_terms(
            n_node,
            indices,
            stencil,
            basis.weight,
            basis.weighted_flux_values,
            basis.angular[:, : 2 * m_max + 1],
            m_max,
            radial_order,
        )

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
        moment_group_size: int | None = DEFAULT_MOMENT_GROUP_SIZE,
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
        # Grouped moments only apply to linear stencils; ``None`` disables them.
        self._moments = (
            [
                [build_grouped_moments(basis, moment_group_size) for basis in row]
                for row in self._bases
            ]
            if moment_group_size is not None and self.radial_order == 1
            else None
        )
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
        moments = self._moments if order == 1 else None
        batches = [
            [
                consistent_kernel_from_basis(
                    basis,
                    radial_nodes,
                    m_max,
                    radial_order=order,
                    moments=None if moments is None else moments[g][d],
                )
                for d, basis in enumerate(geometry_bases)
            ]
            for g, geometry_bases in enumerate(self._bases)
        ]
        self.build_seconds += time.perf_counter() - started
        self.call_count += 1
        return batches
