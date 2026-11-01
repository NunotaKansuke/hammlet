"""Fixed-stencil radial interpolation of stored angular Fourier coefficients.

The angular FFT search contracts one map-independent event kernel against every
map in the atlas.  That structure requires the radial interpolation weights to
depend only on the observation radii and the atlas radial nodes, never on the
map coefficients themselves.  Every scheme in this module is therefore a fixed
linear operator; slope-limited schemes such as PCHIP are deliberately excluded.

``order=1`` reproduces the original two-point linear interpolation.  ``order=3``
uses the cubic polynomial through the four nodes bracketing each observation,
which removes most of the residual chi-square discrepancy against direct
``VBMicrolensing`` evaluation without changing the stored atlas.  Wider
stencils were measured to be worse: node spacing near a caustic is comparable
to the source radius, so a higher-degree polynomial rings instead of
converging.

Every scheme interpolates the stored excess ``A - 1`` relative to the point
lens: the polynomial runs through ``(A - 1) / g`` at the nodes and the result
is multiplied by ``g`` at the observation, with ``g(r) = A_PL(r) - 1``.  A
binary map's excess follows the point-lens ``1 / r`` cusp near the centre, and
the atlas nodes there are spaced by a sizeable fraction of their radius, so
interpolating the excess itself overshoots by several percent at high
magnification.  The ratio is smooth across the cusp, and the factor is still a
fixed weight per (observation, node) pair, so the search kernels keep their
map-independent form.
"""

from __future__ import annotations

import numpy as np

DEFAULT_RADIAL_ORDER = 1
SUPPORTED_RADIAL_ORDERS = (1, 3)


def validate_radial_order(order: int) -> int:
    order = int(order)
    if order not in SUPPORTED_RADIAL_ORDERS:
        raise ValueError(
            f"radial_order must be one of {SUPPORTED_RADIAL_ORDERS}; got {order}"
        )
    return order


def stencil_width(order: int) -> int:
    """Number of radial nodes each observation reads."""
    return validate_radial_order(order) + 1


def inverse_point_lens_excess(radius: np.ndarray) -> np.ndarray:
    """Return ``1 / (A_PL(r) - 1)``, which is finite and zero at ``r = 0``.

    ``A_PL - 1 = 4 / (r q (r**2 + 2 + r q))`` with ``q = sqrt(r**2 + 4)``,
    written without the cancellation of the textbook form at large radius.
    """
    radius = np.asarray(radius, dtype=np.float64)
    root = radius * np.sqrt(radius * radius + 4.0)
    return 0.25 * root * (radius * radius + 2.0 + root)


def point_lens_scale(
    radius: np.ndarray, node_radius: np.ndarray
) -> np.ndarray:
    """Factor ``g(r) / g(r_k)`` that turns a ratio stencil into excess weights.

    ``radius`` broadcasts against ``node_radius``.  An observation exactly at
    ``r = 0`` (only reachable on a node at zero) keeps its plain weights.
    """
    observation = inverse_point_lens_excess(radius)
    node = inverse_point_lens_excess(node_radius)
    safe = np.where(observation > 0.0, observation, 1.0)
    return np.where(observation > 0.0, node / safe, 1.0)


def clip_to_radial_range(
    radius: np.ndarray, radial_nodes: np.ndarray
) -> np.ndarray:
    """Clip observation radii onto the atlas range, rejecting real excursions."""
    radius = np.asarray(radius, dtype=np.float64)
    nodes = np.asarray(radial_nodes, dtype=np.float64)
    if nodes.ndim != 1 or nodes.size < 2 or np.any(np.diff(nodes) <= 0.0):
        raise ValueError("radial_nodes must be a strictly increasing 1-D array")
    tolerance = 32.0 * np.finfo(np.float64).eps * max(
        1.0, abs(float(nodes[0])), abs(float(nodes[-1]))
    )
    if np.any(radius < nodes[0] - tolerance) or np.any(
        radius > nodes[-1] + tolerance
    ):
        raise ValueError(
            "event trajectory is outside the atlas radial range; rebuild the atlas "
            "with radial_nodes covering all fitted observations"
        )
    return np.clip(radius, nodes[0], nodes[-1])


def radial_stencil(
    radius: np.ndarray,
    radial_nodes: np.ndarray,
    order: int = DEFAULT_RADIAL_ORDER,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(indices, weights)`` of shape ``(n_observation, order + 1)``.

    The weights are the Lagrange basis of the selected nodes times
    ``point_lens_scale``, so they reproduce ``g(r)`` times any polynomial up to
    ``order`` exactly, and in particular the point lens itself.  Near either end
    of the node array the stencil is shifted inward rather than reduced in
    order, which keeps a single dense shape for the batched kernels.
    """
    indices, weights = lagrange_stencil(radius, radial_nodes, order)
    radius = clip_to_radial_range(radius, radial_nodes)
    nodes = np.asarray(radial_nodes, dtype=np.float64)
    return indices, weights * point_lens_scale(radius[:, None], nodes[indices])


def lagrange_stencil(
    radius: np.ndarray,
    radial_nodes: np.ndarray,
    order: int = DEFAULT_RADIAL_ORDER,
) -> tuple[np.ndarray, np.ndarray]:
    """Plain Lagrange stencil: weights sum to one, no point-lens factor."""
    order = validate_radial_order(order)
    width = order + 1
    nodes = np.asarray(radial_nodes, dtype=np.float64)
    radius = clip_to_radial_range(radius, nodes)
    if nodes.size < width:
        raise ValueError(
            f"radial_order={order} needs at least {width} radial nodes; "
            f"the atlas provides {nodes.size}"
        )

    upper = np.clip(np.searchsorted(nodes, radius, side="right"), 1, nodes.size - 1)
    lower = upper - 1
    if order == 1:
        fraction = (radius - nodes[lower]) / (nodes[upper] - nodes[lower])
        indices = np.stack((lower, upper), axis=1)
        weights = np.stack((1.0 - fraction, fraction), axis=1)
        return indices, weights

    # Centre the stencil on the bracketing cell, then shift it inside the array.
    start = np.clip(lower - (width // 2 - 1), 0, nodes.size - width)
    indices = start[:, None] + np.arange(width)[None, :]
    stencil_nodes = nodes[indices]
    weights = np.ones_like(stencil_nodes)
    for target in range(width):
        for other in range(width):
            if other == target:
                continue
            weights[:, target] *= (radius - stencil_nodes[:, other]) / (
                stencil_nodes[:, target] - stencil_nodes[:, other]
            )
    return indices, weights


def interpolate_coefficients(
    coefficients: np.ndarray,
    radius: np.ndarray,
    radial_nodes: np.ndarray,
    order: int = DEFAULT_RADIAL_ORDER,
) -> np.ndarray:
    """Interpolate ``(radial_node, mode)`` coefficients onto observation radii."""
    coefficients = np.asarray(coefficients)
    if coefficients.ndim != 2:
        raise ValueError("coefficients must have shape (radial_node, mode)")
    nodes = np.asarray(radial_nodes, dtype=np.float64)
    if coefficients.shape[0] != nodes.size:
        raise ValueError("coefficients and radial_nodes shapes do not match")
    indices, weights = radial_stencil(radius, nodes, order)
    return np.einsum(
        "nk,nkm->nm", weights.astype(coefficients.dtype), coefficients[indices]
    )


def segmented_angular_sums(
    n_rows: int,
    indices: np.ndarray,
    weights: np.ndarray,
    angular: np.ndarray,
) -> np.ndarray:
    """Accumulate ``weights[:, p] * angular`` into radial nodes, for every ``p``.

    ``indices`` and ``weights`` both have shape ``(n_observation, n_term)``;
    term ``p`` sends observation ``i`` to node ``indices[i, p]`` with real weight
    ``weights[i, p]``.  The result has shape ``(n_term, n_rows, n_mode)``.

    Doing this one term at a time costs a full ``(n_observation, n_mode)``
    temporary per term, which for a cubic stencil at the full mode budget runs
    to hundreds of megabytes of traffic per event kernel.  Folding every term
    into one sparse operator instead reads ``angular`` exactly once and leaves a
    single sparse-times-dense product, which measured roughly nine times faster
    on the largest benchmark event.
    """
    try:
        from scipy import sparse
    except ImportError as error:
        raise ImportError(
            "segmented_angular_sums requires SciPy; install the search "
            "compatibility environment before running map searches"
        ) from error

    indices = np.asarray(indices)
    weights = np.asarray(weights, dtype=np.float64)
    angular = np.asarray(angular)
    if indices.ndim != 2 or indices.shape != weights.shape:
        raise ValueError("indices and weights must share shape (observation, term)")
    if angular.ndim != 2 or angular.shape[0] != indices.shape[0]:
        raise ValueError("angular must have one row per observation")
    n_observation, n_term = indices.shape
    n_rows = int(n_rows)
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= n_rows):
        raise ValueError("indices fall outside the requested row range")

    # Term-major ordering keeps the row indices ascending whenever the caller
    # supplied observations in ascending radius, which skips the sort below.
    rows = (
        indices.T.astype(np.int64, copy=False)
        + (np.arange(n_term, dtype=np.int64) * n_rows)[:, None]
    ).ravel()
    columns = np.tile(np.arange(n_observation, dtype=np.int64), n_term)
    data = np.ascontiguousarray(weights.T).ravel()
    if np.any(np.diff(rows) < 0):
        order = np.argsort(rows, kind="stable")
        rows, columns, data = rows[order], columns[order], data[order]
    indptr = np.searchsorted(rows, np.arange(n_term * n_rows + 1))
    operator = sparse.csr_matrix(
        (data, columns, indptr), shape=(n_term * n_rows, n_observation)
    )
    n_mode = angular.shape[1]
    if np.iscomplexobj(angular):
        # scipy's real kernel is roughly 1.5x faster than its complex one on the
        # same bytes, and the operator is real, so the interleaved real view of
        # the angular basis is the cheaper input even after the copy that makes
        # a sliced basis contiguous.
        real_angular = np.ascontiguousarray(
            angular, dtype=np.complex128
        ).view(np.float64)
        product = np.asarray(operator @ real_angular).view(np.complex128)
    else:
        product = np.asarray(operator @ angular)
    return product.reshape(n_term, n_rows, n_mode)
