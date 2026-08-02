from __future__ import annotations

import numpy as np

from hammlet._core.certification import (
    certify_sampled_ring,
    radial_interval_to_node_envelope,
    radial_piecewise_linear_residual_bound,
)
from hammlet._core.radial import (
    interpolate_coefficients,
    radial_stencil,
)


def _series(coefficients: np.ndarray, angles: np.ndarray) -> np.ndarray:
    modes = np.arange(len(coefficients), dtype=np.float64)
    return np.real(
        coefficients[0]
        + 2.0
        * np.sum(
            coefficients[None, 1:]
            * np.exp(1j * angles[:, None] * modes[None, 1:]),
            axis=1,
        )
    )


def test_certificate_covers_every_point_of_sampled_reference() -> None:
    count = 64
    angles = np.arange(count) * (2.0 * np.pi / count)
    values = (
        0.4
        + 0.3 * np.cos(3.0 * angles)
        - 0.2 * np.sin(7.0 * angles)
        + 0.15 * np.cos(11.0 * angles)
    )
    diagnostic = np.fft.rfft(values)[:17] / count
    certificate = certify_sampled_ring(diagnostic, angles, values)
    retained_m_max = 4
    retained = diagnostic[: retained_m_max + 1]

    dense = np.arange(32768) * (2.0 * np.pi / 32768)
    extended_angles = np.concatenate((angles, [2.0 * np.pi]))
    extended_values = np.concatenate((values, values[:1]))
    sampled_reference = np.interp(dense, extended_angles, extended_values)
    actual = np.max(np.abs(sampled_reference - _series(retained, dense)))

    assert actual <= certificate.error_bound(retained_m_max)


def test_certificate_frontier_shrinks_monotonically_with_m() -> None:
    count = 128
    angles = np.arange(count) * (2.0 * np.pi / count)
    distance = np.angle(np.exp(1j * (angles - 0.37)))
    values = 0.2 + 3.0 * np.exp(-0.5 * (distance / 0.08) ** 2)
    diagnostic = np.fft.rfft(values)[:49] / count
    certificate = certify_sampled_ring(diagnostic, angles, values)

    frontier = certificate.frontier(np.asarray([4, 8, 16, 32, 48]))

    assert np.all(np.diff(frontier) <= 0.0)
    assert frontier[-1] == certificate.unresolved_bound


def test_radial_certificate_covers_complete_piecewise_linear_reference() -> None:
    stored_nodes = np.asarray([0.0, 0.3, 0.9, 1.8, 3.0])
    reference_nodes = np.sort(
        np.concatenate(
            (stored_nodes, 0.5 * (stored_nodes[:-1] + stored_nodes[1:]))
        )
    )

    def coefficients(radius):
        radius = np.asarray(radius)
        return np.stack(
            (
                0.2 + 0.1 * radius**2,
                (0.03 + 0.02j) * np.sin(1.3 * radius),
                (0.01 - 0.015j) * radius**3,
            ),
            axis=-1,
        )

    exact_stored = coefficients(stored_nodes)
    # Exercise the storage-rounding term through the same path as production.
    stored = exact_stored.astype(np.complex64)
    reference = coefficients(reference_nodes)
    reference_error = np.linspace(1.0e-5, 3.0e-5, len(reference_nodes))
    phase = np.linspace(0.0, 2.0 * np.pi, 97, endpoint=False)

    for order in (1, 3):
        interval = radial_piecewise_linear_residual_bound(
            stored_nodes,
            stored,
            reference_nodes,
            reference,
            reference_error,
            order=order,
        )
        node_envelope = radial_interval_to_node_envelope(interval, order=order)
        for segment in range(len(reference_nodes) - 1):
            radii = np.linspace(
                reference_nodes[segment], reference_nodes[segment + 1], 31
            )
            fraction = (radii - radii[0]) / (radii[-1] - radii[0])
            declared = reference[segment] + fraction[:, None] * (
                reference[segment + 1] - reference[segment]
            )
            reconstructed = interpolate_coefficients(
                stored, radii, stored_nodes, order=order
            )
            delta = declared - reconstructed
            actual = np.max(
                np.abs(
                    np.real(
                        delta[:, :1]
                        + 2.0
                        * np.sum(
                            delta[:, None, 1:]
                            * np.exp(
                                1j
                                * phase[None, :, None]
                                * np.arange(1, delta.shape[1])[None, None, :]
                            ),
                            axis=-1,
                        )
                    )
                ),
                axis=1,
            )
            indices, weights = radial_stencil(radii, stored_nodes, order)
            propagated = np.sum(np.abs(weights) * node_envelope[indices], axis=1)
            angular = np.maximum(
                reference_error[segment], reference_error[segment + 1]
            )
            assert np.all(actual + angular <= propagated + 1.0e-12)
