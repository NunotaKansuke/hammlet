from __future__ import annotations

import numpy as np

from hammlet._core.certification import certify_sampled_ring


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
