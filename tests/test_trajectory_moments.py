from __future__ import annotations

import numpy as np
import pytest

from hamagri._core.config import PSPLGeometry
from hamagri._core.trajectory import ConsistentKernelFactory, PhotometricDataset

_FIELDS = (
    "weighted",
    "weighted_flux",
    "weighted_square_lags",
    "error_weighted_abs_centered_flux",
    "error_weighted_square_lags",
)


def _datasets() -> list[PhotometricDataset]:
    rng = np.random.default_rng(3)
    datasets = []
    for size in (301, 57):
        time = np.sort(rng.uniform(-3.0, 3.0, size))
        datasets.append(
            PhotometricDataset(
                time=time,
                flux=1.0 + rng.normal(0.0, 0.05, size),
                error=rng.uniform(0.01, 0.1, size),
            )
        )
    return datasets


@pytest.mark.parametrize("group_size", [1, 4, 7, 64])
def test_grouped_moments_reproduce_the_observation_sums(group_size: int) -> None:
    datasets = _datasets()
    geometries = [PSPLGeometry(0.0, 0.1, 1.0), PSPLGeometry(0.3, 0.05, 1.2)]
    exact = ConsistentKernelFactory(
        datasets, geometries, 6, radial_order=1, moment_group_size=None
    )
    grouped = ConsistentKernelFactory(
        datasets, geometries, 6, radial_order=1, moment_group_size=group_size
    )
    # Narrow cells far from the origin are where global prefix sums would lose
    # precision; the outer nodes sit exactly on the extreme radii so the
    # clipped observations are exercised too.
    radius = np.concatenate([basis.radius for row in exact._bases for basis in row])
    nodes = np.unique(
        np.concatenate(
            (
                [radius.min(), radius.max()],
                np.linspace(radius.min(), radius.max(), 9)[1:-1],
                2.0 + np.linspace(0.0, 1.0e-3, 5),
            )
        )
    )
    for m_max in (6, 3):
        for exact_row, grouped_row in zip(exact(nodes, m_max), grouped(nodes, m_max)):
            for expected, actual in zip(exact_row, grouped_row):
                for name in _FIELDS:
                    reference = getattr(expected, name)
                    np.testing.assert_allclose(
                        getattr(actual, name),
                        reference,
                        rtol=0.0,
                        atol=1.0e-12 * np.abs(reference).max(),
                        err_msg=name,
                    )


def test_grouped_moments_are_skipped_for_cubic_stencils() -> None:
    factory = ConsistentKernelFactory(
        _datasets(), [PSPLGeometry(0.0, 0.1, 1.0)], 4, radial_order=3
    )
    assert factory._moments is None
