from dataclasses import dataclass

import numpy as np
import pytest

from hammlet import Dataset, Geometry, Maps, SearchConfig
from hammlet._core.atlas_builder import MapBuildSpec, PolarAtlasBuilder


@dataclass
class ToyEvaluator:
    amplitude: float

    def magnification(self, x, y):
        radius2 = x * x + y * y
        return 1.0 + self.amplitude * np.exp(-2.0 * radius2) * (
            1.0 + 0.15 * np.cos(2.0 * np.arctan2(y, x))
        )


def toy_maps(path):
    nodes = np.linspace(0.0, 1.5, 24)
    builder = PolarAtlasBuilder(
        nodes, n_phi_build=32, m_max=3, core_m_max=1, shard_size=2
    )
    specs = [
        MapBuildSpec(i, 0.0, -3.0 + i, -3.0, ToyEvaluator(0.2 + 0.1 * i))
        for i in range(3)
    ]
    builder.build(path, specs)
    return Maps.open(path)


def test_reconstructs_cartesian_magnification(tmp_path):
    maps = toy_maps(tmp_path / "maps")
    x = np.asarray([0.2, 0.5])
    y = np.asarray([0.1, -0.2])
    actual = maps.magnification(1, x, y, m_max=3)
    expected = ToyEvaluator(0.3).magnification(x, y)
    np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=5e-3)


def test_end_to_end_search_returns_physical_parameters(tmp_path):
    maps = toy_maps(tmp_path / "maps")
    time = np.linspace(-1.0, 1.0, 60)
    geometry = Geometry(t0=0.0, u0=0.2, tE=1.0)
    tau = time
    x, y = -tau, np.full_like(tau, -0.2)
    flux = 1.7 * ToyEvaluator(0.3).magnification(x, y) + 0.4
    dataset = Dataset(time, flux, np.full_like(time, 0.01))
    result = maps.search(
        [dataset],
        [geometry],
        config=SearchConfig(
            base_m_max=1,
            full_m_max=3,
            base_n_alpha=8,
            full_n_alpha=16,
            top_count=2,
            risk_count=1,
            candidate_count=3,
        ),
    )
    assert result.candidates
    assert result.candidates[0].map_id == 1
    assert result.candidates[0].q == pytest.approx(1e-2)
