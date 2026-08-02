from dataclasses import dataclass

import numpy as np

from hammlet import Dataset, Geometry, Maps, SearchConfig
from hammlet._core.atlas_builder import MapBuildSpec, PolarAtlasBuilder
from hammlet._core.bucketed_atlas import write_bucket_manifest


@dataclass
class ToyEvaluator:
    harmonic: float

    def magnification(self, x, y):
        radius2 = x * x + y * y
        return 1.0 + np.exp(-2.0 * radius2) * (
            0.2 + self.harmonic * np.cos(2.0 * np.arctan2(y, x))
        )


def test_bucketed_search_accepts_distinct_radial_layouts(tmp_path):
    root = tmp_path / "maps"
    entries = []
    for map_id, (nodes, harmonic) in enumerate(
        (
            (np.linspace(0.0, 1.5, 24), 0.04),
            (np.linspace(0.0, 1.5, 24) ** 1.25 / 1.5**0.25, 0.12),
        )
    ):
        child = root / f"bucket-{map_id}"
        PolarAtlasBuilder(
            nodes, n_phi_build=32, m_max=3, core_m_max=1, shard_size=1
        ).build(
            child,
            [
                MapBuildSpec(
                    map_id,
                    0.1 * map_id,
                    -3.0 + map_id,
                    -3.0,
                    ToyEvaluator(harmonic),
                )
            ],
        )
        entries.append(
            {
                "name": child.name,
                "path": child.name,
                "count": 1,
                "logs": [0.1 * map_id, 0.1 * map_id],
                "logq": [-3.0 + map_id, -3.0 + map_id],
                "map_ids": [map_id],
            }
        )
    write_bucket_manifest(root, entries)

    maps = Maps.open(root)
    geometry = Geometry(t0=0.0, u0=0.2, tE=1.0)
    time = np.linspace(-1.0, 1.0, 60)
    x, y = -time, np.full_like(time, -0.2)
    flux = 1.7 * ToyEvaluator(0.12).magnification(x, y) + 0.4
    result = maps.search(
        [Dataset(time, flux, np.full_like(time, 0.01))],
        [geometry],
        config=SearchConfig(
            base_m_max=1,
            full_m_max=3,
            base_n_alpha=8,
            full_n_alpha=16,
            top_count=2,
            risk_count=1,
            candidate_count=2,
            refine=False,
        ),
    )

    assert result.maps_scanned == 2
    assert result.candidates
    assert maps.radial_nodes_for(0).shape == maps.radial_nodes_for(1).shape
    assert not np.array_equal(maps.radial_nodes_for(0), maps.radial_nodes_for(1))
