import json

import numpy as np

from hammlet import Maps, merge_maps
from hammlet._core.atlas_builder import MapBuildSpec, PolarAtlasBuilder
from hammlet._core.bucketed_atlas import write_bucket_manifest


class ConstantEvaluator:
    def __init__(self, value):
        self.value = value

    def magnification(self, x, y):
        return np.full(np.broadcast(x, y).shape, self.value)


def write_part(root, index, map_ids):
    part = root / "parts" / f"part-{index:05d}-of-00002"
    child = part / f"s{index:03d}_q000"
    builder = PolarAtlasBuilder(
        np.linspace(0.0, 2.0, 8),
        n_phi_build=8,
        m_max=2,
        core_m_max=1,
        shard_size=1,
    )
    builder.build(
        child,
        [
            MapBuildSpec(
                map_id,
                0.1 * map_id,
                -3.0,
                -3.0,
                ConstantEvaluator(1.0 + map_id),
            )
            for map_id in map_ids
        ],
    )
    write_bucket_manifest(
        part,
        [
            {
                "name": child.name,
                "path": child.name,
                "count": len(map_ids),
                "logs": [0.0, 0.2],
                "logq": [-3.0, -3.0],
                "map_ids": map_ids,
            }
        ],
        {"radial_layout": {"test": True}},
    )
    metadata = {
        "grid": {"s": [1.0, 1.1, 1.2], "q": [0.001], "rho": [0.001]},
        "config": {"m_max": 2},
        "partition": {"index": index, "count": 2},
        "selected_map_ids": map_ids,
    }
    (part / "hammlet-build.json").write_text(json.dumps(metadata))


def test_merge_validates_and_preserves_global_ids(tmp_path):
    write_part(tmp_path, 0, [0, 1])
    write_part(tmp_path, 1, [2])
    destination = merge_maps(tmp_path)
    maps = Maps.open(destination)
    np.testing.assert_array_equal(maps.map_ids, [0, 1, 2])
    assert maps._core.parameters_for(2)[0] == 0.2
