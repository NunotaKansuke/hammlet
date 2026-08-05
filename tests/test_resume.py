import json

import numpy as np

from hammlet._core.atlas_builder import MapBuildSpec, PolarAtlasBuilder


class CountingEvaluator:
    def __init__(self, value, calls):
        self.value = value
        self.calls = calls

    def magnification(self, x, y):
        self.calls.append(self.value)
        return np.full(np.broadcast(x, y).shape, self.value)


def test_resume_reuses_durable_map_shards(tmp_path):
    nodes = np.linspace(0.0, 2.0, 8)
    builder = PolarAtlasBuilder(
        nodes,
        n_phi_build=8,
        m_max=2,
        core_m_max=1,
        shard_size=1,
    )
    first_calls = []
    builder.build(
        tmp_path,
        [MapBuildSpec(0, 0.1, -3.0, -3.0, CountingEvaluator(1.0, first_calls))],
    )
    assert first_calls == [1.0]

    resumed_calls = []
    builder.build(
        tmp_path,
        [
            MapBuildSpec(0, 0.1, -3.0, -3.0, CountingEvaluator(2.0, resumed_calls)),
            MapBuildSpec(1, 0.2, -3.0, -3.0, CountingEvaluator(3.0, resumed_calls)),
        ],
        resume=True,
    )

    assert resumed_calls == [3.0]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["n_maps"] == 2
    assert {tuple(entry["map_ids"]) for entry in manifest["shards"]} == {(0,), (1,)}
