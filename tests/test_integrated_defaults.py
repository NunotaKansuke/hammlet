import numpy as np

from hammlet import MapConfig, ParameterGrid, SearchConfig
from hammlet.build import _parameter_buckets


def test_production_defaults_keep_high_modes_and_use_cubic_full_pass():
    maps = MapConfig()
    search = SearchConfig()
    assert maps.m_max == 512
    assert maps.core_m_max == 128
    assert search.full_m_max == 128
    assert search.handoff_m_max == 512
    assert search.refine_deep_levels == 10
    assert search.base_radial_order == 1
    assert search.full_radial_order == 3
    assert search.certified_selection
    assert search.refine


def test_radial_buckets_never_split_the_rho_axis():
    grid = ParameterGrid(
        s=(0.8, 1.0, 1.2),
        q=(1e-4, 1e-3),
        rho=(1e-4, 3e-4, 1e-3),
    )
    config = MapConfig(radial_s_buckets=2, radial_q_buckets=2)
    buckets = _parameter_buckets(grid.table(), config)
    for bucket in buckets:
        rows = np.asarray(bucket["rows"])
        for logs in np.unique(rows[:, 1]):
            for logq in np.unique(rows[:, 2]):
                local = rows[(rows[:, 1] == logs) & (rows[:, 2] == logq)]
                np.testing.assert_allclose(
                    np.sort(local[:, 3]), np.log10(np.asarray(grid.rho))
                )
