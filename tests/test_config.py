import numpy as np
import pytest

from hammlet import ParameterGrid, Partition
from hammlet.config import default_radial_nodes


def test_parameter_grid_has_stable_order():
    grid = ParameterGrid(s=(0.8, 1.2), q=(1e-4, 1e-3), rho=(1e-3,))
    table = grid.table()
    assert table.shape == (4, 4)
    np.testing.assert_array_equal(table[:, 0], np.arange(4))
    np.testing.assert_allclose(table[2, 1:], np.log10([1.2, 1e-4, 1e-3]))


def test_partitions_are_complete_and_disjoint():
    selected = [
        np.arange(17)[Partition(index, 4).rows(17)] for index in range(4)
    ]
    np.testing.assert_array_equal(np.concatenate(selected), np.arange(17))


def test_radial_default_preserves_node_count_and_outer_coverage():
    nodes = default_radial_nodes(64, 2.5)
    assert len(nodes) == 64
    assert nodes[0] == 0.0
    assert nodes[-1] == pytest.approx(2.5)
    assert np.all(np.diff(nodes) > 0.0)

