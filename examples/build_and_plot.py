"""Build one small atlas and reproduce the README magnification map."""

from pathlib import Path

from hammlet import Atlas, AtlasConfig, ParameterGrid, build_atlas
from hammlet.plot import plot_magnification_map


ROOT = Path(__file__).resolve().parents[1]
output = ROOT / "example_output" / "single_map_atlas"

grid = ParameterGrid(s=(1.6,), q=(1.0e-3,), rho=(8.0e-4,))
config = AtlasConfig(
    m_max=64,
    core_m_max=32,
    radial_nodes=64,
    min_n_phi=128,
    caustic_base_n_phi=256,
    max_n_phi=512,
    diagnostic_m_max=128,
)

if not output.exists():
    build_atlas(output, grid, config=config)

atlas = Atlas.open(output)
plot_magnification_map(
    atlas,
    map_id=0,
    output=ROOT / "assets" / "example_magnification_map.png",
    extent=1.25,
    pixels=420,
)

