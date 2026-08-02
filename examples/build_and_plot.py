"""Build one map and reproduce the README VBM comparison."""

from pathlib import Path

from hammlet import MapConfig, Maps, ParameterGrid, build_maps
from hammlet.plot import plot_vbm_comparison


ROOT = Path(__file__).resolve().parents[1]
output = ROOT / "example_output" / "resonant_maps"

grid = ParameterGrid(s=(1.0,), q=(1.0e-2,), rho=(3.0e-3,))
config = MapConfig(
    m_max=384,
    core_m_max=96,
    radial_nodes=256,
    min_n_phi=512,
    caustic_base_n_phi=2048,
    max_n_phi=4096,
    diagnostic_m_max=768,
)

if not output.exists():
    build_maps(output, grid, config=config)

maps = Maps.open(output)
plot_vbm_comparison(
    maps,
    map_id=0,
    output=ROOT / "assets" / "vbm_fourier_residual.png",
    extent=0.36,
    pixels=260,
)
