# hamagri

`hamagri` builds Fourier atlases for binary microlensing and searches them for
candidate seeds. Re-evaluate candidates with direct VBMicrolensing before
scientific inference.

## Install

Python 3.10 or later and Git:

```bash
python -m pip install --upgrade "hamagri[all] @ git+https://github.com/NunotaKansuke/hamagri.git@main"
```

## Quick start

```python
import numpy as np
from hamagri import Atlas, Dataset, Geometry

data = np.loadtxt("lightcurve.dat")  # time, flux, uncertainty
dataset = Dataset(data[:, 0], data[:, 1], data[:, 2])
atlas = Atlas.open("hamagri-atlas")
geometry = Geometry(
    t0=2459000.0, u0=0.08, tE=24.0,
    coordinate_frame=atlas.coordinate_frame,
)
best = atlas.search([dataset], [geometry]).candidates[0]
print(best.s, best.q, best.rho, best.alpha, best.chi2)
```

The curated atlas (~72 GB) is prepared as a separate data release; public download hosting is being arranged. See [Atlas data](docs/atlas-data.md) for its format and release status.

## Related project

[AdaMGrid](https://github.com/zhangjiyuan22/AdaMGrid) is an adaptive magnification-map grid-search project for binary and triple microlensing. `hamagri` focuses on Fourier-atlas searches for binary lenses; its map-generation guide includes the AdaMGrid default parameter grid.

## Documentation

[API](docs/api.md) · [Atlas data](docs/atlas-data.md) · [Accuracy](docs/accuracy.md) ·
[Coordinate frames](docs/coordinate-frames.md) · [Theory](docs/theory.md) ·
[Map generation](docs/map-generation.md) ·
[Distributed generation](docs/distributed-generation.md)

## Acknowledgments

OpenAI Codex and Anthropic Claude Code were used as coding assistants during development.

## License

MIT. Follow the license and citation requirements of VBMicrolensing and JAX.
