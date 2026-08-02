"""Minimal end-to-end search of an existing atlas."""

import numpy as np

from hammlet import Atlas, Dataset, Geometry, SearchConfig


atlas = Atlas.open("atlas")
data = Dataset(
    time=np.loadtxt("lightcurve.dat")[:, 0],
    flux=np.loadtxt("lightcurve.dat")[:, 1],
    error=np.loadtxt("lightcurve.dat")[:, 2],
    name="survey",
)
result = atlas.search(
    [data],
    [Geometry(t0=2459000.0, u0=0.08, tE=24.0)],
    config=SearchConfig(candidate_count=300),
)

for seed in result.candidates[:10]:
    print(seed)

