# Python API

## ParameterGrid

`ParameterGrid(s=..., q=..., rho=...)` takes physical positive values.
`ParameterGrid.from_log10(log_s=..., log_q=..., log_rho=...)` is the logarithmic
equivalent. Axes must be strictly increasing; the stable map ID order is
`s -> q -> rho`.

## MapConfig

Important defaults:

| field | default | meaning |
|---|---:|---|
| `m_max` | 512 | maximum stored angular mode |
| `core_m_max` | 128 | modes in the small core coefficient file |
| `radial_nodes` | 256 | samples shared inside each `(s,q)` bucket |
| `radial_max` | 3.5 | largest map-frame radius |
| `min_n_phi` | 512 | initial smooth-ring samples |
| `caustic_base_n_phi` | 2048 | initial guarded-ring samples |
| `max_n_phi` | 8192 | maximum uniform angular work |
| `diagnostic_m_max` | 1024 | modes used for tail/error diagnosis |
| `radial_s_buckets` | 4 | number of radial-layout groups across s |
| `radial_q_buckets` | 4 | number of radial-layout groups across q |
| `radial_pilot_maps` | 8 | sharp-rho pilot maps per bucket |
| `radial_adaptive_fraction` | 0.65 | radial nodes allocated by measured difficulty |

The power-of-two angular limits are validated by the numerical core.

## build_maps

```python
build_maps(path, grid, config=MapConfig(), partition=Partition())
```

The function requires the optional VBMicrolensing dependency. An existing
destination is never overwritten. Failed builds remove only their own private
partial directory.

## Maps

`Maps.open(path)` memory-maps the stored arrays. Useful members are
`map_ids`, `parameters`, and `m_max`. Radial nodes vary by bucket and are
available through `radial_nodes_for(map_id)`.

```python
A = maps.magnification(map_id, x, y, m_max=96)
```

reconstructs at arbitrary Cartesian points. Coordinates must lie inside the
stored radial range.

## Dataset and Geometry

`Dataset(time, flux, error, name="")` validates one photometric dataset.
`Geometry(t0, u0, tE)` specifies a rectilinear trajectory seed; angles are
radians and `tE` must be positive.

## SearchConfig and Maps.search

```python
result = maps.search(datasets, geometries, config=SearchConfig())
```

The default scan uses `M=32, N_alpha=128` with linear radial interpolation for
global ranking and `M=128, N_alpha=512` with cubic interpolation for selected
maps. Certified chi-square intervals decide which maps cannot yet be discarded.

Up to 300 independent candidates are then evaluated at `M=512` (or the largest
mode stored by older/smaller maps). All candidates
receive two batched pattern-search levels. The best 32 additionally compare up
to 26 neighbouring maps and receive one pairwise plus ten deep pattern levels.
Candidate fields include starting/final map IDs, refined `(t0,u0,tE,alpha)`,
physical `(s,q,rho)`, FFT and refined chi-square, FFT lower/upper bounds, and
spectral risk.

The first call includes JAX compilation. Benchmark steady-state throughput
separately from cold-start latency.
