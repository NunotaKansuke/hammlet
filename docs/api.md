# Python API

## ParameterGrid

`ParameterGrid(s=..., q=..., rho=...)` takes physical positive values.
`ParameterGrid.from_log10(log_s=..., log_q=..., log_rho=...)` is the logarithmic
equivalent. Axes must be strictly increasing; the stable map ID order is
`s -> q -> rho`.

## AtlasConfig

Important defaults:

| field | default | meaning |
|---|---:|---|
| `m_max` | 384 | maximum stored angular mode |
| `core_m_max` | 96 | modes in the small core coefficient file |
| `radial_nodes` | 256 | common radial samples per map |
| `radial_max` | 3.5 | largest map-frame radius |
| `min_n_phi` | 512 | initial smooth-ring samples |
| `caustic_base_n_phi` | 2048 | initial guarded-ring samples |
| `max_n_phi` | 8192 | maximum uniform angular work |
| `diagnostic_m_max` | 768 | modes used for tail/error diagnosis |

The power-of-two angular limits are validated by the numerical core.

## build_atlas

```python
build_atlas(path, grid, config=AtlasConfig(), partition=Partition())
```

The function requires the optional VBMicrolensing dependency. An existing
destination is never overwritten. Failed builds remove only their own private
partial directory.

## Atlas

`Atlas.open(path)` memory-maps the stored arrays. Useful members are
`map_ids`, `parameters`, `radial_nodes`, and `m_max`.

```python
A = atlas.magnification(map_id, x, y, m_max=96)
```

reconstructs at arbitrary Cartesian points. Coordinates must lie inside the
stored radial range.

## Dataset and Geometry

`Dataset(time, flux, error, name="")` validates one photometric dataset.
`Geometry(t0, u0, tE)` specifies a rectilinear trajectory seed; angles are
radians and `tE` must be positive.

## SearchConfig and Atlas.search

```python
result = atlas.search(datasets, geometries, config=SearchConfig())
```

The default two-pass scan uses `M=32, N_alpha=128` globally and
`M=96, N_alpha=512` on selected candidates. `candidate_count=300` controls the
handoff list, not the number of map evaluations. Candidate fields are physical
`s,q,rho,alpha`, geometry index, central chi-square, lower/upper bounds, and
spectral risk.

The first call includes JAX compilation. Benchmark steady-state throughput
separately from cold-start latency.

