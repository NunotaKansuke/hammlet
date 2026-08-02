# Numerical validation plan

This document separates implemented guarantees from empirical acceptance tests.
Passing these tests supports production seed-search use; it does not turn an
empirical comparison into a mathematical certificate.

## 1. Radial interpolation error

The stored certificate now evaluates deterministic dyadic radial holdouts in
every interval and propagates the linear/cubic interpolation envelope through
the existing chi-square bounds. Its declared reference is piecewise linear
between evaluated rings. Validation must measure how well that sampled
reference represents direct VBM between those rings; a fully continuous proof
still requires a solver-supplied radial derivative enclosure.

Report at least:

- maximum and quantiles of $|A_{\rm VBM}-A_{\rm map}|$ at radial holdouts;
- the same residual normalized by photometric uncertainty along trajectories;
- observed coverage of the resulting chi-square interval;
- generation-time and search-time overhead separately.

Hammlet must continue to describe its interval as conditional on the sampled
radial representation.

## 2. Caustic and small-source recall

Measure top-$K$ seed recall against direct VBMicrolensing for a stratified suite:

- central caustics;
- resonant caustics;
- planetary caustics on both close and wide branches;
- progressively smaller $\rho$, including caustic crossings and grazing events;
- smooth non-caustic controls.

Sweep stored mode budget, runtime mode budget, radial-node count, and angular
generation limit rather than validating only the defaults. For each stratum,
report whether the direct-VBM optimum is represented in the retained independent
seeds and the direct-VBM $\Delta\chi^2$ of the best retained seed. Do not claim a
recall guarantee from a single event or a single $(s,q,\rho)$ map.

## 3. Performance accounting

Benchmark these phases independently:

1. map manifest and memory-map opening;
2. cold JAX compilation;
3. event-kernel construction;
4. coefficient contraction;
5. inverse FFT and candidate selection;
6. host-to-device transfer, when a GPU is used;
7. high-mode seed refinement;
8. steady-state repeat search with compilation and disk cache warm.

Record CPU model, core affinity, memory bandwidth, storage medium, JAX backend,
device, map count, radial-node count, mode budgets, observations, geometries,
and alpha-grid sizes. The expected contraction scaling is

$$
O\!\left(N_{\rm map}N_rM\right),
$$

so wall-clock conclusions must distinguish arithmetic throughput from memory
bandwidth and I/O effects.

## Acceptance gate for changing defaults

Do not change production defaults from `M=512`, 256 radial nodes, or
`max_n_phi=8192` until the stratified recall suite and phase-separated benchmark
have been run. A release report must link its exact input manifests, software
revision, raw result tables, and plotting script.
