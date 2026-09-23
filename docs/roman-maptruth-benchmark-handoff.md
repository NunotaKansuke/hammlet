# Actual Hammlet search configuration for the Roman truth-centered benchmark

This document records the search that was actually executed for the Roman/GULLS
benchmark. It is intended as the source of truth for the paper and for an LLM
that needs to distinguish the executed run from Hammlet's generic/default
workflow.

## Executive summary

The generated sample contains 200 event IDs. The truth-centered FFT search was
completed for 199 events (`9910000`--`9910099` and `9920000`--`9920098`);
`9920099` was not part of this batch. The subsequent LM run was applied to all
199 FFT reports. Of those 199 events, 175 pass the display/analysis cut
\(\Delta\chi^2_{\rm PSPL-truth}>100\); the remaining 24 were searched and
refined, but are omitted from the published q/s and per-event figure pages.

The run is truth-assisted: the center of the trajectory stencil was computed
from the injected truth parameters in the atlas working frame. Therefore this
is a controlled map/search diagnostic, not a blind recovery experiment.

## Event data

- One `time-flux-error` light curve per event.
- `23,104` photometric epochs per event.
- The FFT and LM stages both used all available epochs.
- The event-strength cut is computed independently as
  \[
  \Delta\chi^2 = \sum_i (A_{{\rm truth},i}-A_{{\rm PSPL},i})^2/\sigma_{A,i}^2
  \]
  over all 23,104 epochs. It is a signal-strength/display cut, not a
  definition of successful parameter recovery.

## Initial \((t_0,u_0,t_E)\) grid

For each event, one center was constructed from the injected truth trajectory
and then expressed in the atlas working frame. A fixed 16-point stencil was
built around that center and used for every lens-map point in that event. The
stencil was not a blind PSPL-fit grid and it was not rebuilt for every
candidate \((s,q,\rho)\).

Let the center be \((t_{0,c},u_{0,c},t_{E,c})\). The steps recorded in the
reports were

\[
\Delta t_0=\max(2,0.1t_{E,c}),\qquad
\Delta u_0=\max(0.03,0.1|u_{0,c}|),\qquad
\Delta\log t_E=0.15.
\]

The 16 dimensionless offsets \((\delta_{t_0},\delta_{u_0},
\delta_{\log t_E})\) were

```text
( 0,  0,  0)
(-1,  0,  0), ( 1,  0,  0)
( 0, -1,  0), ( 0,  1,  0)
( 0,  0, -1), ( 0,  0,  1)
(-1, -1, -1), (-1, -1,  1), (-1,  1, -1), (-1,  1,  1)
( 1, -1, -1), ( 1, -1,  1), ( 1,  1, -1), ( 1,  1,  1)
(-0.5, -0.5, 0.5)
```

The physical values are therefore

\[
t_0=t_{0,c}+\delta_{t_0}\Delta t_0,\quad
u_0=u_{0,c}+\delta_{u_0}\Delta u_0,\quad
t_E=t_{E,c}\exp(0.15\,\delta_{\log t_E}).
\]

Thus the search used 16 trajectory combinations per event, rather than a full
Cartesian 3-by-3-by-3 grid. The exact center and all 16 values are stored in
each event's `report.json` under `geometry_stencil` and `geometries`.

## FFT/map search

The search used the existing read-only atlas; no lens maps were regenerated.
The atlas task universe contained 33,993 pre-generated map entries, of which
33,903 were readable and selected in the immutable snapshot used by every
event. Ninety expected entries were unavailable. The atlas task list is a
non-rectangular subset in parameter space, with the following coordinate
sampling:

- \(\log_{10}s\): `-1.5` to `1.5` in steps of `0.05`;
- \(\log_{10}q\): `-6.0` to `4.0` in steps of `0.1`;
- \(\log_{10}\rho\): `-4.0` to `-1.6` in steps of `0.3`.

The actual FFT settings were identical in all 199 reports:

| Setting | Executed value |
|---|---:|
| Fourier mode limit \(M\) | 128 |
| uniform \(\alpha\) samples | 540 over \([0,2\pi)\) |
| radial order | 1 |
| map filter | none |
| batch size | 1024 |
| arithmetic | float64 |
| tail mode | automatic point-lens far-field fallback when needed |
| saved candidate list | 100 per event |

All 33,903 readable maps were scanned for every one of the 16 trajectory
geometries and 540 angles. The nominal profile-\(\chi^2\) search volume was
therefore

\[
33{,}903\times16\times540=292{,}921{,}920
\]

map/trajectory/angle combinations per event. The saved 100-candidate list was
not used to launch 100 LM fits: the LM stage used the single global
`best_fft` entry from each event.

This was a single-resolution \(M=128\) search. No executed 32-to-128-to-512
multiresolution sequence should be claimed for this benchmark, and the FFT
reports do not contain certified reconstruction-error intervals.

## FFT parallel execution

There is no single parallel configuration for all final event reports. The
report metadata gives:

| Final event reports | Requested/used workers | Threads per worker | CPU affinity |
|---:|---:|---:|---|
| 123 | 16 | 1 | CPUs 0--15 |
| 76 | 4 | 1 | CPUs 0--3 |

The 4-worker reports are not a separate scientific configuration: several
events encountered host/memory failures and were resumed with the smaller
worker budget. Consequently, the run should not be described as a uniform
16-core timing benchmark. If a paper quotes a speed, a clean rerun with one
fixed allocation is needed; the final per-event `report.json` files are the
authoritative record of the configurations actually used.

## LM refinement

Every one of the 199 FFT reports was passed to a direct-VBMicrolensing local
refinement. The initial point was the event's single global `best_fft` seed,
not all 100 saved candidates. At each residual evaluation, source and blend
fluxes were profiled analytically in the same likelihood convention as the
FFT stage.

The fitted parameter vector was

\[
(t_0,u_0,t_E,s,q,\rho,\alpha).
\]

Internally the optimizer used the transformed vector

```text
(t0 / tE, u0, log(tE), log(s), log(q), log(rho), alpha)
```

with the initial seed as the zero point. The executed optimizer and numerical
settings were:

| Setting | Executed value |
|---|---:|
| optimizer | `scipy.optimize.least_squares(method="lm")` |
| maximum function evaluations | 25 |
| finite-difference step | `1e-3` |
| `ftol`, `xtol`, `gtol` | `1e-8` each |
| `x_scale` | `"jac"` |
| VBM absolute tolerance | `1e-3` |
| VBM relative tolerance | `1e-4` |
| event-level worker processes | 24 |
| BLAS/OpenMP threads per worker | 1 |

The LM output accounting is:

- result records: `199/199`;
- worker errors: `0`;
- non-negative direct-VBM \(\chi^2\) improvements: `199/199`;
- SciPy convergence flag `success=True`: `3/199`;
- evaluation-cap termination: `196/199`.

The last point is important for the paper: all 199 events have valid final
records and non-worsening objective values, but only 3 runs reached a SciPy
termination condition before the evaluation cap. The LM stage should be
described as a short local refinement pass, not as 199 fully converged fits.
The LM artifact records 24 worker processes, but it does not record a fixed
CPU affinity, so “24 cores” should not be claimed from this run alone.

## What can and cannot be called “recovered”

The current artifacts define:

1. `best_fft`: the minimum profile-\(\chi^2\) seed found by the fixed FFT/map
   search;
2. `optimized`: the final point returned by the short direct-VBM LM pass; and
3. the 175-event sample: events with injected binary signal strength
   \(\Delta\chi^2_{\rm PSPL-truth}>100\).

They do **not** yet define a parameter-recovery label such as “the truth is
within a tolerance,” “the truth is in the top-N candidates,” or “the final
fit is within a specified \(\Delta\chi^2\).” Therefore the q/s page is an
injection-strength-selected result display, not a measured recovery fraction.
A paper claiming recovery completeness or accuracy needs that additional
definition and a per-event table derived from the stored truth and optimized
parameters.

## Reproducibility artifacts

- FFT event reports and timing:
  `results/roman_local/roman_maptruth_fft_batch200_m128_16w1t/events/*/report.json`
- FFT minima and 100-candidate lists:
  `results/roman_local/roman_maptruth_fft_batch200_m128_16w1t/events/*/minima.npz`
  and `candidates.json`
- LM configuration, counts, and all event records:
  `results/roman_local/roman_maptruth_lm_batch200/summary.json`
- 175-event selection metadata:
  `assets/roman-lm-q-s-maptruth-dchi2gt100.json`
- Executed FFT launcher:
  `scripts/run-roman-maptruth-event-sge.sh`
- Executed LM implementation:
  `scripts/roman-lm-refine.py`
- Code snapshot for the current result page:
  `c40e55e` on `codex/roman-coordinate-fix`

## Paper-ready short description

> We searched 199 of the 200 injected Roman/GULLS events using a read-only
> atlas of 33,903 available binary-lens maps. For each event, the injected
> truth trajectory was used only to define a truth-centered 16-point stencil
> in \((t_0,u_0,t_E)\); each stencil point was evaluated over 540 trajectory
> angles and all atlas maps with a single-resolution Fourier search
> (\(M=128\)). The best FFT seed from each event was then refined with a
> short seven-parameter direct-VBMicrolensing Levenberg--Marquardt fit,
> profiling the linear source and blend fluxes at each iteration. The FFT
> stage used either 16 or 4 single-threaded worker processes depending on the
> resumed event job, while the LM stage used 24 event-level worker processes.
> Of the 199 searched events, 175 had
> \(\Delta\chi^2_{\rm PSPL-truth}>100\) and were retained for the displayed
> strong-signal sample.

