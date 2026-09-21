# Roman/Hammlet test-case taxonomy

This document gives a numerical classification of the 200 final GULLS
light curves available for the Roman/Hammlet tests. It is intended to
help choose a small set of cases for the paper. It does not classify
the success or failure of `lcbinint`, `VBMicrolensing`, or an MCMC run.

The classification uses only the GULLS manifests and, where available,
the binary-batch signal-screen metadata.

## Dataset scope

The final test pool consists of two 100-event batches:

| batch | event IDs | description |
|---|---:|---|
| `roman_static_bound_planet_hammlet_batch100` | `9910000`--`9910099` | stratified bound-planet batch; `q <= 1e-2`, `0.55 <= s <= 1.8` |
| `roman_static_binary_hammlet_signal_batch100` | `9920000`--`9920099` | signal-selected broad binary/planet batch; `q ~= 1.1e-4`--`0.96`, `s ~= 0.30`--`2.99` |

Both final batches contain 23,104 rows per event and passed their GULLS
QC checks. The 500-event candidate pool used to select the second batch
is not counted as an additional final test set.

The existing result report for the second batch is the source of the
per-event diagnostic convention and event grouping:

`docs/roman-binary-hammlet-signal-batch100-results.md`

## Classification rules

The following labels are deliberately numerical and reproducible.

### Mass-ratio class

| label | definition |
|---|---:|
| `planetary` | `q < 1e-2` |
| `intermediate` | `1e-2 <= q < 0.1` |
| `stellar` | `q >= 0.1` |

The labels describe the injected mass-ratio range only. They are not a
claim about the astrophysical population represented by the sample.

### Separation regime

| label | definition |
|---|---:|
| `very_close` | `s < 0.45` |
| `close` | `0.45 <= s < 0.70` |
| `resonant_range` | `0.70 <= s < 1.35` |
| `wide` | `1.35 <= s < 2.20` |
| `very_wide` | `s >= 2.20` |

`resonant_range` refers only to the separation range used for the
proposal. It does not prove a particular caustic topology or a
caustic crossing.

### Source-size class

Using the existing log10(rho) bins, the compact labels are:

| label | definition |
|---|---:|
| `small_rho` | `log10(rho) < -3.4` |
| `medium_rho` | `-3.4 <= log10(rho) < -2.5` |
| `large_rho` | `log10(rho) >= -2.5` |

These labels are useful for distinguishing sharp finite-source
structure from strongly smoothed structure.

### Signal-screen class

This class is available only for the `992xxxx` batch. It uses the
stored `signal_max_sigma_fixed_pspl` value only as a descriptive
screening metric:

| label | definition |
|---|---:|
| `near_threshold` | maximum screen sigma `< 10` |
| `moderate_screen` | maximum screen sigma `10--50` |
| `strong_screen` | maximum screen sigma `>= 50` |

This is not a formal binary-detection significance. The selector's
residual convention is source-flux weighted and uses a fixed truth
geometry PSPL.

### Geometry proposal class

For the `992xxxx` batch, the manifest records:

- `central_random`;
- `outer_caustic_target`;
- `central_fallback`.

These are generator proposal labels. `outer_caustic_target` means that
the trajectory was directed toward an approximate caustic location; it
does not establish a geometric finite-source caustic crossing.

## Counts over the 200-event pool

### Mass-ratio class

| class | count |
|---|---:|
| planetary | 106 |
| intermediate | 49 |
| stellar | 45 |

### Separation regime

| regime | count |
|---|---:|
| very close | 20 |
| close | 36 |
| resonant range | 84 |
| wide | 40 |
| very wide | 20 |

### Coarse q x s coverage

| mass-ratio class | very close | close | resonant range | wide | very wide | total |
|---|---:|---:|---:|---:|---:|---:|
| planetary | 0 | 18 | 64 | 24 | 0 | 106 |
| intermediate | 8 | 9 | 16 | 8 | 8 | 49 |
| stellar | 12 | 9 | 4 | 8 | 12 | 45 |

### Source-size class

| class | count |
|---|---:|
| small rho | 45 |
| medium rho | 81 |
| large rho | 74 |

### Additional metadata for the `992xxxx` batch

| quantity | counts |
|---|---|
| geometry proposal | 44 `central_random`, 55 `outer_caustic_target`, 1 `central_fallback` |
| screen strength | 2 `near_threshold`, 6 `moderate_screen`, 92 `strong_screen` |
| companion class | 26 planetary, 29 intermediate, 45 stellar-binary |

## Numerically selected candidate events

The following are provisional candidates selected by a deterministic
numeric-medoid rule within the indicated class. The rule uses the event
closest to the class median in log(q), log(s), log(rho), log(tE), and
log(abs(u0)). These are not yet declared good or failed inference
cases; their MCMC/map-search outcomes must be checked separately.

| proposed type | count | candidate | q | s | tE [day] | rho | u0 | batch |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| planetary + close | 18 | `9910023` | 1.0e-3 | 0.5995 | 16.51 | 9.12e-4 | 0.0120 | first |
| planetary + resonant range | 64 | `9910081` | 3.0e-4 | 1.0186 | 17.81 | 3.95e-4 | 0.0060 | first |
| planetary + wide | 24 | `9910092` | 3.0e-3 | 1.6364 | 17.71 | 4.96e-4 | 0.0120 | first |
| intermediate + resonant range | 16 | `9910041` | 1.0e-2 | 1.1414 | 18.78 | 2.19e-4 | 0.0300 | first |
| stellar + close | 21 | `9920005` | 0.3818 | 0.3588 | 9.19 | 4.80e-3 | 0.0106 | second |
| stellar + wide | 20 | `9920078` | 0.2410 | 2.1392 | 15.24 | 3.69e-3 | 0.0157 | second |
| small-rho example | 45 | `9910036` | 3.0e-3 | 0.8257 | 23.08 | 2.93e-4 | 0.0060 | first |
| large-rho example | 74 | `9920042` | 0.0226 | 0.7713 | 17.02 | 4.01e-3 | 0.0065 | second |
| weakest screen candidate | 2 | `9920020` | 1.97e-3 | 0.4551 | 15.00 | 6.92e-4 | 0.0176 | second |

The second weakest-screen candidate is `9920047`: q = 2.72e-4,
s = 0.9892, rho = 5.60e-4, and maximum screen sigma = 9.10.

## Suggested paper-level subset

Before looking at inference outcomes, a compact coverage-oriented
shortlist would be:

1. `9910023`: low-q, close;
2. `9910081`: low-q, resonant-range;
3. `9910092`: low-q, wide;
4. `9910041`: intermediate-q, resonant-range;
5. `9920005`: high-q, very-close;
6. `9920078`: high-q, wide;
7. `9920042`: large-rho, central-random proposal;
8. `9920020`: near-threshold screen candidate.

This is a coverage shortlist, not a claim that these are the best
light curves or the most representative astrophysical events. After
inference results are available, one or two of these can be replaced by
actual failure cases while preserving the same parameter coverage.

## Failure-status boundary

The GULLS manifests and QC files do not contain `lcbinint`/MCMC
success or failure outcomes. Therefore this document must not label any
event as a failed inference case based only on the GULLS metadata.

To add failure labels later, join this taxonomy to a result table with
at least:

- event ID;
- map-search completion status;
- MCMC completion/convergence status;
- direct-VBM comparison status;
- recovered-parameter error;
- runtime or timeout status.

Only after that join should the paper select a failure example.
