# PSPL Geometry, Map Coordinate Frames, and the FFT Search

Status: diagnosis of the Roman binary-lens local-search discrepancy (2026-09-22)

Scope: GULLS truth light curves, the existing Fourier atlas, and the Roman
local FFT scorer. This document records the coordinate contract and the
minimum correction needed by a future implementation or LLM agent.

## Executive conclusion

The existing Fourier maps do not need to be regenerated based on this
diagnosis.

The main failure is a coordinate-frame mismatch between the map builder and
the FFT trajectory scorer:

1. GULLS supplies a source trajectory in the native VBMicrolensing frame.
2. The atlas is generated in the AdaMGrid **map frame**.
3. The FFT scorer constructs a trajectory from `(t0, u0, tE, alpha)` but
   treats it as if it were already in the map frame.
4. For wide binaries (`s > 1`), the two frames differ by a map-specific
   horizontal origin shift.

Therefore the input GULLS `t0`, `u0`, and `tE` are not intrinsically wrong.
The missing operation is the transformation of the same physical trajectory
into the coordinate frame of each map.

The design problem is an incomplete coordinate-frame contract, not evidence
that the stored map coefficients are physically invalid.

There is a second, practical consequence for the current Roman search: the
initial `Geometry` stencil is built before the map scan and its trajectory
kernel is reused for every map. Therefore the coordinate conversion has to be
handled when the initial geometry grid is defined or when the map-specific
kernel is built. Translating only the final selected seed is too late; it
cannot recover a truth trajectory that was absent from the initial FFT grid.

## Coordinate definitions

### GULLS/native source trajectory

For the static binary events used here, the GULLS trajectory can be written
as

```text
tau       = (time - t0) / tE
x_native  = tau * cos(alpha) - u0 * sin(alpha)
y_native  = tau * sin(alpha) + u0 * cos(alpha)
```

These coordinates use the native VBMicrolensing convention. The GULLS truth
values for `t0`, `u0`, and `tE` describe this physical source motion.

### AdaMGrid/Hammlet map frame

The map evaluator uses a map-frame point `(x_map, y_map)` and calls
VBMicrolensing at

```text
x_native = x_map + shift_x(s, q)
y_native = y_map
```

For the current convention,

```text
shift_x(s, q) = -(s - 1/s) * q / (1 + q),  for s > 1
shift_x(s, q) = 0,                            for s <= 1
```

Therefore a native source position must be supplied to the map as

```text
x_map = x_native - shift_x(s, q)
y_map = y_native
```

The shift is a fixed Cartesian translation for a given `(s, q)`. It is not a
physical change in the source trajectory.

The implementation documents this conversion in
[`caustics.py`](../src/hammlet/_core/caustics.py#L11-L30), and the map-frame
direct evaluator applies the inverse conversion in
[`direct_vbm.py`](../src/hammlet/_core/direct_vbm.py#L510-L585).

## What is and is not wrong

| Component | Diagnosis |
| --- | --- |
| GULLS truth `t0`, `u0`, `tE` | Correct in the native/source convention. |
| Stored Fourier coefficients | Consistent with the configured map frame. No blanket map regeneration is required. |
| `shift_x(s,q)` | A required map/native coordinate conversion for `s > 1`. |
| FFT trajectory scorer | Incomplete: it uses the unshifted PSPL trajectory when reading a map-frame atlas. |
| `t0/u0` values found after the bad search | May be compensating values, not the physical GULLS truth. |
| Alpha sign/180-degree convention | A convention difference that can be absorbed by an alpha plus pi transformation; it is not the main failure. |

The core FFT refinement currently computes the radial coordinate and angle as

```python
radius = hypot(tau, u0)
angle = atan2(-u0, -tau) + alpha
```

without applying the map-specific origin conversion. See
[`jax_seed_refine.py`](../src/hammlet/_core/jax_seed_refine.py#L117-L136).

## Why `t0` and `u0` can appear to depend on `(s,q)`

This is a coordinate representation effect.

For a fixed alpha, translating a straight trajectory by a fixed vector can be
represented by changing its along-track and perpendicular-track coordinates.
For the native convention above, with

```text
d = (-shift_x, 0)
```

the equivalent changes are

```text
delta_t0 =  shift_x * tE * cos(alpha)
delta_u0 =  shift_x * sin(alpha)
```

The signs change if the alternate Hammlet trajectory convention is used, but
the geometry is the same: the fixed Cartesian shift is projected onto the
trajectory-parallel and trajectory-perpendicular axes.

This does **not** mean that the physical GULLS truth changes with `(s,q)`.
It means that the same line has different effective PSPL coordinates when
written relative to a different map origin.

The important distinction is:

```text
physical/native t0,u0: fixed truth trajectory
map-local effective t0,u0: representation of that trajectory in one map frame
```

The current search implicitly treated the first as if it were always the
second.

## The initial geometry grid is part of the problem

The Roman runner passes the supplied geometry center to `geometry_stencil`.
The FFT path then computes

```python
radius = hypot((time - t0) / tE, u0)
phase = atan2(-u0, -(time - t0) / tE)
```

once for each geometry and reuses that basis across map IDs. In other words,
the current implementation is not independently optimizing `t0/u0` for each
map during the first scan. It is asking whether the fixed initial geometry
stencil contains the correct trajectory for every map frame.

For a truth-centered diagnostic, the natural procedure is therefore:

```text
1. Take the injected/native truth (s, q, t0, u0, tE, alpha).
2. Compute shift_x(s, q).
3. Express that same truth trajectory in the map frame.
4. Build the local fixed geometry stencil around the transformed truth.
```

This uses the truth as an explicit benchmark/oracle; it is not a blind
inference procedure. It is nevertheless the right test of whether the FFT
can recover the truth once the frame contract is made consistent.

For the native trajectory convention written above, the map-local PSPL
parameters at a fixed alpha are

```text
t0_map = t0 + shift_x * tE * cos(alpha)
u0_map = u0 + shift_x * sin(alpha)
```

The repository's low-level `trajectory_xy` helper uses the equivalent
trajectory with a 180-degree alpha convention. In that convention the same
conversion is written with both signs reversed:

```text
t0_map = t0 - shift_x * tE * cos(alpha)
u0_map = u0 - shift_x * sin(alpha)
```

These are not different physical corrections; they are the same Cartesian
translation after the alpha convention is made explicit.

The alpha dependence here is important. `shift_x` itself depends only on
`(s,q)`, but its decomposition into the PSPL coordinates `t0/u0` depends on
the trajectory angle. Consequently, one fixed `t0/u0` stencil centered at the
truth alpha is exact only for that alpha. A full alpha scan needs either
alpha-indexed transformed stencils/kernels, a Cartesian position-based
parameterization, or a different map-frame FFT construction.

For a blind search, replace the injected truth by the best available PSPL
estimate and transform the whole seed stencil, not just the eventual winning
row. The estimate need not equal the truth, but the seed must cover the
map-frame representation of the truth. This is the initial-estimation issue;
candidate-wise coordinate conversion by itself does not solve it.

## Why the issue affects the FFT implementation

The alpha FFT is based on a useful factorization. Without a Cartesian origin
translation,

```text
r_i   = sqrt(tau_i^2 + u0^2)            # independent of alpha
phi_i = phi0_i + alpha                  # alpha is an angular rotation
```

The radial interpolation weights can therefore be prepared once, and the
alpha dependence can be evaluated as a Fourier series.

If the existing map coefficients are kept fixed and the trajectory is
translated point-by-point instead,

```text
x_i(map) = x_i(alpha) - shift_x(s,q)
y_i(map) = y_i(alpha)
r_i(map) = hypot(x_i(map), y_i(map))
```

The shift itself is alpha-independent, but `x_i(alpha)` rotates with alpha,
so the radius sampled in the map becomes alpha-dependent. This is why the
**current shared radial kernel** cannot simply be reused unchanged.

This does not make FFT mathematically impossible. There are three valid
implementation choices:

1. Translate/re-express each map once in the native/source-centered polar
   frame, cache the resulting map-specific Fourier kernel, and then use the
   ordinary alpha FFT.
2. Keep the maps as they are and perform exact map-frame trajectory evaluation
   for the candidate alpha values without assuming one shared radial kernel.
3. Keep the current fast FFT as a provisional candidate filter, then rescore
   and refine the top candidates with the exact map-frame transformation.

The first choice preserves the alpha FFT most directly, but adds a
map-specific kernel-preparation step. The third choice is the lowest-risk
short-term correction.

## Numerical evidence

Each event has 23,104 data points. Using the GULLS truth parameters with the
correct native-to-map transformation gives a direct-VBM profiled chi-square
close to the number of data points. Using the same truth trajectory without
the transformation fails catastrophically.

| Event | `shift_x` | Correct map-frame truth chi-square | Unshifted truth chi-square |
| --- | ---: | ---: | ---: |
| 9920084 | `-0.900071` | `22,973.9` | `337,574,881.5` |
| 9920099 | `-1.230840` | `23,132.8` | `62,148,097.3` |

This is the decisive distinction: the exact truth curve is recovered without
changing the map files or the physical `t0/u0`; only the coordinate conversion
is changed.

Simply translating already-selected FFT seeds does not repair their fit. The
seeds were selected under the wrong coordinate contract, so the corrected
initial grid/search or a corrected exact refinement must be run to obtain
valid candidates. A direct candidate evaluator can apply the conversion to
any candidate `(t0,u0)`, but that is a forward-model operation; it does not
retroactively add a missing point to the initial FFT stencil.

## Required implementation contract

Every consumer of a Hammlet atlas must explicitly know its coordinate frame.

For an atlas with `coordinate_frame="map"`:

```text
1. Construct the physical/native source trajectory from the PSPL geometry.
2. Convert every source position to map coordinates using (s,q).
3. Evaluate the Fourier map at those map coordinates.
```

For an atlas with `coordinate_frame="native"`:

```text
1. Construct the physical/native source trajectory.
2. Evaluate the Fourier map directly at those coordinates.
```

The map builder already passes the configured coordinate frame into the VBM
evaluator in [`build.py`](../src/hammlet/build.py#L46-L61), and the default
configuration is `coordinate_frame="map"` in
[`config.py`](../src/hammlet/config.py#L195-L243). The missing work is to
propagate that same contract into the FFT search and all direct-reference
helpers.

## Recommended tests before changing the production search

1. For random map-frame points, compare the stored-map reconstruction against
   direct VBM using `x_native = x_map + shift_x`.
2. For each Roman truth event, compare the GULLS/native direct curve against
   the map-frame direct curve after applying `x_map = x_native - shift_x`.
3. Confirm that `s <= 1` remains unchanged because `shift_x = 0` there.
4. Compare a corrected alpha scan against direct VBM for one fixed map before
   scaling to the full atlas.
5. Record `coordinate_frame` in every atlas/report artifact and reject a
   search when the scorer and atlas frames are unspecified or inconsistent.

## Bottom line for future agents

Do not regenerate all maps just because this issue was found.

Do not overwrite the physical GULLS truth `t0/u0` as a global fix. For a
truth-centered diagnostic, construct a separate map-local copy and center the
fixed initial grid on that copy.

Fix the map/native coordinate adapter, then make the initial geometry grid and
kernel strategy consistent with it. The broken assumption was that one common
PSPL trajectory representation could be passed unchanged to maps whose local
origins depend on `(s,q)`.
