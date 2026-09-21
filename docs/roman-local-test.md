# Local Roman test

`tools/roman_local` is a local diagnostic path for running one Roman event
against an already-built Hammlet atlas. It is intentionally separate from the
production map builders and from the public multi-resolution search API.

The default profile is:

```text
M = 128                 linear Fourier modes
n_alpha = 540           alpha samples
radial_order = 1       fixed linear radial interpolation
cores = 4               process affinity and numerical thread budget
chi2 guarantee = off   no reconstruction/certified error sidecars are read
```

The tool never writes to the `--atlas` path. Results are written to a separate
directory, and an existing result directory is not overwritten.

## Catalog-shaped Roman data

The adapter accepts the sample catalog layout used by the Roman handoff. The
default band is `W146`, whose file is read as
`magnitude, magnitude_error, time` and converted to flux space.

First inspect paths and completed-map coverage:

```bash
bash scripts/run-roman-local.sh \
  --catalog /path/to/OMPLDG_croin_cassan.sample.csv \
  --simulation-root /path/to/sample_rtmodel_v2.4 \
  --atlas /path/to/hammlet-maps/run-004 \
  --index 160 \
  --dry-run
```

An anomaly JSON is optional. With `--anomaly-product`, its `fit.params`
`t0/u0/tE` values center the geometry stencil; without it, the catalog truth
trajectory is used. The catalog's `t0lens1` receives the historical `8234.0`
time offset by default; override it with `--time-offset` if the supplied data
uses another convention.

By default the tool reads every valid row in the supplied data file. The
loaded event window recorded in `report.json` is the actual minimum-to-maximum
time span of those rows. To run a deliberately shorter diagnostic, pass
`--window-tE N` or an explicit `--window-start TMIN --window-end TMAX` pair.

For a small smoke scan, add `--max-maps 32 --geometry-stencil center` and an
explicit output directory. Remove `--max-maps` for the complete set of
currently completed maps:

```bash
bash scripts/run-roman-local.sh \
  --catalog /path/to/catalog.csv \
  --simulation-root /path/to/simulation-root \
  --atlas /path/to/hammlet-maps/run-004 \
  --index 160 \
  --output /path/to/local-results/roman-160
```

The atlas argument may also be a parent containing both `run-003` and
`run-004`. In the current production layout, `run-004` is a restart of the
same task universe as `run-003`: the reader uses `run-004/tasks.json` for the
expected total and takes the union of completed map IDs from both runs. Thus a
map ID is counted once even when the two runs are inspected together. Completed
immutable parts and completed `map-*` results are combined read-only;
the restart's `seed/index.json` is also used to read coefficient shards from
partial parts, excluding IDs already present in a completed part. Result
directories without a manifest (including currently running maps) remain
excluded.

## User-provided data files

Catalog metadata is not required when the data and PSPL center are supplied
directly:

```bash
bash scripts/run-roman-local.sh \
  --data-file /path/to/RomanW146sat1.dat \
  --data-format roman-mag \
  --t0 8757.5 --u0 0.01 --tE 7.0 \
  --atlas /path/to/hammlet-maps/run-004 \
  --event-name my-event \
  --output /path/to/local-results/my-event
```

For already converted data, use `--data-format time-flux-error` with columns
`time, flux, error`. Repeat `--data-file` for multiple data sets and provide
matching `--band-name` values. `--s --q --rho` are optional; when supplied they
enable the nearest-reference-map diagnostic in `report.json`.

## GULLS `.all.lc`

The local runner also accepts a GULLS observed light curve and its manifest
truth row directly:

```bash
bash scripts/run-roman-local.sh \
  --gulls-lightcurve /path/to/event.all.lc \
  --gulls-manifest /path/to/manifest.csv \
  --event-id 9900402 \
  --atlas /path/to/hammlet-maps \
  --output /path/to/local-results/gulls-9900402
```

GULLS `time_day`/`Aobs`/`Aerr` are used as time/data/error, and saturated rows
(`satflag != 0`) are excluded. The manifest's `t0_day`, `u0`, `tE_days`,
`planet_s`, `planet_q`, and `rho` define the truth-centered seed stencil;
`t0_jd` is not substituted because the `.all.lc` time column is relative days.
By default every valid unsaturated row in the `.all.lc` is used. If a shorter
event window is scientifically intended, pass `--window-tE N` or an explicit
`--window-start/--window-end` pair. When all seasons extend past the stored
radial maximum (3.5 in the current atlas), the local reader appends a
map-specific point-lens far-field tail in memory so the full data set can be
used; the atlas files remain untouched. This approximation is recorded in
`report.json` and is not a chi-square guarantee. Use `--tail-mode off` only for
a controlled timing comparison with a window that stays inside the stored
radial support; it is not a valid full-data fallback.

## Artifacts and plots

Each run writes only to its result directory:

* `minima.npz`: map minima, winning geometry/alpha indices, and the geometry
  table;
* `candidates.json`: the best finite FFT seeds;
* `report.json`: event provenance, map coverage, CPU settings, phase timing
  (including coefficient I/O, tail construction, kernel binding, and JAX
  calls), and the explicit no-certificate limitation.

Make the chi-square map and the best-seed light curve with:

```bash
bash scripts/plot-roman-local.sh /path/to/local-results/roman-160
```

The light-curve view is focused on the magnification peak by default. Its
half-width is `5 t_eff`, where `t_eff=max(|u0|,rho)tE`; this follows the actual
high-magnification timescale instead of using the full `tE`. Use
`--zoom-t-eff VALUE` (or the legacy alias `--zoom-tE VALUE`) to widen or narrow
that time window.

The chi-square map uses an adaptive linear color scale by default: its upper
limit is the Δχ² value of the `N`th-best `(s,q)` cell after minimizing over
`rho` (default `N=200`). Change that width with `--delta-chi2-top-cells N`.
For a fixed diagnostic scale, override it explicitly with
`--delta-chi2-max VALUE` (for example `5000`).

Use `--map-id MAP_ID` to make the light curve for a particular scanned map
(for example, the close/wide counterpart), while keeping the global chi-square
map unchanged:

```bash
bash scripts/plot-roman-local.sh \
  /path/to/local-results/roman-160 \
  --map-id MAP_ID
```

For a specific anomaly window, use `--time-start TMIN --time-end TMAX`; the
output filename is kept separate from the ordinary light-curve plot.

Add `--direct-vbm` when the optional `VBMicrolensing` package is available to
overlay a direct-VBM profile. The plotter reads the original atlas; it does not
modify it.

The radial range is checked rather than silently clipped. With the default
`--tail-mode auto`, any rows beyond the finite atlas support receive the
map-specific in-memory point-lens tail described above. With
`--tail-mode off`, an event window beyond the atlas radial nodes stops with an
actionable error; use that mode only for a controlled diagnostic.

To inspect the actual FFT reconstruction of the best scanned map as a simple
Cartesian `jet` image:

```bash
bash scripts/plot-roman-map.sh /path/to/local-results/roman-160
```

To put one event's light curve and Δχ² map side by side in one image:

```bash
python scripts/plot-roman-local-event.py \
  /path/to/local-results/roman-160/events/9910002
```

To make that one-image-per-event output for every event currently marked
complete in a batch:

```bash
python scripts/plot-roman-local-completed.py \
  /path/to/local-results/roman-events \
  --workers 4
```

The default is the scan's `M` (normally 128), the full stored radial support,
and a percentile-clipped logarithmic magnification scale. Select another map
or color limit with `--map-id MAP_ID`, `--vmax VALUE`, or
`--vmax-percentile P`. Use `--scale linear` for the linear version. This only
reads the atlas and writes the PNG below the result directory's `figures/`
folder.

## Packed buckets and process parallelism

When the atlas contains many small coefficient shards, build a separate local
snapshot with one uncompressed, memmap-able coefficient array per scan bucket:

```bash
bash scripts/pack-roman-local.sh \
  --atlas /path/to/hammlet-maps \
  --output /path/to/local-results/packed-atlas-m512 \
  --m-max 512
```

The packer reads the atlas only; it never writes, deletes, or replaces map
files. The resulting `manifest.json` records the source map-ID/parameter
digests, expected count, and readable count. If more maps become readable
later, the parallel runner refuses the stale snapshot until a new cache is
built. `--resume` is available for an interrupted pack whose manifest still
matches the same source snapshot.

The packed runner assigns whole radial buckets to spawned processes. The
recommended first comparison on a four-core machine is four workers with one
thread each:

```bash
bash scripts/run-roman-local-parallel.sh \
  --gulls-lightcurve /path/to/event.all.lc \
  --gulls-manifest /path/to/manifest.csv \
  --event-id 9900403 \
  --atlas /path/to/hammlet-maps \
  --packed-cache /path/to/local-results/packed-atlas-m512 \
  --output /path/to/local-results/roman-9900403-4w1t \
  --m-max 128 --n-alpha 540 --radial-order 1 \
  --cores 4 --workers 4 --threads-per-worker 1
```

Each finished bucket is saved below `bucket-results/`, so `--resume` can
reuse validated bucket results after an interrupted local run. `report.json`
contains per-bucket timing, worker affinity/process time, and measured
aggregate busy time for the selected CPUs. The CPU figure is system busy time
on those CPUs (100% means all selected CPUs were busy), not an unsupported
claim about one process's utilization. A two-worker/two-thread run can be
tested with `--workers 2 --threads-per-worker 2`; spawning one process per
bucket is not the default because it multiplies JAX startup/compilation and
does not fit a four-core budget.

For a group of converted light curves, use the local batch driver. It
preflights the declared row counts, keeps one result/log directory per event,
and resumes completed or partial events without touching the atlas:

```bash
bash scripts/run-roman-local-batch.sh \
  --input-root /path/to/roman_static_bound_planet_hammlet_batch100 \
  --truth /path/to/roman_static_bound_planet_hammlet_batch100/hammlet_input/truth.csv \
  --atlas /path/to/hammlet-maps \
  --packed-cache /path/to/local-results/packed-atlas-m512 \
  --output-root /path/to/local-results/roman-hammlet-batch100 \
  --cores 16 --workers 16 --threads-per-worker 1
```

The batch driver uses all valid input rows by default and places a shared
`jax-compilation-cache/` beside the event results. The cache lets separate
event processes reuse identical JAX executables; packed coefficient arrays
remain read-only memory-mapped files shared by the operating system.
