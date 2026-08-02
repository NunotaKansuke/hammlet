# Distributed generation and storage format

## Ordered parameter space

The Cartesian parameter grid is flattened deterministically with `rho` varying
fastest, then `q`, then `s`. Before partitioning, Hammlet forms `(s,q)` radial
layout buckets. A bucket always contains every rho value for its `(s,q)` cells,
because its sharpest/smallest-rho maps determine where radial resolution is
needed. `Partition(i,n)` divides the ordered bucket list, never an individual
rho axis.

Contiguous assignment keeps nearby lens parameters together and requires no
inter-machine messaging. `part_count` cannot exceed the number of non-empty
radial buckets.

## Scheduler example

For a Slurm array:

```bash
#SBATCH --array=0-31
#SBATCH --cpus-per-task=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
hammlet build-maps config.json /shared/run-001 \
  --part-index "$SLURM_ARRAY_TASK_ID" --part-count 32
```

VBMicrolensing generation is scalar inside each process. Use one scheduler job
per partition; do not allow each job to spawn a second uncontrolled thread
pool. Different machines may use local scratch and copy only a completed part
directory to shared storage before merge, provided directory names remain
unchanged.

## Restart semantics

- A completed part is immutable and a repeated build fails visibly.
- An interrupted job has only a hidden `.partial-PID` directory.
- Remove a stale partial directory only after confirming its process/job is no
  longer alive, then resubmit the same part index.
- `merge-maps` never deletes or modifies input parts.

## Maps directory

```text
maps/
  manifest.json
  hammlet-build.json
  s000_q000/
    manifest.json
    radial_nodes.npy
    radial_pilot_map_ids.npy
    radial_pilot_radii.npy
    radial_pilot_difficulty.npy
    map_ids.npy
    map_parameters.npy
    shard_0000/
      x_coeff.npy
      x_coeff_extension.npy
      certified_error.npy
      certified_error_full.npy
    direct_diagnostics.npz
```

Core and extension mode files allow a low-mode pass to avoid reading production
high modes. Coefficients are complex64 by default and arrays are NumPy `.npy`
files so they can be memory-mapped. `manifest.json` records normalization,
mode budgets, builder settings, certificate semantics, and shard membership.
The certified-error arrays already combine angular reconstruction, radial
holdout interpolation, and storage rounding; search does not read a separate
radial-error tensor.

`hammlet-build.json` records the physical grid, generation config, partition,
and selected global map IDs. Merge validates these records before copying
shards into the final maps namespace.
