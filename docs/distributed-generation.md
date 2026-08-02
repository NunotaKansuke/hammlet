# Distributed generation and storage format

## Ordered parameter space

The Cartesian parameter grid is flattened deterministically with `rho` varying
fastest, then `q`, then `s`. `Partition(i,n)` divides that ordered table into
`n` contiguous, nearly equal slices using integer linspace boundaries. The
slices are disjoint and their concatenation is the original table.

Contiguous assignment makes logs easy to audit and generally keeps nearby lens
parameters on the same machine. It does not require inter-machine messaging.

## Scheduler example

For a Slurm array:

```bash
#SBATCH --array=0-31
#SBATCH --cpus-per-task=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
hammlet build config.json /shared/run-001 \
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
- `merge` never deletes or modifies input parts.

## Atlas directory

```text
atlas/
  manifest.json
  hammlet-build.json
  radial_nodes.npy
  map_ids.npy
  map_parameters.npy
  shard_00000/
    map_ids.npy
    x_coeff.npy
    x_coeff_extension.npy
    x2_coeff.npy
    x2_coeff_extension.npy
    certified_error.npy
    certified_error_full.npy
    reconstruction_error.npy
    reconstruction_error_full.npy
    deviation_envelope.npy
```

Core and extension mode files allow a low-mode pass to avoid reading production
high modes. Coefficients are complex64 by default and arrays are NumPy `.npy`
files so they can be memory-mapped. `manifest.json` records normalization,
mode budgets, builder settings, certificate semantics, and shard membership.

`hammlet-build.json` records the physical grid, generation config, partition,
and selected global map IDs. Merge validates these records before copying
shards into the final atlas namespace.

