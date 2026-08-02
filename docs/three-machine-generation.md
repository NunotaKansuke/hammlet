# Three-machine production generation handoff

## Readiness decision

The production default is `radial_certificate_levels=1`. It evaluates one
direct-VBM holdout ring at the midpoint of every adjacent pair of stored radial
nodes. The resulting radial residual is combined with the angular and storage
certificates before the map is written. Increasing the level is not required
before starting the first production build.

This is a practical production-candidate setting, not a proof about arbitrary
continuous VBM behaviour between sampled rings. The generated certificate is a
deterministic bound relative to the sampled piecewise-linear VBM reference.
Direct-VBM re-evaluation of retained seeds remains required for scientific
inference.

## Freeze the run inputs

Choose these two paths once on the coordinating machine:

```text
CONFIG=/absolute/path/to/production-build.json
RUN_ROOT=/absolute/path/to/hammlet-maps/run-001
```

The configuration must contain the final physical grid. Use
`examples/distributed_build.json` only as a schema example; its small grid is
not a production recommendation. Keep one canonical copy of the JSON and copy
it byte-for-byte to all machines. Do not independently edit three copies.

All machines must use commit:

```text
2490810fea98edf6ee77ee8430fdc38f5b365e60
```

Before submission, record the following in the run log:

```bash
sha256sum /absolute/path/to/production-build.json
git -C /absolute/path/to/hammlet rev-parse HEAD
```

The JSON may omit `radial_certificate_levels`; at the pinned commit its default
is `1`. For a long-lived production record, making it explicit is clearer:

```json
{
  "maps": {
    "m_max": 512,
    "core_m_max": 128,
    "radial_nodes": 256,
    "radial_certificate_levels": 1
  }
}
```

These keys belong beside the complete `grid` object and any other chosen map
settings; the fragment is not a complete configuration file.

## Assign the three machines

Every machine receives the same configuration, output root, and
`--part-count 3`. Only `--part-index` differs:

| Machine | Part index | Final directory |
| --- | ---: | --- |
| A | 0 | `parts/part-00000-of-00003` |
| B | 1 | `parts/part-00001-of-00003` |
| C | 2 | `parts/part-00002-of-00003` |

The split is made over complete `(s,q)` radial-layout buckets, so a rho axis is
never split across machines. The grid must produce at least three buckets.

If `RUN_ROOT` is on storage shared by all three machines, each machine can
write directly to it. Otherwise, use the same logical root on local scratch and
copy each completed `part-.....-of-00003` directory into the coordinator's
`RUN_ROOT/parts/` directory. Copy only completed visible part directories;
never copy a hidden `.partial-*` directory.

## Prompt to give each machine's Codex

Replace `INDEX`, `CONFIG`, and `RUN_ROOT`, then send this prompt independently
to the three Codex sessions:

```text
Generate Hammlet production map partition INDEX of 3.

Repository: /rogue1_8/nunota/hammlet
Required commit: 2490810fea98edf6ee77ee8430fdc38f5b365e60
Config: CONFIG
Output root: RUN_ROOT

First verify that the repository commit is exact, the config exists, its
SHA-256 agrees with the coordinator, radial_certificate_levels resolves to 1,
and RUN_ROOT/parts/part-INDEX-of-00003 does not already exist. Install the
generation dependencies if needed. Then launch the following as a background
job with OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1:

hammlet build-maps CONFIG RUN_ROOT --part-index INDEX --part-count 3

Write stdout/stderr to a persistent per-part log. Report the PID, log path,
config SHA-256, commit, and the exact destination. Do not delete or overwrite
an existing completed part. Do not merge maps on this machine.
```

Use zero-padded directory names only when checking paths; the CLI indices are
plain `0`, `1`, and `2`.

## Monitor and restart

A running build writes to a hidden `.part-.....partial-PID` directory. The
completed part becomes visible atomically only after all its buckets finish.
Therefore, absence of the final directory while the process is alive is normal.

If a process fails, inspect its log first. A stale `.partial-*` directory may be
removed only after confirming that its process is no longer running. Resubmit
the same part index. Never remove a completed part to make a retry appear to
work; preserve it and investigate any configuration mismatch.

## Merge and verify

After all three final part directories are present on the coordinator:

```bash
hammlet merge-maps /absolute/path/to/hammlet-maps/run-001
```

This creates `RUN_ROOT/maps`. Merge validates that all three indices are
present, their grids and map configurations match, and map IDs/buckets are not
duplicated. It refuses to overwrite an existing merged destination.

Confirm the certificate metadata after merging:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("/absolute/path/to/hammlet-maps/run-001/maps")
build = json.loads((root / "hammlet-build.json").read_text())
assert build["config"]["radial_certificate_levels"] == 1

bucket = next(path for path in root.iterdir() if path.is_dir())
manifest = json.loads((bucket / "manifest.json").read_text())
assert manifest["error_certificate"] == (
    "piecewise-linear-vbm-angular-and-radial-reference"
)
print(root)
print("maps:", len(build["selected_map_ids"]))
print("radial certificate: OK")
PY
```

Keep the canonical JSON, its SHA-256, the pinned commit, and all three logs with
the generated maps.
