#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/extern/moao42_9/nunota/hammlet-maps/run-003}"
REPO="${REPO:-/extern/moao38_7/nunota/hammlet}"
PYTHON="${PYTHON:-/whome/nunota/.pyenv/versions/anaconda3-5.3.0/envs/jaxenv/bin/python}"
CONFIG="${CONFIG:-$ROOT/config/production-build.json}"
PART_COUNT="${PART_COUNT:-161}"
WORKER_COUNT="${WORKER_COUNT:-60}"
WORKER_ID="${1:?usage: $0 WORKER_ID}"

if ! [[ "$WORKER_ID" =~ ^[0-9]+$ && "$WORKER_ID" -lt "$WORKER_COUNT" ]]; then
  echo "require 0 <= WORKER_ID < WORKER_COUNT (got $WORKER_ID/$WORKER_COUNT)" >&2
  exit 2
fi
if [ ! -f "$CONFIG" ]; then
  echo "missing production config: $CONFIG" >&2
  exit 1
fi
if [ -n "$(cd "$REPO" && git status --porcelain)" ]; then
  echo "repository has uncommitted changes; refusing to launch" >&2
  exit 1
fi

"$PYTHON" - "$CONFIG" "$PART_COUNT" "$REPO" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
part_count = int(sys.argv[2])
sys.path.insert(0, str(Path(sys.argv[3]) / "src"))
from hammlet.build import _parameter_buckets
from hammlet.cli import _config
grid, config = _config(path)
if len(grid.table()) != 33993:
    raise SystemExit(f"expected 33993 maps, got {len(grid.table())}")
bucket_count = len(_parameter_buckets(grid.table(), config))
if bucket_count != part_count:
    raise SystemExit(f"expected one partition per bucket: buckets={bucket_count}, parts={part_count}")
if config.radial_certificate_levels != 1:
    raise SystemExit("radial_certificate_levels must be 1")
print(f"preflight maps={len(grid.table())} buckets={bucket_count} parts={part_count}")
PY

mkdir -p "$ROOT/parts" "$ROOT/logs" "$ROOT/aborted-restart"
for ((index = WORKER_ID; index < PART_COUNT; index += WORKER_COUNT)); do
  part_name="part-$(printf '%05d' "$index")-of-$(printf '%05d' "$PART_COUNT")"
  destination="$ROOT/parts/$part_name"
  if [ -d "$destination" ]; then
    echo "skip completed $part_name"
    continue
  fi

  while IFS= read -r -d '' stale; do
    echo "quarantine stale $stale"
    mv -- "$stale" "$ROOT/aborted-restart/"
  done < <(find "$ROOT/parts" -mindepth 1 -maxdepth 1 -type d \
    -name ".${part_name}.partial-*" -print0)

  log="$ROOT/logs/$part_name-$(date +%Y%m%dT%H%M%S).log"
  {
    echo "worker_id=$WORKER_ID"
    echo "part_index=$index"
    echo "part_count=$PART_COUNT"
    echo "config_sha256=$(sha256sum "$CONFIG" | awk '{print $1}')"
    echo "commit=$(cd "$REPO" && git rev-parse HEAD)"
    date
    env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$REPO/src" \
      "$PYTHON" -c 'from hammlet.cli import main; raise SystemExit(main())' \
      build-maps "$CONFIG" "$ROOT" --part-index "$index" --part-count "$PART_COUNT"
    date
  } >"$log" 2>&1
  echo "completed $part_name log=$log"
done
