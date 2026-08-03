#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/extern/moao42_9/nunota/hammlet-maps/run-002}"
REPO="${REPO:-/extern/moao38_7/nunota/hammlet}"
PYTHON="${PYTHON:-/whome/nunota/.pyenv/versions/anaconda3-5.3.0/envs/jaxenv/bin/python}"
CONFIG="${CONFIG:-$ROOT/config/production-build.json}"
PART_COUNT="${PART_COUNT:-60}"
PART_INDEX="${1:?usage: $0 PART_INDEX [CONFIG_PATH]}"
if [ "${2-}" ]; then CONFIG="$2"; fi

if ! [[ "$PART_INDEX" =~ ^[0-9]+$ && "$PART_COUNT" =~ ^[1-9][0-9]*$ &&
        "$PART_INDEX" -lt "$PART_COUNT" ]]; then
  echo "require 0 <= PART_INDEX < PART_COUNT (got $PART_INDEX/$PART_COUNT)" >&2
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
if bucket_count < part_count:
    raise SystemExit(f"part_count={part_count} needs at least {part_count} buckets; got {bucket_count}")
if config.radial_certificate_levels != 1:
    raise SystemExit("radial_certificate_levels must be 1")
print(f"preflight maps={len(grid.table())} buckets={bucket_count} parts={part_count}")
PY

PART_DIR="$ROOT/parts/part-$(printf '%05d' "$PART_INDEX")-of-$(printf '%05d' "$PART_COUNT")"
if [ -e "$PART_DIR" ]; then
  echo "completed part already exists; refusing to overwrite: $PART_DIR" >&2
  exit 1
fi

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/part-$(printf '%05d' "$PART_INDEX")-$(date +%Y%m%dT%H%M%S).log"
SHA256="$(sha256sum "$CONFIG" | awk '{print $1}')"
COMMIT="$(cd "$REPO" && git rev-parse HEAD)"
nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$REPO/src" \
  "$PYTHON" -c 'from hammlet.cli import main; raise SystemExit(main())' \
  build-maps "$CONFIG" "$ROOT" --part-index "$PART_INDEX" --part-count "$PART_COUNT" \
  >"$LOG" 2>&1 < /dev/null &
PID=$!
echo "pid=$PID"
echo "log=$LOG"
echo "config_sha256=$SHA256"
echo "commit=$COMMIT"
echo "destination=$PART_DIR"
