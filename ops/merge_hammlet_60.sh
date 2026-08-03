#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/extern/moao42_9/nunota/hammlet-maps/run-002}"
REPO="${REPO:-/extern/moao38_7/nunota/hammlet}"
PYTHON="${PYTHON:-/whome/nunota/.pyenv/versions/anaconda3-5.3.0/envs/jaxenv/bin/python}"
PART_COUNT="${PART_COUNT:-60}"
if [ -n "$(cd "$REPO" && git status --porcelain)" ]; then
  echo "repository has uncommitted changes; refusing to merge" >&2
  exit 1
fi
for index in $(seq 0 $((PART_COUNT - 1))); do
  part="$ROOT/parts/part-$(printf '%05d' "$index")-of-$(printf '%05d' "$PART_COUNT")"
  [ -d "$part" ] || { echo "missing completed part: $part" >&2; exit 1; }
done
PYTHONPATH="$REPO/src" "$PYTHON" -c \
  'from hammlet.cli import main; raise SystemExit(main())' merge-maps "$ROOT"
