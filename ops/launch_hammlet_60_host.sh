#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/extern/moao42_9/nunota/hammlet-maps/run-002}"
PART_COUNT="${PART_COUNT:-60}"
JOBS_PER_HOST="${JOBS_PER_HOST:-20}"
HOST_INDEX="${1:?usage: $0 HOST_INDEX [CONFIG_PATH]}"
if [ "$PART_COUNT" -ne 60 ] || [ "$JOBS_PER_HOST" -ne 20 ]; then
  echo "this launcher requires PART_COUNT=60 and JOBS_PER_HOST=20" >&2
  exit 2
fi
case "$HOST_INDEX" in 0|1|2) ;; *) echo "host index must be 0, 1, or 2" >&2; exit 2 ;; esac

first=$((HOST_INDEX * JOBS_PER_HOST))
for offset in $(seq 0 $((JOBS_PER_HOST - 1))); do
  index=$((first + offset))
  if [ "${2-}" ]; then
    PART_COUNT="$PART_COUNT" ROOT="$ROOT" "$ROOT/launch-part.sh" "$index" "$2"
  else
    PART_COUNT="$PART_COUNT" ROOT="$ROOT" "$ROOT/launch-part.sh" "$index"
  fi
done
