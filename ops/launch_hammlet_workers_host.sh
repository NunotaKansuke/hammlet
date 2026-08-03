#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/extern/moao42_9/nunota/hammlet-maps/run-003}"
WORKERS_PER_HOST="${WORKERS_PER_HOST:-20}"
HOST_INDEX="${1:?usage: $0 HOST_INDEX}"
if [ "$WORKERS_PER_HOST" -ne 20 ]; then
  echo "this launcher requires WORKERS_PER_HOST=20" >&2
  exit 2
fi
case "$HOST_INDEX" in 0|1|2) ;; *) echo "host index must be 0, 1, or 2" >&2; exit 2 ;; esac

mkdir -p "$ROOT/logs"
first=$((HOST_INDEX * WORKERS_PER_HOST))
for ((offset = 0; offset < WORKERS_PER_HOST; offset++)); do
  worker_id=$((first + offset))
  log="$ROOT/logs/worker-$(printf '%02d' "$worker_id")-$(date +%Y%m%dT%H%M%S).log"
  nohup "$ROOT/worker.sh" "$worker_id" >"$log" 2>&1 < /dev/null &
  echo "worker_id=$worker_id pid=$! log=$log"
done
