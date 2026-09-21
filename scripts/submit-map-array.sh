#!/usr/bin/env bash
set -euo pipefail

# Submit one SGE array job.  SGE creates one array task per pending map and
# schedules those tasks against the three requested queues; this script does
# not split the submission into host-specific batches.

RUN_ROOT=${1:?usage: submit-map-array.sh RUN_ROOT [--hold-jid JOB_ID]}
shift
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
QSUB=${QSUB:-/whome/share/64bit/gridware/sge/bin/lx24-amd64/qsub}
PYTHON=${HAMMLET_PYTHON:-/extern/whome/nunota/.pyenv/versions/anaconda3-5.3.0/envs/jaxenv/bin/python}
QUEUE_LIST=${HAMMLET_QUEUE_LIST:-moao3.q,moao4.q,moao8.q,moao12.q,moao13.q,moao14.q,moao16.q,moao19.q,moao20.q,moao21.q,moao22.q,moao23.q,moao25.q,moao26.q,moao28.q,moao29.q,moao30.q,moao31.q,moao32.q,moao33.q,moao34.q,moao37.q,moao38.q,moao39.q,moao40.q,moao41.q,moao42.q}
HOLD_JID=
if [ "$#" -gt 0 ]; then
    if [ "$#" -ne 2 ] || [ "$1" != "--hold-jid" ]; then
        echo "usage: submit-map-array.sh RUN_ROOT [--hold-jid JOB_ID]" >&2
        exit 64
    fi
    HOLD_JID=$2
fi
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"

export PYTHONPATH="${HAMMLET_NUMPY_OVERLAY:-/extern/moao42_9/nunota/hammlet-maps/compat-build-20260807/numpy-2.2.4-overlay}:${HAMMLET_VBM_COMPAT:-/extern/moao42_9/nunota/hammlet-maps/compat-build-20260807/vbm5-glibc212-py310-retry2}:$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
status_json=$("$PYTHON" "$REPO_DIR/scripts/map-run-status.py" "$RUN_ROOT")
pending=$(printf '%s\n' "$status_json" | "$PYTHON" -c 'import json, sys; print(json.load(sys.stdin)["pending"])')
layouts=$(printf '%s\n' "$status_json" | "$PYTHON" -c 'import json, sys; print(json.load(sys.stdin)["layout_pending_buckets"])')

if [ "$layouts" -ne 0 ] && [ -z "$HOLD_JID" ]; then
    echo "refusing map submission: $layouts radial-layout buckets are still missing" >&2
    exit 2
fi
if [ "$pending" -le 0 ]; then
    echo "no pending maps; nothing to submit"
    exit 0
fi

HOLD_ARGS=()
if [ -n "$HOLD_JID" ]; then
    HOLD_ARGS=(-hold_jid "$HOLD_JID")
fi

"$QSUB" \
    -S /bin/bash \
    -cwd \
    -q "$QUEUE_LIST" \
    -t "1-${pending}" \
    -N hammlet-map-array \
    -o "$LOG_DIR" \
    -j y \
    "${HOLD_ARGS[@]}" \
    "$REPO_DIR/scripts/run-map-task.sh" "$RUN_ROOT"
