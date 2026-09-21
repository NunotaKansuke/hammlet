#!/usr/bin/env bash
set -euo pipefail

# Submit the missing-layout array first, then submit the one-map-per-task
# array held on that layout job.  Both worker wrappers apply nice 19.

RUN_ROOT=${1:?usage: launch-production-arrays.sh RUN_ROOT}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
QSUB=${QSUB:-/whome/share/64bit/gridware/sge/bin/lx24-amd64/qsub}
QSTAT=${QSTAT:-/whome/share/64bit/gridware/sge/bin/lx24-amd64/qstat}
PYTHON=${HAMMLET_PYTHON:-/extern/whome/nunota/.pyenv/versions/anaconda3-5.3.0/envs/jaxenv/bin/python}
QUEUE_LIST=${HAMMLET_QUEUE_LIST:-moao3.q,moao4.q,moao8.q,moao12.q,moao13.q,moao14.q,moao16.q,moao19.q,moao20.q,moao21.q,moao22.q,moao23.q,moao25.q,moao26.q,moao28.q,moao29.q,moao30.q,moao31.q,moao32.q,moao33.q,moao34.q,moao37.q,moao38.q,moao39.q,moao40.q,moao41.q,moao42.q}
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"

if timeout 10 "$QSTAT" -u "$USER" 2>/dev/null | awk 'NR > 2 && ($3 == "hammlet-layout-array" || $3 == "hammlet-map-array") { found=1 } END { exit !found }'; then
    echo "refusing duplicate Hammlet array submission" >&2
    exit 2
fi

export PYTHONPATH="${HAMMLET_NUMPY_OVERLAY:-/extern/moao42_9/nunota/hammlet-maps/compat-build-20260807/numpy-2.2.4-overlay}:${HAMMLET_VBM_COMPAT:-/extern/moao42_9/nunota/hammlet-maps/compat-build-20260807/vbm5-glibc212-py310-retry2}:$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
status_json=$("$PYTHON" "$REPO_DIR/scripts/map-run-status.py" "$RUN_ROOT")
layout_pending=$(printf '%s\n' "$status_json" | "$PYTHON" -c 'import json, sys; print(json.load(sys.stdin)["layout_pending_buckets"])')
pending=$(printf '%s\n' "$status_json" | "$PYTHON" -c 'import json, sys; print(json.load(sys.stdin)["pending"])')

if [ "$pending" -le 0 ]; then
    echo "no pending maps; nothing to submit"
    exit 0
fi

hold_jid=
if [ "$layout_pending" -gt 0 ]; then
    layout_output=$("$QSUB" \
        -S /bin/bash \
        -cwd \
        -q "$QUEUE_LIST" \
        -t "1-${layout_pending}" \
        -N hammlet-layout-array \
        -o "$LOG_DIR" \
        -j y \
        "$REPO_DIR/scripts/run-layout-task.sh" "$RUN_ROOT")
    printf '%s\n' "$layout_output"
    hold_jid=$(printf '%s\n' "$layout_output" | sed -n 's/^[Yy]our job-array \([0-9][0-9]*\).*/\1/p')
    if [ -z "$hold_jid" ]; then
        echo "could not parse layout array job ID; map array was not submitted" >&2
        exit 1
    fi
fi

if [ -n "$hold_jid" ]; then
    "$REPO_DIR/scripts/submit-map-array.sh" "$RUN_ROOT" --hold-jid "$hold_jid"
else
    "$REPO_DIR/scripts/submit-map-array.sh" "$RUN_ROOT"
fi
