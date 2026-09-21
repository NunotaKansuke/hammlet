#!/usr/bin/env bash
# Grid Engine task wrapper for ge-map-benchmark.py.
set -euo pipefail

REPO="${REPO:-/extern/moao38_7/nunota/hammlet}"
PYTHON="${PYTHON:-/whome/nunota/.pyenv/versions/anaconda3-5.3.0/envs/jaxenv/bin/python}"
NUMPY_OVERLAY="${NUMPY_OVERLAY:-}"
VBM_COMPAT="${VBM_COMPAT:-}"
NICE_LEVEL="${NICE_LEVEL:-19}"
MANIFEST="${MANIFEST:?set MANIFEST to the benchmark manifest}"
OUTPUT="${OUTPUT:?set OUTPUT to the benchmark output root}"

echo "host=$(hostname -f)"
echo "sge_job_id=${JOB_ID:-unknown}"
echo "sge_task_id=${SGE_TASK_ID:-unknown}"
echo "start=$(date -Is)"
echo "map_manifest=$MANIFEST"
echo "output=$OUTPUT"
echo "commit=$(cd "$REPO" && git rev-parse HEAD)"
config_path=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"])' "$MANIFEST")
echo "config_sha256=$(sha256sum "$config_path" | awk '{print $1}')"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$REPO/src"
if [ -n "$VBM_COMPAT" ]; then
  export PYTHONPATH="$VBM_COMPAT:$PYTHONPATH"
fi
if [ -n "$NUMPY_OVERLAY" ]; then
  export PYTHONPATH="$NUMPY_OVERLAY:$PYTHONPATH"
fi

/usr/bin/time -v nice -n "$NICE_LEVEL" "$PYTHON" "$REPO/scripts/ge-map-benchmark.py" run \
  "$MANIFEST" "$OUTPUT"

echo "end=$(date -Is)"
