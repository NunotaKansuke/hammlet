#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${HAMMLET_REPO_DIR:-/extern/moao38_7/nunota/hammlet}
PYTHON=${HAMMLET_PYTHON:-/extern/whome/nunota/.pyenv/versions/anaconda3-5.3.0/envs/jaxenv/bin/python}
NUMPY_OVERLAY=${HAMMLET_NUMPY_OVERLAY:-/extern/moao42_9/nunota/hammlet-maps/compat-build-20260807/numpy-2.2.4-overlay}
VBM_COMPAT=${HAMMLET_VBM_COMPAT:-/extern/moao42_9/nunota/hammlet-maps/compat-build-20260807/vbm5-glibc212-py310-retry2}
NICE_LEVEL=${HAMMLET_NICE_LEVEL:-19}

case "$(hostname -s)" in
    moao3|moao4|moao8|moao12|moao13|moao14|moao16|moao19|moao20|moao21|moao22|moao23|moao25|moao26|moao28|moao29|moao30|moao31|moao32|moao33|moao34|moao37|moao38|moao39|moao40|moao41|moao42) ;;
    *) echo "unsupported Hammlet map host for compatibility stack: $(hostname -s)" >&2; exit 64 ;;
esac

export PYTHONPATH="$NUMPY_OVERLAY:$VBM_COMPAT:$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
exec nice -n "$NICE_LEVEL" "$PYTHON" "$REPO_DIR/scripts/run-map-task.py" "$@"
