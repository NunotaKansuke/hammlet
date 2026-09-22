#!/usr/bin/env bash
set -euo pipefail

repo_dir="${HAMMLET_REPO_DIR:-/extern/moao38_7/nunota/hammlet}"
python_bin="${HAMMLET_PYTHON:-/extern/whome/nunota/.pyenv/versions/anaconda3-5.3.0/envs/jaxenv/bin/python}"
numpy_overlay="${HAMMLET_NUMPY_OVERLAY:-/extern/moao42_9/nunota/hammlet-maps/compat-build-20260807/numpy-2.2.4-overlay}"
vbm_compat="${HAMMLET_VBM_COMPAT:-/extern/moao42_9/nunota/hammlet-maps/compat-build-20260807/vbm5-glibc212-py310-retry2}"

export PYTHONPATH="${numpy_overlay}:${vbm_compat}:${repo_dir}/src:${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${python_bin}" "${repo_dir}/scripts/plot-roman-lm-q-s.py" "$@"
