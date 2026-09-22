#!/usr/bin/env bash
set -euo pipefail

# Run one truth-centered map-frame Roman event from an SGE array task.
# Arguments: INPUT_ROOT TRUTH_CSV ATLAS PACKED_CACHE OUTPUT_ROOT

repo_dir="${HAMMLET_REPO_DIR:-/extern/moao38_7/nunota/hammlet}"
python_bin="${HAMMLET_PYTHON:-/extern/whome/nunota/.pyenv/versions/anaconda3-5.3.0/envs/jaxenv/bin/python}"
numpy_overlay="${HAMMLET_NUMPY_OVERLAY:-/extern/moao42_9/nunota/hammlet-maps/compat-build-20260807/numpy-2.2.4-overlay}"
vbm_compat="${HAMMLET_VBM_COMPAT:-/extern/moao42_9/nunota/hammlet-maps/compat-build-20260807/vbm5-glibc212-py310-retry2}"

input_root="${1:?usage: run-roman-maptruth-event-sge.sh INPUT_ROOT TRUTH_CSV ATLAS PACKED_CACHE OUTPUT_ROOT [TASK_ID]}"
truth_csv="${2:?usage: run-roman-maptruth-event-sge.sh INPUT_ROOT TRUTH_CSV ATLAS PACKED_CACHE OUTPUT_ROOT [TASK_ID]}"
atlas="${3:?usage: run-roman-maptruth-event-sge.sh INPUT_ROOT TRUTH_CSV ATLAS PACKED_CACHE OUTPUT_ROOT [TASK_ID]}"
packed_cache="${4:?usage: run-roman-maptruth-event-sge.sh INPUT_ROOT TRUTH_CSV ATLAS PACKED_CACHE OUTPUT_ROOT [TASK_ID]}"
output_root="${5:?usage: run-roman-maptruth-event-sge.sh INPUT_ROOT TRUTH_CSV ATLAS PACKED_CACHE OUTPUT_ROOT [TASK_ID]}"

# Individual SGE jobs pass the row explicitly as argument 6.  Keep the
# array/environment fallbacks for manual use and backward compatibility.
task_id="${6:-${SGE_TASK_ID:-${MAPTRUTH_TASK_ID:?SGE_TASK_ID, MAPTRUTH_TASK_ID, or TASK_ID is required}}}"
cores="${MAPTRUTH_CORES:-16}"
workers="${MAPTRUTH_WORKERS:-16}"
export PYTHONPATH="${numpy_overlay}:${vbm_compat}:${repo_dir}/src:${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"

event_fields="$(${python_bin} - "${truth_csv}" "${input_root}" "${task_id}" <<'PY'
import csv
import math
import pathlib
import sys

truth_path = pathlib.Path(sys.argv[1])
input_root = pathlib.Path(sys.argv[2])
task_id = int(sys.argv[3])
with truth_path.open(encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
if not 1 <= task_id <= len(rows):
    raise SystemExit(f"task {task_id} is outside truth rows 1..{len(rows)}")
row = rows[task_id - 1]
data_path = (input_root / "hammlet_input" / row["hammlet_lightcurve"]).resolve()
if not data_path.is_file():
    raise SystemExit(f"missing lightcurve: {data_path}")
values = (
    row["event_id"],
    str(data_path),
    row["t0_day"],
    row["u0"],
    row["tE_days"],
    row["planet_s"],
    row["planet_q"],
    row["rho"],
    repr(math.radians(float(row["alpha_deg"])) % (2.0 * math.pi)),
)
print("\t".join(values))
PY
)"
IFS=$'\t' read -r event_id data_file t0 u0 tE separation mass_ratio rho alpha <<< "${event_fields}"

event_output="${output_root}/events/${event_id}"
mkdir -p "${output_root}/events" "${output_root}/logs" "${output_root}/jax-compilation-cache"
export JAX_COMPILATION_CACHE_DIR="${output_root}/jax-compilation-cache"
log_path="${output_root}/logs/${event_id}.log"

echo "[maptruth] start ${event_id} task=${task_id} host=$(hostname)" | tee -a "${log_path}"
exec "${python_bin}" -m tools.roman_local.parallel_run \
    --data-file "${data_file}" \
    --data-format time-flux-error \
    --t0 "${t0}" \
    --u0 "${u0}" \
    --tE "${tE}" \
    --s "${separation}" \
    --q "${mass_ratio}" \
    --rho "${rho}" \
    --alpha "${alpha}" \
    --geometry-center map-truth \
    --atlas "${atlas}" \
    --packed-cache "${packed_cache}" \
    --output "${event_output}" \
    --event-name "${event_id}" \
    --m-max 128 \
    --n-alpha 540 \
    --radial-order 1 \
    --tail-mode auto \
    --batch-size 1024 \
    --progress-every 10 \
    --candidate-count 100 \
    --cores "${cores}" \
    --workers "${workers}" \
    --threads-per-worker 1 \
    --resume \
    >> "${log_path}" 2>&1
