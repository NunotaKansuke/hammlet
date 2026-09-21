#!/usr/bin/env bash
set -euo pipefail

roman_tool_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${roman_tool_root}/src:${roman_tool_root}:${PYTHONPATH}"
else
    export PYTHONPATH="${roman_tool_root}/src:${roman_tool_root}"
fi
exec python "${roman_tool_root}/scripts/run-roman-local-batch.py" "$@"
