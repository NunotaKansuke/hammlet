#!/usr/bin/env bash
# Build a non-AVX NumPy overlay for the glibc used by old moao hosts.
set -euo pipefail

BASE_PYTHON="${BASE_PYTHON:-/extern/whome/nunota/.pyenv/versions/anaconda3-5.3.0}"
JAXENV="${JAXENV:-$BASE_PYTHON/envs/jaxenv}"
PYTHON="${PYTHON:-$JAXENV/bin/python}"
CXX="${CXX:-$JAXENV/bin/x86_64-conda-linux-gnu-c++}"
CC="${CC:-$JAXENV/bin/x86_64-conda-linux-gnu-cc}"
NUMPY_CFLAGS="${NUMPY_CFLAGS:--O2 -fPIC -march=x86-64 -mtune=generic}"
NUMPY_CXXFLAGS="${NUMPY_CXXFLAGS:--O2 -fPIC -march=x86-64 -mtune=generic}"
NUMPY_LDFLAGS="${NUMPY_LDFLAGS:--Wl,-rpath,$JAXENV/lib}"
SOURCE="${1:?usage: $0 NUMPY_SOURCE_TARBALL OUTPUT_DIRECTORY}"
OUTPUT="${2:?usage: $0 NUMPY_SOURCE_TARBALL OUTPUT_DIRECTORY}"

if [ -e "$OUTPUT" ]; then
  echo "refusing to overwrite existing output: $OUTPUT" >&2
  exit 2
fi
if [ ! -f "$SOURCE" ]; then
  echo "NumPy source tarball is missing: $SOURCE" >&2
  exit 1
fi
for tool in "$PYTHON" "$CC" "$CXX" unzip; do
  if ! command -v "$tool" >/dev/null 2>&1 && [ ! -x "$tool" ]; then
    echo "required build tool is missing: $tool" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT/wheels" "$OUTPUT/pip-cache"

env \
  CC="$CC" \
  CXX="$CXX" \
  CFLAGS="$NUMPY_CFLAGS" \
  CXXFLAGS="$NUMPY_CXXFLAGS" \
  LDFLAGS="$NUMPY_LDFLAGS" \
  PIP_CACHE_DIR="$OUTPUT/pip-cache" \
  "$PYTHON" -m pip wheel "$SOURCE" --no-deps \
    -w "$OUTPUT/wheels" \
    --config-settings=setup-args=-Dcpu-dispatch=none \
    --config-settings=setup-args=-Dblas=none \
    --config-settings=setup-args=-Dlapack=none

WHEEL="$(find "$OUTPUT/wheels" -maxdepth 1 -name 'numpy-*.whl' -print -quit)"
if [ -z "$WHEEL" ]; then
  echo "NumPy wheel was not produced in $OUTPUT/wheels" >&2
  exit 1
fi
unzip -q "$WHEEL" -d "$OUTPUT"

PYTHONPATH="$OUTPUT" "$PYTHON" - <<'PY'
import numpy as np

print("NumPy compatibility overlay: OK")
print("version:", np.__version__)
print("sum:", np.sum(np.arange(10, dtype=np.float64)))
print("fft:", np.fft.rfft(np.arange(8.0))[0])
PY

echo "built NumPy compatibility overlay: $OUTPUT"
