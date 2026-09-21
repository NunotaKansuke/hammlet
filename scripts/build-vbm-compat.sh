#!/usr/bin/env bash
# Build VBMicrolensing 5.5 for the glibc used by the older moao hosts.
set -euo pipefail

BASE_PYTHON="${BASE_PYTHON:-/extern/whome/nunota/.pyenv/versions/anaconda3-5.3.0}"
JAXENV="${JAXENV:-$BASE_PYTHON/envs/jaxenv}"
SOURCE_PACKAGE="${SOURCE_PACKAGE:-$JAXENV/lib/python3.10/site-packages/VBMicrolensing}"
PYTHON="${PYTHON:-$BASE_PYTHON/bin/python}"
# Use the conda-forge toolchain shipped with jaxenv.  It targets the old
# system glibc; /usr/local/bin/g++ on moao38 produces GLIBC_2.14 references.
CXX="${CXX:-$JAXENV/bin/x86_64-conda-linux-gnu-c++}"
OUTPUT="${1:?usage: $0 OUTPUT_DIRECTORY}"

if [ -e "$OUTPUT" ]; then
  echo "refusing to overwrite existing output: $OUTPUT" >&2
  exit 2
fi
if [ ! -f "$SOURCE_PACKAGE/lib/python_bindings.cpp" ]; then
  echo "VBMicrolensing 5.5 source is missing: $SOURCE_PACKAGE" >&2
  exit 1
fi
if [ ! -x "$CXX" ]; then
  echo "compatible C++ compiler is missing: $CXX" >&2
  exit 1
fi

mkdir -p "$OUTPUT/VBMicrolensing" "$OUTPUT/vbmicrolensing-5.5.dist-info"
cp "$SOURCE_PACKAGE/lib/python_bindings.cpp" "$OUTPUT/"
cp "$SOURCE_PACKAGE/lib/VBMicrolensingLibrary.cpp" "$OUTPUT/"
cp "$SOURCE_PACKAGE/lib/VBMicrolensingLibrary.h" "$OUTPUT/"
cp "$SOURCE_PACKAGE/__init__.py" "$OUTPUT/VBMicrolensing/"
cp -a "$SOURCE_PACKAGE/data" "$OUTPUT/VBMicrolensing/"
cp "$JAXENV/lib/python3.10/site-packages/vbmicrolensing-5.5.dist-info/METADATA" \
  "$OUTPUT/vbmicrolensing-5.5.dist-info/"

# Python 3.7 has the backport as importlib_metadata, not importlib.metadata.
# Keep the standard-library import for Python 3.10 builds.
if ! "$PYTHON" -c 'from importlib.metadata import version' >/dev/null 2>&1; then
  sed -i 's/from importlib\.metadata import version/from importlib_metadata import version/' \
    "$OUTPUT/VBMicrolensing/__init__.py"
fi

# VBMicrolensing 5.5 uses filesystem::exists only for an optional table path.
# The rest of the extension is C++11-compatible; access(2) works on glibc 2.12
# and avoids requiring a newer libstdc++ filesystem implementation.
sed -i \
  -e 's/#include <filesystem>/#include <unistd.h>/' \
  -e 's/std::filesystem::exists(sattabledir)/(access(sattabledir, F_OK) == 0)/' \
  "$OUTPUT/python_bindings.cpp"

PYTHON_INCLUDE="$($PYTHON -c 'import sysconfig; print(sysconfig.get_path("include"))')"
NUMPY_INCLUDE="$($PYTHON -c 'import numpy; print(numpy.get_include())')"
# pybind11 3.x in jaxenv only supports Python >=3.8.  The torch wheel carries
# the last pybind11 line that supports the base Python 3.7 runtime.
PYBIND_INCLUDE="${PYBIND_INCLUDE:-$JAXENV/lib/python3.10/site-packages/torch/include}"
EXT_SUFFIX="$($PYTHON -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"

"$CXX" -O3 -fPIC -shared -std=c++11 \
  -I"$PYTHON_INCLUDE" -I"$NUMPY_INCLUDE" -I"$PYBIND_INCLUDE" \
  -Wl,-rpath,"$JAXENV/lib" \
  "$OUTPUT/python_bindings.cpp" "$OUTPUT/VBMicrolensingLibrary.cpp" \
  -o "$OUTPUT/VBMicrolensing/VBMicrolensing$EXT_SUFFIX"

PYTHONPATH="$OUTPUT" "$PYTHON" - <<'PY'
import VBMicrolensing
print("VBMicrolensing import: OK")
print("VBMicrolensing class:", VBMicrolensing.VBMicrolensing)
PY

echo "built VBMicrolensing 5.5 compatibility tree: $OUTPUT"
