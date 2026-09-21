# VBMicrolensing compatibility build for old moao hosts

The current generation environment uses VBMicrolensing 5.5, but the
pre-built extension in `jaxenv` is not loadable on hosts with glibc 2.12.
This procedure builds a NumPy overlay and rebuilds the same VBMicrolensing
5.5 sources against the existing Python 3.10 interpreter in `jaxenv`.

The resulting stack is sufficient for direct-VBM map generation. JAX remains
a separate compatibility problem; see [Limitations](#limitations).

## Build the NumPy overlay

Build NumPy on an old-glibc host so its extension modules do not carry newer
glibc requirements. The build deliberately disables optional BLAS/LAPACK and
CPU dispatch. Map generation needs NumPy FFT and array operations, and this
keeps the result usable on hosts without AVX.

```bash
cd /extern/moao38_7/nunota/hammlet

BUILD=/extern/moao42_9/nunota/hammlet-maps/compat-build-YYYYMMDD
mkdir -p "$BUILD/sources"
curl -fL \
  'https://files.pythonhosted.org/packages/e1/78/31103410a57bc2c2b93a3597340a8119588571f6a4539067546cb9a0bfac/numpy-2.2.4.tar.gz' \
  -o "$BUILD/sources/numpy-2.2.4.tar.gz"
sha256sum "$BUILD/sources/numpy-2.2.4.tar.gz"

env -u CC -u CXX -u CFLAGS -u CXXFLAGS -u LDFLAGS \
  scripts/build-numpy-compat.sh \
  "$BUILD/sources/numpy-2.2.4.tar.gz" \
  "$BUILD/numpy-2.2.4-overlay"
```

The expected source SHA-256 is
`9ba03692a45d3eef66559efe1d1096c4b9b75c0986b5dff5530c378fb8331d4f`.
The overlay is a `PYTHONPATH` tree; it is not installed into `jaxenv`.

## Build

Run the build on a host where the shared `jaxenv` files and the old-glibc
Python environment are visible, for example `moao38`:

```bash
cd /extern/moao38_7/nunota/hammlet

BASE=/extern/whome/nunota/.pyenv/versions/anaconda3-5.3.0
JAXENV="$BASE/envs/jaxenv"
BUILD=/extern/moao42_9/nunota/hammlet-maps/compat-build-YYYYMMDD
OVERLAY="$BUILD/numpy-2.2.4-overlay"
OUT="$BUILD/vbm5-glibc212-py310"

# Do not inherit a host/conda compiler or linker flags accidentally.
env -u CC -u CXX -u CFLAGS -u CXXFLAGS -u LDFLAGS \
  PYTHONPATH="$OVERLAY" PYTHON="$JAXENV/bin/python" \
  scripts/build-vbm-compat.sh "$OUT"
```

The script copies the VBM 5.5 C++ sources, Python package data, and metadata
from:

```text
$JAXENV/lib/python3.10/site-packages/VBMicrolensing
```

It then makes only two compatibility substitutions:

1. `std::filesystem::exists` is replaced with the glibc-2.12-compatible
   `access(2)` check.
2. Python 3.7's missing `importlib.metadata` is replaced by the installed
   `importlib_metadata` backport. Python 3.10 builds keep the standard-library
   import.

The compiler defaults to the conda-forge compiler shipped with `jaxenv`.
That toolchain produces old-glibc-compatible references and the resulting
extension uses the shared `jaxenv/lib` C++ runtime through its RPATH. Do not
replace it with `/usr/local/bin/g++` unless the resulting ELF dependencies
have been checked; the latter can emit `GLIBC_2.14` references.

The output directory is intentionally required to be new. The script refuses
to overwrite an existing build.

## Verify the build

First check the package and run one representative VBM calculation with the
Python 3.10 interpreter and the NumPy overlay:

```bash
PYTHONPATH="$OVERLAY:$OUT" "$JAXENV/bin/python" - <<'PY'
import VBMicrolensing

vbm = VBMicrolensing.VBMicrolensing()
vbm.Tol = 1.0e-3
vbm.RelTol = 1.0e-4
vbm.a1 = 0.0

print("version:", VBMicrolensing.__version__)
print("BinaryMag2:", vbm.BinaryMag2(1.1, 0.3, 0.0, 0.0, 0.01))
PY
```

The representative value should be approximately
`5.67744192615535`. For a cross-host smoke test, use the same shared output
directory on an old node:

```bash
ssh moao39 "OUT=/extern/moao42_9/nunota/hammlet-maps/vbm5-glibc212-YYYYMMDD; \
BASE=/extern/whome/nunota/.pyenv/versions/anaconda3-5.3.0; \
OVERLAY=/extern/moao42_9/nunota/hammlet-maps/compat-build-YYYYMMDD/numpy-2.2.4-overlay; \
PYTHONPATH=\"\$OVERLAY:\$OUT\" \"\$BASE/envs/jaxenv/bin/python\" -c \
'import VBMicrolensing; print(VBMicrolensing.__version__)'"
```

Check the extension's dynamic requirements as well:

```bash
SO="$OUT/VBMicrolensing/VBMicrolensing.cpython-310-x86_64-linux-gnu.so"
readelf --version-info "$SO" | grep -E 'Name: (GLIBC|GLIBCXX|CXXABI)'
ldd "$SO"
```

The extension must not require `GLIBC_2.14`. The complete output tree must be
kept together: the Python wrapper, `data/ESPL.tbl`, `data/SunEphemeris.txt`,
the shared object, and the 5.5 metadata are all needed.

## Verify map generation

Use the same `PYTHONPATH` ordering for Hammlet. The repository's map builder
does not need JAX or SciPy; SciPy is imported only when the sparse search
kernel is called.

```bash
REPO=/extern/moao38_7/nunota/hammlet
PYTHONPATH="$OVERLAY:$OUT:$REPO/src" nice -n 19 "$JAXENV/bin/python" - <<'PY'
from pathlib import Path

from hammlet import MapConfig, ParameterGrid, build_maps

config = MapConfig(
    m_max=8,
    core_m_max=4,
    diagnostic_m_max=8,
    radial_nodes=16,
    min_n_phi=16,
    caustic_base_n_phi=32,
    max_n_phi=32,
    caustic_local_levels=1,
    radial_pilot_bins=16,
    radial_pilot_phi=32,
    radial_pilot_m_max=8,
    radial_pilot_maps=1,
    radial_s_buckets=1,
    radial_q_buckets=1,
    shard_size=1,
)
grid = ParameterGrid(s=(1.1,), q=(0.3,), rho=(0.01,))
print(build_maps(Path("/extern/moao42_9/nunota/hammlet-maps/map-smoke"), grid,
                 config=config))
PY
```

The production configuration must of course be used for the real run; this
small configuration only verifies the old-host runtime and file-writing path.
For the Grid Engine benchmark wrapper, set `NUMPY_OVERLAY` and `VBM_COMPAT`
to the same two directories; the wrapper prepends them to `PYTHONPATH` and
runs the process at `nice 19` by default.

## Limitations

The NumPy overlay plus the Python 3.10 VBM build is sufficient for the
current direct-VBM map-generation path on the tested moao3--moao42 hosts.
The production wrapper deliberately uses this same stack on newer hosts too,
so the map coefficients do not depend on each host's native NumPy/VBM build.
JAX is not yet
compatible there:

- The current `jaxlib` requires `GLIBC_2.14` on `moao39` and below.
- `moao3` additionally lacks the required AVX support for that JAX binary.
- SciPy is lazy-imported only by the search/trajectory sparse kernel; it is
  not needed for map generation. Install the search extra after a compatible
  SciPy build is available.

The current map array excludes `moao1` (glibc 2.5 cannot start the shared
Python 3.10 interpreter) and queues currently reported unreachable by Grid
Engine. It uses the compatible queues explicitly listed by
`scripts/submit-map-array.sh`. Keep JAX-based search separate from this map
array until a source-built old-glibc `jaxlib` is produced.
