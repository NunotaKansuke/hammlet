#!/usr/bin/env python3
"""Render one stored Fourier map as a simple Cartesian ``jet`` image.

This is a local diagnostic for the coefficient-only atlas used by the Roman
test.  It reads one map's radial Fourier coefficients, evaluates the angular
series with an inverse FFT, linearly interpolates the radial rings onto an
``(x, y)`` grid, and writes a PNG below the separate Roman result directory.
The atlas is opened read-only and is never modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="roman_local result directory")
    parser.add_argument(
        "--atlas",
        type=Path,
        help="override the atlas path recorded in report.json",
    )
    parser.add_argument(
        "--map-id",
        type=int,
        help="map ID to render (default: best finite scanned map)",
    )
    parser.add_argument(
        "--m-max",
        type=int,
        help="Fourier modes to reconstruct (default: the scan setting, usually 128)",
    )
    parser.add_argument(
        "--n-phi",
        type=int,
        default=2048,
        help="angular samples used by the inverse FFT (default: 2048)",
    )
    parser.add_argument(
        "--pixels",
        type=int,
        default=700,
        help="pixels per Cartesian axis (default: 700)",
    )
    parser.add_argument(
        "--extent",
        type=float,
        help="plot half-width in theta_E units (default: stored outer radius)",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=1.0,
        help="lower linear magnification color limit (default: 1)",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        help="upper linear magnification color limit (default: percentile)",
    )
    parser.add_argument(
        "--vmax-percentile",
        type=float,
        default=99.5,
        help="percentile for the default upper color limit (default: 99.5)",
    )
    parser.add_argument(
        "--scale",
        choices=("log", "linear"),
        default="log",
        help="color scale for magnification (default: log)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG path (default: result/figures/map-<id>-m<m>.png)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n_phi < 2 or args.n_phi % 2:
        raise SystemExit("--n-phi must be a positive even integer")
    if args.pixels < 2:
        raise SystemExit("--pixels must be at least 2")
    if args.vmin < 0.0:
        raise SystemExit("--vmin must be nonnegative")
    if args.vmax is not None and args.vmax <= args.vmin:
        raise SystemExit("--vmax must be greater than --vmin")
    if args.scale == "log" and args.vmin <= 0.0:
        raise SystemExit("--vmin must be positive for --scale log")
    if not 0.0 < args.vmax_percentile <= 100.0:
        raise SystemExit("--vmax-percentile must be in (0, 100]")

    result = args.result.expanduser().resolve()
    report = _read_json(result / "report.json")
    minima = _read_minima(result / "minima.npz")
    map_id = _select_map_id(report, minima, args.map_id)
    scan_m_max = int(report.get("search_settings", {}).get("m_max", 128))
    m_max = scan_m_max if args.m_max is None else int(args.m_max)
    if m_max < 0:
        raise SystemExit("--m-max must be nonnegative")

    atlas_path = (args.atlas or Path(report["atlas"]["path"])).expanduser().resolve()
    from tools.roman_local.atlas import open_readonly_atlas

    atlas = open_readonly_atlas(atlas_path)
    if m_max > atlas.m_max:
        raise SystemExit(
            f"requested m_max={m_max} exceeds the atlas limit m_max={atlas.m_max}"
        )
    radial_nodes, parameters, coefficients = atlas.coefficient_row(
        map_id, m_max=m_max
    )
    if args.n_phi // 2 < m_max:
        raise SystemExit(
            f"--n-phi={args.n_phi} is too small for m_max={m_max}; "
            f"use at least {2 * m_max}"
        )

    import numpy as np

    extent = float(radial_nodes[-1]) if args.extent is None else float(args.extent)
    if not np.isfinite(extent) or extent <= 0.0:
        raise SystemExit("--extent must be positive and finite")
    values = reconstruct_cartesian_map(
        radial_nodes,
        coefficients,
        extent=extent,
        pixels=args.pixels,
        n_phi=args.n_phi,
    )
    geometry, alpha = _trajectory_for_map(report, minima, map_id)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise SystemExit("FFT reconstruction produced no finite pixels")
    vmin = float(args.vmin)
    if args.vmax is None:
        vmax = float(np.nanpercentile(values[finite], args.vmax_percentile))
    else:
        vmax = float(args.vmax)
    vmax = max(vmax, vmin + np.finfo(float).eps)

    output = (
        result / "figures" / f"map-{map_id:08d}-m{m_max}.png"
        if args.output is None
        else args.output.expanduser().resolve()
    )
    _save_map_figure(
        values,
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        output=output,
        map_id=map_id,
        parameters=parameters,
        m_max=m_max,
        radial_nodes=radial_nodes,
        scale=args.scale,
        geometry=geometry,
        alpha=alpha,
    )
    logs, logq, logrho = (float(value) for value in parameters)
    payload = {
        "output": str(output.resolve()),
        "map_id": int(map_id),
        "parameters": [logs, logq, logrho],
        "physical_parameters": {
            "s": float(10.0**logs),
            "q": float(10.0**logq),
            "rho": float(10.0**logrho),
        },
        "m_max": int(m_max),
        "n_phi": int(args.n_phi),
        "pixels": int(args.pixels),
        "radial_support": [float(radial_nodes[0]), float(radial_nodes[-1])],
        "extent": extent,
        "trajectory": {
            "geometry": [float(value) for value in geometry],
            "alpha": float(alpha),
        },
        "color_scale": {
            "cmap": "jet",
            "scale": args.scale,
            "vmin": vmin,
            "vmax": vmax,
        },
        "atlas": str(atlas_path),
        "atlas_read_only": True,
    }
    print(json.dumps(payload, indent=2))
    return 0


def reconstruct_cartesian_map(
    radial_nodes,
    coefficients,
    *,
    extent: float,
    pixels: int,
    n_phi: int,
):
    """Reconstruct ``A(x, y)`` from ``c_m(r)`` using an inverse FFT."""
    import numpy as np

    nodes = np.asarray(radial_nodes, dtype=np.float64)
    coefficients = np.asarray(coefficients)
    if nodes.ndim != 1 or nodes.size < 2 or np.any(np.diff(nodes) <= 0.0):
        raise ValueError("radial_nodes must be a strictly increasing 1-D array")
    if coefficients.ndim != 2 or coefficients.shape[0] != nodes.size:
        raise ValueError("coefficients must have shape (radial_node, mode)")
    n_phi = int(n_phi)
    m_max = coefficients.shape[-1] - 1
    if n_phi < 2 * max(m_max, 1) or n_phi % 2:
        raise ValueError("n_phi must be even and large enough for all modes")

    from hammlet._core.alpha_fft import fourier_series_values

    # Stored coefficients describe X=A-1.  The helper applies the rFFT
    # normalization used by the map builder before evaluating the rings.
    rings = 1.0 + fourier_series_values(coefficients[None, ...], n_phi)[0]
    axis = np.linspace(-float(extent), float(extent), int(pixels))
    x, y = np.meshgrid(axis, axis, indexing="xy")
    radius = np.hypot(x, y)
    valid = (radius >= nodes[0]) & (radius <= nodes[-1])
    safe_radius = np.clip(radius, nodes[0], nodes[-1])

    upper = np.clip(
        np.searchsorted(nodes, safe_radius, side="right"), 1, nodes.size - 1
    )
    lower = upper - 1
    fraction_r = (safe_radius - nodes[lower]) / (nodes[upper] - nodes[lower])

    phase = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    angular_coordinate = phase * n_phi / (2.0 * np.pi)
    angular_lower = np.floor(angular_coordinate).astype(np.int64)
    fraction_phi = angular_coordinate - angular_lower
    angular_upper = (angular_lower + 1) % n_phi

    lower_ring = (1.0 - fraction_phi) * rings[lower, angular_lower]
    lower_ring += fraction_phi * rings[lower, angular_upper]
    upper_ring = (1.0 - fraction_phi) * rings[upper, angular_lower]
    upper_ring += fraction_phi * rings[upper, angular_upper]
    values = (1.0 - fraction_r) * lower_ring + fraction_r * upper_ring
    return np.where(valid, values, np.nan)


def _save_map_figure(
    values,
    *,
    extent: float,
    vmin: float,
    vmax: float,
    output: Path,
    map_id: int,
    parameters,
    m_max: int,
    radial_nodes,
    scale: str,
    geometry,
    alpha: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import numpy as np

    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad("white")
    image = np.ma.masked_invalid(np.clip(values, vmin, vmax))
    normalization = (
        LogNorm(vmin=vmin, vmax=vmax)
        if scale == "log"
        else None
    )
    logs, logq, logrho = (float(value) for value in parameters)
    figure, axis = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    image_options = {
        "origin": "lower",
        "extent": (-extent, extent, -extent, extent),
        "cmap": cmap,
        "interpolation": "nearest",
    }
    if normalization is None:
        image_options.update(vmin=vmin, vmax=vmax)
    else:
        image_options["norm"] = normalization
    artist = axis.imshow(image, **image_options)
    from hammlet._core.map_adapter import trajectory_xy

    t0, u0, tE = (float(value) for value in geometry)
    trajectory_time = np.linspace(
        t0 - (extent + abs(u0) + 0.05) * tE,
        t0 + (extent + abs(u0) + 0.05) * tE,
        4096,
    )
    trajectory_x, trajectory_y = trajectory_xy(
        trajectory_time, t0, u0, tE, float(alpha)
    )
    trajectory_inside = (
        np.hypot(trajectory_x, trajectory_y) <= float(extent)
    )
    axis.plot(
        trajectory_x[trajectory_inside],
        trajectory_y[trajectory_inside],
        color="black",
        linewidth=3.0,
        zorder=4,
    )
    axis.plot(
        trajectory_x[trajectory_inside],
        trajectory_y[trajectory_inside],
        color="white",
        linewidth=1.4,
        label=rf"best trajectory ($\alpha={float(alpha):.3f}$ rad)",
        zorder=5,
    )
    axis.set(
        xlabel=r"$x/\theta_E$",
        ylabel=r"$y/\theta_E$",
        aspect="equal",
        title=(
            f"FFT map {map_id:08d} (M={m_max})\n"
            f"$s={10.0**logs:.5g}$, $q={10.0**logq:.5g}$, "
            f"$\u03c1={10.0**logrho:.5g}$"
        ),
    )
    colorbar_label = (
        r"magnification $A$ (log scale)"
        if scale == "log"
        else r"magnification $A$"
    )
    figure.colorbar(artist, ax=axis, label=colorbar_label)
    axis.legend(loc="upper right", framealpha=0.8)
    axis.text(
        0.02,
        0.02,
        f"radial support: 0–{float(radial_nodes[-1]):g} $\u03b8_E$",
        transform=axis.transAxes,
        color="black",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _read_json(path: Path) -> dict:
    import json

    if not path.is_file():
        raise SystemExit(f"missing JSON artifact: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SystemExit(f"expected a JSON object: {path}")
    return value


def _read_minima(path: Path) -> dict[str, object]:
    import numpy as np

    if not path.is_file():
        raise SystemExit(f"missing minima artifact: {path}")
    with np.load(path) as arrays:
        required = ("map_ids", "chi2", "scanned")
        missing = [name for name in required if name not in arrays]
        if missing:
            raise SystemExit(f"minima artifact is missing: {', '.join(missing)}")
        result = {
            "map_ids": np.asarray(arrays["map_ids"], dtype=np.int64),
            "chi2": np.asarray(arrays["chi2"], dtype=np.float64),
            "scanned": np.asarray(arrays["scanned"], dtype=bool),
        }
        for name in ("geometry_index", "alpha_index", "geometries"):
            if name in arrays:
                result[name] = np.asarray(arrays[name])
        return result


def _select_map_id(report: dict, minima: dict[str, object], requested: int | None) -> int:
    import numpy as np

    map_ids = np.asarray(minima["map_ids"], dtype=np.int64)
    chi2 = np.asarray(minima["chi2"], dtype=np.float64)
    scanned = np.asarray(minima["scanned"], dtype=bool)
    if requested is not None:
        return int(requested)
    best = report.get("best_fft", {}).get("map_id")
    if best is not None:
        return int(best)
    finite = scanned & np.isfinite(chi2)
    if not np.any(finite):
        raise SystemExit("no finite scanned map is available")
    return int(map_ids[np.flatnonzero(finite)[np.argmin(chi2[finite])]])


def _trajectory_for_map(
    report: dict, minima: dict[str, object], map_id: int
) -> tuple[object, float]:
    import numpy as np

    map_ids = np.asarray(minima["map_ids"], dtype=np.int64)
    rows = np.flatnonzero(map_ids == int(map_id))
    if not rows.size:
        raise SystemExit(
            f"map {map_id} is not present in minima.npz; cannot recover its best trajectory"
        )
    row = int(rows[0])
    scanned = np.asarray(minima["scanned"], dtype=bool)
    geometry_index = np.asarray(minima.get("geometry_index"), dtype=np.int64)
    alpha_index = np.asarray(minima.get("alpha_index"), dtype=np.int64)
    geometries = np.asarray(minima.get("geometries"), dtype=np.float64)
    if (
        row >= len(scanned)
        or not scanned[row]
        or geometry_index.ndim != 1
        or alpha_index.ndim != 1
        or geometries.ndim != 2
        or geometries.shape[1] != 3
    ):
        raise SystemExit(f"map {map_id} has no scanned best trajectory")
    geometry_row = int(geometry_index[row])
    if not 0 <= geometry_row < len(geometries):
        raise SystemExit(f"map {map_id} has an invalid geometry index")
    n_alpha = int(report.get("search_settings", {}).get("n_alpha", 0))
    if n_alpha <= 0:
        raise SystemExit("report.json has no valid n_alpha for trajectory recovery")
    return geometries[geometry_row], 2.0 * np.pi * int(alpha_index[row]) / n_alpha


if __name__ == "__main__":
    raise SystemExit(main())
