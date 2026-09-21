#!/usr/bin/env python3
"""Make lightweight diagnostics for a ``roman_local`` result directory.

The chi-square map uses only ``minima.npz``.  The light-curve panel additionally
opens the original atlas read-only to reconstruct the best FFT seed.  Direct
VBMicrolensing comparison is optional and is never needed for the search run.
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
        "--data-root",
        type=Path,
        help=(
            "override the directory containing the data files; report path "
            "basenames are resolved under this directory"
        ),
    )
    parser.add_argument(
        "--map-id",
        type=int,
        help="plot this map's best light curve (default: globally best map)",
    )
    parser.add_argument(
        "--direct-vbm",
        action="store_true",
        help="add a direct VBMicrolensing curve to the light-curve plot",
    )
    parser.add_argument(
        "--zoom-t-eff",
        "--zoom-tE",
        dest="zoom_t_eff",
        type=float,
        default=5.0,
        help=(
            "half-width of the light-curve plot around the best t0, in "
            "t_eff=max(|u0|,rho)*tE (default: 5)"
        ),
    )
    parser.add_argument(
        "--min-zoom-half-width",
        type=float,
        default=1.0,
        help=(
            "minimum half-width in days for the light-curve panel; the "
            "window also recenters on the observed peak when the best t0 "
            "would miss it (default: 1 day)"
        ),
    )
    parser.add_argument(
        "--time-start",
        type=float,
        help="explicit lower time limit for the light-curve plot",
    )
    parser.add_argument(
        "--time-end",
        type=float,
        help="explicit upper time limit for the light-curve plot",
    )
    parser.add_argument(
        "--delta-chi2-max",
        type=float,
        default=None,
        help=(
            "fixed linear chi2 color scale upper limit; omit for an adaptive "
            "upper limit based on the top cells"
        ),
    )
    parser.add_argument(
        "--delta-chi2-top-cells",
        type=int,
        default=200,
        help=(
            "number of best (s,q) cells used to choose the adaptive color "
            "limit (default: 200)"
        ),
    )
    parser.add_argument(
        "--skip-lightcurve",
        action="store_true",
        help="make only chi2_map.png; avoids reopening the atlas",
    )
    parser.add_argument(
        "--skip-map",
        action="store_true",
        help="make only lightcurve.png; reuses an existing chi2_map.png",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.zoom_t_eff <= 0.0:
        raise SystemExit("--zoom-t-eff must be positive")
    if args.min_zoom_half_width <= 0.0:
        raise SystemExit("--min-zoom-half-width must be positive")
    if (args.time_start is None) != (args.time_end is None):
        raise SystemExit("--time-start and --time-end must be supplied together")
    if (
        args.time_start is not None
        and args.time_end is not None
        and args.time_end <= args.time_start
    ):
        raise SystemExit("--time-end must be greater than --time-start")
    if args.delta_chi2_max is not None and args.delta_chi2_max <= 0.0:
        raise SystemExit("--delta-chi2-max must be positive")
    if args.delta_chi2_top_cells < 1:
        raise SystemExit("--delta-chi2-top-cells must be positive")
    result = args.result.expanduser().resolve()
    data_root = (
        None if args.data_root is None else args.data_root.expanduser().resolve()
    )
    report = _read_json(result / "report.json")
    minima_path = result / "minima.npz"
    if not minima_path.is_file():
        raise SystemExit(f"missing minima artifact: {minima_path}")
    import numpy as np

    with np.load(minima_path) as arrays:
        map_ids = np.asarray(arrays["map_ids"], dtype=np.int64)
        parameters = np.asarray(arrays["parameters"], dtype=np.float64)
        chi2 = np.asarray(arrays["chi2"], dtype=np.float64)
        scanned = np.asarray(arrays["scanned"], dtype=bool)
        geometries = np.asarray(arrays["geometries"], dtype=np.float64)
        geometry_index = np.asarray(arrays["geometry_index"], dtype=np.int64)
        alpha_index = np.asarray(arrays["alpha_index"], dtype=np.int64)
    if parameters.shape != (len(map_ids), 3):
        raise SystemExit("minima parameters have an unexpected shape")
    output = result / "figures"
    output.mkdir(parents=True, exist_ok=True)
    paths = {
    }
    if not args.skip_map:
        chi2_path = output / "chi2_map.png"
        chi2_scale = _plot_chi2_map(
            report,
            parameters,
            map_ids,
            chi2,
            scanned,
            chi2_path,
            max_delta_chi2=args.delta_chi2_max,
            top_cells=args.delta_chi2_top_cells,
        )
        paths.update(
            {
                "chi2_map": str(chi2_path.resolve()),
                "chi2_scale": chi2_scale,
            }
        )
    elif not (output / "chi2_map.png").is_file():
        raise SystemExit(
            f"--skip-map requested but existing chi2_map.png is missing: {output}"
        )
    if not args.skip_lightcurve:
        atlas_path = args.atlas or Path(report["atlas"]["path"])
        if args.time_start is not None:
            if args.map_id is None:
                lightcurve_path = output / (
                    f"lightcurve-t{args.time_start:g}-{args.time_end:g}.png"
                )
            else:
                lightcurve_path = output / (
                    f"lightcurve-map-{args.map_id:08d}-"
                    f"t{args.time_start:g}-{args.time_end:g}.png"
                )
        else:
            lightcurve_path = (
                output / "lightcurve.png"
                if args.map_id is None
                else output / f"lightcurve-map-{args.map_id:08d}.png"
            )
        _plot_lightcurve(
            report,
            atlas_path,
            map_ids,
            parameters,
            chi2,
            scanned,
            geometries,
            geometry_index,
            alpha_index,
            lightcurve_path,
            direct_vbm=args.direct_vbm,
            zoom_t_eff=args.zoom_t_eff,
            min_zoom_half_width=args.min_zoom_half_width,
            map_id=args.map_id,
            data_root=data_root,
            time_range=(args.time_start, args.time_end)
            if args.time_start is not None
            else None,
        )
        paths["lightcurve"] = str(lightcurve_path.resolve())
    print(json.dumps(paths, indent=2))
    return 0


def _plot_chi2_map(
    report,
    parameters,
    map_ids,
    chi2,
    scanned,
    output: Path,
    *,
    max_delta_chi2: float | None,
    top_cells: int,
) -> dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    finite = np.asarray(scanned, dtype=bool) & np.isfinite(chi2)
    if not np.any(finite):
        raise SystemExit("no finite scanned map chi2 values to plot")
    grouped: dict[tuple[float, float], tuple[float, int]] = {}
    for row in np.flatnonzero(finite):
        key = (float(parameters[row, 0]), float(parameters[row, 1]))
        previous = grouped.get(key)
        if previous is None or chi2[row] < previous[0]:
            grouped[key] = (float(chi2[row]), int(row))
    rows = np.asarray([value[1] for value in grouped.values()], dtype=np.int64)
    best_chi2 = float(np.min(chi2[finite]))
    raw_delta_chi2 = np.maximum(chi2[rows] - best_chi2, 0.0)
    if max_delta_chi2 is None:
        selected_cells = min(int(top_cells), len(raw_delta_chi2))
        rank = selected_cells - 1
        vmax = float(np.partition(raw_delta_chi2, rank)[rank])
        vmax = max(vmax, np.finfo(float).eps)
        scale_info = {
            "mode": "adaptive-top-cells",
            "top_cells": selected_cells,
            "available_cells": int(len(raw_delta_chi2)),
            "vmax": vmax,
        }
        scale_label = (
            f"adaptive vmax from best {selected_cells} (Δχ²={vmax:g})"
        )
    else:
        vmax = float(max_delta_chi2)
        scale_info = {
            "mode": "fixed",
            "top_cells": None,
            "available_cells": int(len(raw_delta_chi2)),
            "vmax": vmax,
        }
        scale_label = f"fixed vmax={vmax:g}"
    delta_chi2 = np.clip(raw_delta_chi2, 0.0, vmax)
    figure, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    image = axis.scatter(
        parameters[rows, 0],
        parameters[rows, 1],
        c=delta_chi2,
        s=16,
        cmap="jet_r",
        vmin=0.0,
        vmax=vmax,
        rasterized=True,
    )
    figure.colorbar(
        image,
        ax=axis,
        label=f"Δχ² per (s,q), {scale_label}",
    )
    reference = report.get("event", {}).get("reference", {})
    truth_s = reference.get("s")
    truth_q = reference.get("q")
    if truth_s is not None and truth_q is not None:
        truth = (np.log10(float(truth_s)), np.log10(float(truth_q)))
        axis.scatter(
            *truth,
            marker="o",
            s=220,
            facecolors="none",
            edgecolors="white",
            linewidths=3.0,
            zorder=4,
        )
        axis.scatter(
            *truth,
            marker="o",
            s=180,
            facecolors="none",
            edgecolors="black",
            linewidths=1.3,
            zorder=5,
        )
    axis.set_xlabel("log10(s)")
    axis.set_ylabel("log10(q)")
    axis.set_title("Roman local FFT Δχ² map")
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return scale_info


def _plot_lightcurve(
    report,
    atlas_path,
    map_ids,
    parameters,
    chi2,
    scanned,
    geometries,
    geometry_index,
    alpha_index,
    output: Path,
    *,
    direct_vbm: bool,
    zoom_t_eff: float,
    min_zoom_half_width: float,
    map_id: int | None,
    data_root: Path | None,
    time_range: tuple[float, float] | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    finite = np.asarray(scanned, dtype=bool) & np.isfinite(chi2)
    if not np.any(finite):
        raise SystemExit("no finite scanned map chi2 values to plot")
    best = _selected_row(map_ids, chi2, finite, map_id)
    settings = report["search_settings"]
    m_max = int(settings["m_max"])
    radial_order = int(settings["radial_order"])
    n_alpha = int(settings["n_alpha"])
    center = geometries[int(geometry_index[best])]
    alpha = 2.0 * np.pi * int(alpha_index[best]) / n_alpha

    from tools.roman_local.atlas import open_readonly_atlas
    from tools.roman_local.roman_event import load_data_file, load_gulls_data_file
    from hammlet._core.config import PSPLGeometry
    from hammlet._core.map_adapter import trajectory_xy
    from hammlet._core.radial import interpolate_coefficients
    from hammlet._core.reference import profile_flux
    from hammlet._core.trajectory import radial_coordinates

    atlas = open_readonly_atlas(atlas_path)
    tail = (
        report.get("atlas", {})
        .get("metadata", {})
        .get("point_lens_tail", {})
    )
    if isinstance(tail, dict) and tail.get("enabled"):
        atlas = atlas.with_point_lens_tail(
            float(tail["max_radius"]),
            radial_order=radial_order,
        )
    nodes, map_parameters, coefficients = atlas.coefficient_row(
        int(map_ids[best]), m_max=m_max
    )
    event = report["event"]
    window = tuple(float(value) for value in event["window"])
    data_format = event.get("data_format", "roman-mag")
    dataset_names = tuple(event.get("datasets", {}).keys())
    data_paths = tuple(
        data_root / Path(path).name if data_root is not None else Path(path)
        for path in event["data_paths"]
    )
    if len(dataset_names) != len(data_paths):
        dataset_names = tuple(
            f"data-{index}" for index in range(len(data_paths))
        )
    if data_format == "gulls-all-lc":
        datasets = [
            load_gulls_data_file(path, name=name, window=window)
            for name, path in zip(dataset_names, data_paths)
        ]
    else:
        datasets = [
            load_data_file(
                path,
                name=name,
                window=window,
                data_format=data_format,
            )
            for name, path in zip(dataset_names, data_paths)
        ]
    all_times = np.concatenate([dataset.time for dataset in datasets])
    all_flux = np.concatenate([dataset.flux for dataset in datasets])
    finite_data = np.isfinite(all_times) & np.isfinite(all_flux)
    if not np.any(finite_data):
        raise SystemExit("no finite data points to plot")
    observed_peak_time = float(all_times[finite_data][np.argmax(all_flux[finite_data])])
    geometry = PSPLGeometry(
        float(center[0]), float(center[1]), float(center[2])
    )
    if time_range is None:
        rho = 10.0 ** float(parameters[best, 2])
        peak_scale = max(abs(float(geometry.u0)), rho)
        effective_timescale = peak_scale * float(geometry.tE)
        zoom_half_width = max(
            float(zoom_t_eff) * effective_timescale,
            float(min_zoom_half_width),
        )
        model_lower = geometry.t0 - zoom_half_width
        model_upper = geometry.t0 + zoom_half_width
        peak_is_well_inside = (
            model_lower <= observed_peak_time <= model_upper
            and abs(observed_peak_time - geometry.t0) <= 0.5 * zoom_half_width
        )
        if peak_is_well_inside:
            focus_time = geometry.t0
            focus_label = "best t0"
        else:
            focus_time = observed_peak_time
            focus_label = "data peak"
        zoom_lower = focus_time - zoom_half_width
        zoom_upper = focus_time + zoom_half_width
        local_times = all_times[finite_data & (all_times >= zoom_lower) & (all_times <= zoom_upper)]
        if local_times.size >= 2:
            plot_lower = float(np.min(local_times))
            plot_upper = float(np.max(local_times))
        else:
            plot_lower = float(np.min(all_times[finite_data]))
            plot_upper = float(np.max(all_times[finite_data]))
            focus_label = "full data fallback"
    else:
        plot_lower, plot_upper = (float(value) for value in time_range)
        focus_label = "explicit range"
    figure, axes = plt.subplots(
        len(datasets), 1, figsize=(9.0, 4.0 * len(datasets)), squeeze=False
    )
    evaluator = None
    if direct_vbm:
        from hammlet._core.direct_vbm import VBMBinaryLensEvaluator
        from hammlet._core.caustics import adamgrid_map_origin_shift

        atlas_metadata = report.get("atlas", {}).get("metadata", {})
        coordinate_frame = (
            atlas_metadata.get("coordinate_frame", "map")
            if isinstance(atlas_metadata, dict)
            else "map"
        )
        if coordinate_frame is None:
            coordinate_frame = "map"
        if coordinate_frame not in ("map", "native"):
            raise SystemExit(f"unsupported atlas coordinate frame: {coordinate_frame!r}")

        evaluator = VBMBinaryLensEvaluator(
            float(10.0 ** map_parameters[0]),
            float(10.0 ** map_parameters[1]),
            float(10.0 ** map_parameters[2]),
            coordinate_frame=coordinate_frame,
        )
    for axis, dataset in zip(axes[:, 0], datasets):
        radius, _ = radial_coordinates(dataset.time, geometry)
        phase = _trajectory_phase(dataset.time, geometry, alpha)
        local = interpolate_coefficients(
            coefficients, radius, nodes, order=radial_order
        )
        mode = np.arange(m_max + 1)
        fft_mag = 1.0 + np.real(
            local[:, 0]
            + 2.0
            * np.sum(
                local[:, 1:]
                * np.exp(1j * phase[:, None] * mode[None, 1:]),
                axis=1,
            )
        )
        fft_profile = profile_flux(fft_mag, dataset)
        fft_model = fft_profile.source_flux * fft_mag + fft_profile.blend_flux
        axis.errorbar(
            dataset.time,
            dataset.flux,
            yerr=dataset.error,
            fmt=".",
            ms=2.2,
            alpha=0.35,
            color="C0",
            zorder=2,
            label=f"{dataset.name} data",
        )
        axis.plot(
            dataset.time,
            fft_model,
            color="C1",
            linewidth=1.4,
            zorder=3,
            label=f"FFT seed (chi2={fft_profile.chi2:.1f})",
        )
        if evaluator is not None:
            x, y = trajectory_xy(
                dataset.time, geometry.t0, geometry.u0, geometry.tE, alpha
            )
            if coordinate_frame == "map":
                x = x - adamgrid_map_origin_shift(
                    float(10.0 ** map_parameters[0]),
                    float(10.0 ** map_parameters[1]),
                )
            direct_mag = evaluator.magnification(x, y)
            direct_profile = profile_flux(direct_mag, dataset)
            direct_model = (
                direct_profile.source_flux * direct_mag + direct_profile.blend_flux
            )
            axis.plot(
                dataset.time,
                direct_model,
                color="tab:orange",
                linewidth=1.0,
                label=f"direct VBM (chi2={direct_profile.chi2:.1f})",
        )
        axis.set_ylabel("flux")
        axis.set_xlim(plot_lower, plot_upper)
        axis.legend(loc="best")
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("time")
    range_label = (
        f"time {plot_lower:g}–{plot_upper:g}"
        if time_range is not None
        else f"adaptive {focus_label}; half-width={zoom_half_width:g} d"
    )
    figure.suptitle(
        f"map {int(map_ids[best])}: log(s,q,rho)="
        f"{parameters[best, 0]:.3f}, {parameters[best, 1]:.3f}, {parameters[best, 2]:.3f} "
        f"({range_label})"
    )
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _trajectory_phase(time, geometry, alpha):
    import numpy as np

    tau = (np.asarray(time, dtype=np.float64) - geometry.t0) / geometry.tE
    x = geometry.u0 * np.sin(alpha) - tau * np.cos(alpha)
    y = -geometry.u0 * np.cos(alpha) - tau * np.sin(alpha)
    return np.arctan2(y, x)


def _best_row(map_ids, chi2, finite):
    import numpy as np

    rows = np.flatnonzero(finite)
    return int(rows[np.argmin(chi2[rows])])


def _selected_row(map_ids, chi2, finite, map_id):
    import numpy as np

    if map_id is None:
        return _best_row(map_ids, chi2, finite)
    rows = np.flatnonzero(np.asarray(map_ids, dtype=np.int64) == int(map_id))
    if not rows.size:
        raise SystemExit(f"map {map_id} is not present in minima.npz")
    row = int(rows[0])
    if not bool(np.asarray(finite, dtype=bool)[row]):
        raise SystemExit(f"map {map_id} has no finite scanned chi2")
    return row


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"could not read report.json: {path}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"report.json must contain an object: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
