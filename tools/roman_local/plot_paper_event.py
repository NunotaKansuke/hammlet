#!/usr/bin/env python3
"""Render a paper-style Roman truth-centered event figure.

This first version keeps the FFT seed and its corresponding caustic geometry
consistent: the inset uses the same best-map ``(s, q, rho)``, map-frame
trajectory geometry, and alpha as the light-curve model.  It is intentionally
separate from the legacy diagnostic plotter while the paper layout is being
iterated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event", help="event result directory or event ID")
    parser.add_argument(
        "--result-root",
        type=Path,
        default=REPO_ROOT / "results/roman_local/roman_maptruth_fft_batch200_m128_16w1t/events",
        help="parent directory when EVENT is an event ID",
    )
    parser.add_argument(
        "--atlas",
        type=Path,
        help="override the atlas path recorded in report.json",
    )
    parser.add_argument(
        "--raw-lightcurve",
        type=Path,
        help="GULLS .all.lc file used to recover the F146 magnitude zero point",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output PNG (default: assets/roman-maptruth-batch200/<event>-paper-v1.png)",
    )
    parser.add_argument(
        "--delta-chi2-top-cells",
        type=int,
        default=200,
        help="best cells used to set the map colour scale (default: 200)",
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _resolve_event(event: str, result_root: Path) -> Path:
    candidate = Path(event).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    return (result_root.expanduser().resolve() / event).resolve()


def _best_row(map_ids, chi2, scanned, requested: int) -> int:
    finite = np.asarray(scanned, dtype=bool) & np.isfinite(chi2)
    rows = np.flatnonzero(np.asarray(map_ids, dtype=np.int64) == requested)
    if not rows.size or not finite[rows[0]]:
        raise ValueError(f"best map {requested} is absent or not finite")
    return int(rows[0])


def _trajectory_phase(time: np.ndarray, geometry: np.ndarray, alpha: float) -> np.ndarray:
    tau = (np.asarray(time, dtype=np.float64) - geometry[0]) / geometry[2]
    x = geometry[1] * np.sin(alpha) - tau * np.cos(alpha)
    y = -geometry[1] * np.cos(alpha) - tau * np.sin(alpha)
    return np.arctan2(y, x)


def _fft_magnification(
    time: np.ndarray,
    geometry: np.ndarray,
    alpha: float,
    nodes: np.ndarray,
    coefficients: np.ndarray,
    *,
    m_max: int,
    radial_order: int,
) -> np.ndarray:
    from hammlet._core.config import PSPLGeometry
    from hammlet._core.radial import interpolate_coefficients
    from hammlet._core.trajectory import radial_coordinates

    radius, _ = radial_coordinates(
        time,
        PSPLGeometry(float(geometry[0]), float(geometry[1]), float(geometry[2])),
    )
    phase = _trajectory_phase(time, geometry, alpha)
    local = interpolate_coefficients(
        coefficients,
        radius,
        nodes,
        order=radial_order,
    )
    modes = np.arange(m_max + 1)
    return 1.0 + np.real(
        local[:, 0]
        + 2.0
        * np.sum(
            local[:, 1:]
            * np.exp(1j * phase[:, None] * modes[None, 1:]),
            axis=1,
        )
    )


def _load_dataset(report: dict):
    from tools.roman_local.roman_event import load_data_file, load_gulls_data_file

    event = report["event"]
    window = tuple(float(value) for value in event["window"])
    names = tuple(event.get("datasets", {}).keys())
    paths = tuple(Path(value) for value in event["data_paths"])
    if len(names) != len(paths):
        names = tuple(f"data-{index}" for index in range(len(paths)))
    data_format = event.get("data_format", "time-flux-error")
    if data_format == "gulls-all-lc":
        loader = load_gulls_data_file
    else:
        loader = load_data_file
    datasets = tuple(
        loader(path, name=name, window=window, data_format=data_format)
        if data_format != "gulls-all-lc"
        else loader(path, name=name, window=window)
        for name, path in zip(names, paths, strict=True)
    )
    return datasets


def _map_points(parameters: np.ndarray, map_ids: np.ndarray, chi2: np.ndarray, scanned: np.ndarray):
    """Keep the best rho row at each (log s, log q) cell."""
    finite = np.asarray(scanned, dtype=bool) & np.isfinite(chi2)
    grouped: dict[tuple[float, float], int] = {}
    for row in np.flatnonzero(finite):
        key = (float(parameters[row, 0]), float(parameters[row, 1]))
        old = grouped.get(key)
        if old is None or chi2[row] < chi2[old]:
            grouped[key] = int(row)
    rows = np.asarray(list(grouped.values()), dtype=np.int64)
    return rows


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "mathtext.fontset": "stix",
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )


def _flux_to_magnitude(flux: np.ndarray) -> np.ndarray:
    return -2.5 * np.log10(np.clip(np.asarray(flux, dtype=np.float64), 1.0e-12, None))


def _find_gulls_raw_lightcurve(event_id: str) -> Path:
    """Find the raw GULLS product corresponding to an exported event."""
    roots = (
        Path("/rogue1_8/nunota/gulls/runs"),
        REPO_ROOT.parent / "gulls" / "runs",
    )
    candidates: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        candidates.extend(root.glob(f"**/gulls/raw/**/*_{event_id}.all.lc"))
    unique = sorted({path.resolve() for path in candidates})
    if not unique:
        raise FileNotFoundError(
            f"could not find the raw GULLS light curve for event {event_id}; "
            "pass --raw-lightcurve"
        )
    if len(unique) > 1:
        raise RuntimeError(
            f"found multiple raw GULLS light curves for event {event_id}: "
            + ", ".join(str(path) for path in unique)
            + "; pass --raw-lightcurve to disambiguate"
        )
    return unique[0]


def _gulls_f146_zero_point(raw_lightcurve: Path) -> tuple[float, float, float]:
    """Return ``(m_source, fs, m_baseline)`` from a GULLS header.

    The second entry of ``#Sourcemag`` is the Roman F146 source magnitude.
    The exported Hammlet flux is A_obs = F_obs/F_baseline, so the absolute
    magnitude of a plotted point is m_baseline - 2.5 log10(A_obs).
    """
    source_magnitudes: np.ndarray | None = None
    source_flux_fraction: float | None = None
    with raw_lightcurve.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("#fs:"):
                source_flux_fraction = float(line.split(":", 1)[1].strip().split()[0])
            elif line.startswith("#Sourcemag:"):
                source_magnitudes = np.fromstring(line.split(":", 1)[1], sep=" ")
            elif not line.startswith("#"):
                break
    if source_magnitudes is None or source_magnitudes.size < 2:
        raise ValueError(f"raw GULLS header has no usable #Sourcemag: {raw_lightcurve}")
    if source_flux_fraction is None or not np.isfinite(source_flux_fraction):
        raise ValueError(f"raw GULLS header has no usable #fs: {raw_lightcurve}")
    source_magnitude = float(source_magnitudes[1])
    baseline_magnitude = source_magnitude + 2.5 * np.log10(source_flux_fraction)
    return source_magnitude, float(source_flux_fraction), baseline_magnitude


def _plot_geometry_inset(axis, evaluator, geometry: np.ndarray, alpha: float) -> None:
    from hammlet._core.map_adapter import trajectory_xy

    all_components = evaluator.caustic_components
    # The distant planetary caustics can be much farther from the origin than
    # the central caustic.  Showing all components in one square inset makes
    # the central structure unreadably small, so select the component whose
    # centroid is closest to the map-frame origin.
    central_index = int(
        np.argmin(
            [np.linalg.norm(np.mean(component, axis=0)) for component in all_components]
        )
    )
    components = (all_components[central_index],)
    caustic_points = components[0]
    source_origin = np.asarray(
        [geometry[1] * np.sin(alpha), -geometry[1] * np.cos(alpha)],
        dtype=np.float64,
    )
    direction = np.asarray([-np.cos(alpha), -np.sin(alpha)], dtype=np.float64)
    caustic_tau = (caustic_points - source_origin) @ direction
    tau_span = max(float(np.ptp(caustic_tau)), 0.02)
    tau_pad = max(0.70 * tau_span, 0.02)
    tau = np.linspace(
        float(np.min(caustic_tau)) - tau_pad,
        float(np.max(caustic_tau)) + tau_pad,
        4000,
    )
    time = geometry[0] + tau * geometry[2]
    path_x, path_y = trajectory_xy(time, *geometry, alpha)

    for index, component in enumerate(components):
        axis.plot(
            component[:, 0],
            component[:, 1],
            color="#202020",
            linewidth=1.05,
            zorder=3,
            label="caustic" if index == 0 else None,
        )
    axis.plot(
        path_x,
        path_y,
        color="#d95f02",
        linewidth=1.25,
        zorder=4,
        label="source trajectory",
    )
    arrow_index = int(0.64 * (len(path_x) - 1))
    axis.annotate(
        "",
        xy=(path_x[arrow_index + 8], path_y[arrow_index + 8]),
        xytext=(path_x[arrow_index - 8], path_y[arrow_index - 8]),
        arrowprops={"arrowstyle": "-|>", "color": "#d95f02", "lw": 0.9},
        zorder=6,
    )
    geometry_points = np.concatenate(
        [caustic_points, np.column_stack((path_x, path_y))], axis=0
    )
    center = 0.5 * (
        np.min(geometry_points, axis=0) + np.max(geometry_points, axis=0)
    )
    half_range = max(
        0.5 * float(np.ptp(geometry_points[:, 0])),
        0.5 * float(np.ptp(geometry_points[:, 1])),
        0.05,
    )
    pad = max(0.018, 0.12 * half_range)
    axis.set_xlim(center[0] - half_range - pad, center[0] + half_range + pad)
    axis.set_ylim(center[1] - half_range - pad, center[1] + half_range + pad)
    axis.set_aspect("equal", adjustable="box")
    axis.tick_params(labelsize=7, length=2)
    axis.grid(False)


def _plot_planet_signal_inset(
    axis,
    data,
    model_time: np.ndarray,
    model_magnitude: np.ndarray,
    t0: float,
    baseline_magnitude: float,
    window: tuple[float, float],
) -> None:
    """Plot the short anomaly window without adding another legend."""
    shifted_time = np.asarray(data.time, dtype=np.float64) - float(t0)
    data_mask = (
        np.isfinite(shifted_time)
        & np.isfinite(data.flux)
        & np.isfinite(data.error)
        & (data.flux > 0.0)
        & (shifted_time >= window[0])
        & (shifted_time <= window[1])
    )
    data_magnitude = baseline_magnitude + _flux_to_magnitude(data.flux[data_mask])
    model_shifted_time = np.asarray(model_time, dtype=np.float64) - float(t0)
    model_mask = (
        (model_shifted_time >= window[0]) & (model_shifted_time <= window[1])
    )
    axis.errorbar(
        shifted_time[data_mask],
        data_magnitude,
        yerr=(2.5 / np.log(10.0)) * data.error[data_mask] / data.flux[data_mask],
        fmt="o",
        ms=2.1,
        alpha=0.78,
        color="#214f86",
        markeredgewidth=0.0,
        elinewidth=0.35,
        capsize=0,
        rasterized=True,
        zorder=6,
    )
    axis.plot(
        model_shifted_time[model_mask],
        model_magnitude[model_mask],
        color="#d95f02",
        linewidth=1.2,
        zorder=7,
    )
    finite_values = np.concatenate(
        [data_magnitude[np.isfinite(data_magnitude)], model_magnitude[model_mask]]
    )
    if finite_values.size:
        axis.set_ylim(
            float(np.min(finite_values) - 0.08),
            float(np.max(finite_values) + 0.08),
        )
    axis.set_xlim(*window)
    axis.invert_yaxis()
    axis.grid(False)
    axis.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    axis.set_xlabel("")
    axis.set_ylabel("")


def render(
    event_dir: Path,
    output: Path,
    *,
    atlas_override: Path | None,
    raw_lightcurve: Path | None,
    top_cells: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from hammlet._core.config import PSPLGeometry
    from hammlet._core.reference import profile_flux
    from hammlet._core.direct_vbm import VBMBinaryLensEvaluator
    from tools.roman_local.atlas import open_readonly_atlas

    report = _load_json(event_dir / "report.json")
    with np.load(event_dir / "minima.npz") as arrays:
        map_ids = np.asarray(arrays["map_ids"], dtype=np.int64)
        parameters = np.asarray(arrays["parameters"], dtype=np.float64)
        chi2 = np.asarray(arrays["chi2"], dtype=np.float64)
        scanned = np.asarray(arrays["scanned"], dtype=bool)
        geometries = np.asarray(arrays["geometries"], dtype=np.float64)
        geometry_index = np.asarray(arrays["geometry_index"], dtype=np.int64)
        alpha_index = np.asarray(arrays["alpha_index"], dtype=np.int64)

    best_meta = report["best_fft"]
    best_row = _best_row(map_ids, chi2, scanned, int(best_meta["map_id"]))
    geometry = geometries[int(geometry_index[best_row])]
    n_alpha = int(report["search_settings"]["n_alpha"])
    alpha = 2.0 * np.pi * int(alpha_index[best_row]) / n_alpha
    logs, logq, logrho = parameters[best_row]
    separation, mass_ratio, source_radius = 10.0 ** np.asarray(
        [logs, logq, logrho], dtype=np.float64
    )

    atlas_path = atlas_override or Path(report["atlas"]["path"])
    atlas = open_readonly_atlas(atlas_path)
    tail = report.get("atlas", {}).get("metadata", {}).get("point_lens_tail", {})
    if isinstance(tail, dict) and tail.get("enabled"):
        atlas = atlas.with_point_lens_tail(
            float(tail["max_radius"]),
            radial_order=int(report["search_settings"]["radial_order"]),
        )
    nodes, map_parameters, coefficients = atlas.coefficient_row(
        int(map_ids[best_row]),
        m_max=int(report["search_settings"]["m_max"]),
    )

    datasets = _load_dataset(report)
    event_id = str(report["event"]["name"])
    raw_path = raw_lightcurve or _find_gulls_raw_lightcurve(event_id)
    source_magnitude, source_flux_fraction, baseline_magnitude = _gulls_f146_zero_point(
        raw_path
    )
    all_time = np.concatenate([dataset.time for dataset in datasets])
    all_flux = np.concatenate([dataset.flux for dataset in datasets])
    all_error = np.concatenate([dataset.error for dataset in datasets])
    valid = np.isfinite(all_time) & np.isfinite(all_flux) & np.isfinite(all_error)
    display_half_width = 5.0
    event_t0 = float(geometry[0])
    plot_lower = max(event_t0 - display_half_width, float(np.min(all_time[valid])))
    plot_upper = min(event_t0 + display_half_width, float(np.max(all_time[valid])))
    model_time = np.linspace(plot_lower, plot_upper, 6000)
    m_max = int(report["search_settings"]["m_max"])
    radial_order = int(report["search_settings"]["radial_order"])
    observed_mag = _fft_magnification(
        all_time,
        geometry,
        alpha,
        nodes,
        coefficients,
        m_max=m_max,
        radial_order=radial_order,
    )
    model_mag = _fft_magnification(
        model_time,
        geometry,
        alpha,
        nodes,
        coefficients,
        m_max=m_max,
        radial_order=radial_order,
    )
    flux_profile = profile_flux(observed_mag, datasets[0])
    model_flux = flux_profile.source_flux * model_mag + flux_profile.blend_flux

    _style()
    figure = plt.figure(figsize=(16.0, 7.4))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 1.0),
        left=0.06,
        right=0.90,
        bottom=0.10,
        top=0.91,
        wspace=0.27,
    )
    light_axis = figure.add_subplot(grid[0, 0])
    map_axis = figure.add_subplot(grid[0, 1])

    data = datasets[0]
    model_magnitude = baseline_magnitude + _flux_to_magnitude(model_flux)
    plot_data = (
        np.isfinite(data.time)
        & np.isfinite(data.flux)
        & np.isfinite(data.error)
        & (data.time >= plot_lower)
        & (data.time <= plot_upper)
        & (data.flux > 0.0)
    )
    plot_time = data.time[plot_data]
    plot_magnitude = baseline_magnitude + _flux_to_magnitude(data.flux[plot_data])
    plot_magnitude_error = (2.5 / np.log(10.0)) * data.error[plot_data] / data.flux[plot_data]
    light_axis.errorbar(
        plot_time - float(geometry[0]),
        plot_magnitude,
        yerr=plot_magnitude_error,
        fmt="o",
        ms=2.7,
        alpha=0.78,
        color="#214f86",
        markeredgewidth=0.0,
        elinewidth=0.45,
        capsize=0,
        rasterized=True,
        zorder=6,
        label="F146 data",
    )
    light_axis.plot(
        model_time - float(geometry[0]),
        model_magnitude,
        color="#d95f02",
        linewidth=1.8,
        zorder=7,
        label="FFT best",
    )
    main_x_pad = 0.12
    main_x_limits = (
        plot_lower - float(geometry[0]) - main_x_pad,
        plot_upper - float(geometry[0]) + main_x_pad,
    )
    light_axis.set(
        xlim=main_x_limits,
        xlabel=r"$t-t_0$ [d]",
        ylabel="F146 Magnitude",
    )
    main_label_size = 14
    main_tick_size = 12.5
    light_axis.xaxis.label.set_size(main_label_size)
    light_axis.yaxis.label.set_size(main_label_size)
    light_axis.tick_params(axis="both", labelsize=main_tick_size)
    finite_curve = np.concatenate(
        [plot_magnitude[np.isfinite(plot_magnitude)], model_magnitude[np.isfinite(model_magnitude)]]
    )
    if finite_curve.size:
        finite_data_magnitude = plot_magnitude[np.isfinite(plot_magnitude)]
        data_tail = (
            float(np.percentile(finite_data_magnitude, 99.5))
            if finite_data_magnitude.size
            else float(np.max(finite_curve))
        )
        model_tail = float(np.max(model_magnitude[np.isfinite(model_magnitude)]))
        y_bottom = 0.5 * np.ceil(2.0 * max(data_tail + 0.15, model_tail + 0.10))
        light_axis.set_ylim(
            float(np.min(finite_curve) - 0.45),
            y_bottom,
        )
    light_axis.invert_yaxis()
    light_axis.grid(False)

    model_shifted_time = model_time - float(geometry[0])
    zoom_candidates = np.isfinite(model_magnitude) & (np.abs(model_shifted_time) <= 1.5)
    if np.any(zoom_candidates):
        zoom_center = float(
            model_shifted_time[zoom_candidates][
                np.argmin(model_magnitude[zoom_candidates])
            ]
        )
    else:
        zoom_center = 0.0
    zoom_left_width = 0.85
    zoom_right_width = 0.65
    planet_window = (
        max(float(model_shifted_time[0]), zoom_center - zoom_left_width),
        min(float(model_shifted_time[-1]), zoom_center + zoom_right_width),
    )
    planet_axis = light_axis.inset_axes([0.035, 0.66, 0.34, 0.29], zorder=10)
    _plot_planet_signal_inset(
        planet_axis,
        data,
        model_time,
        model_magnitude,
        float(geometry[0]),
        baseline_magnitude,
        planet_window,
    )

    evaluator = VBMBinaryLensEvaluator(
        separation,
        mass_ratio,
        source_radius,
        coordinate_frame="map",
    )
    geometry_axis = light_axis.inset_axes([0.72, 0.69, 0.25, 0.25], zorder=10)
    _plot_geometry_inset(geometry_axis, evaluator, geometry, alpha)
    geometry_axis.set_xlabel("")
    geometry_axis.set_ylabel("")
    geometry_axis.patch.set_facecolor("white")
    geometry_axis.patch.set_alpha(0.96)

    rows = _map_points(parameters, map_ids, chi2, scanned)
    best_chi2 = float(np.min(chi2[scanned & np.isfinite(chi2)]))
    delta = np.maximum(chi2[rows] - best_chi2, 0.0)
    vmax = float(np.partition(delta, min(top_cells, len(delta)) - 1)[min(top_cells, len(delta)) - 1])
    vmax = max(vmax, np.finfo(float).eps)
    colour = np.clip(delta, 0.0, vmax)
    # Draw the high-chi2 cells first and the good cells last, so overlapping
    # square markers always leave the best chi2 visible on top.
    draw_order = np.argsort(delta)[::-1]
    map_rows = rows[draw_order]
    map_colour = colour[draw_order]
    image = map_axis.scatter(
        parameters[map_rows, 0],
        parameters[map_rows, 1],
        c=map_colour,
        s=64,
        marker="s",
        cmap="jet_r",
        norm=Normalize(vmin=0.0, vmax=vmax),
        linewidths=0.0,
        alpha=0.96,
        rasterized=True,
        zorder=2,
    )
    truth = report["event"]["reference"]
    truth_xy = (np.log10(float(truth["s"])), np.log10(float(truth["q"])))
    map_axis.scatter(
        *truth_xy,
        s=120,
        facecolors="none",
        edgecolors="white",
        linewidths=3.0,
        zorder=5,
    )
    map_axis.scatter(
        *truth_xy,
        s=120,
        facecolors="none",
        edgecolors="black",
        linewidths=1.1,
        zorder=6,
    )
    best_xy = parameters[best_row, :2]
    # These are the paper-display limits used by the accepted 9910003
    # figure.  Keep the renderer identical and change only the view window
    # when a different event needs a different crop.
    s_min = -0.90
    s_max = 0.90
    # Keep the q display window fixed across events, matching the broad
    # low-q range used for 9910003 (log10(q) ~= -5.8 ... 0).
    q_min = -5.80
    q_max = 0.0
    map_axis.set(
        xlabel=r"$\log_{10}(s)$",
        ylabel=r"$\log_{10}(q)$",
        xlim=(s_min, s_max),
        ylim=(q_min, q_max),
    )
    # The GridSpec cells, rather than the colourbar, define the two main
    # panel sizes.  Keep the map cell at its full height and width so the
    # left and right panels remain identical in physical dimensions.
    map_axis.set_aspect("auto")
    map_axis.xaxis.label.set_size(main_label_size)
    map_axis.yaxis.label.set_size(main_label_size)
    map_axis.tick_params(axis="both", labelsize=main_tick_size)
    map_axis.grid(alpha=0.14, linewidth=0.5)
    map_axis.legend(
        handles=[
            Line2D([], [], marker="o", color="none", markerfacecolor="none", markeredgecolor="black", markersize=11, label="truth"),
        ],
        loc="upper left",
        framealpha=0.92,
        handlelength=1.0,
        fontsize=12,
        borderpad=0.45,
    )
    # Keep the large inter-panel gap, but place the colour bar immediately
    # beside the map instead of making it another GridSpec column.  This
    # preserves identical main-panel sizes and gives the colour bar exactly
    # the map height.
    figure.canvas.draw()
    map_bbox = map_axis.get_position()
    color_axis = figure.add_axes(
        [map_bbox.x1 + 0.008, map_bbox.y0, 0.018, map_bbox.height]
    )
    colourbar = figure.colorbar(image, cax=color_axis, label=r"$\Delta\chi^2$")
    colourbar.ax.tick_params(labelsize=11.5)
    colourbar.set_label(r"$\Delta\chi^2$", size=14)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=260, facecolor="white")
    plt.close(figure)
    print(json.dumps({
        "output": str(output.resolve()),
        "event": event_id,
        "map_id": int(map_ids[best_row]),
        "parameters": [float(value) for value in map_parameters],
        "geometry": [float(value) for value in geometry],
        "alpha": float(alpha),
        "color_vmax": vmax,
        "raw_lightcurve": str(raw_path),
        "source_magnitude_f146": source_magnitude,
        "source_flux_fraction": source_flux_fraction,
        "baseline_magnitude_f146": baseline_magnitude,
    }, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.delta_chi2_top_cells < 1:
        raise SystemExit("--delta-chi2-top-cells must be positive")
    event_dir = _resolve_event(args.event, args.result_root)
    output = args.output or (
        REPO_ROOT / "assets" / "roman-maptruth-batch200" / f"{event_dir.name}-paper-v1.png"
    )
    render(
        event_dir,
        output.expanduser().resolve(),
        atlas_override=None if args.atlas is None else args.atlas.expanduser().resolve(),
        raw_lightcurve=(
            None if args.raw_lightcurve is None else args.raw_lightcurve.expanduser().resolve()
        ),
        top_cells=args.delta_chi2_top_cells,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
