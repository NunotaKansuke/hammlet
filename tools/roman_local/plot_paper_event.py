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
    parser.add_argument(
        "--q-log-max",
        type=float,
        default=0.0,
        help="upper displayed log10(q) limit (default: 0, q=1)",
    )
    parser.add_argument(
        "--time-window",
        nargs=2,
        type=float,
        metavar=("LEFT", "RIGHT"),
        help=(
            "display limits in t-t0 days; when omitted, use the default "
            "symmetric five-day window"
        ),
    )
    parser.add_argument(
        "--show-all-caustics",
        action="store_true",
        help="show every caustic component in the geometry inset",
    )
    parser.add_argument(
        "--fallback-result",
        type=Path,
        help="fallback event JSON whose final model is overlaid in green",
    )
    parser.add_argument(
        "--lm-result",
        type=Path,
        help="normal short/long-LM event JSON whose polished model is overlaid",
    )
    parser.add_argument(
        "--inset-time-window",
        nargs=2,
        type=float,
        metavar=("LEFT", "RIGHT"),
        help="time limits in t-t0 days for the light-curve inset",
    )
    parser.add_argument(
        "--no-inset",
        action="store_true",
        help="omit the light-curve and geometry inset panels from the main figure",
    )
    parser.add_argument(
        "--zoom-output",
        type=Path,
        help=(
            "also write a standalone zoom-plus-geometry figure; use with "
            "--no-inset for a main figure without inset panels"
        ),
    )
    parser.add_argument(
        "--compare-close-wide",
        action="store_true",
        help="overlay the best scanned close and wide branch models",
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


def _best_branch_row(parameters, chi2, scanned, *, close: bool) -> int:
    finite = np.asarray(scanned, dtype=bool) & np.isfinite(chi2)
    branch = np.asarray(parameters[:, 0], dtype=np.float64) < 0.0
    if not close:
        branch = ~branch
    rows = np.flatnonzero(finite & branch)
    if not rows.size:
        name = "close" if close else "wide"
        raise ValueError(f"no finite {name} branch rows")
    return int(rows[np.argmin(np.asarray(chi2)[rows])])


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


def _coordinate_frame(report: dict) -> str:
    metadata = report.get("atlas", {}).get("metadata", {})
    if not isinstance(metadata, dict):
        return "map"
    frame = metadata.get("coordinate_frame")
    if frame is None and isinstance(metadata.get("source_metadata"), dict):
        frame = metadata["source_metadata"].get("coordinate_frame")
    if frame is None:
        return "map"
    if frame not in ("map", "native"):
        raise ValueError(f"unsupported coordinate frame: {frame!r}")
    return str(frame)


def _direct_vbm_model(
    parameters: dict,
    dataset,
    times: np.ndarray,
    baseline_magnitude: float,
    *,
    coordinate_frame: str,
) -> tuple[np.ndarray, object]:
    """Evaluate one optimized direct-VBM model and its evaluator."""
    from hammlet._core.caustics import adamgrid_map_origin_shift
    from hammlet._core.direct_vbm import VBMBinaryLensEvaluator
    from hammlet._core.map_adapter import trajectory_xy
    from hammlet._core.reference import profile_flux

    s = float(parameters["s"])
    q = float(parameters["q"])
    rho = float(parameters["rho"])
    t0 = float(parameters["t0"])
    u0 = float(parameters["u0"])
    tE = float(parameters["tE"])
    alpha = float(parameters["alpha"])
    evaluator = VBMBinaryLensEvaluator(
        s,
        q,
        rho,
        tolerance=1.0e-3,
        relative_tolerance=1.0e-4,
        coordinate_frame=coordinate_frame,
    )
    x_data, y_data = trajectory_xy(dataset.time, t0, u0, tE, alpha)
    x_curve, y_curve = trajectory_xy(times, t0, u0, tE, alpha)
    if coordinate_frame == "map":
        shift = adamgrid_map_origin_shift(s, q)
        x_data = x_data - shift
        x_curve = x_curve - shift
    magnification_data = evaluator.magnification(x_data, y_data)
    profile = profile_flux(magnification_data, dataset)
    magnification_curve = evaluator.magnification(x_curve, y_curve)
    model_flux = profile.source_flux * magnification_curve + profile.blend_flux
    model_magnitude = baseline_magnitude + _flux_to_magnitude(model_flux)
    return np.asarray(model_magnitude, dtype=np.float64), evaluator


def _load_lm_comparison(path: Path) -> tuple[dict, dict, str, dict]:
    """Load the raw FFT seed and final model from a normal-LM record."""
    payload = _load_json(path.expanduser().resolve())
    if isinstance(payload.get("best_short"), dict):
        seed_section = payload["best_short"]
        final_section = payload.get("long_lm")
    else:
        seed_section = payload.get("initial_seed")
        final_section = payload.get("optimized")
    if not isinstance(seed_section, dict) or not isinstance(final_section, dict):
        raise ValueError(f"normal-LM result has no seed/final pair: {path}")
    seed_parameters = seed_section.get("parameters")
    final_parameters = final_section.get("optimized_parameters")
    if final_parameters is None:
        final_parameters = final_section.get("parameters")
    if not isinstance(seed_parameters, dict) or not isinstance(final_parameters, dict):
        raise ValueError(f"normal-LM result has unusable parameters: {path}")
    required = {"t0", "u0", "tE", "s", "q", "rho", "alpha"}
    if not required.issubset(seed_parameters) or not required.issubset(final_parameters):
        raise ValueError(f"normal-LM result is missing model parameters: {path}")
    coordinate_frame = str(payload.get("coordinate_frame", "map"))
    if coordinate_frame not in ("map", "native"):
        raise ValueError(f"unsupported normal-LM coordinate frame: {coordinate_frame!r}")
    return seed_parameters, final_parameters, coordinate_frame, seed_section


def _find_lm_seed_row(
    map_ids: np.ndarray,
    geometry_indices: np.ndarray,
    alpha_indices: np.ndarray,
    chi2: np.ndarray,
    scanned: np.ndarray,
    seed_section: dict,
) -> int:
    """Find the minima row corresponding to the normal-LM FFT seed."""
    finite = np.asarray(scanned, dtype=bool) & np.isfinite(chi2)
    requested_map = seed_section.get("map_id")
    requested_geometry = seed_section.get("geometry_index")
    requested_alpha = seed_section.get("alpha_index")
    if requested_map is None:
        raise ValueError("normal-LM seed has no map_id")
    candidates = finite & (np.asarray(map_ids, dtype=np.int64) == int(requested_map))
    if requested_geometry is not None:
        narrowed = candidates & (
            np.asarray(geometry_indices, dtype=np.int64) == int(requested_geometry)
        )
        if np.any(narrowed):
            candidates = narrowed
    if requested_alpha is not None:
        narrowed = candidates & (
            np.asarray(alpha_indices, dtype=np.int64) == int(requested_alpha)
        )
        if np.any(narrowed):
            candidates = narrowed
    rows = np.flatnonzero(candidates)
    if not rows.size:
        raise ValueError(
            "normal-LM seed is absent from the supplied FFT minima: "
            f"map_id={requested_map}, geometry_index={requested_geometry}, "
            f"alpha_index={requested_alpha}"
        )
    return int(rows[np.argmin(np.asarray(chi2)[rows])])


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
            "font.size": 12.5,
            "axes.labelsize": 15,
            "axes.titlesize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 11,
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


def _plot_geometry_inset(
    axis,
    evaluator,
    geometry: np.ndarray,
    alpha: float,
    *,
    show_all_caustics: bool = False,
    overlay_models: tuple[dict, ...] = (),
    caustic_color: str = "#202020",
    path_color: str = "#d95f02",
    path_linestyle: str = "-",
) -> None:
    from hammlet._core.map_adapter import trajectory_xy

    model_specs = [
        {
            "evaluator": evaluator,
            "geometry": geometry,
            "alpha": alpha,
            "caustic_color": caustic_color,
            "path_color": path_color,
            "path_linestyle": path_linestyle,
        },
        *overlay_models,
    ]
    plotted_models = []
    for model in model_specs:
        all_components = model["evaluator"].caustic_components
        # The default paper figure shows the central caustic, which keeps the
        # inset legible for ordinary single-feature events.  Wide events can
        # request every component explicitly.
        if show_all_caustics:
            components = tuple(all_components)
        else:
            central_index = int(
                np.argmin(
                    [
                        np.linalg.norm(np.mean(component, axis=0))
                        for component in all_components
                    ]
                )
            )
            components = (all_components[central_index],)
        caustic_points = np.concatenate(components, axis=0)
        model_geometry = np.asarray(model["geometry"], dtype=np.float64)
        model_alpha = float(model["alpha"])
        source_origin = np.asarray(
            [
                model_geometry[1] * np.sin(model_alpha),
                -model_geometry[1] * np.cos(model_alpha),
            ],
            dtype=np.float64,
        )
        direction = np.asarray(
            [-np.cos(model_alpha), -np.sin(model_alpha)], dtype=np.float64
        )
        caustic_tau = (caustic_points - source_origin) @ direction
        tau_span = max(float(np.ptp(caustic_tau)), 0.02)
        tau_pad_fraction = 0.08 if show_all_caustics else 0.70
        tau_pad = max(tau_pad_fraction * tau_span, 0.02)
        tau = np.linspace(
            float(np.min(caustic_tau)) - tau_pad,
            float(np.max(caustic_tau)) + tau_pad,
            4000,
        )
        time = model_geometry[0] + tau * model_geometry[2]
        path_x, path_y = trajectory_xy(time, *model_geometry, model_alpha)
        plotted_models.append(
            {
                "components": components,
                "caustic_points": caustic_points,
                "path_x": path_x,
                "path_y": path_y,
                "caustic_color": model["caustic_color"],
                "path_color": model["path_color"],
                "path_linestyle": model["path_linestyle"],
            }
        )

    for model in plotted_models:
        for index, component in enumerate(model["components"]):
            axis.plot(
                component[:, 0],
                component[:, 1],
                color=model["caustic_color"],
                linewidth=1.05,
                zorder=3,
                label="caustic" if index == 0 else None,
            )
        axis.plot(
            model["path_x"],
            model["path_y"],
            color=model["path_color"],
            linewidth=1.25,
            linestyle=model["path_linestyle"],
            zorder=4,
            label="source trajectory",
        )
        arrow_index = int(0.64 * (len(model["path_x"]) - 1))
        axis.annotate(
            "",
            xy=(model["path_x"][arrow_index + 8], model["path_y"][arrow_index + 8]),
            xytext=(
                model["path_x"][arrow_index - 8],
                model["path_y"][arrow_index - 8],
            ),
            arrowprops={
                "arrowstyle": "-|>",
                "color": model["path_color"],
                "lw": 0.9,
            },
            zorder=6,
        )
    geometry_points = np.concatenate(
        [
            np.concatenate(
                [model["caustic_points"] for model in plotted_models], axis=0
            ),
            np.concatenate(
                [
                    np.column_stack((model["path_x"], model["path_y"]))
                    for model in plotted_models
                ],
                axis=0,
            ),
        ],
        axis=0,
    )
    center = 0.5 * (
        np.min(geometry_points, axis=0) + np.max(geometry_points, axis=0)
    )
    x_span = max(float(np.ptp(geometry_points[:, 0])) * 1.18, 0.05)
    y_span = max(float(np.ptp(geometry_points[:, 1])) * 1.18, 0.05)
    if show_all_caustics:
        # Match the data limits to the deliberately horizontal inset.  This
        # keeps equal data scaling while avoiding the very large empty
        # vertical range produced by the old square framing of wide systems.
        box = axis.get_position()
        box_ratio = max(float(box.width / box.height), 1.0)
        if x_span / y_span > box_ratio:
            y_span = x_span / box_ratio
        else:
            x_span = y_span * box_ratio
        axis.set_xlim(center[0] - 0.5 * x_span, center[0] + 0.5 * x_span)
        axis.set_ylim(center[1] - 0.5 * y_span, center[1] + 0.5 * y_span)
    else:
        half_range = max(0.5 * x_span, 0.5 * y_span, 0.05)
        pad = max(0.018, 0.12 * half_range)
        axis.set_xlim(center[0] - half_range - pad, center[0] + half_range + pad)
        axis.set_ylim(center[1] - half_range - pad, center[1] + half_range + pad)
    axis.set_aspect("equal", adjustable="box")
    axis.tick_params(labelsize=10, length=2.5)
    axis.grid(False)


def _plot_planet_signal_inset(
    axis,
    data,
    model_time: np.ndarray,
    model_magnitude: np.ndarray,
    t0: float,
    baseline_magnitude: float,
    window: tuple[float, float],
    *,
    model_color: str = "#d95f02",
    draw_data: bool = True,
    update_limits: bool = True,
    invert_axis: bool = True,
    show_ticks: bool = False,
    model_linestyle: str = "-",
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
    if draw_data:
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
        color=model_color,
        linewidth=1.2,
        linestyle=model_linestyle,
        zorder=7,
    )
    finite_values = [model_magnitude[model_mask]]
    if draw_data:
        finite_values.insert(0, data_magnitude[np.isfinite(data_magnitude)])
    finite_values = np.concatenate(finite_values)
    if update_limits and finite_values.size:
        axis.set_ylim(
            float(np.min(finite_values) - 0.08),
            float(np.max(finite_values) + 0.08),
        )
    axis.set_xlim(*window)
    if invert_axis:
        axis.invert_yaxis()
    axis.grid(False)
    if show_ticks:
        axis.tick_params(
            axis="both",
            which="both",
            bottom=True,
            top=False,
            left=True,
            right=False,
            labelbottom=True,
            labelleft=True,
            labelsize=10,
            length=3.0,
            width=0.8,
        )
    else:
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
    fallback_result: Path | None,
    lm_result: Path | None,
    top_cells: int,
    q_log_max: float,
    compare_close_wide: bool,
    time_window: tuple[float, float] | None,
    show_all_caustics: bool,
    inset_time_window: tuple[float, float] | None,
    no_inset: bool,
    zoom_output: Path | None,
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

    if fallback_result is not None and lm_result is not None:
        raise ValueError("--fallback-result and --lm-result are mutually exclusive")
    lm_seed_parameters = None
    lm_final_parameters = None
    lm_coordinate_frame = "map"
    lm_seed_section = None
    if lm_result is not None:
        (
            lm_seed_parameters,
            lm_final_parameters,
            lm_coordinate_frame,
            lm_seed_section,
        ) = _load_lm_comparison(lm_result)

    best_meta = report["best_fft"]
    best_row = _best_row(map_ids, chi2, scanned, int(best_meta["map_id"]))
    if lm_seed_section is not None:
        best_row = _find_lm_seed_row(
            map_ids,
            geometry_index,
            alpha_index,
            chi2,
            scanned,
            lm_seed_section,
        )
        branch_rows = [("FFT seed", best_row, "#7f7f7f")]
    else:
        branch_rows = [("FFT best", best_row, "#d95f02")]
    if compare_close_wide:
        close_row = _best_branch_row(parameters, chi2, scanned, close=True)
        wide_row = _best_branch_row(parameters, chi2, scanned, close=False)
        branch_rows = [
            ("close", close_row, "#d95f02"),
            ("wide", wide_row, "#32a852"),
        ]
        # Use the global best branch as the reference x-coordinate.  For the
        # 9920090 comparison both branch candidates share this geometry.
        best_row = close_row if chi2[close_row] <= chi2[wide_row] else wide_row
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
    branch_models = []
    for label, row, colour in branch_rows:
        branch_geometry = geometries[int(geometry_index[row])]
        branch_alpha = 2.0 * np.pi * int(alpha_index[row]) / n_alpha
        branch_logs, branch_logq, branch_logrho = parameters[row]
        branch_s, branch_q, branch_rho = 10.0 ** np.asarray(
            [branch_logs, branch_logq, branch_logrho], dtype=np.float64
        )
        nodes, map_parameters, coefficients = atlas.coefficient_row(
            int(map_ids[row]),
            m_max=int(report["search_settings"]["m_max"]),
        )
        branch_models.append(
            {
                "label": label,
                "row": int(row),
                "colour": colour,
                "geometry": branch_geometry,
                "alpha": branch_alpha,
                "s": float(branch_s),
                "q": float(branch_q),
                "rho": float(branch_rho),
                "map_parameters": map_parameters,
                "nodes": nodes,
                "coefficients": coefficients,
                "line_style": "--" if lm_seed_section is not None else "-",
            }
        )
    primary_model = branch_models[0]
    map_parameters = primary_model["map_parameters"]

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
    event_t0 = float(geometry[0])
    if time_window is None:
        display_half_width = 5.0
        requested_lower = event_t0 - display_half_width
        requested_upper = event_t0 + display_half_width
    else:
        requested_lower = event_t0 + float(time_window[0])
        requested_upper = event_t0 + float(time_window[1])
    plot_lower = max(requested_lower, float(np.min(all_time[valid])))
    plot_upper = min(requested_upper, float(np.max(all_time[valid])))
    if plot_upper <= plot_lower:
        raise ValueError("requested time window does not overlap the data")
    model_time = np.linspace(plot_lower, plot_upper, 6000)
    m_max = int(report["search_settings"]["m_max"])
    radial_order = int(report["search_settings"]["radial_order"])
    for model in branch_models:
        observed_mag = _fft_magnification(
            all_time,
            model["geometry"],
            model["alpha"],
            model["nodes"],
            model["coefficients"],
            m_max=m_max,
            radial_order=radial_order,
        )
        model_mag = _fft_magnification(
            model_time,
            model["geometry"],
            model["alpha"],
            model["nodes"],
            model["coefficients"],
            m_max=m_max,
            radial_order=radial_order,
        )
        flux_profile = profile_flux(observed_mag, datasets[0])
        model_flux = flux_profile.source_flux * model_mag + flux_profile.blend_flux
        model["model_magnitude"] = baseline_magnitude + _flux_to_magnitude(model_flux)

    fallback_model = None
    if fallback_result is not None:
        fallback_payload = _load_json(fallback_result.expanduser().resolve())
        if str(fallback_payload.get("event_id")) != event_id:
            raise ValueError(
                f"fallback result event {fallback_payload.get('event_id')!r} "
                f"does not match {event_id}"
            )
        best_fallback = fallback_payload.get("best")
        if not isinstance(best_fallback, dict):
            raise ValueError("fallback result has no best candidate")
        fallback_parameters = best_fallback.get("optimized_parameters")
        if not isinstance(fallback_parameters, dict):
            raise ValueError("fallback result has no optimized_parameters")
        fallback_magnitude, fallback_evaluator = _direct_vbm_model(
            fallback_parameters,
            datasets[0],
            model_time,
            baseline_magnitude,
            coordinate_frame=_coordinate_frame(report),
        )
        fallback_model = {
            "label": "Fallback final",
            "colour": "#2ca02c",
            "parameters": fallback_parameters,
            "model_magnitude": fallback_magnitude,
            "evaluator": fallback_evaluator,
            "geometry": np.asarray(
                [
                    float(fallback_parameters["t0"]),
                    float(fallback_parameters["u0"]),
                    float(fallback_parameters["tE"]),
                ],
                dtype=np.float64,
            ),
            "alpha": float(fallback_parameters["alpha"]),
        }

    polish_model = None
    if lm_final_parameters is not None:
        polished_magnitude, polished_evaluator = _direct_vbm_model(
            lm_final_parameters,
            datasets[0],
            model_time,
            baseline_magnitude,
            coordinate_frame=lm_coordinate_frame,
        )
        polish_model = {
            "label": "LM polished",
            "colour": "#d95f02",
            "line_style": "-",
            "parameters": lm_final_parameters,
            "model_magnitude": polished_magnitude,
            "evaluator": polished_evaluator,
            "geometry": np.asarray(
                [
                    float(lm_final_parameters["t0"]),
                    float(lm_final_parameters["u0"]),
                    float(lm_final_parameters["tE"]),
                ],
                dtype=np.float64,
            ),
            "alpha": float(lm_final_parameters["alpha"]),
        }
    comparison_model = fallback_model if fallback_model is not None else polish_model

    _style()
    figure = plt.figure(figsize=(16.0, 7.4))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 1.0),
        left=0.06,
        right=0.90,
        bottom=0.13,
        top=0.91,
        wspace=0.16,
    )
    light_axis = figure.add_subplot(grid[0, 0])
    map_axis = figure.add_subplot(grid[0, 1])

    data = datasets[0]
    model_magnitude = primary_model["model_magnitude"]
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
    for model in branch_models:
        light_axis.plot(
            model_time - float(geometry[0]),
            model["model_magnitude"],
            color=model["colour"],
            linewidth=1.8,
            linestyle=model["line_style"],
            zorder=8,
            label=model["label"] if compare_close_wide else "FFT best",
        )
    if comparison_model is not None:
        light_axis.plot(
            model_time - float(geometry[0]),
            comparison_model["model_magnitude"],
            color=comparison_model["colour"],
            linewidth=1.8,
            zorder=9,
            linestyle=comparison_model["line_style"],
        )
    if no_inset:
        main_legend_models = [*branch_models]
        if comparison_model is not None:
            main_legend_models.append(comparison_model)
        light_axis.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    color=model["colour"],
                    linewidth=1.8,
                    linestyle=model["line_style"],
                    label=model["label"],
                )
                for model in main_legend_models
            ],
            loc="upper left",
            frameon=False,
            fontsize=14,
            handlelength=1.6,
            handletextpad=0.45,
            columnspacing=1.0,
        )
    if compare_close_wide:
        legend_handles = [
            Line2D([], [], color=colour, linewidth=2.0, label=label)
            for label, _, colour in branch_rows
        ]
        legend = light_axis.legend(
            handles=legend_handles,
            frameon=False,
            fontsize=15,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.025),
            ncol=2,
            handlelength=1.5,
            handletextpad=0.45,
            columnspacing=1.0,
        )
        for text in legend.get_texts():
            text.set_color("black")
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
    main_label_size = 19
    main_tick_size = 16
    light_axis.xaxis.label.set_size(main_label_size)
    light_axis.yaxis.label.set_size(main_label_size)
    light_axis.tick_params(axis="both", labelsize=main_tick_size)
    finite_models = [
        model["model_magnitude"][np.isfinite(model["model_magnitude"])]
        for model in branch_models
    ]
    if comparison_model is not None:
        finite_models.append(
            comparison_model["model_magnitude"][
                np.isfinite(comparison_model["model_magnitude"])
            ]
        )
    finite_curve = np.concatenate(
        [plot_magnitude[np.isfinite(plot_magnitude)], *finite_models]
    )
    if finite_curve.size:
        finite_data_magnitude = plot_magnitude[np.isfinite(plot_magnitude)]
        data_tail = (
            float(np.percentile(finite_data_magnitude, 99.5))
            if finite_data_magnitude.size
            else float(np.max(finite_curve))
        )
        model_tail = max(float(np.max(values)) for values in finite_models)
        y_bottom = 0.5 * np.ceil(2.0 * max(data_tail + 0.15, model_tail + 0.10))
        light_axis.set_ylim(
            float(np.min(finite_curve) - 0.45),
            y_bottom,
        )
    light_axis.invert_yaxis()
    light_axis.grid(False)

    model_shifted_time = model_time - float(geometry[0])
    if inset_time_window is not None:
        inset_left, inset_right = (float(value) for value in inset_time_window)
        if inset_right <= inset_left:
            raise ValueError("inset time window must have RIGHT > LEFT")
        planet_window = (
            max(float(model_shifted_time[0]), inset_left),
            min(float(model_shifted_time[-1]), inset_right),
        )
        if planet_window[1] <= planet_window[0]:
            raise ValueError("inset time window does not overlap the model")
    else:
        # Prefer the strongest feature near t0, but fall back to the
        # brightest modeled point in the observed window when t0 itself lies
        # outside the available data (common for edge-truncated events).
        zoom_candidates = np.isfinite(model_magnitude)
        near_t0 = zoom_candidates & (np.abs(model_shifted_time) <= 1.5)
        if np.any(near_t0):
            zoom_candidates = near_t0
        if not np.any(zoom_candidates):
            raise ValueError("model has no finite points for the zoom window")
        zoom_center = float(
            model_shifted_time[zoom_candidates][
                np.argmin(model_magnitude[zoom_candidates])
            ]
        )
        zoom_left_width = 0.85
        zoom_right_width = 0.65
        planet_window = (
            max(float(model_shifted_time[0]), zoom_center - zoom_left_width),
            min(float(model_shifted_time[-1]), zoom_center + zoom_right_width),
        )
        if planet_window[1] <= planet_window[0]:
            planet_window = (
                float(model_shifted_time[0]),
                float(model_shifted_time[-1]),
            )
    if comparison_model is not None:
        planet_position = [0.33, 0.65, 0.42, 0.24]
        geometry_positions = [
            [0.12, 0.47, 0.84, 0.09],
            [0.12, 0.34, 0.84, 0.09],
        ]
    elif show_all_caustics:
        planet_position = [0.33, 0.65, 0.42, 0.24]
        geometry_positions = [[0.12, 0.38, 0.84, 0.12]]
    else:
        planet_position = [0.035, 0.66, 0.34, 0.29]
        geometry_positions = [[0.72, 0.69, 0.25, 0.25]]
    if not no_inset:
        planet_axis = light_axis.inset_axes(planet_position, zorder=10)
        _plot_planet_signal_inset(
            planet_axis,
            data,
            model_time,
            model_magnitude,
            float(geometry[0]),
            baseline_magnitude,
            planet_window,
            model_color=primary_model["colour"],
            show_ticks=show_all_caustics,
            model_linestyle=primary_model["line_style"],
        )
        for model in branch_models[1:]:
            _plot_planet_signal_inset(
                planet_axis,
                data,
                model_time,
                model["model_magnitude"],
                float(geometry[0]),
                baseline_magnitude,
                planet_window,
                model_color=model["colour"],
                draw_data=False,
                update_limits=False,
                invert_axis=False,
                show_ticks=show_all_caustics,
                model_linestyle=model["line_style"],
            )
        if comparison_model is not None:
            _plot_planet_signal_inset(
                planet_axis,
                data,
                model_time,
                comparison_model["model_magnitude"],
                float(geometry[0]),
                baseline_magnitude,
                planet_window,
                model_color=comparison_model["colour"],
                draw_data=False,
                update_limits=False,
                invert_axis=False,
                show_ticks=show_all_caustics,
                model_linestyle=comparison_model["line_style"],
            )
            planet_axis.legend(
                handles=[
                    Line2D(
                        [],
                        [],
                        color=primary_model["colour"],
                        linewidth=1.8,
                        linestyle=primary_model["line_style"],
                        label=primary_model["label"],
                    ),
                    Line2D(
                        [],
                        [],
                        color=comparison_model["colour"],
                        linewidth=1.8,
                        linestyle=comparison_model["line_style"],
                        label=comparison_model["label"],
                    ),
                ],
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=2,
                frameon=False,
                fontsize=11,
                handlelength=1.5,
                handletextpad=0.4,
                columnspacing=1.0,
                borderaxespad=0.0,
            )

    evaluator = VBMBinaryLensEvaluator(
        separation,
        mass_ratio,
        source_radius,
        coordinate_frame="map",
    )
    geometry_models = [
        (
            primary_model["label"],
            evaluator,
            geometry,
            alpha,
            primary_model["colour"],
            primary_model["line_style"],
        )
    ]
    if comparison_model is not None:
        geometry_models.append(
            (
                comparison_model["label"],
                comparison_model["evaluator"],
                comparison_model["geometry"],
                comparison_model["alpha"],
                comparison_model["colour"],
                comparison_model["line_style"],
            )
        )
    if not no_inset:
        geometry_axes = []
        for position, (
            label,
            model_evaluator,
            model_geometry,
            model_alpha,
            path_color,
            path_linestyle,
        ) in zip(geometry_positions, geometry_models, strict=True):
            geometry_axis = light_axis.inset_axes(position, zorder=10)
            _plot_geometry_inset(
                geometry_axis,
                model_evaluator,
                model_geometry,
                model_alpha,
                show_all_caustics=show_all_caustics,
                caustic_color="#202020",
                path_color=path_color,
                path_linestyle=path_linestyle,
            )
            geometry_axis.set_xlabel("")
            geometry_axis.set_ylabel("")
            geometry_axis.patch.set_facecolor("white")
            geometry_axis.patch.set_alpha(0.96)
            geometry_axes.append(geometry_axis)
        common_xlim = (
            min(float(axis.get_xlim()[0]) for axis in geometry_axes),
            max(float(axis.get_xlim()[1]) for axis in geometry_axes),
        )
        common_ylim = (
            min(float(axis.get_ylim()[0]) for axis in geometry_axes),
            max(float(axis.get_ylim()[1]) for axis in geometry_axes),
        )
        for geometry_axis in geometry_axes:
            geometry_axis.set_xlim(*common_xlim)
            geometry_axis.set_ylim(*common_ylim)
            geometry_axis.set_aspect("equal", adjustable="box")

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
    q_max = float(q_log_max)
    if q_max <= q_min:
        raise ValueError("q-log-max must be greater than the fixed lower q limit")
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
        fontsize=16,
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
    colourbar.ax.tick_params(labelsize=15)
    colourbar.set_label(r"$\Delta\chi^2$", size=18)

    if zoom_output is not None:
        zoom_figure = plt.figure(figsize=(13.0, 5.8))
        zoom_grid = zoom_figure.add_gridspec(
            1,
            2,
            width_ratios=(1.25, 1.0),
            left=0.08,
            right=0.98,
            bottom=0.16,
            top=0.84,
            wspace=0.20,
        )
        zoom_axis = zoom_figure.add_subplot(zoom_grid[0, 0])
        _plot_planet_signal_inset(
            zoom_axis,
            data,
            model_time,
            model_magnitude,
            float(geometry[0]),
            baseline_magnitude,
            planet_window,
            model_color=primary_model["colour"],
            show_ticks=True,
            model_linestyle=primary_model["line_style"],
        )
        zoom_models = [*branch_models[1:]]
        if comparison_model is not None:
            zoom_models.append(comparison_model)
        for model in zoom_models:
            _plot_planet_signal_inset(
                zoom_axis,
                data,
                model_time,
                model["model_magnitude"],
                float(geometry[0]),
                baseline_magnitude,
                planet_window,
                model_color=model["colour"],
                draw_data=False,
                update_limits=False,
                invert_axis=False,
                show_ticks=True,
                model_linestyle=model["line_style"],
            )
        zoom_axis.set_xlabel(r"$t-t_0$ [d]", fontsize=17)
        zoom_axis.set_ylabel("F146 Magnitude", fontsize=17)
        zoom_axis.tick_params(axis="both", labelsize=13)
        legend_models = [primary_model, *zoom_models]
        zoom_axis.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    color=model["colour"],
                    linewidth=1.8,
                    linestyle=model["line_style"],
                    label=model["label"],
                )
                for model in legend_models
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=min(3, len(legend_models)),
            frameon=False,
            fontsize=11,
            handlelength=1.5,
            handletextpad=0.4,
            columnspacing=1.0,
            borderaxespad=0.0,
        )
        geometry_axis = zoom_figure.add_subplot(zoom_grid[0, 1])
        for index, (
            label,
            model_evaluator,
            model_geometry,
            model_alpha,
            path_color,
            path_linestyle,
        ) in enumerate(geometry_models):
            _plot_geometry_inset(
                geometry_axis,
                model_evaluator,
                model_geometry,
                model_alpha,
                show_all_caustics=show_all_caustics,
                caustic_color="#202020",
                path_color=path_color,
                path_linestyle=path_linestyle,
            )
            if index == 0:
                geometry_xlim = geometry_axis.get_xlim()
                geometry_ylim = geometry_axis.get_ylim()
        geometry_axis.set_xlabel("")
        geometry_axis.set_ylabel("")
        geometry_axis.tick_params(labelsize=11)
        geometry_axis.set_aspect("equal", adjustable="box")
        zoom_output = zoom_output.expanduser().resolve()
        zoom_output.parent.mkdir(parents=True, exist_ok=True)
        zoom_figure.savefig(zoom_output, dpi=260, facecolor="white")
        plt.close(zoom_figure)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=260, facecolor="white")
    plt.close(figure)
    print(json.dumps({
        "output": str(output.resolve()),
        "zoom_output": None if zoom_output is None else str(zoom_output.resolve()),
        "event": event_id,
        "map_id": int(map_ids[best_row]),
        "parameters": [float(value) for value in map_parameters],
        "geometry": [float(value) for value in geometry],
        "alpha": float(alpha),
        "color_vmax": vmax,
        "compare_close_wide": compare_close_wide,
        "branch_models": [
            {
                "label": model["label"],
                "map_id": int(map_ids[model["row"]]),
                "chi2": float(chi2[model["row"]]),
                "parameters": [model["s"], model["q"], model["rho"]],
                "alpha": float(model["alpha"]),
            }
            for model in branch_models
        ],
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
        fallback_result=(
            None
            if args.fallback_result is None
            else args.fallback_result.expanduser().resolve()
        ),
        lm_result=(
            None if args.lm_result is None else args.lm_result.expanduser().resolve()
        ),
        top_cells=args.delta_chi2_top_cells,
        q_log_max=args.q_log_max,
        compare_close_wide=args.compare_close_wide,
        time_window=(tuple(args.time_window) if args.time_window is not None else None),
        show_all_caustics=args.show_all_caustics,
        inset_time_window=(
            tuple(args.inset_time_window)
            if args.inset_time_window is not None
            else None
        ),
        no_inset=args.no_inset,
        zoom_output=(
            None if args.zoom_output is None else args.zoom_output.expanduser().resolve()
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
