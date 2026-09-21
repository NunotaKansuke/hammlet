#!/usr/bin/env python3
"""Compare stored FFT seeds after converting the source path to map coordinates."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


RESULT_ROOT = (
    REPO_ROOT
    / "results/roman_local/roman_binary_hammlet_signal_batch100_full_m128_16w1t/events"
)
TRUTH_PATH = Path(
    "/rogue1_8/nunota/gulls/runs/examples/roman_static_binary_hammlet_signal_batch100/"
    "hammlet_input/truth.json"
)
EVENT_IDS = ("9920084", "9920099")
LABELS = ("q>1 best", "q<1 best", "truth-nearest")


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as arrays:
        return {
            name: np.asarray(arrays[name])
            for name in (
                "map_ids",
                "parameters",
                "chi2",
                "scanned",
                "geometries",
                "geometry_index",
                "alpha_index",
            )
        }


def _select_rows(arrays: dict[str, np.ndarray], truth: dict[str, float]) -> dict[str, int]:
    parameters = arrays["parameters"]
    chi2 = arrays["chi2"]
    finite = arrays["scanned"].astype(bool) & np.isfinite(chi2)
    q_gt = finite & (parameters[:, 1] > 0.0)
    q_lt = finite & (parameters[:, 1] < 0.0)
    rows = {
        "q>1 best": int(np.flatnonzero(q_gt)[np.argmin(chi2[q_gt])]),
        "q<1 best": int(np.flatnonzero(q_lt)[np.argmin(chi2[q_lt])]),
    }
    truth_log = np.log10(
        [truth["planet_s"], truth["planet_q"], truth["rho"]]
    )
    distance = np.sum((parameters - truth_log[None, :]) ** 2, axis=1)
    distance[~finite] = np.inf
    rows["truth-nearest"] = int(np.argmin(distance))
    return rows


def _corrected_fft(
    dataset,
    geometry,
    alpha: float,
    map_id: int,
    map_parameters: np.ndarray,
    atlas,
):
    from hammlet._core.caustics import adamgrid_map_origin_shift
    from hammlet._core.map_adapter import trajectory_xy
    from hammlet._core.radial import interpolate_coefficients
    from hammlet._core.reference import profile_flux

    separation, mass_ratio, _ = 10.0 ** np.asarray(map_parameters, dtype=np.float64)
    shift_x = adamgrid_map_origin_shift(separation, mass_ratio)
    x_native_path, y_native_path = trajectory_xy(
        dataset.time, geometry.t0, geometry.u0, geometry.tE, alpha
    )
    x_map = x_native_path - shift_x
    y_map = y_native_path
    radius = np.hypot(x_map, y_map)
    phase = np.arctan2(y_map, x_map)
    nodes, _, coefficients = atlas.coefficient_row(map_id, m_max=128)
    if np.any(radius < nodes[0]) or np.any(radius > nodes[-1]):
        raise RuntimeError(
            f"map {map_id}: corrected path is outside radial support "
            f"({radius.min()}..{radius.max()} vs {nodes[0]}..{nodes[-1]})"
        )
    local = interpolate_coefficients(coefficients, radius, nodes, order=1)
    modes = np.arange(coefficients.shape[1], dtype=np.float64)
    magnification = 1.0 + np.real(
        local[:, 0]
        + 2.0
        * np.sum(
            local[:, 1:] * np.exp(1j * phase[:, None] * modes[None, 1:]),
            axis=1,
        )
    )
    profile = profile_flux(magnification, dataset)
    return magnification, profile, shift_x


def _exact_truth(dataset, truth: dict[str, float]):
    from hammlet._core.caustics import adamgrid_map_origin_shift
    from hammlet._core.config import PSPLGeometry
    from hammlet._core.direct_vbm import VBMBinaryLensEvaluator
    from hammlet._core.reference import profile_flux

    separation = float(truth["planet_s"])
    mass_ratio = float(truth["planet_q"])
    source_radius = float(truth["rho"])
    geometry = PSPLGeometry(
        float(truth["t0_day"]), float(truth["u0"]), float(truth["tE_days"])
    )
    alpha = np.deg2rad(float(truth["alpha_deg"]))
    tau = (dataset.time - geometry.t0) / geometry.tE
    x_native = tau * np.cos(alpha) - geometry.u0 * np.sin(alpha)
    y_native = tau * np.sin(alpha) + geometry.u0 * np.cos(alpha)
    shift_x = adamgrid_map_origin_shift(separation, mass_ratio)
    evaluator = VBMBinaryLensEvaluator(separation, mass_ratio, source_radius)
    magnification = evaluator.magnification(x_native - shift_x, y_native)
    profile = profile_flux(magnification, dataset)
    return magnification, profile, shift_x


def _plot_event(event_id: str, truth: dict[str, float], atlas) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from hammlet._core.caustics import adamgrid_map_origin_shift
    from hammlet._core.config import PSPLGeometry
    from hammlet._core.map_adapter import trajectory_xy
    from tools.roman_local.roman_event import load_data_file

    event_dir = RESULT_ROOT / event_id
    report = json.loads((event_dir / "report.json").read_text(encoding="utf-8"))
    arrays = _load_arrays(event_dir / "minima.npz")
    rows = _select_rows(arrays, truth)
    n_alpha = int(report["search_settings"]["n_alpha"])
    radial_order = int(report["search_settings"]["radial_order"])
    dataset = load_data_file(
        Path(report["event"]["data_paths"][0]),
        name="data-0",
        window=tuple(report["event"]["window"]),
        data_format=report["event"]["data_format"],
    )

    candidate_info = {}
    required_radius = 0.0
    for label, row in rows.items():
        map_id = int(arrays["map_ids"][row])
        map_parameters = arrays["parameters"][row]
        geometry = PSPLGeometry(
            *[float(value) for value in arrays["geometries"][arrays["geometry_index"][row]]]
        )
        alpha = 2.0 * np.pi * int(arrays["alpha_index"][row]) / n_alpha
        separation, mass_ratio, _ = 10.0 ** map_parameters
        shift_x = adamgrid_map_origin_shift(separation, mass_ratio)
        x_path, y_path = trajectory_xy(
            dataset.time, geometry.t0, geometry.u0, geometry.tE, alpha
        )
        required_radius = max(
            required_radius, float(np.nanmax(np.hypot(x_path - shift_x, y_path)))
        )
        candidate_info[label] = (
            row,
            map_id,
            map_parameters,
            geometry,
            alpha,
        )

    stored_tail = float(report["atlas"]["metadata"]["point_lens_tail"]["max_radius"])
    atlas_for_event = atlas.with_point_lens_tail(
        max(stored_tail, required_radius + 0.25), radial_order=radial_order
    )
    print(
        f"{event_id}: radial tail {max(stored_tail, required_radius + 0.25):g}; "
        f"required {required_radius:g}",
        flush=True,
    )

    curves = {}
    for label, (row, map_id, map_parameters, geometry, alpha) in candidate_info.items():
        magnification, profile, shift_x = _corrected_fft(
            dataset, geometry, alpha, map_id, map_parameters, atlas_for_event
        )
        curves[label] = {
            "row": row,
            "map_id": map_id,
            "geometry": geometry,
            "magnification": magnification,
            "profile": profile,
            "shift_x": shift_x,
        }
        print(
            f"{event_id}: {label} map {map_id}, Δx_map={-shift_x:+.6f}, "
            f"corrected chi2={profile.chi2:.6f}",
            flush=True,
        )

    truth_magnification, truth_profile, truth_shift = _exact_truth(dataset, truth)
    print(
        f"{event_id}: exact truth VBM Δx_map={-truth_shift:+.6f}, "
        f"chi2={truth_profile.chi2:.6f}",
        flush=True,
    )

    best = curves["q>1 best"]
    best_row = best["row"]
    best_geometry = best["geometry"]
    best_rho = 10.0 ** float(arrays["parameters"][best_row, 2])
    half_width = max(
        5.0 * max(abs(best_geometry.u0), best_rho) * best_geometry.tE, 1.0
    )
    finite = np.isfinite(dataset.time) & np.isfinite(dataset.flux)
    observed_peak = float(dataset.time[finite][np.argmax(dataset.flux[finite])])
    model_lower = best_geometry.t0 - half_width
    model_upper = best_geometry.t0 + half_width
    peak_inside = (
        model_lower <= observed_peak <= model_upper
        and abs(observed_peak - best_geometry.t0) <= 0.5 * half_width
    )
    focus = best_geometry.t0 if peak_inside else observed_peak
    focus_label = "best t0" if peak_inside else "data peak"
    lower, upper = focus - half_width, focus + half_width
    local = finite & (dataset.time >= lower) & (dataset.time <= upper)
    if np.count_nonzero(local) >= 2:
        plot_lower = float(np.min(dataset.time[local]))
        plot_upper = float(np.max(dataset.time[local]))
    else:
        plot_lower = float(np.min(dataset.time[finite]))
        plot_upper = float(np.max(dataset.time[finite]))
        focus_label = "full data fallback"
    if event_id == "9920099":
        plot_lower -= 5.0
        plot_upper += 5.0
        local = finite & (dataset.time >= plot_lower) & (dataset.time <= plot_upper)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15.0, 5.7),
        gridspec_kw={"width_ratios": [1.06, 1.0]},
        constrained_layout=True,
    )
    axis = axes[0]
    data_mask = local & np.isfinite(dataset.error)
    data_ms = 3.2 if event_id == "9920099" else 2.0
    data_alpha = 0.68 if event_id == "9920099" else 0.32
    data_zorder = 5 if event_id == "9920099" else 1
    axis.errorbar(
        dataset.time[data_mask],
        dataset.flux[data_mask],
        yerr=dataset.error[data_mask],
        fmt=".",
        ms=data_ms,
        alpha=data_alpha,
        color="tab:blue",
        zorder=data_zorder,
        label="data",
    )
    styles = {
        "q>1 best": ("tab:orange", "-", 1.7),
        "q<1 best": ("tab:green", "--", 1.7),
        "truth-nearest": ("tab:purple", "-.", 1.7),
    }
    display_names = {
        "q>1 best": "q>1 best FFT",
        "q<1 best": "q<1 best FFT",
        "truth-nearest": "truth-nearest FFT",
    }
    if event_id != "9920099":
        for label in LABELS:
            curve = curves[label]
            profile = curve["profile"]
            color, linestyle, linewidth = styles[label]
            model = profile.source_flux * curve["magnification"] + profile.blend_flux
            axis.plot(
                dataset.time[local],
                model[local],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                zorder=3,
                label=f"{display_names[label]} (χ²={profile.chi2:.1f})",
            )
    truth_model = truth_profile.source_flux * truth_magnification + truth_profile.blend_flux
    axis.plot(
        dataset.time[local],
        truth_model[local],
        color="tab:orange" if event_id == "9920099" else "black",
        linestyle="-" if event_id == "9920099" else ":",
        linewidth=1.8 if event_id == "9920099" else 1.35,
        zorder=3 if event_id == "9920099" else 4,
        label=(
            f"truth VBM (χ²={truth_profile.chi2:.1f})"
            if event_id == "9920099"
            else f"exact truth VBM (χ²={truth_profile.chi2:.1f})"
        ),
    )
    axis.set_xlim(plot_lower, plot_upper)
    axis.set_xlabel("time")
    axis.set_ylabel("flux")
    axis.grid(alpha=0.2)
    axis.legend(loc="best", fontsize=9, framealpha=0.90)
    if event_id == "9920099":
        shift_text = f"truth: Δx={-truth_shift:+.3f}"
    else:
        shift_text = ", ".join(
            f"{label}: Δx={-curves[label]['shift_x']:+.3f}" for label in LABELS
        )
    axis.text(
        0.015,
        0.015,
        "map frame: x_map = x_native − shift_x\n" + shift_text,
        transform=axis.transAxes,
        fontsize=8.1,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "0.8"},
    )

    map_path = event_dir / "figures" / "chi2_map_q_lt1_baseline.png"
    axes[1].imshow(plt.imread(map_path))
    axes[1].axis("off")
    title = (
        "truth trajectory in corrected coordinates"
        if event_id == "9920099"
        else "same models in corrected coordinates"
    )
    figure.suptitle(
        f"Roman local event {event_id} — {title}\n"
        f"left: x_map=x_native−shift_x; right: q<1-baselined χ² map; "
        f"{focus_label}, window={plot_lower:g}–{plot_upper:g} d",
        fontsize=15,
    )
    output = event_dir / "figures" / f"{event_id}_q_lt1_baseline_corrected_coords.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"{event_id}: saved {output}", flush=True)


def main() -> None:
    from tools.roman_local.atlas import open_readonly_atlas

    truth_by_id = {
        str(row["event_id"]): row
        for row in json.loads(TRUTH_PATH.read_text(encoding="utf-8"))["events"]
    }
    report = json.loads(
        (RESULT_ROOT / EVENT_IDS[0] / "report.json").read_text(encoding="utf-8")
    )
    atlas_path = Path(report["atlas"]["path"])
    print(f"opening atlas: {atlas_path}", flush=True)
    atlas = open_readonly_atlas(atlas_path)
    print(f"atlas maps: {atlas.total_maps}", flush=True)
    for event_id in EVENT_IDS:
        _plot_event(event_id, truth_by_id[event_id], atlas)


if __name__ == "__main__":
    main()
