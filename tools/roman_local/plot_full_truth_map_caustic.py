#!/usr/bin/env python3
"""Plot the full FFT map beside the truth caustics and source path."""

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


EVENT_ID = "9920099"
MAP_ID = 20931
RESULT_ROOT = (
    REPO_ROOT
    / "results/roman_local/roman_binary_hammlet_signal_batch100_full_m128_16w1t/events"
)
TRUTH_PATH = Path(
    "/rogue1_8/nunota/gulls/runs/examples/roman_static_binary_hammlet_signal_batch100/"
    "hammlet_input/truth.json"
)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    from hammlet._core.caustics import adamgrid_map_origin_shift
    from hammlet._core.direct_vbm import VBMBinaryLensEvaluator
    from tools.roman_local.atlas import open_readonly_atlas
    from tools.roman_local.map_plot import reconstruct_cartesian_map

    event_dir = RESULT_ROOT / EVENT_ID
    report = json.loads((event_dir / "report.json").read_text(encoding="utf-8"))
    truth = next(
        row
        for row in json.loads(TRUTH_PATH.read_text(encoding="utf-8"))["events"]
        if str(row["event_id"]) == EVENT_ID
    )

    atlas_path = Path(report["atlas"]["path"])
    print(f"opening atlas: {atlas_path}", flush=True)
    atlas = open_readonly_atlas(atlas_path)
    radial_nodes, map_parameters, coefficients = atlas.coefficient_row(
        MAP_ID, m_max=128
    )
    extent = float(radial_nodes[-1])
    print(f"map {MAP_ID}: radial support 0..{extent:g}", flush=True)

    values = reconstruct_cartesian_map(
        radial_nodes,
        coefficients,
        extent=extent,
        pixels=900,
        n_phi=2048,
    )
    finite = np.isfinite(values)
    log_values = np.full(values.shape, np.nan, dtype=np.float64)
    log_values[finite] = np.log10(np.clip(values[finite], 1.0, None))
    vmax = float(np.nanpercentile(log_values, 99.5))
    vmax = max(vmax, 1.0e-6)

    separation = float(truth["planet_s"])
    mass_ratio = float(truth["planet_q"])
    source_radius = float(truth["rho"])
    t0 = float(truth["t0_day"])
    u0 = float(truth["u0"])
    t_eff = float(truth["tE_days"])
    alpha = np.deg2rad(float(truth["alpha_deg"]))
    peak_time = float(truth["truth_peak_time_day"])
    shift_x = adamgrid_map_origin_shift(separation, mass_ratio)

    evaluator = VBMBinaryLensEvaluator(separation, mass_ratio, source_radius)
    components = evaluator.caustic_components

    # Sample the physical GULLS/native path, then transform it to the map frame.
    tau = np.linspace(-extent - 0.08, extent + 0.08, 12000)
    x_native = tau * np.cos(alpha) - u0 * np.sin(alpha)
    y_native = tau * np.sin(alpha) + u0 * np.cos(alpha)
    x_source = x_native - shift_x
    y_source = y_native
    inside = np.hypot(x_source, y_source) <= extent

    tau_peak = (peak_time - t0) / t_eff
    x_peak = tau_peak * np.cos(alpha) - u0 * np.sin(alpha) - shift_x
    y_peak = tau_peak * np.sin(alpha) + u0 * np.cos(alpha)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14.0, 6.3),
        gridspec_kw={"width_ratios": [1.0, 1.0]},
        constrained_layout=True,
    )

    image = axes[0].imshow(
        np.ma.masked_invalid(log_values),
        origin="lower",
        extent=(-extent, extent, -extent, extent),
        cmap="jet",
        norm=Normalize(vmin=0.0, vmax=vmax),
        interpolation="nearest",
    )
    axes[0].set(
        xlabel=r"$x/\theta_E$",
        ylabel=r"$y/\theta_E$",
        aspect="equal",
        title=f"Full FFT reconstruction (map {MAP_ID}, M=128)",
    )
    figure.colorbar(image, ax=axes[0], label=r"$\log_{10} A$")

    for index, component in enumerate(components):
        axes[1].plot(
            component[:, 0],
            component[:, 1],
            color="black",
            linewidth=1.7,
            label="truth caustic" if index == 0 else None,
        )
    axes[1].plot(
        x_source[inside],
        y_source[inside],
        color="tab:orange",
        linewidth=2.0,
        label="truth source trajectory",
    )
    source_disk = plt.Circle(
        (x_peak, y_peak),
        source_radius,
        facecolor="tab:orange",
        edgecolor="tab:orange",
        alpha=0.28,
        linewidth=1.0,
        label="source disk at truth peak",
    )
    axes[1].add_patch(source_disk)
    axes[1].scatter(
        [x_peak],
        [y_peak],
        color="tab:orange",
        edgecolor="black",
        s=28,
        zorder=5,
    )
    axes[1].set(
        xlim=(-extent, extent),
        ylim=(-extent, extent),
        xlabel=r"$x/\theta_E$",
        ylabel=r"$y/\theta_E$",
        aspect="equal",
        title="Truth caustics + source trajectory (map frame)",
    )
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="upper right", framealpha=0.9)

    logs, logq, logrho = (float(value) for value in map_parameters)
    figure.suptitle(
        f"Roman event {EVENT_ID}: full FFT map + truth geometry\n"
        f"truth: s={separation:.5g}, q={mass_ratio:.5g}, rho={source_radius:.5g}, "
        f"alpha={np.rad2deg(alpha):.4g} deg, shift_x={shift_x:+.5g}\n"
        f"nearest FFT map {MAP_ID}: s={10**logs:.5g}, q={10**logq:.5g}, rho={10**logrho:.5g}",
        fontsize=12,
    )
    output = event_dir / "figures" / "truth_reconstruction_caustic_full.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
