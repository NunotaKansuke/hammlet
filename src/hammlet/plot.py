"""Reproducible plotting helpers used by the examples and documentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .maps import Maps


def plot_magnification_map(
    maps: Maps,
    map_id: int,
    output: str | Path,
    *,
    extent: float = 1.5,
    pixels: int = 500,
    m_max: int | None = None,
    vmax_percentile: float = 99.5,
) -> Path:
    """Render ``log10(A)`` on a Cartesian plane and return the PNG path."""
    import matplotlib.pyplot as plt

    axis = np.linspace(-extent, extent, pixels)
    x, y = np.meshgrid(axis, axis)
    magnification = maps.magnification(map_id, x, y, m_max=m_max)
    image = np.log10(np.maximum(magnification, 1.0))
    vmax = max(float(np.percentile(image[np.isfinite(image)], vmax_percentile)), 1e-6)
    figure, axes = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    artist = axes.imshow(
        image,
        origin="lower",
        extent=(-extent, extent, -extent, extent),
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )
    axes.set(xlabel=r"$x/\theta_E$", ylabel=r"$y/\theta_E$", aspect="equal")
    figure.colorbar(artist, ax=axes, label=r"$\log_{10} A$")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_vbm_comparison(
    maps: Maps,
    map_id: int,
    output: str | Path,
    *,
    extent: float = 0.4,
    pixels: int = 260,
    m_max: int | None = None,
) -> Path:
    """Plot direct VBM, Fourier reconstruction, residual, and coefficients.

    This deliberately recomputes the Cartesian reference with VBMicrolensing;
    it does not reuse samples taken while building the Fourier maps.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    from ._core.direct_vbm import VBMBinaryLensEvaluator

    modes = maps.m_max if m_max is None else int(m_max)
    logs, logq, logrho = maps._core.parameters_for(map_id)
    s, q, rho = 10.0**logs, 10.0**logq, 10.0**logrho
    axis = np.linspace(-extent, extent, pixels)
    x, y = np.meshgrid(axis, axis)
    evaluator = VBMBinaryLensEvaluator(s, q, rho)
    direct = evaluator.magnification(x, y)
    reconstructed = maps.magnification(map_id, x, y, m_max=modes)
    residual = direct - reconstructed
    finite = np.isfinite(direct) & np.isfinite(reconstructed)

    direct_log = np.log10(np.maximum(direct, 1.0))
    reconstructed_log = np.log10(np.maximum(reconstructed, 1.0))
    color_max = float(np.percentile(direct_log[finite], 99.8))
    residual_limit = max(float(np.percentile(np.abs(residual[finite]), 99.5)), 1e-5)
    residual_norm = SymLogNorm(
        linthresh=max(residual_limit * 1.0e-3, 1.0e-6),
        linscale=0.7,
        vmin=-residual_limit,
        vmax=residual_limit,
        base=10,
    )

    figure, axes = plt.subplots(2, 2, figsize=(12.8, 10.4), constrained_layout=True)
    map_extent = (-extent, extent, -extent, extent)
    images = []
    for axis_object, values, title in (
        (axes[0, 0], direct_log, "Direct VBMicrolensing"),
        (axes[0, 1], reconstructed_log, f"Fourier reconstruction (M={modes})"),
    ):
        image = axis_object.imshow(
            values,
            origin="lower",
            extent=map_extent,
            cmap="turbo",
            vmin=0.0,
            vmax=color_max,
            interpolation="nearest",
        )
        images.append(image)
        axis_object.set_title(title)
        axis_object.set(xlabel=r"$x/\theta_E$", ylabel=r"$y/\theta_E$", aspect="equal")
    figure.colorbar(
        images[0], ax=axes[0, :], label=r"$\log_{10} A$", shrink=0.92
    )

    residual_image = axes[1, 0].imshow(
        np.clip(residual, -residual_limit, residual_limit),
        origin="lower",
        extent=map_extent,
        cmap="coolwarm",
        norm=residual_norm,
        interpolation="nearest",
    )
    axes[1, 0].set_title("Residual: direct VBM − Fourier")
    axes[1, 0].set(
        xlabel=r"$x/\theta_E$", ylabel=r"$y/\theta_E$", aspect="equal"
    )
    figure.colorbar(residual_image, ax=axes[1, 0], label=r"$\Delta A$")

    coefficients = maps._core.coefficient_rows([map_id], m_max=modes)[map_id][1]
    amplitude = np.maximum(np.abs(coefficients), 1.0e-12)
    coefficient_image = axes[1, 1].pcolormesh(
        np.arange(modes + 1),
        maps.radial_nodes,
        np.log10(amplitude),
        cmap="magma",
        shading="nearest",
        rasterized=True,
    )
    axes[1, 1].set_yscale("symlog", linthresh=1.0e-4)
    axes[1, 1].set_title(r"Stored coefficient amplitude $|c_m(r)|$")
    axes[1, 1].set(xlabel="Fourier mode m", ylabel=r"radius $r/\theta_E$")
    figure.colorbar(
        coefficient_image, ax=axes[1, 1], label=r"$\log_{10}|c_m|$"
    )

    absolute = np.abs(residual[finite])
    relative = absolute / np.maximum(np.abs(direct[finite]), 1.0)
    figure.suptitle(
        "Resonant caustic: direct VBM versus Hammlet maps\n"
        f"s={s:g}, q={q:g}, rho={rho:g}; "
        f"median |residual|={np.median(absolute):.3g}, "
        f"99% relative={np.percentile(relative, 99):.2%}",
        fontsize=15,
    )
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)
    return output
