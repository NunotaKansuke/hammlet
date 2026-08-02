"""Reproducible plotting helpers used by the examples and documentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .atlas import Atlas


def plot_magnification_map(
    atlas: Atlas,
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
    magnification = atlas.magnification(map_id, x, y, m_max=m_max)
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

