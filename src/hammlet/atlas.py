"""High-level, memory-mapped atlas interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._core.atlas import PolarAtlas
from ._core.radial import interpolate_coefficients


class Atlas:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self._core = PolarAtlas(self.path)

    @classmethod
    def open(cls, path: str | Path) -> "Atlas":
        return cls(path)

    @property
    def parameters(self) -> np.ndarray:
        """Rows ``(log10(s), log10(q), log10(rho))`` in stable map-ID order."""
        return np.asarray(self._core.map_parameters)

    @property
    def map_ids(self) -> np.ndarray:
        return np.asarray(self._core.map_ids)

    @property
    def radial_nodes(self) -> np.ndarray:
        return np.asarray(self._core.radial_nodes)

    @property
    def m_max(self) -> int:
        return self._core.m_max

    def magnification(
        self,
        map_id: int,
        x: np.ndarray,
        y: np.ndarray,
        *,
        m_max: int | None = None,
        radial_order: int = 1,
    ) -> np.ndarray:
        """Reconstruct a magnification map at arbitrary Cartesian points."""
        modes = self.m_max if m_max is None else int(m_max)
        coefficients = self._core.coefficient_rows([map_id], m_max=modes)[int(map_id)][1]
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        radius = np.hypot(x, y).ravel()
        phase = np.arctan2(y, x).ravel()
        local = interpolate_coefficients(
            coefficients, radius, self.radial_nodes, order=radial_order
        )
        mode = np.arange(modes + 1)
        excess = np.real(
            local[:, 0]
            + 2.0 * np.sum(local[:, 1:] * np.exp(1j * phase[:, None] * mode[None, 1:]), axis=1)
        )
        return (1.0 + excess).reshape(x.shape)

    def search(self, datasets, geometries, *, config=None):
        from .search import search

        return search(self, datasets, geometries, config=config)

