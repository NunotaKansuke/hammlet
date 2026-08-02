"""Validated public configuration objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


def _axis(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or not array.size or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional axis")
    if np.any(array <= 0.0) or np.any(np.diff(array) <= 0.0):
        raise ValueError(f"{name} must be positive and strictly increasing")
    return array


@dataclass(frozen=True)
class ParameterGrid:
    """Cartesian grid in physical ``s``, ``q`` and ``rho`` values."""

    s: tuple[float, ...]
    q: tuple[float, ...]
    rho: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in ("s", "q", "rho"):
            object.__setattr__(self, name, tuple(_axis(getattr(self, name), name)))

    @classmethod
    def from_log10(
        cls,
        *,
        log_s: Iterable[float],
        log_q: Iterable[float],
        log_rho: Iterable[float],
    ) -> "ParameterGrid":
        return cls(
            tuple(10.0 ** np.asarray(tuple(log_s))),
            tuple(10.0 ** np.asarray(tuple(log_q))),
            tuple(10.0 ** np.asarray(tuple(log_rho))),
        )

    def table(self) -> np.ndarray:
        """Return stable rows ``(map_id, log10(s), log10(q), log10(rho))``."""
        rows = []
        map_id = 0
        for s in self.s:
            for q in self.q:
                for rho in self.rho:
                    rows.append((map_id, np.log10(s), np.log10(q), np.log10(rho)))
                    map_id += 1
        return np.asarray(rows, dtype=np.float64)

    def to_dict(self) -> dict[str, list[float]]:
        return {name: list(getattr(self, name)) for name in ("s", "q", "rho")}


@dataclass(frozen=True)
class Partition:
    """One zero-based contiguous part of an ordered parameter table."""

    index: int = 0
    count: int = 1

    def __post_init__(self) -> None:
        if self.count < 1 or not 0 <= self.index < self.count:
            raise ValueError("require count >= 1 and 0 <= index < count")

    def rows(self, size: int) -> slice:
        bounds = np.linspace(0, int(size), self.count + 1, dtype=np.int64)
        return slice(int(bounds[self.index]), int(bounds[self.index + 1]))


def default_radial_nodes(n: int = 256, maximum: float = 3.5) -> np.ndarray:
    """Dense near the origin, with every outer annulus still represented."""
    if n < 16 or maximum <= 0.0:
        raise ValueError("n must be at least 16 and maximum must be positive")
    # Geometric nodes resolve source-sized central structure.  A linear tail
    # prevents the outer trajectory from developing excessively wide annuli.
    inner_count = int(round(0.72 * (n - 1)))
    inner = np.geomspace(1.0e-5, min(0.35, maximum), inner_count)
    outer = np.linspace(inner[-1], maximum, n - inner_count)[1:]
    return np.concatenate(([0.0], inner, outer))


@dataclass(frozen=True)
class MapConfig:
    """Production defaults for direct, adaptive VBM map generation."""

    m_max: int = 384
    core_m_max: int = 96
    radial_nodes: int = 256
    radial_max: float = 3.5
    min_n_phi: int = 512
    caustic_base_n_phi: int = 2048
    max_n_phi: int = 8192
    diagnostic_m_max: int = 768
    coefficient_rtol: float = 2.0e-4
    coefficient_atol: float = 2.0e-7
    caustic_guard_rho: float = 4.0
    caustic_local_levels: int = 8
    caustic_points_per_side: int = 4
    vbm_tolerance: float = 1.0e-3
    vbm_relative_tolerance: float = 1.0e-4
    shard_size: int = 256

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SearchConfig:
    base_m_max: int = 32
    full_m_max: int = 96
    base_n_alpha: int = 128
    full_n_alpha: int = 512
    top_count: int = 500
    risk_count: int = 500
    candidate_count: int = 300
    radial_order: int = 1
    certified_selection: bool = True
    shards_per_call: int = 4
