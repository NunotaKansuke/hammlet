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

    m_max: int = 512
    core_m_max: int = 128
    radial_nodes: int = 256
    radial_max: float = 3.5
    min_n_phi: int = 512
    caustic_base_n_phi: int = 2048
    max_n_phi: int = 8192
    diagnostic_m_max: int = 1024
    coefficient_rtol: float = 2.0e-4
    coefficient_atol: float = 2.0e-7
    caustic_guard_rho: float = 4.0
    caustic_local_levels: int = 8
    caustic_points_per_side: int = 4
    vbm_tolerance: float = 1.0e-3
    vbm_relative_tolerance: float = 1.0e-4
    shard_size: int = 256
    radial_s_buckets: int = 4
    radial_q_buckets: int = 4
    radial_pilot_maps: int = 8
    radial_pilot_bins: int = 1025
    radial_pilot_phi: int = 64
    radial_pilot_m_max: int = 24
    radial_adaptive_fraction: float = 0.65

    def __post_init__(self) -> None:
        positive_ints = (
            "m_max",
            "core_m_max",
            "radial_nodes",
            "min_n_phi",
            "caustic_base_n_phi",
            "max_n_phi",
            "diagnostic_m_max",
            "shard_size",
            "radial_s_buckets",
            "radial_q_buckets",
            "radial_pilot_maps",
            "radial_pilot_bins",
            "radial_pilot_phi",
            "radial_pilot_m_max",
        )
        if any(int(getattr(self, name)) < 1 for name in positive_ints):
            raise ValueError("map generation counts and mode budgets must be positive")
        if not self.core_m_max <= self.m_max <= self.diagnostic_m_max:
            raise ValueError("require core_m_max <= m_max <= diagnostic_m_max")
        if not 0.0 <= self.radial_adaptive_fraction <= 1.0:
            raise ValueError("radial_adaptive_fraction must be between zero and one")
        if self.radial_pilot_bins < self.radial_nodes:
            raise ValueError("radial_pilot_bins must be at least radial_nodes")
        if self.radial_pilot_phi < 2 * (self.radial_pilot_m_max + 1):
            raise ValueError("radial_pilot_phi is too small for radial_pilot_m_max")
        if self.radial_max <= 0.0:
            raise ValueError("radial_max must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SearchConfig:
    base_m_max: int = 32
    full_m_max: int = 128
    base_n_alpha: int = 128
    full_n_alpha: int = 512
    top_count: int = 500
    risk_count: int = 500
    candidate_count: int = 300
    base_radial_order: int = 1
    full_radial_order: int = 3
    certified_selection: bool = True
    shards_per_call: int = 4
    handoff_m_max: int = 512
    cluster_log_s: float = 0.075
    cluster_log_q: float = 0.15
    cluster_log_rho: float = 0.30
    cluster_alpha_bins: float = 2.0
    refine: bool = True
    refine_batch_size: int = 32
    refine_shallow_levels: int = 2
    refine_deep_levels: int = 10
    refine_pairwise_levels: int = 1
    refine_deep_count: int = 32
    refine_neighbor_limit: int = 26
    refine_t0_half_width: float = 0.5
    refine_u0_half_width: float = 0.01
    refine_log_tE_half_width: float = 0.05

    def __post_init__(self) -> None:
        if not 0 <= self.base_m_max < self.full_m_max <= self.handoff_m_max:
            raise ValueError("require base_m_max < full_m_max <= handoff_m_max")
        if self.base_n_alpha < 4 * self.base_m_max:
            raise ValueError("base_n_alpha must be at least 4*base_m_max")
        if self.full_n_alpha < 4 * self.full_m_max:
            raise ValueError("full_n_alpha must be at least 4*full_m_max")
        if self.full_n_alpha % self.base_n_alpha:
            raise ValueError("full_n_alpha must be a multiple of base_n_alpha")
        if self.base_radial_order not in (1, 3) or self.full_radial_order not in (1, 3):
            raise ValueError("radial orders must be 1 or 3")
        for name in (
            "top_count",
            "risk_count",
            "candidate_count",
            "shards_per_call",
            "refine_batch_size",
            "refine_deep_count",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
