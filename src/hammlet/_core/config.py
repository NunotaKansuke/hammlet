from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

SearchMode = Literal["pspl", "hybrid", "caustic_clock", "auto"]


@dataclass(frozen=True)
class PSPLGeometry:
    """Trajectory seed. Angles exposed by this package are in radians."""

    t0: Optional[float]
    u0: Optional[float]
    tE: Optional[float]
    covariance: Optional[np.ndarray] = None
    reliable: bool = True
    origin: Literal["pspl", "caustic_clock"] = "pspl"

    def validate(self, require_timing: bool = True) -> None:
        values = (self.t0, self.u0, self.tE)
        if require_timing and any(value is None for value in values):
            raise ValueError("t0, u0, and tE are required for this search mode")
        if self.tE is not None and (not np.isfinite(self.tE) or self.tE <= 0):
            raise ValueError("tE must be finite and positive")
        for name, value in zip(("t0", "u0"), values[:2]):
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.covariance is not None:
            covariance = np.asarray(self.covariance)
            if covariance.shape != (3, 3):
                raise ValueError("covariance must have shape (3, 3) for (t0, u0, log(tE))")
        if self.origin not in {"pspl", "caustic_clock"}:
            raise ValueError("origin must be 'pspl' or 'caustic_clock'")


@dataclass(frozen=True)
class TimingHints:
    anomaly_center: Optional[float] = None
    caustic_entry: Optional[float] = None
    caustic_exit: Optional[float] = None
    sigma_entry: Optional[float] = None
    sigma_exit: Optional[float] = None
    required_anomaly_deviation: Optional[float] = None

    @property
    def has_any(self) -> bool:
        return any(
            value is not None
            for value in (self.anomaly_center, self.caustic_entry, self.caustic_exit)
        )

    @property
    def has_crossing(self) -> bool:
        return self.caustic_entry is not None and self.caustic_exit is not None

    def validate(self) -> None:
        if self.has_crossing and self.caustic_exit <= self.caustic_entry:
            raise ValueError("caustic_exit must be later than caustic_entry")
        if self.required_anomaly_deviation is not None and self.required_anomaly_deviation < 0:
            raise ValueError("required_anomaly_deviation must be non-negative")


@dataclass(frozen=True)
class BinaryAlphaFFTConfig:
    mode: SearchMode = "auto"
    n_alpha: int = 256
    m_max: int = 48
    radial_bins: int = 64
    map_chunk_size: int = 256
    top_k_fft: int = 200
    per_map_candidates: int = 4
    cluster_log_tolerances: tuple[float, float, float] = (0.075, 0.15, 0.45)
    cluster_alpha_bins: float = 2.0
    top_k_refine: int = 30
    top_k_exact: int = 5
    refine_map_neighbors: bool = True
    max_map_neighbors: int = 26
    finite_source_mode: Literal["precomputed_map"] = "precomputed_map"
    backend: Literal["numpy"] = "numpy"
    coefficient_dtype: Literal["complex64", "complex128"] = "complex64"
    profile_dtype: Literal["float64"] = "float64"
    singular_rtol: float = 1e-12
    alpha_refine_bins: float = 2.0
    t0_half_width_tE: float = 0.1
    u0_relative_half_width: float = 0.1
    u0_min_half_width: float = 0.01
    tE_factor: float = 1.5

    def __post_init__(self) -> None:
        if self.mode not in {"pspl", "hybrid", "caustic_clock", "auto"}:
            raise ValueError(f"unsupported mode: {self.mode}")
        if self.n_alpha < 4 or self.n_alpha % 2:
            raise ValueError("n_alpha must be an even integer >= 4")
        if not 0 <= 2 * self.m_max <= self.n_alpha // 2:
            raise ValueError(
                "n_alpha must be at least 4*m_max so squared-map modes fit"
            )
        for name in (
            "radial_bins",
            "map_chunk_size",
            "top_k_fft",
            "per_map_candidates",
            "top_k_refine",
            "top_k_exact",
            "max_map_neighbors",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.tE_factor <= 1:
            raise ValueError("tE_factor must be greater than one")
        if any(value <= 0 for value in self.cluster_log_tolerances):
            raise ValueError("cluster_log_tolerances must be positive")
        if self.cluster_alpha_bins <= 0:
            raise ValueError("cluster_alpha_bins must be positive")

    def resolve_mode(
        self, geometry: Optional[PSPLGeometry], timing: Optional[TimingHints]
    ) -> SearchMode:
        timing = timing or TimingHints()
        timing.validate()
        if self.mode != "auto":
            return self.mode
        if geometry is not None and geometry.reliable:
            return "hybrid" if timing.has_any else "pspl"
        if timing.has_crossing:
            return "caustic_clock"
        raise ValueError(
            "auto mode needs a reliable PSPL seed or both caustic entry and exit times"
        )
