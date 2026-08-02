"""Short public API for multi-resolution alpha-FFT seed search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .maps import Maps
from .config import SearchConfig
from ._core.config import PSPLGeometry
from ._core.jax_backend import JAXConsistentGeometryBatchScanner
from ._core.multires import scan_atlas_multiresolution
from ._core.trajectory import PhotometricDataset, build_consistent_event_kernel


Dataset = PhotometricDataset
Geometry = PSPLGeometry


@dataclass(frozen=True)
class Candidate:
    map_id: int
    s: float
    q: float
    rho: float
    alpha: float
    geometry_index: int
    chi2: float
    chi2_lower: float
    chi2_upper: float
    spectral_risk: float


@dataclass(frozen=True)
class SearchResult:
    candidates: tuple[Candidate, ...]
    base_seconds: float
    full_seconds: float
    maps_scanned: int
    maps_rescanned: int


def _kernels(datasets, geometries, nodes, m_max, radial_order):
    return [
        [
            build_consistent_event_kernel(
                dataset, geometry, nodes, m_max, radial_order=radial_order
            )
            for dataset in datasets
        ]
        for geometry in geometries
    ]


def search(
    maps: Maps,
    datasets: Sequence[Dataset],
    geometries: Sequence[Geometry],
    *,
    config: SearchConfig | None = None,
) -> SearchResult:
    """Search all maps and alpha, returning independent downstream seeds."""
    config = config or SearchConfig()
    datasets, geometries = tuple(datasets), tuple(geometries)
    if not datasets or not geometries:
        raise ValueError("at least one dataset and one geometry are required")
    if config.full_m_max > maps.m_max:
        raise ValueError("full_m_max exceeds the modes stored in these maps")
    nodes = maps.radial_nodes
    base = JAXConsistentGeometryBatchScanner(
        _kernels(datasets, geometries, nodes, config.base_m_max, config.radial_order),
        n_alpha=config.base_n_alpha,
    )
    full = JAXConsistentGeometryBatchScanner(
        _kernels(datasets, geometries, nodes, config.full_m_max, config.radial_order),
        n_alpha=config.full_n_alpha,
    )
    raw = scan_atlas_multiresolution(
        maps._core,
        base,
        full,
        base_m_max=config.base_m_max,
        full_m_max=config.full_m_max,
        top_count=config.top_count,
        risk_count=config.risk_count,
        shards_per_call=config.shards_per_call,
        certified_selection=config.certified_selection,
    )
    records = raw.candidates(
        n_alpha=config.full_n_alpha,
        count=config.candidate_count,
        rescanned_only=True,
    )
    candidates = tuple(
        Candidate(
            map_id=item.map_id,
            s=10.0**item.logs,
            q=10.0**item.logq,
            rho=10.0**item.logrho,
            alpha=item.alpha,
            geometry_index=item.geometry_index,
            chi2=item.chi2,
            chi2_lower=float(item.chi2_lower),
            chi2_upper=float(item.chi2_upper),
            spectral_risk=item.spectral_risk,
        )
        for item in records
    )
    return SearchResult(
        candidates=candidates,
        base_seconds=raw.base_seconds,
        full_seconds=raw.full_seconds,
        maps_scanned=len(raw.map_ids),
        maps_rescanned=int(np.sum(raw.rescanned)),
    )
