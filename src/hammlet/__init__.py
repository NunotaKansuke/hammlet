"""Hammlet: fast binary-microlensing map building and seed searches."""

from .build import build_maps, merge_maps
from .config import MapConfig, ParameterGrid, Partition, SearchConfig
from .maps import Maps
from .search import Candidate, Dataset, Geometry, SearchResult, search

__all__ = [
    "MapConfig",
    "Maps",
    "Candidate",
    "Dataset",
    "Geometry",
    "ParameterGrid",
    "Partition",
    "SearchConfig",
    "SearchResult",
    "build_maps",
    "merge_maps",
    "search",
]
