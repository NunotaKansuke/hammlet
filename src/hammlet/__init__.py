"""Hammlet: fast binary-microlensing Fourier atlases and seed searches."""

from .atlas import Atlas
from .build import build_atlas, merge_parts
from .config import AtlasConfig, ParameterGrid, Partition, SearchConfig
from .search import Candidate, Dataset, Geometry, SearchResult, search

__all__ = [
    "Atlas",
    "AtlasConfig",
    "Candidate",
    "Dataset",
    "Geometry",
    "ParameterGrid",
    "Partition",
    "SearchConfig",
    "SearchResult",
    "build_atlas",
    "merge_parts",
    "search",
]

