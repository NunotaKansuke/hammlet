"""Manifest-backed collections of :class:`PolarAtlas` instances.

Bucketed atlases keep independently-built polar atlases in parameter-space
bins.  The small manifest in the parent directory is deliberately explicit:
it makes the routing bounds inspectable without opening every child atlas,
while the child manifests remain the source of the spectral data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from os import replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .atlas import PolarAtlas

_FORMAT = "adamgrid-bucketed-polar-atlas"
_VERSION = 1


@dataclass(frozen=True)
class BucketRecord:
    """One parameter-space bucket and its loaded child atlas."""

    name: str
    path: Path
    logs: tuple[float, float]
    logq: tuple[float, float]
    atlas: PolarAtlas

    @property
    def map_ids(self) -> np.ndarray:
        """The map identifiers contained in this bucket."""
        return self.atlas.map_ids

    @property
    def count(self) -> int:
        return int(self.atlas.map_ids.size)


# Kept as a descriptive alias for callers that prefer the atlas-specific name.
PolarAtlasBucket = BucketRecord


class BucketedPolarAtlas:
    """Read a validated collection of parameter-space-bucketed polar atlases.

    The parent ``manifest.json`` has format ``adamgrid-bucketed-polar-atlas``
    version 1.  Every entry in ``buckets`` contains ``name``, relative ``path``,
    ``count``, ``logs`` and ``logq``.  Bounds are inclusive and represented as
    two-element ``[minimum, maximum]`` arrays.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_dir():
            raise FileNotFoundError(
                f"bucketed atlas directory does not exist: {self.path}"
            )
        manifest_path = self.path / "manifest.json"
        try:
            self.manifest: dict[str, Any] = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except FileNotFoundError:
            raise
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"invalid bucketed atlas manifest: {manifest_path}"
            ) from error
        if not isinstance(self.manifest, dict) or (
            self.manifest.get("format") != _FORMAT
            or self.manifest.get("version") != _VERSION
        ):
            raise ValueError("unsupported bucketed polar atlas format")

        entries = self.manifest.get("buckets")
        if not isinstance(entries, list) or not entries:
            raise ValueError(
                "bucketed atlas manifest must contain a non-empty buckets list"
            )

        root = self.path.resolve()
        seen_names: set[str] = set()
        seen_ids: set[int] = set()
        buckets: list[BucketRecord] = []
        expected_m_max: int | None = None
        for index, entry in enumerate(entries):
            bucket = self._load_bucket(entry, index, root)
            if bucket.name in seen_names:
                raise ValueError(f"duplicate bucket name: {bucket.name!r}")
            seen_names.add(bucket.name)
            ids = [int(map_id) for map_id in bucket.atlas.map_ids]
            if len(ids) != len(set(ids)):
                raise ValueError(f"bucket {bucket.name!r} contains duplicate map IDs")
            overlap = seen_ids.intersection(ids)
            if overlap:
                raise ValueError(
                    f"map IDs occur in more than one bucket: {sorted(overlap)}"
                )
            seen_ids.update(ids)
            if expected_m_max is None:
                expected_m_max = bucket.atlas.m_max
            elif bucket.atlas.m_max != expected_m_max:
                raise ValueError(
                    "all bucket atlases must use the same m_max "
                    f"({expected_m_max} != {bucket.atlas.m_max})"
                )
            buckets.append(bucket)

        declared_total = self.manifest.get("n_maps", len(seen_ids))
        if not _is_int(declared_total) or int(declared_total) != len(seen_ids):
            raise ValueError("bucketed atlas n_maps does not match bucket contents")
        declared_m_max = self.manifest.get("m_max", expected_m_max)
        if not _is_int(declared_m_max) or int(declared_m_max) != expected_m_max:
            raise ValueError("bucketed atlas m_max does not match child atlases")
        self._buckets = tuple(buckets)
        self._map_ids = frozenset(seen_ids)
        self._m_max = int(expected_m_max)  # entries is known non-empty
        self._bucket_for_map = {
            int(map_id): bucket for bucket in self._buckets for map_id in bucket.map_ids
        }
        self._ordered_map_ids = np.concatenate(
            [np.asarray(bucket.map_ids, dtype=np.int64) for bucket in self._buckets]
        )
        self._map_parameters = np.concatenate(
            [np.asarray(bucket.atlas.map_parameters) for bucket in self._buckets],
            axis=0,
        )
        self._parameter_index = {
            int(map_id): row for row, map_id in enumerate(self._ordered_map_ids)
        }

    def _load_bucket(self, entry: object, index: int, root: Path) -> BucketRecord:
        if not isinstance(entry, Mapping):
            raise ValueError(f"bucket {index} must be an object")
        name = entry.get("name")
        relative_path = entry.get("path")
        if not isinstance(name, str) or not name:
            raise ValueError(f"bucket {index} has an invalid name")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"bucket {name!r} has an invalid path")
        child_path = Path(relative_path)
        if child_path.is_absolute() or ".." in child_path.parts:
            raise ValueError(f"bucket {name!r} path must be relative to the manifest")
        resolved = (self.path / child_path).resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"bucket {name!r} path escapes the manifest directory")
        if not resolved.is_dir():
            raise FileNotFoundError(
                f"bucket {name!r} atlas directory does not exist: {resolved}"
            )
        count = entry.get("count")
        if not _is_int(count) or int(count) < 1:
            raise ValueError(f"bucket {name!r} has an invalid count")
        logs = _bounds(entry.get("logs"), f"bucket {name!r} logs")
        logq = _bounds(entry.get("logq"), f"bucket {name!r} logq")
        atlas = PolarAtlas(resolved)
        if atlas.map_ids.size != int(count):
            raise ValueError(
                f"bucket {name!r} count ({count}) does not match child atlas "
                f"({atlas.map_ids.size})"
            )
        declared_ids = entry.get("map_ids")
        if declared_ids is not None:
            if not isinstance(declared_ids, list) or not all(
                _is_int(value) for value in declared_ids
            ):
                raise ValueError(f"bucket {name!r} map_ids must be an integer list")
            actual_ids = {int(value) for value in atlas.map_ids}
            if len(declared_ids) != len(actual_ids) or set(declared_ids) != actual_ids:
                raise ValueError(f"bucket {name!r} map_ids do not match child atlas")
        return BucketRecord(name, resolved, logs, logq, atlas)

    def __iter__(self) -> Iterator[BucketRecord]:
        return iter(self._buckets)

    def iter_buckets(self) -> Iterator[BucketRecord]:
        """Iterate over bucket records in manifest order."""
        return iter(self)

    @property
    def buckets(self) -> tuple[BucketRecord, ...]:
        return self._buckets

    @property
    def total_maps(self) -> int:
        return len(self._map_ids)

    @property
    def n_maps(self) -> int:
        """Alias for :attr:`total_maps`, matching the manifest field."""
        return self.total_maps

    @property
    def map_parameters(self) -> np.ndarray:
        """``(logs, logq, logrho)`` for every map, in bucket iteration order."""
        return np.asarray(self._map_parameters, dtype=np.float64)

    def parameters_for(self, map_id: int) -> np.ndarray:
        """Return ``(logs, logq, logrho)`` for one global map identifier."""
        try:
            row = self._parameter_index[int(map_id)]
        except KeyError as error:
            raise KeyError(
                f"map id is not present in the bucketed atlas: {map_id}"
            ) from error
        return np.asarray(self._map_parameters[row], dtype=np.float64)

    def neighboring_maps(
        self, map_id: int, limit: int = 26
    ) -> list[tuple[int, np.ndarray]]:
        """Return immediate parameter-grid neighbours, including other buckets."""
        if not _is_int(limit) or int(limit) < 1:
            raise ValueError("neighbor limit must be a positive integer")
        center = self.parameters_for(map_id)
        allowed: list[np.ndarray] = []
        scales = []
        for axis in range(3):
            values = np.unique(self._map_parameters[:, axis])
            position = int(np.searchsorted(values, center[axis]))
            choices = [values[position]]
            if position:
                choices.append(values[position - 1])
            if position + 1 < len(values):
                choices.append(values[position + 1])
            allowed.append(np.asarray(choices))
            differences = np.diff(values)
            scales.append(
                max(float(np.min(differences)), 1.0e-12) if differences.size else 1.0
            )
        mask = np.ones(len(self._map_parameters), dtype=bool)
        for axis in range(3):
            mask &= np.isin(self._map_parameters[:, axis], allowed[axis])
        mask[self._parameter_index[int(map_id)]] = False
        rows = np.flatnonzero(mask)
        if len(rows) > limit:
            distance = np.sum(
                ((self._map_parameters[rows] - center) / np.asarray(scales)) ** 2,
                axis=1,
            )
            rows = rows[np.argsort(distance, kind="stable")[: int(limit)]]
        return [
            (
                int(self._ordered_map_ids[row]),
                np.asarray(self._map_parameters[row], dtype=np.float64),
            )
            for row in rows
        ]

    def coefficient_row(
        self, map_id: int, *, m_max: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load one map's ``(radial_nodes, parameters, x_coeff)`` record."""
        try:
            bucket = self._bucket_for_map[int(map_id)]
        except KeyError as error:
            raise KeyError(
                f"map id is not present in the bucketed atlas: {map_id}"
            ) from error
        parameters, coefficients = bucket.atlas.coefficient_rows(
            [int(map_id)], m_max=m_max
        )[int(map_id)]
        return (
            np.asarray(bucket.atlas.radial_nodes, dtype=np.float64),
            parameters,
            coefficients,
        )

    def coefficient_rows(
        self, map_ids: Sequence[int], *, m_max: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bulk-load rows in caller order, reading each extension shard once."""
        requested = [int(map_id) for map_id in map_ids]
        if not requested:
            return (
                np.empty((0, 0), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 0, m_max + 1), dtype=np.complex64),
            )
        missing = set(requested).difference(self._map_ids)
        if missing:
            raise KeyError(
                f"map ids are not present in the bucketed atlas: {sorted(missing)}"
            )
        unique_by_bucket: dict[BucketRecord, list[int]] = {}
        for map_id in dict.fromkeys(requested):
            unique_by_bucket.setdefault(self._bucket_for_map[map_id], []).append(map_id)
        loaded: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for bucket, local_ids in unique_by_bucket.items():
            rows = bucket.atlas.coefficient_rows(local_ids, m_max=m_max)
            nodes = np.asarray(bucket.atlas.radial_nodes, dtype=np.float64)
            for map_id, (parameters, coefficients) in rows.items():
                loaded[map_id] = (nodes, parameters, coefficients)
        node_shapes = {loaded[map_id][0].shape for map_id in requested}
        coefficient_shapes = {loaded[map_id][2].shape for map_id in requested}
        if len(node_shapes) != 1 or len(coefficient_shapes) != 1:
            raise ValueError("bulk JIT rows require equal atlas dimensions")
        return (
            np.stack([loaded[map_id][0] for map_id in requested]),
            np.stack([loaded[map_id][1] for map_id in requested]),
            np.stack([loaded[map_id][2] for map_id in requested]),
        )

    @property
    def m_max(self) -> int:
        return self._m_max


def write_bucket_manifest(
    output: str | Path,
    entries: Sequence[Mapping[str, object] | BucketRecord],
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Write a version-1 bucketed-atlas manifest for a builder.

    ``entries`` use the same required fields accepted by :class:`BucketedPolarAtlas`.
    ``metadata`` may add builder-specific fields but cannot replace format,
    version, bucket records, ``n_maps``, or ``m_max``.
    """
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    normalized: list[dict[str, object]] = []
    total = 0
    m_max: int | None = None
    for index, source in enumerate(entries):
        if isinstance(source, BucketRecord):
            relative = _relative_child_path(source.path, root)
            entry: Mapping[str, object] = {
                "name": source.name,
                "path": relative,
                "count": source.count,
                "logs": list(source.logs),
                "logq": list(source.logq),
                "map_ids": [int(value) for value in source.map_ids],
            }
            child_m_max = source.atlas.m_max
        elif isinstance(source, Mapping):
            entry = source
            child_m_max = None
        else:
            raise ValueError(f"bucket {index} must be a mapping or BucketRecord")
        record = _normalise_entry(entry, index)
        child = destination / str(record["path"])
        if not child.is_dir():
            raise FileNotFoundError(
                f"bucket {record['name']!r} atlas directory does not exist: {child}"
            )
        child_atlas = PolarAtlas(child)
        if child_atlas.map_ids.size != int(record["count"]):
            raise ValueError(
                f"bucket {record['name']!r} count does not match child atlas"
            )
        if "map_ids" in record and set(record["map_ids"]) != {
            int(value) for value in child_atlas.map_ids
        }:
            raise ValueError(
                f"bucket {record['name']!r} map_ids do not match child atlas"
            )
        child_m_max = child_atlas.m_max
        if m_max is None:
            m_max = child_m_max
        elif m_max != child_m_max:
            raise ValueError("all bucket atlases must use the same m_max")
        total += int(record["count"])
        normalized.append(record)
    if not normalized:
        raise ValueError("cannot write an empty bucket manifest")
    payload = dict(metadata or {})
    reserved = {"format", "version", "buckets", "n_maps", "m_max"}
    collision = reserved.intersection(payload)
    if collision:
        raise ValueError(
            f"metadata cannot override manifest fields: {sorted(collision)}"
        )
    payload.update(
        {
            "format": _FORMAT,
            "version": _VERSION,
            "n_maps": total,
            "m_max": m_max,
            "buckets": normalized,
        }
    )
    manifest = destination / "manifest.json"
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination, delete=False
    ) as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
        temporary = Path(handle.name)
    replace(temporary, manifest)
    return manifest


def _normalise_entry(entry: Mapping[str, object], index: int) -> dict[str, object]:
    name = entry.get("name")
    path = entry.get("path")
    count = entry.get("count")
    if not isinstance(name, str) or not name:
        raise ValueError(f"bucket {index} has an invalid name")
    if (
        not isinstance(path, str)
        or not path
        or Path(path).is_absolute()
        or ".." in Path(path).parts
    ):
        raise ValueError(f"bucket {name!r} path must be a relative child path")
    if not _is_int(count) or int(count) < 1:
        raise ValueError(f"bucket {name!r} has an invalid count")
    result: dict[str, object] = {
        "name": name,
        "path": path,
        "count": int(count),
        "logs": list(_bounds(entry.get("logs"), f"bucket {name!r} logs")),
        "logq": list(_bounds(entry.get("logq"), f"bucket {name!r} logq")),
    }
    if "map_ids" in entry:
        ids = entry["map_ids"]
        if not isinstance(ids, list) or not all(_is_int(value) for value in ids):
            raise ValueError(f"bucket {name!r} map_ids must be an integer list")
        result["map_ids"] = [int(value) for value in ids]
    return result


def _bounds(value: object, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be a two-element numeric range")
    lower, upper = value
    if isinstance(lower, bool) or isinstance(upper, bool):
        raise ValueError(f"{label} must be a two-element numeric range")
    try:
        bounds = (float(lower), float(upper))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a two-element numeric range") from error
    if not np.all(np.isfinite(bounds)) or bounds[0] > bounds[1]:
        raise ValueError(f"{label} must have finite increasing bounds")
    return bounds


def _is_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _relative_child_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError as error:
        raise ValueError("bucket path must be below the manifest directory") from error
