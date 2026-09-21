"""Local-only packed bucket cache for Roman map scans.

The production atlas is deliberately left untouched.  This module reads the
currently completed map union once and writes one uncompressed, memmap-able
``.npy`` coefficient array per local scan bucket.  The cache is a snapshot: it
records the map-ID digest and the completed/expected counts so a later map
completion cannot be silently mistaken for part of the same snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterator, Mapping

import numpy as np

from .atlas import LocalMapAtlas, open_readonly_atlas


PACKED_FORMAT = "hammlet-roman-local-packed-buckets"
PACKED_VERSION = 2


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _safe_child(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} must be a non-empty relative path")
    child = Path(relative)
    if child.is_absolute() or ".." in child.parts:
        raise ValueError(f"{label} must stay below the packed-cache root")
    resolved = (root / child).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} escapes the packed-cache root")
    return resolved


@dataclass(frozen=True)
class PackedBucket:
    """One packed coefficient array and its shared radial grid."""

    index: int
    name: str
    path: Path
    radial_nodes: np.ndarray
    map_ids: np.ndarray
    parameters: np.ndarray
    m_max: int
    point_lens_tail_nodes: np.ndarray | None = None
    storage_rows: np.ndarray | None = None

    def __post_init__(self) -> None:
        nodes = np.asarray(self.radial_nodes, dtype=np.float64)
        ids = np.asarray(self.map_ids, dtype=np.int64)
        parameters = np.asarray(self.parameters, dtype=np.float64)
        if nodes.ndim != 1 or nodes.size < 2 or np.any(np.diff(nodes) <= 0.0):
            raise ValueError("packed radial_nodes must be strictly increasing")
        if ids.ndim != 1 or not ids.size or len(set(map(int, ids))) != len(ids):
            raise ValueError("packed map_ids must be a unique non-empty vector")
        if parameters.shape != (len(ids), 3) or not np.all(np.isfinite(parameters)):
            raise ValueError("packed parameters must have shape (n_map, 3)")
        if int(self.m_max) < 1:
            raise ValueError("packed m_max must be positive")
        if self.storage_rows is not None:
            storage_rows = np.asarray(self.storage_rows, dtype=np.int64)
            if storage_rows.shape != (len(ids),) or np.any(storage_rows < 0):
                raise ValueError("packed storage_rows must match map_ids")
            object.__setattr__(self, "storage_rows", storage_rows)
        if self.point_lens_tail_nodes is not None:
            tail = np.asarray(self.point_lens_tail_nodes, dtype=np.float64)
            if tail.ndim != 1 or not tail.size or np.any(np.diff(tail) <= 0.0):
                raise ValueError("packed tail nodes must be strictly increasing")
            base_count = nodes.size - tail.size
            if base_count < 2 or not np.array_equal(nodes[base_count:], tail):
                raise ValueError("packed tail nodes must be the stored-grid suffix")
            if tail[0] <= nodes[base_count - 1]:
                raise ValueError("packed tail nodes must follow the stored grid")
            object.__setattr__(self, "point_lens_tail_nodes", tail)
        object.__setattr__(self, "radial_nodes", nodes)
        object.__setattr__(self, "map_ids", ids)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "m_max", int(self.m_max))
        object.__setattr__(self, "path", Path(self.path).resolve())

    @property
    def base_n_r(self) -> int:
        if self.point_lens_tail_nodes is None:
            return int(self.radial_nodes.size)
        return int(self.radial_nodes.size - self.point_lens_tail_nodes.size)

    @property
    def n_r(self) -> int:
        return int(self.radial_nodes.size)

    def with_point_lens_tail(
        self, max_radius: float, *, radial_order: int
    ) -> "PackedBucket":
        """Return an in-memory view with the same local tail as the Roman reader."""
        max_radius = float(max_radius)
        if not np.isfinite(max_radius) or max_radius <= self.radial_nodes[-1]:
            return self
        count = int(radial_order) + 1
        if count < 2:
            raise ValueError("radial_order must provide at least two tail nodes")
        epsilon = max(
            1.0e-8,
            1.0e-6 * max(1.0, abs(float(self.radial_nodes[-1]))),
        )
        if max_radius <= self.radial_nodes[-1] + epsilon:
            return self
        tail = np.linspace(
            float(self.radial_nodes[-1]) + epsilon,
            max_radius,
            count,
            dtype=np.float64,
        )
        return PackedBucket(
            index=self.index,
            name=self.name,
            path=self.path,
            radial_nodes=np.concatenate((self.radial_nodes, tail)),
            map_ids=self.map_ids,
            parameters=self.parameters,
            m_max=self.m_max,
            point_lens_tail_nodes=tail,
            storage_rows=self.storage_rows,
        )

    def load_coefficients(self, m_max: int) -> np.ndarray:
        """Load requested modes from a memmap and optionally append the tail."""
        requested = int(m_max)
        if not 0 <= requested <= self.m_max:
            raise ValueError(
                f"requested m_max={requested} is outside packed range 0..{self.m_max}"
            )
        coefficients_map = np.load(self.path, mmap_mode="r", allow_pickle=False)
        expected_tail = (self.base_n_r, self.m_max + 1)
        if coefficients_map.ndim != 3 or coefficients_map.shape[1:] != expected_tail:
            raise ValueError(f"packed coefficient shape is invalid: {self.path}")
        if self.storage_rows is None:
            if coefficients_map.shape[0] != len(self.map_ids):
                raise ValueError(f"packed coefficient row count is invalid: {self.path}")
            selected_rows = slice(None)
        else:
            if np.any(self.storage_rows >= coefficients_map.shape[0]):
                raise ValueError(f"packed storage row is outside the payload: {self.path}")
            selected_rows = self.storage_rows
        # Materialise only the requested mode prefix.  The memmap is closed by
        # dropping the view before a worker returns; the JAX call owns its copy.
        coefficients = np.asarray(
            coefficients_map[selected_rows, ..., : requested + 1]
        )
        del coefficients_map
        if self.point_lens_tail_nodes is not None:
            from .atlas import _append_point_lens_tail

            coefficients = _append_point_lens_tail(
                coefficients,
                self.point_lens_tail_nodes,
                self.parameters,
            )
        return coefficients


@dataclass(frozen=True)
class PackedAtlas:
    """Read-only view of a packed atlas snapshot."""

    path: Path
    buckets: tuple[PackedBucket, ...]
    expected_maps: int
    completed_maps: int
    m_max: int
    metadata: Mapping[str, Any]

    @property
    def kind(self) -> str:
        return "packed-local-cache"

    @property
    def total_maps(self) -> int:
        return int(sum(len(bucket.map_ids) for bucket in self.buckets))

    @property
    def map_ids(self) -> np.ndarray:
        return np.concatenate(
            [np.asarray(bucket.map_ids, dtype=np.int64) for bucket in self.buckets]
        )

    @property
    def map_parameters(self) -> np.ndarray:
        return np.concatenate(
            [np.asarray(bucket.parameters, dtype=np.float64) for bucket in self.buckets],
            axis=0,
        )

    def iter_buckets(self) -> Iterator[PackedBucket]:
        return iter(self.buckets)

    def with_point_lens_tail(
        self, max_radius: float, *, radial_order: int
    ) -> "PackedAtlas":
        buckets = tuple(
            bucket.with_point_lens_tail(max_radius, radial_order=radial_order)
            for bucket in self.buckets
        )
        if all(bucket.point_lens_tail_nodes is None for bucket in buckets):
            return self
        metadata = dict(self.metadata)
        metadata["point_lens_tail"] = {
            "enabled": True,
            "max_radius": float(max_radius),
            "radial_order": int(radial_order),
            "model": "map-specific-point-lens-fallback",
        }
        return PackedAtlas(
            path=self.path,
            buckets=buckets,
            expected_maps=self.expected_maps,
            completed_maps=self.completed_maps,
            m_max=self.m_max,
            metadata=metadata,
        )


def open_packed_atlas(
    path: str | Path, *, max_maps: int | None = None
) -> PackedAtlas:
    """Open a packed snapshot without reading coefficient payloads eagerly."""
    root = Path(path).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError(f"packed cache manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid packed cache manifest: {manifest_path}") from error
    if manifest.get("format") != PACKED_FORMAT or manifest.get("version") != PACKED_VERSION:
        raise ValueError(f"unsupported packed cache format: {manifest_path}")
    raw_buckets = manifest.get("buckets")
    if not isinstance(raw_buckets, list) or not raw_buckets:
        raise ValueError("packed cache has no bucket entries")
    index_path = _safe_child(root, manifest.get("index"), "packed index")
    if not index_path.is_file():
        raise FileNotFoundError(f"packed index does not exist: {index_path}")
    with np.load(index_path, allow_pickle=False) as index:
        bucket_offsets = np.asarray(index["bucket_offsets"], dtype=np.int64)
        node_offsets = np.asarray(index["node_offsets"], dtype=np.int64)
        all_ids = np.asarray(index["map_ids"], dtype=np.int64)
        all_parameters = np.asarray(index["parameters"], dtype=np.float64)
        all_nodes = np.asarray(index["radial_nodes"], dtype=np.float64)
    if bucket_offsets.shape != (len(raw_buckets) + 1,):
        raise ValueError("packed bucket offsets have the wrong shape")
    if node_offsets.shape != (len(raw_buckets) + 1,):
        raise ValueError("packed radial offsets have the wrong shape")
    if all_parameters.shape != (len(all_ids), 3):
        raise ValueError("packed parameter index has the wrong shape")
    if int(bucket_offsets[-1]) != len(all_ids) or int(node_offsets[-1]) != len(all_nodes):
        raise ValueError("packed index offsets do not match their arrays")

    m_max = int(manifest["m_max"])
    buckets: list[PackedBucket] = []
    seen: set[int] = set()
    for expected_index, entry in enumerate(raw_buckets):
        if not isinstance(entry, Mapping):
            raise ValueError(f"packed bucket {expected_index} is not an object")
        index = int(entry.get("index", expected_index))
        if index != expected_index:
            raise ValueError("packed bucket indices must be consecutive")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"packed bucket {index} has no name")
        bucket_path = _safe_child(root, entry.get("path"), f"packed bucket {name!r}")
        if not bucket_path.is_file():
            raise FileNotFoundError(f"packed bucket file does not exist: {bucket_path}")
        map_start, map_stop = map(int, bucket_offsets[index : index + 2])
        node_start, node_stop = map(int, node_offsets[index : index + 2])
        ids = all_ids[map_start:map_stop]
        parameters = all_parameters[map_start:map_stop]
        radial_nodes = all_nodes[node_start:node_stop]
        declared_count = int(entry.get("count", len(ids)))
        declared_n_r = int(entry.get("n_r", len(radial_nodes)))
        if declared_count != len(ids) or declared_n_r != len(radial_nodes):
            raise ValueError(f"packed bucket index metadata disagrees: {bucket_path}")
        overlap = seen.intersection(map(int, ids))
        if overlap:
            raise ValueError(f"packed cache has duplicate map IDs: {sorted(overlap)}")
        seen.update(map(int, ids))
        buckets.append(
            PackedBucket(
                index=index,
                name=name,
                path=bucket_path,
                radial_nodes=radial_nodes,
                map_ids=ids,
                parameters=parameters,
                m_max=m_max,
            )
        )
    if int(manifest.get("n_maps", len(seen))) != len(seen):
        raise ValueError("packed cache n_maps does not match its bucket contents")
    expected = int(manifest.get("expected_maps", len(seen)))
    completed = int(manifest.get("completed_maps", len(seen)))
    atlas = PackedAtlas(
        path=root,
        buckets=tuple(buckets),
        expected_maps=expected,
        completed_maps=completed,
        m_max=m_max,
        metadata=dict(manifest.get("metadata", {})),
    )
    if max_maps is None or int(max_maps) >= atlas.total_maps:
        return atlas
    count = int(max_maps)
    if count < 1:
        raise ValueError("max_maps must be positive")
    selected = set(map(int, np.sort(atlas.map_ids)[:count]))
    selected_buckets: list[PackedBucket] = []
    for bucket in atlas.buckets:
        rows = [
            index for index, map_id in enumerate(bucket.map_ids)
            if int(map_id) in selected
        ]
        if rows:
            selected_buckets.append(
                PackedBucket(
                    index=bucket.index,
                    name=bucket.name,
                    path=bucket.path,
                    radial_nodes=bucket.radial_nodes,
                    map_ids=bucket.map_ids[rows],
                    parameters=bucket.parameters[rows],
                    m_max=bucket.m_max,
                    storage_rows=(
                        np.asarray(rows, dtype=np.int64)
                        if bucket.storage_rows is None
                        else bucket.storage_rows[rows]
                    ),
                )
            )
    metadata = dict(atlas.metadata)
    metadata["max_maps"] = count
    return PackedAtlas(
        path=atlas.path,
        buckets=tuple(selected_buckets),
        expected_maps=atlas.expected_maps,
        completed_maps=atlas.completed_maps,
        m_max=atlas.m_max,
        metadata=metadata,
    )


def validate_packed_source(packed: PackedAtlas, source: LocalMapAtlas) -> None:
    """Verify that a packed snapshot still describes the requested atlas.

    This is intentionally strict.  If more map results become readable after a
    cache was built, the caller must build a new cache instead of silently
    mixing generations.  Only metadata arrays are inspected here; coefficient
    payloads in the production atlas are never opened.
    """
    if packed.metadata.get("source_path"):
        recorded = Path(str(packed.metadata["source_path"])).expanduser().resolve()
        if recorded != source.path.resolve():
            raise ValueError(
                "packed cache was built from a different atlas path: "
                f"{recorded} != {source.path.resolve()}"
            )
    if source.total_maps != packed.completed_maps:
        raise ValueError(
            "packed cache is stale or incomplete: source has "
            f"{source.total_maps} readable maps, cache records "
            f"{packed.completed_maps}"
        )
    recorded_ids = packed.metadata.get("source_map_id_sha256")
    recorded_parameters = packed.metadata.get("source_parameter_sha256")
    if recorded_ids and str(recorded_ids) != _sha256_array(source.map_ids):
        raise ValueError(
            "packed cache map-ID digest differs from the current read-only atlas"
        )
    if recorded_parameters and str(recorded_parameters) != _sha256_array(
        source.map_parameters
    ):
        raise ValueError(
            "packed cache parameter digest differs from the current read-only atlas"
        )


def pack_atlas(
    source: str | Path | LocalMapAtlas,
    output: str | Path,
    *,
    m_max: int | None = None,
    max_maps: int | None = None,
    resume: bool = False,
) -> Path:
    """Pack the currently readable map union into one memmap file per bucket."""
    source_atlas = (
        open_readonly_atlas(source, max_maps=max_maps)
        if not isinstance(source, LocalMapAtlas)
        else source.select_first(max_maps)
    )
    destination = Path(output).expanduser().resolve()
    source_path = source_atlas.path.resolve()
    if destination == source_path or source_path in destination.parents:
        raise ValueError("packed cache must be separate from the source atlas")
    if destination.exists() and any(destination.iterdir()) and not resume:
        raise FileExistsError(
            f"packed cache is not empty; use --resume or choose another path: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    stored_m_max = source_atlas.m_max if m_max is None else int(m_max)
    if not 0 <= stored_m_max <= source_atlas.m_max:
        raise ValueError("requested packed m_max is outside the source atlas range")

    if resume and (destination / "manifest.json").is_file():
        try:
            old_manifest = json.loads(
                (destination / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("cannot resume from an invalid packed manifest") from error
        old_metadata = old_manifest.get("metadata", {})
        if (
            old_manifest.get("format") != PACKED_FORMAT
            or old_manifest.get("version") != PACKED_VERSION
            or int(old_manifest.get("m_max", -1)) != stored_m_max
            or int(old_manifest.get("n_maps", -1)) != source_atlas.total_maps
            or old_metadata.get("source_map_id_sha256")
            != _sha256_array(source_atlas.map_ids)
            or old_metadata.get("source_parameter_sha256")
            != _sha256_array(source_atlas.map_parameters)
        ):
            raise ValueError(
                "existing packed manifest describes a different source snapshot; "
                "choose a new destination"
            )

    bucket_entries: list[dict[str, object]] = []
    bucket_ids: list[np.ndarray] = []
    bucket_parameters: list[np.ndarray] = []
    bucket_nodes: list[np.ndarray] = []
    started = time.perf_counter()
    for bucket_index, bucket in enumerate(source_atlas.iter_buckets()):
        filename = f"bucket-{bucket_index:05d}.npy"
        bucket_path = destination / filename
        expected_ids = np.asarray(bucket.map_ids, dtype=np.int64)
        expected_shape = (
            len(expected_ids),
            len(bucket.radial_nodes),
            stored_m_max + 1,
        )
        can_resume = False
        if resume and bucket_path.is_file():
            try:
                existing = np.load(bucket_path, mmap_mode="r", allow_pickle=False)
                can_resume = existing.shape == expected_shape
                del existing
            except (OSError, ValueError):
                can_resume = False
        if not can_resume:
            temporary = destination / f".{filename}.partial-{time.time_ns()}"
            coefficients = np.lib.format.open_memmap(
                temporary,
                mode="w+",
                dtype=np.complex64,
                shape=expected_shape,
            )
            row_for_id = {int(map_id): row for row, map_id in enumerate(expected_ids)}
            filled = np.zeros(len(expected_ids), dtype=bool)
            for shard in bucket.iter_shards(stored_m_max, append_tail=False):
                shard_ids = np.asarray(shard.map_ids, dtype=np.int64)
                rows = np.asarray([row_for_id[int(map_id)] for map_id in shard_ids])
                local = np.asarray(shard.x_coeff)
                if local.shape != (len(rows), len(bucket.radial_nodes), stored_m_max + 1):
                    raise ValueError(
                        f"source coefficient shape is invalid for bucket {bucket.name!r}"
                    )
                coefficients[rows] = local
                filled[rows] = True
            if not np.all(filled):
                missing = expected_ids[~filled]
                del coefficients
                Path(temporary).unlink(missing_ok=True)
                raise RuntimeError(
                    f"source bucket {bucket.name!r} did not provide map IDs: "
                    f"{missing[:5].tolist()}"
                )
            coefficients.flush()
            del coefficients
            Path(temporary).replace(bucket_path)
        bucket_entries.append(
            {
                "index": bucket_index,
                "name": bucket.name,
                "path": filename,
                "count": int(len(expected_ids)),
                "n_r": int(len(bucket.radial_nodes)),
            }
        )
        bucket_ids.append(expected_ids)
        bucket_parameters.append(np.asarray(bucket.parameters, dtype=np.float64))
        bucket_nodes.append(np.asarray(bucket.radial_nodes, dtype=np.float64))
        print(
            f"[pack] bucket {bucket_index + 1}/{len(source_atlas.buckets)} "
            f"maps={len(expected_ids)} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    all_ids = np.concatenate(bucket_ids)
    all_parameters = np.concatenate(bucket_parameters, axis=0)
    bucket_offsets = np.concatenate(
        ([0], np.cumsum([len(values) for values in bucket_ids], dtype=np.int64))
    )
    node_offsets = np.concatenate(
        ([0], np.cumsum([len(values) for values in bucket_nodes], dtype=np.int64))
    )
    all_nodes = np.concatenate(bucket_nodes)
    temporary_index = destination / f".index.npz.partial-{time.time_ns()}"
    with temporary_index.open("wb") as stream:
        np.savez(
            stream,
            bucket_offsets=bucket_offsets,
            node_offsets=node_offsets,
            map_ids=all_ids,
            parameters=all_parameters,
            radial_nodes=all_nodes,
        )
    temporary_index.replace(destination / "index.npz")

    metadata = {
        "source_path": str(source_path),
        "source_kind": source_atlas.kind,
        "coordinate_frame": str(
            source_atlas.metadata.get("coordinate_frame", "map")
        ),
        "completion_count_exact": bool(
            source_atlas.metadata.get("completion_count_exact", True)
        ),
        "source_metadata": source_atlas.metadata,
        "source_map_id_sha256": _sha256_array(source_atlas.map_ids),
        "source_parameter_sha256": _sha256_array(source_atlas.map_parameters),
        "snapshot_note": "Only map IDs readable when this cache was built are included.",
        "coefficient_storage": "uncompressed-npy-memmap",
    }
    manifest = {
        "format": PACKED_FORMAT,
        "version": PACKED_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "m_max": int(stored_m_max),
        "n_maps": int(source_atlas.total_maps),
        "expected_maps": int(source_atlas.expected_maps),
        "completed_maps": int(source_atlas.completed_maps),
        "bucket_count": len(bucket_entries),
        "index": "index.npz",
        "buckets": bucket_entries,
        "metadata": metadata,
    }
    temporary_manifest = destination / f".manifest.json.partial-{time.time_ns()}"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(destination / "manifest.json")
    print(
        f"[pack] completed maps={source_atlas.total_maps} "
        f"buckets={len(bucket_entries)} elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return destination
