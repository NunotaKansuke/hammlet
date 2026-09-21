"""Read-only views of Hammlet production map outputs.

The production builders have two useful on-disk shapes for a local Roman
test:

* a normal ``hammlet-bucketed-fourier-maps`` directory;
* a restartable ``tasks.json`` + ``results/map-*`` run directory.

This adapter deliberately reads only ``x_coeff`` and its extension.  The
reconstruction-error and certified-error sidecars are not opened, because the
local Roman profile is a single-resolution search without a chi-square
guarantee.  No method in this module writes to the atlas path.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
from typing import Any, Iterator, Mapping, MutableMapping, Sequence

import numpy as np


_POLAR_FORMATS = {"hammlet-fourier-maps", "hammlet-fourier-atlas"}
_BUCKETED_FORMAT = "hammlet-bucketed-fourier-maps"
_MAP_DIRECTORY = re.compile(r"^map-(?P<map_id>[0-9]+)$")


def _add_timing(
    timing: MutableMapping[str, float | int] | None,
    key: str,
    value: float | int,
) -> None:
    if timing is None:
        return
    timing[key] = timing.get(key, 0) + value


@dataclass(frozen=True)
class LocalAtlasShard:
    """A coefficient-only shard used by the no-certificate scanner."""

    map_ids: np.ndarray
    parameters: np.ndarray
    x_coeff: np.ndarray


@dataclass(frozen=True)
class _PolarSource:
    root: Path
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class LocalAtlasBucket:
    """One radial grid plus one or more read-only polar sources."""

    name: str
    radial_nodes: np.ndarray
    map_ids: np.ndarray
    parameters: np.ndarray
    m_max: int
    core_m_max: int
    sources: tuple[_PolarSource, ...]
    selected_ids: frozenset[int] | None = None
    point_lens_tail_nodes: np.ndarray | None = None

    def with_selected_ids(self, map_ids: Sequence[int]) -> "LocalAtlasBucket":
        selected = frozenset(int(map_id) for map_id in map_ids)
        actual = {int(map_id) for map_id in self.map_ids}
        if not selected.issubset(actual):
            missing = sorted(selected.difference(actual))
            raise KeyError(f"map IDs are not present in bucket {self.name!r}: {missing}")
        rows = [
            index
            for index, map_id in enumerate(self.map_ids)
            if int(map_id) in selected
        ]
        return LocalAtlasBucket(
            name=self.name,
            radial_nodes=self.radial_nodes,
            map_ids=np.asarray(self.map_ids[rows], dtype=np.int64),
            parameters=np.asarray(self.parameters[rows], dtype=np.float64),
            m_max=self.m_max,
            core_m_max=self.core_m_max,
            sources=self.sources,
            selected_ids=selected,
            point_lens_tail_nodes=self.point_lens_tail_nodes,
        )

    def iter_shards(
        self,
        m_max: int,
        *,
        timing: MutableMapping[str, float | int] | None = None,
        append_tail: bool = True,
    ) -> Iterator[LocalAtlasShard]:
        requested_m_max = int(m_max)
        if not 0 <= requested_m_max <= self.m_max:
            raise ValueError(
                f"requested m_max={requested_m_max} is outside the stored range"
            )
        parameter_index = {
            int(map_id): index for index, map_id in enumerate(self.map_ids)
        }
        allowed = self.selected_ids
        for source in self.sources:
            source_nodes = np.load(source.root / "radial_nodes.npy", mmap_mode="r")
            stored_nodes = self.radial_nodes[: source_nodes.size]
            if source_nodes.shape != stored_nodes.shape or not np.array_equal(
                source_nodes, stored_nodes
            ):
                raise ValueError(
                    f"radial nodes differ inside bucket {self.name!r}: {source.root}"
                )
            for entry in _manifest_shards(source.manifest, source.root):
                shard_started = time.perf_counter()
                shard_path = source.root / str(entry["path"])
                ids = np.asarray(
                    np.load(shard_path / "map_ids.npy", mmap_mode="r"),
                    dtype=np.int64,
                )
                rows = np.arange(len(ids), dtype=np.int64)
                if allowed is not None:
                    rows = np.asarray(
                        [
                            index
                            for index, map_id in enumerate(ids)
                            if int(map_id) in allowed
                        ],
                        dtype=np.int64,
                    )
                if not rows.size:
                    _add_timing(timing, "shards_filtered_empty", 1)
                    continue
                if np.any(
                    [int(map_id) not in parameter_index for map_id in ids[rows]]
                ):
                    raise ValueError(
                        f"source shard contains an unknown map ID: {shard_path}"
                    )
                selected_ids = ids[rows]
                parameters = np.asarray(
                    [parameter_index[int(map_id)] for map_id in selected_ids],
                    dtype=np.int64,
                )
                map_parameters = np.asarray(
                    self.parameters[parameters], dtype=np.float64
                )
                coefficient_started = time.perf_counter()
                coefficients = _load_x_coefficients(
                    shard_path,
                    requested_m_max,
                    self.core_m_max,
                    rows,
                )
                _add_timing(
                    timing,
                    "coefficient_load_seconds",
                    time.perf_counter() - coefficient_started,
                )
                if append_tail and self.point_lens_tail_nodes is not None:
                    tail_started = time.perf_counter()
                    coefficients = _append_point_lens_tail(
                        coefficients,
                        self.point_lens_tail_nodes,
                        map_parameters,
                    )
                    _add_timing(
                        timing,
                        "point_lens_tail_seconds",
                        time.perf_counter() - tail_started,
                    )
                _add_timing(
                    timing,
                    "shard_seconds",
                    time.perf_counter() - shard_started,
                )
                _add_timing(timing, "shards_yielded", 1)
                yield LocalAtlasShard(
                    map_ids=selected_ids,
                    parameters=map_parameters,
                    x_coeff=coefficients,
                )

    def with_point_lens_tail(
        self, max_radius: float, *, radial_order: int
    ) -> "LocalAtlasBucket":
        """Extend this view in memory with a map-specific point-lens tail."""
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
        tail_nodes = np.linspace(
            float(self.radial_nodes[-1]) + epsilon,
            max_radius,
            count,
            dtype=np.float64,
        )
        return LocalAtlasBucket(
            name=self.name,
            radial_nodes=np.concatenate((self.radial_nodes, tail_nodes)),
            map_ids=self.map_ids,
            parameters=self.parameters,
            m_max=self.m_max,
            core_m_max=self.core_m_max,
            sources=self.sources,
            selected_ids=self.selected_ids,
            point_lens_tail_nodes=tail_nodes,
        )


class LocalMapAtlas:
    """A deterministic, coefficient-only atlas view for local experiments."""

    def __init__(
        self,
        path: Path,
        buckets: Sequence[LocalAtlasBucket],
        *,
        expected_maps: int,
        completed_maps: int,
        kind: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not buckets:
            raise ValueError("atlas contains no completed map buckets")
        normalized = tuple(buckets)
        all_ids = [np.asarray(bucket.map_ids, dtype=np.int64) for bucket in normalized]
        map_ids = np.concatenate(all_ids)
        if len(set(map(int, map_ids))) != len(map_ids):
            raise ValueError("atlas contains duplicate map IDs")
        m_values = {bucket.m_max for bucket in normalized}
        if len(m_values) != 1:
            raise ValueError("all local atlas buckets must use the same m_max")
        parameter_values = [
            np.asarray(bucket.parameters, dtype=np.float64) for bucket in normalized
        ]
        parameters = np.concatenate(parameter_values, axis=0)
        if parameters.shape != (len(map_ids), 3):
            raise ValueError("atlas parameter rows must have shape (n_map, 3)")
        self.path = path.resolve()
        self._buckets = normalized
        self._map_ids = map_ids
        self._parameters = parameters
        self._parameter_index = {
            int(map_id): index for index, map_id in enumerate(map_ids)
        }
        self._bucket_for_map = {
            int(map_id): bucket
            for bucket in normalized
            for map_id in bucket.map_ids
        }
        self.expected_maps = int(expected_maps)
        self.completed_maps = int(completed_maps)
        self.kind = str(kind)
        self.metadata = dict(metadata or {})

    @property
    def m_max(self) -> int:
        return self._buckets[0].m_max

    @property
    def total_maps(self) -> int:
        return int(self._map_ids.size)

    @property
    def n_maps(self) -> int:
        return self.total_maps

    @property
    def map_ids(self) -> np.ndarray:
        return np.asarray(self._map_ids, dtype=np.int64)

    @property
    def map_parameters(self) -> np.ndarray:
        return np.asarray(self._parameters, dtype=np.float64)

    @property
    def buckets(self) -> tuple[LocalAtlasBucket, ...]:
        return self._buckets

    def __iter__(self) -> Iterator[LocalAtlasBucket]:
        return iter(self._buckets)

    def iter_buckets(self) -> Iterator[LocalAtlasBucket]:
        return iter(self._buckets)

    def parameters_for(self, map_id: int) -> np.ndarray:
        try:
            row = self._parameter_index[int(map_id)]
        except KeyError as error:
            raise KeyError(f"map ID is not present in the atlas: {map_id}") from error
        return np.asarray(self._parameters[row], dtype=np.float64)

    def coefficient_row(
        self, map_id: int, *, m_max: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            bucket = self._bucket_for_map[int(map_id)]
        except KeyError as error:
            raise KeyError(f"map ID is not present in the atlas: {map_id}") from error
        for shard in bucket.iter_shards(m_max):
            rows = np.flatnonzero(shard.map_ids == int(map_id))
            if rows.size:
                row = int(rows[0])
                return (
                    np.asarray(bucket.radial_nodes, dtype=np.float64),
                    np.asarray(shard.parameters[row], dtype=np.float64),
                    np.asarray(shard.x_coeff[row]),
                )
        raise RuntimeError(f"map ID disappeared while reading the atlas: {map_id}")

    def select_first(self, count: int | None) -> "LocalMapAtlas":
        """Return a view limited to the first map IDs without copying map data."""
        if count is None:
            return self
        count = int(count)
        if count < 1:
            raise ValueError("max_maps must be positive")
        if count >= self.total_maps:
            return self
        selected = np.sort(self._map_ids)[:count]
        selected_set = {int(map_id) for map_id in selected}
        buckets = [
            bucket.with_selected_ids(
                [
                    map_id
                    for map_id in bucket.map_ids
                    if int(map_id) in selected_set
                ]
            )
            for bucket in self._buckets
        ]
        buckets = [bucket for bucket in buckets if bucket.map_ids.size]
        metadata = dict(self.metadata)
        metadata["max_maps"] = count
        return LocalMapAtlas(
            self.path,
            buckets,
            expected_maps=self.expected_maps,
            completed_maps=self.completed_maps,
            kind=self.kind,
            metadata=metadata,
        )

    def with_point_lens_tail(
        self, max_radius: float, *, radial_order: int
    ) -> "LocalMapAtlas":
        """Return a read-only view with a point-lens tail added in memory.

        The stored atlas remains untouched.  This is useful for a full GULLS
        light curve when later observing seasons extend beyond the finite
        radial support of the stored binary-lens coefficients.
        """
        buckets = [
            bucket.with_point_lens_tail(max_radius, radial_order=radial_order)
            for bucket in self._buckets
        ]
        if all(
            bucket.point_lens_tail_nodes is None for bucket in buckets
        ):
            return self
        metadata = dict(self.metadata)
        metadata["point_lens_tail"] = {
            "enabled": True,
            "max_radius": float(max_radius),
            "radial_order": int(radial_order),
            "model": "map-specific-point-lens-fallback",
        }
        return LocalMapAtlas(
            self.path,
            buckets,
            expected_maps=self.expected_maps,
            completed_maps=self.completed_maps,
            kind=self.kind,
            metadata=metadata,
        )


def open_readonly_atlas(path: str | Path, *, max_maps: int | None = None) -> LocalMapAtlas:
    """Open a map directory without ever writing to it.

    A directory containing both ``run-003`` and ``run-004`` is treated as the
    handoff's combined parent.  ``run-004`` is a restart/continuation of
    ``run-003``: its task list is the authoritative task universe, while the
    readable map coefficients are the union of completed IDs from both runs.
    For ``run-003``, only atomic ``parts/part-*/`` directories with a manifest
    are included; temporary dot-directories are ignored.
    """
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"atlas directory does not exist: {root}")
    if (root / "run-003").is_dir() and (root / "run-004").is_dir():
        # Apply the smoke-test limit only after the continuation union is
        # assembled.  Limiting each source independently would report the
        # wrong completed count and could hide IDs that should be selected.
        run003 = _open_path(root / "run-003", max_maps=None)
        run004 = _open_path(root / "run-004", max_maps=None)
        parts = [
            run003,
            run004,
        ]
        seed_atlas = _from_seed_index(
            root / "run-004", exclude_ids=set(map(int, run003.map_ids))
        )
        if seed_atlas is not None:
            parts.append(seed_atlas)
        atlas = _combine_continuation_atlases(
            root, parts, kind="combined-generations"
        )
    else:
        atlas = _open_path(root, max_maps=None)
    return atlas.select_first(max_maps)


def _open_path(root: Path, *, max_maps: int | None = None) -> LocalMapAtlas:
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        if manifest.get("format") == _BUCKETED_FORMAT:
            return _from_bucketed(root, manifest)
        if manifest.get("format") in _POLAR_FORMATS:
            return _from_polar(root, manifest, name=root.name)
        raise ValueError(f"unsupported atlas manifest format: {manifest_path}")
    if (root / "tasks.json").exists() or (root / "results").is_dir():
        return _from_standalone(root, max_completed=max_maps)
    if (root / "parts").is_dir():
        return _from_parts(root)
    if root.name == "results" and (root.parent / "tasks.json").exists():
        return _from_standalone(root.parent)
    raise FileNotFoundError(
        f"atlas must contain manifest.json or tasks.json/results: {root}"
    )


def _from_polar(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    name: str,
    parameters_override: np.ndarray | None = None,
    ids_override: np.ndarray | None = None,
) -> LocalMapAtlas:
    _validate_polar_manifest(manifest, root)
    radial_nodes = np.load(root / "radial_nodes.npy", mmap_mode="r")
    map_ids = np.asarray(
        np.load(root / "map_ids.npy", mmap_mode="r")
        if ids_override is None
        else ids_override,
        dtype=np.int64,
    )
    parameters = np.asarray(
        np.load(root / "map_parameters.npy", mmap_mode="r")
        if parameters_override is None
        else parameters_override,
        dtype=np.float64,
    )
    bucket = LocalAtlasBucket(
        name=name,
        radial_nodes=radial_nodes,
        map_ids=map_ids,
        parameters=parameters,
        m_max=_manifest_m_max(manifest),
        core_m_max=_manifest_core_m_max(manifest),
        sources=(_PolarSource(root, manifest),),
    )
    declared = int(manifest.get("n_maps", len(map_ids)))
    if declared != len(map_ids):
        raise ValueError(f"polar atlas n_maps does not match its map_ids: {root}")
    return LocalMapAtlas(
        root,
        [bucket],
        expected_maps=len(map_ids),
        completed_maps=len(map_ids),
        kind="manifest",
    )


def _from_bucketed(root: Path, manifest: Mapping[str, Any]) -> LocalMapAtlas:
    if manifest.get("version") != 1:
        raise ValueError(f"unsupported bucketed atlas version: {root}")
    entries = manifest.get("buckets")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"bucketed atlas has no buckets: {root}")
    buckets: list[LocalAtlasBucket] = []
    seen: set[int] = set()
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"bucket entry {index} is not an object: {root}")
        name = raw_entry.get("name")
        relative = raw_entry.get("path")
        if not isinstance(name, str) or not name:
            raise ValueError(f"bucket entry {index} has no name: {root}")
        child = _safe_child(root, relative, f"bucket {name!r}")
        child_manifest = _read_json(child / "manifest.json")
        if child_manifest.get("format") not in _POLAR_FORMATS:
            raise ValueError(f"bucket {name!r} is not a polar atlas: {child}")
        bucket_atlas = _polar_bucket(child, child_manifest, name)
        overlap = seen.intersection(map(int, bucket_atlas.map_ids))
        if overlap:
            raise ValueError(f"duplicate map IDs in bucketed atlas: {sorted(overlap)}")
        seen.update(map(int, bucket_atlas.map_ids))
        buckets.append(bucket_atlas)
    declared = int(manifest.get("n_maps", len(seen)))
    if declared != len(seen):
        raise ValueError(f"bucketed atlas n_maps does not match contents: {root}")
    return LocalMapAtlas(
        root,
        buckets,
        expected_maps=declared,
        completed_maps=declared,
        kind="bucketed-manifest",
    )


def _from_parts(root: Path) -> LocalMapAtlas:
    part_atlases: list[LocalMapAtlas] = []
    partial = 0
    part_directories = sorted(
        part
        for part in (root / "parts").iterdir()
        if part.is_dir()
        and (part.name.startswith("part-") or part.name.startswith(".part-"))
    )
    for part in part_directories:
        if not (part / "manifest.json").exists():
            partial += 1
            continue
        manifest = _read_json(part / "manifest.json")
        if manifest.get("format") != _BUCKETED_FORMAT:
            raise ValueError(f"unsupported completed part format: {part}")
        part_atlases.append(_from_bucketed(part, manifest))
    if not part_atlases:
        raise FileNotFoundError(f"no completed map parts found under: {root / 'parts'}")
    atlas = _combine_atlases(root, part_atlases, kind="distributed-parts")
    metadata = dict(atlas.metadata)
    metadata["partial_part_directories"] = partial
    metadata["completion_count_exact"] = partial == 0
    return LocalMapAtlas(
        atlas.path,
        atlas.buckets,
        expected_maps=atlas.expected_maps,
        completed_maps=atlas.completed_maps,
        kind=atlas.kind,
        metadata=metadata,
    )


def _from_standalone(
    root: Path, *, max_completed: int | None = None
) -> LocalMapAtlas:
    tasks_path = root / "tasks.json"
    if not tasks_path.exists() and root.name == "results":
        root = root.parent
        tasks_path = root / "tasks.json"
    if not tasks_path.exists():
        raise FileNotFoundError(f"standalone map run has no tasks.json: {root}")
    payload = _read_json(tasks_path)
    raw_tasks = payload.get("tasks") if isinstance(payload, Mapping) else payload
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError(f"tasks.json must contain a non-empty task list: {tasks_path}")
    tasks: dict[int, tuple[str, np.ndarray]] = {}
    order: list[int] = []
    for index, raw_task in enumerate(raw_tasks):
        if not isinstance(raw_task, Mapping):
            raise ValueError(f"task {index} is not an object: {tasks_path}")
        map_id = _integer(raw_task.get("map_id"), f"task {index} map_id")
        if map_id in tasks:
            raise ValueError(f"tasks.json contains duplicate map ID {map_id}")
        try:
            parameters = np.asarray(
                [
                    float(raw_task["logs"]),
                    float(raw_task["logq"]),
                    float(raw_task["logrho"]),
                ],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"task {map_id} has invalid map parameters") from error
        if not np.all(np.isfinite(parameters)):
            raise ValueError(f"task {map_id} has non-finite map parameters")
        bucket_name = str(raw_task.get("bucket", "all"))
        tasks[map_id] = (bucket_name, parameters)
        order.append(map_id)

    results_root = root / "results"
    if not results_root.is_dir():
        results_root = root if root.name == "results" else results_root
    completed: list[tuple[int, Path, Mapping[str, Any]]] = []
    ignored = 0
    for result in sorted(results_root.glob("map-*")):
        if not result.is_dir():
            continue
        match = _MAP_DIRECTORY.fullmatch(result.name)
        if match is None:
            ignored += 1
            continue
        map_id = int(match.group("map_id"))
        manifest_path = result / "manifest.json"
        if not manifest_path.exists():
            # A missing manifest is a partial/in-progress result and is not a
            # map that this local search is allowed to read.
            ignored += 1
            continue
        if max_completed is not None and len(completed) >= int(max_completed):
            break
        if map_id not in tasks:
            ignored += 1
            continue
        manifest = _read_json(manifest_path)
        if manifest.get("format") not in _POLAR_FORMATS or manifest.get("version") != 1:
            raise ValueError(f"unsupported completed map manifest: {manifest_path}")
        if int(manifest.get("n_maps", 0)) != 1:
            raise ValueError(f"standalone result is not one map: {manifest_path}")
        completed.append((map_id, result, manifest))
    if not completed:
        raise FileNotFoundError(f"no completed map results found under: {results_root}")
    completed.sort(key=lambda item: item[0])

    grouped: dict[str, list[tuple[int, Path, Mapping[str, Any]]]] = {}
    first_order: list[str] = []
    for item in completed:
        bucket_name = tasks[item[0]][0]
        if bucket_name not in grouped:
            grouped[bucket_name] = []
            first_order.append(bucket_name)
        grouped[bucket_name].append(item)

    buckets: list[LocalAtlasBucket] = []
    for bucket_name in sorted(first_order):
        items = grouped[bucket_name]
        first_root = items[0][1]
        first_manifest = items[0][2]
        _validate_polar_manifest(first_manifest, first_root)
        radial_nodes = np.load(first_root / "radial_nodes.npy", mmap_mode="r")
        m_max = _manifest_m_max(first_manifest)
        core_m_max = _manifest_core_m_max(first_manifest)
        sources: list[_PolarSource] = []
        map_ids: list[int] = []
        parameters: list[np.ndarray] = []
        for map_id, result, manifest in items:
            _validate_polar_manifest(manifest, result)
            if _manifest_m_max(manifest) != m_max:
                raise ValueError(f"map results disagree on m_max: {result}")
            declared_n_r = int(manifest.get("n_r", radial_nodes.size))
            if declared_n_r != radial_nodes.size:
                raise ValueError(
                    f"radial node count differs inside standalone bucket {bucket_name!r}: {result}"
                )
            if _manifest_core_m_max(manifest) != core_m_max:
                raise ValueError(f"map results disagree on core_m_max: {result}")
            map_ids.append(map_id)
            parameters.append(tasks[map_id][1])
            sources.append(_PolarSource(result, manifest))
        buckets.append(
            LocalAtlasBucket(
                name=bucket_name,
                radial_nodes=radial_nodes,
                map_ids=np.asarray(map_ids, dtype=np.int64),
                parameters=np.asarray(parameters, dtype=np.float64),
                m_max=m_max,
                core_m_max=core_m_max,
                sources=tuple(sources),
            )
        )

    metadata = {
        "tasks_path": str(tasks_path.resolve()),
        "results_path": str(results_root.resolve()),
        "ignored_result_directories": ignored,
        "task_order_maps": len(order),
        "completion_count_exact": max_completed is None,
    }
    config_path = root / "maps-config.json"
    if config_path.is_file():
        try:
            config = _read_json(config_path)
        except (OSError, ValueError):
            config = {}
        if isinstance(config, Mapping):
            metadata["coordinate_frame"] = str(
                config.get("coordinate_frame", "map")
            )
            metadata["maps_config_path"] = str(config_path.resolve())
    return LocalMapAtlas(
        root,
        buckets,
        expected_maps=len(tasks),
        completed_maps=len(completed),
        kind="standalone-results",
        metadata=metadata,
    )


def _from_seed_index(
    root: Path, *, exclude_ids: set[int] | None = None
) -> LocalMapAtlas | None:
    """Read completed seed shards recorded by a restart's seed index.

    A restart can preserve coefficient shards inside an otherwise incomplete
    ``.part-*`` directory.  Those shards have no atomic part manifest, so the
    normal part reader intentionally ignores them.  ``seed/index.json`` is the
    builder's explicit completed-row index for this case; we use it as a
    read-only source and exclude IDs already supplied by completed parts.
    """
    seed_path = root / "seed" / "index.json"
    if not seed_path.exists():
        return None
    payload = _read_json(seed_path)
    raw_maps = payload.get("maps")
    if not isinstance(raw_maps, Mapping) or not raw_maps:
        raise ValueError(f"seed index has no map entries: {seed_path}")

    tasks_path = root / "tasks.json"
    if not tasks_path.exists():
        raise FileNotFoundError(
            f"seed index cannot resolve map parameters without tasks.json: {root}"
        )
    task_payload = _read_json(tasks_path)
    raw_tasks = (
        task_payload.get("tasks")
        if isinstance(task_payload, Mapping)
        else task_payload
    )
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError(f"tasks.json must contain a non-empty task list: {tasks_path}")
    task_parameters: dict[int, np.ndarray] = {}
    for index, raw_task in enumerate(raw_tasks):
        if not isinstance(raw_task, Mapping):
            raise ValueError(f"task {index} is not an object: {tasks_path}")
        map_id = _integer(raw_task.get("map_id"), f"task {index} map_id")
        if map_id in task_parameters:
            raise ValueError(f"tasks.json contains duplicate map ID {map_id}")
        try:
            parameters = np.asarray(
                [
                    float(raw_task["logs"]),
                    float(raw_task["logq"]),
                    float(raw_task["logrho"]),
                ],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"task {map_id} has invalid map parameters") from error
        if not np.all(np.isfinite(parameters)):
            raise ValueError(f"task {map_id} has non-finite map parameters")
        task_parameters[map_id] = parameters

    source_root_raw = payload.get("source_root")
    source_root = (
        Path(source_root_raw).expanduser().resolve()
        if isinstance(source_root_raw, str) and source_root_raw
        else (root.parent / "run-003").resolve()
    )
    excluded = set() if exclude_ids is None else {int(map_id) for map_id in exclude_ids}
    groups: dict[Path, list[dict[str, Any]]] = {}
    included_ids: set[int] = set()
    excluded_count = 0
    for index, raw_entry in raw_maps.items():
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"seed map entry {index} is not an object: {seed_path}")
        map_id = _integer(raw_entry.get("map_id", index), f"seed map {index}")
        if map_id in included_ids:
            raise ValueError(f"seed index contains duplicate map ID {map_id}")
        if map_id in excluded:
            excluded_count += 1
            continue
        if map_id not in task_parameters:
            raise ValueError(f"seed map {map_id} is absent from tasks.json")
        shard = _resolve_seed_path(
            raw_entry.get("shard"),
            source_root=source_root,
            seed_path=seed_path,
            label=f"seed map {map_id} shard",
        )
        radial_nodes = _resolve_seed_path(
            raw_entry.get("radial_nodes"),
            source_root=source_root,
            seed_path=seed_path,
            label=f"seed map {map_id} radial_nodes",
        )
        if not shard.is_dir():
            raise FileNotFoundError(f"seed shard directory does not exist: {shard}")
        if not radial_nodes.is_file():
            raise FileNotFoundError(f"seed radial nodes do not exist: {radial_nodes}")
        if not (shard / "map_ids.npy").is_file() or not (shard / "x_coeff.npy").is_file():
            raise FileNotFoundError(f"seed shard is missing coefficient files: {shard}")
        bucket_root = radial_nodes.parent.resolve()
        try:
            relative_shard = shard.resolve().relative_to(bucket_root)
        except ValueError as error:
            raise ValueError(
                f"seed shard is not below its radial-node bucket: {shard}"
            ) from error
        groups.setdefault(bucket_root, []).append(
            {
                "map_id": map_id,
                "parameters": task_parameters[map_id],
                "shard": shard.resolve(),
                "relative_shard": relative_shard,
                "radial_nodes": radial_nodes.resolve(),
            }
        )
        included_ids.add(map_id)

    if not included_ids:
        return None

    buckets: list[LocalAtlasBucket] = []
    partial_count = 0
    complete_count = 0
    inferred_limits: tuple[int, int] | None = None
    for bucket_root in sorted(groups):
        entries = sorted(groups[bucket_root], key=lambda item: int(item["map_id"]))
        radial_paths = {Path(entry["radial_nodes"]) for entry in entries}
        if len(radial_paths) != 1:
            raise ValueError(f"seed bucket has inconsistent radial nodes: {bucket_root}")
        radial_path = next(iter(radial_paths))
        radial_nodes = np.load(radial_path, mmap_mode="r")
        source_shards = sorted(
            {Path(entry["relative_shard"]) for entry in entries},
            key=lambda value: value.as_posix(),
        )
        manifest_shards = [
            {"path": relative.as_posix(), "count": 1}
            for relative in source_shards
        ]
        first_shard = bucket_root / source_shards[0]
        core = np.load(first_shard / "x_coeff.npy", mmap_mode="r")
        if core.ndim != 3:
            raise ValueError(f"seed coefficient array is not 3D: {first_shard}")
        core_m_max = int(core.shape[-1] - 1)
        extension_path = first_shard / "x_coeff_extension.npy"
        extension_modes = 0
        if extension_path.exists():
            extension = np.load(extension_path, mmap_mode="r")
            if extension.ndim != 3:
                raise ValueError(f"seed extension array is not 3D: {first_shard}")
            extension_modes = int(extension.shape[-1])
        limits = (core_m_max + extension_modes, core_m_max)
        if inferred_limits is None:
            inferred_limits = limits
        elif limits != inferred_limits:
            raise ValueError("seed coefficient shards disagree on mode limits")
        for entry in entries:
            if "/.part-" in str(entry["shard"]):
                partial_count += 1
            else:
                complete_count += 1
        buckets.append(
            LocalAtlasBucket(
                name=f"seed/{bucket_root.name}",
                radial_nodes=radial_nodes,
                map_ids=np.asarray(
                    [int(entry["map_id"]) for entry in entries], dtype=np.int64
                ),
                parameters=np.asarray(
                    [entry["parameters"] for entry in entries], dtype=np.float64
                ),
                m_max=limits[0],
                core_m_max=limits[1],
                sources=(
                    _PolarSource(
                        bucket_root,
                        {
                            "format": "hammlet-fourier-maps",
                            "version": 1,
                            "n_maps": len(entries),
                            "n_r": int(radial_nodes.size),
                            "m_max": limits[0],
                            "core_m_max": limits[1],
                            "shards": manifest_shards,
                        },
                    ),
                ),
            )
        )

    metadata = {
        "seed_index_path": str(seed_path.resolve()),
        "seed_source_root": str(source_root),
        "seed_index_maps": len(raw_maps),
        "seed_index_included_maps": len(included_ids),
        "seed_index_excluded_existing_maps": excluded_count,
        "seed_index_partial_maps": partial_count,
        "seed_index_complete_part_maps": complete_count,
        "completion_count_exact": True,
    }
    return LocalMapAtlas(
        seed_path.parent,
        buckets,
        expected_maps=len(included_ids),
        completed_maps=len(included_ids),
        kind="seed-index",
        metadata=metadata,
    )


def _polar_bucket(
    root: Path, manifest: Mapping[str, Any], name: str
) -> LocalAtlasBucket:
    _validate_polar_manifest(manifest, root)
    map_ids = np.asarray(np.load(root / "map_ids.npy", mmap_mode="r"), dtype=np.int64)
    parameters = np.asarray(
        np.load(root / "map_parameters.npy", mmap_mode="r"), dtype=np.float64
    )
    if len(map_ids) != len(parameters) or parameters.shape != (len(map_ids), 3):
        raise ValueError(f"invalid map-parameter arrays in {root}")
    declared = int(manifest.get("n_maps", len(map_ids)))
    if declared != len(map_ids):
        raise ValueError(f"polar bucket n_maps does not match arrays: {root}")
    return LocalAtlasBucket(
        name=name,
        radial_nodes=np.load(root / "radial_nodes.npy", mmap_mode="r"),
        map_ids=map_ids,
        parameters=parameters,
        m_max=_manifest_m_max(manifest),
        core_m_max=_manifest_core_m_max(manifest),
        sources=(_PolarSource(root, manifest),),
    )


def _combine_continuation_atlases(
    path: Path, atlases: Sequence[LocalMapAtlas], *, kind: str
) -> LocalMapAtlas:
    """Combine a completed run with its restart without double-counting tasks.

    The restart's ``tasks.json`` describes the full grid.  The earlier run is
    a completed subset of that same grid, so its map count must not be added to
    the restart's task count when reporting missing maps.
    """
    combined = _combine_atlases(path, atlases, kind=kind)
    task_atlases = [
        atlas
        for atlas in atlases
        if atlas.kind == "standalone-results"
        and "task_order_maps" in atlas.metadata
    ]
    if len(task_atlases) != 1:
        return combined
    task_atlas = task_atlases[0]
    expected = int(task_atlas.expected_maps)
    if combined.total_maps > expected:
        raise ValueError(
            "continuation atlas has more completed map IDs than its task universe"
        )
    metadata = dict(combined.metadata)
    metadata.update(
        {
            "continuation": True,
            "task_universe_source": task_atlas.metadata.get("tasks_path"),
            "task_universe_maps": expected,
            "completed_union_maps": combined.total_maps,
            "missing_task_maps": expected - combined.total_maps,
        }
    )
    seed_atlases = [
        atlas for atlas in atlases if atlas.kind == "seed-index"
    ]
    if seed_atlases:
        metadata["seed_index"] = dict(seed_atlases[0].metadata)
    return LocalMapAtlas(
        path,
        combined.buckets,
        expected_maps=expected,
        completed_maps=combined.total_maps,
        kind=combined.kind,
        metadata=metadata,
    )


def _combine_atlases(
    path: Path, atlases: Sequence[LocalMapAtlas], *, kind: str
) -> LocalMapAtlas:
    raw_buckets: list[LocalAtlasBucket] = []
    seen: set[int] = set()
    for atlas_index, atlas in enumerate(atlases):
        for bucket in atlas.buckets:
            overlap = seen.intersection(map(int, bucket.map_ids))
            if overlap:
                raise ValueError(f"combined atlas has duplicate map IDs: {sorted(overlap)}")
            seen.update(map(int, bucket.map_ids))
            raw_buckets.append(
                LocalAtlasBucket(
                    name=f"source{atlas_index}/{bucket.name}",
                    radial_nodes=bucket.radial_nodes,
                    map_ids=bucket.map_ids,
                    parameters=bucket.parameters,
                    m_max=bucket.m_max,
                    core_m_max=bucket.core_m_max,
                    sources=bucket.sources,
                    selected_ids=bucket.selected_ids,
                )
            )
    buckets = _coalesce_equal_radial_buckets(raw_buckets)
    metadata = {
        "sources": [str(atlas.path) for atlas in atlases],
        "source_kinds": [atlas.kind for atlas in atlases],
        "completion_count_exact": all(
            bool(atlas.metadata.get("completion_count_exact", True))
            for atlas in atlases
        ),
    }
    return LocalMapAtlas(
        path,
        buckets,
        expected_maps=sum(atlas.expected_maps for atlas in atlases),
        completed_maps=sum(atlas.completed_maps for atlas in atlases),
        kind=kind,
        metadata=metadata,
    )


def _coalesce_equal_radial_buckets(
    buckets: Sequence[LocalAtlasBucket],
) -> list[LocalAtlasBucket]:
    """Merge views that share the exact same radial grid.

    A restart can place maps from one radial grid in both an immutable part and
    standalone result directories.  They can share one kernel binding and one
    JAX scan batch; only their coefficient sources need to remain separate.
    """
    grouped: dict[tuple[int, int, bytes], int] = {}
    merged: list[LocalAtlasBucket] = []
    for bucket in buckets:
        nodes = np.asarray(bucket.radial_nodes)
        key = (bucket.m_max, bucket.core_m_max, nodes.tobytes())
        target = grouped.get(key)
        if target is None:
            grouped[key] = len(merged)
            merged.append(bucket)
            continue
        previous = merged[target]
        if previous.point_lens_tail_nodes is None:
            tail_nodes = bucket.point_lens_tail_nodes
        elif bucket.point_lens_tail_nodes is None:
            tail_nodes = previous.point_lens_tail_nodes
        elif np.array_equal(
            previous.point_lens_tail_nodes, bucket.point_lens_tail_nodes
        ):
            tail_nodes = previous.point_lens_tail_nodes
        else:
            raise ValueError("equal radial grids have different tail nodes")
        selected_ids: frozenset[int] | None
        if previous.selected_ids is None and bucket.selected_ids is None:
            selected_ids = None
        else:
            selected_ids = frozenset(
                (previous.selected_ids or frozenset()).union(
                    bucket.selected_ids or frozenset()
                )
            )
        merged[target] = LocalAtlasBucket(
            name=previous.name,
            radial_nodes=previous.radial_nodes,
            map_ids=np.concatenate((previous.map_ids, bucket.map_ids)),
            parameters=np.concatenate((previous.parameters, bucket.parameters)),
            m_max=previous.m_max,
            core_m_max=previous.core_m_max,
            sources=previous.sources + bucket.sources,
            selected_ids=selected_ids,
            point_lens_tail_nodes=tail_nodes,
        )
    return merged


def _load_x_coefficients(
    shard_path: Path,
    requested_m_max: int,
    core_m_max: int,
    rows: np.ndarray,
) -> np.ndarray:
    core_modes = min(requested_m_max, core_m_max) + 1
    core = np.load(shard_path / "x_coeff.npy", mmap_mode="r")[rows, ..., :core_modes]
    if requested_m_max <= core_m_max:
        return np.asarray(core)
    extension_modes = requested_m_max - core_m_max
    extension = np.load(shard_path / "x_coeff_extension.npy", mmap_mode="r")[
        rows, ..., :extension_modes
    ]
    return np.concatenate((np.asarray(core), np.asarray(extension)), axis=-1)


def _append_point_lens_tail(
    coefficients: np.ndarray,
    tail_nodes: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    """Append the coefficient representation of the far-field fallback."""
    coefficients = np.asarray(coefficients)
    tail_nodes = np.asarray(tail_nodes, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64)
    if coefficients.ndim != 3 or parameters.shape != (len(coefficients), 3):
        raise ValueError("invalid coefficient/parameter shapes for point-lens tail")
    if tail_nodes.ndim != 1 or not len(tail_nodes):
        raise ValueError("point-lens tail needs at least one radial node")
    radius_squared = np.broadcast_to(tail_nodes[None, :] ** 2, (len(coefficients), len(tail_nodes))).copy()
    wide = np.power(10.0, parameters[:, 0]) > 4.25
    if np.any(wide):
        radius_squared[wide] *= 1.0 + np.power(10.0, parameters[wide, 1])[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        magnification = (radius_squared + 2.0) / np.sqrt(
            radius_squared * (radius_squared + 4.0)
        )
    tail = np.zeros(
        (len(coefficients), len(tail_nodes), coefficients.shape[-1]),
        dtype=coefficients.dtype,
    )
    tail[:, :, 0] = np.asarray(magnification - 1.0, dtype=coefficients.real.dtype)
    return np.concatenate((coefficients, tail), axis=1)


def _manifest_shards(
    manifest: Mapping[str, Any], root: Path
) -> Iterator[Mapping[str, Any]]:
    entries = manifest.get("shards")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"polar manifest has no shards: {root}")
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"shard {index} is not an object: {root}")
        relative = entry.get("path")
        _safe_child(root, relative, f"shard {index}")
        yield entry


def _validate_polar_manifest(manifest: Mapping[str, Any], root: Path) -> None:
    if manifest.get("format") not in _POLAR_FORMATS or manifest.get("version") != 1:
        raise ValueError(f"unsupported polar atlas format: {root}")
    m_max = _manifest_m_max(manifest)
    core_m_max = _manifest_core_m_max(manifest)
    if m_max < 0 or core_m_max < 0 or core_m_max > m_max:
        raise ValueError(f"invalid mode limits in polar manifest: {root}")
    _manifest_shards(manifest, root)


def _manifest_m_max(manifest: Mapping[str, Any]) -> int:
    try:
        value = int(manifest["m_max"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("polar manifest has no integer m_max") from error
    return value


def _manifest_core_m_max(manifest: Mapping[str, Any]) -> int:
    return int(manifest.get("core_m_max", manifest["m_max"]))


def _safe_child(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} has an invalid relative path")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label} path escapes its atlas root")
    resolved = (root / candidate).resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError(f"{label} path escapes its atlas root")
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {resolved}")
    return resolved


def _resolve_seed_path(
    raw_path: object,
    *,
    source_root: Path,
    seed_path: Path,
    label: str,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} has an invalid path")
    candidate = Path(raw_path).expanduser()
    options = [candidate] if candidate.is_absolute() else [source_root / candidate]
    if not candidate.is_absolute():
        options.append(seed_path.parent / candidate)
    for option in options:
        resolved = option.resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"{label} does not exist: {raw_path}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON file: {path}") from error
    if not isinstance(payload, dict) and path.name != "tasks.json":
        raise ValueError(f"JSON object expected: {path}")
    return payload


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        integer = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer") from error
    if isinstance(value, float) and integer != value:
        raise ValueError(f"{label} must be an integer")
    return integer
