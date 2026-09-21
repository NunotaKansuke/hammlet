"""Production-facing map-level queue preparation and task metadata."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
from typing import Iterable

import numpy as np

from ..config import MapConfig, ParameterGrid
from ..build import _parameter_buckets
from .map_queue import MapQueue


@dataclass(frozen=True)
class MapTask:
    map_id: int
    logs: float
    logq: float
    logrho: float
    bucket: str

    def to_dict(self) -> dict[str, object]:
        return {
            "map_id": self.map_id,
            "logs": self.logs,
            "logq": self.logq,
            "logrho": self.logrho,
            "bucket": self.bucket,
        }


def production_tasks(
    grid: ParameterGrid, config: MapConfig
) -> tuple[list[MapTask], dict[str, dict[str, object]]]:
    """Return map tasks and bucket metadata in stable grid order."""
    table = grid.table()
    buckets = _parameter_buckets(table, config)
    tasks: list[MapTask] = []
    metadata: dict[str, dict[str, object]] = {}
    for bucket in buckets:
        name = str(bucket["name"])
        rows = np.asarray(bucket["rows"], dtype=np.float64)
        metadata[name] = {
            "logs": list(bucket["logs"]),
            "logq": list(bucket["logq"]),
            "map_ids": [int(value) for value in rows[:, 0]],
        }
        tasks.extend(
            MapTask(
                map_id=int(map_id),
                logs=float(logs),
                logq=float(logq),
                logrho=float(logrho),
                bucket=name,
            )
            for map_id, logs, logq, logrho in rows
        )
    tasks.sort(key=lambda task: task.map_id)
    if [task.map_id for task in tasks] != list(range(len(tasks))):
        raise ValueError("production tasks must cover contiguous map IDs")
    return tasks, metadata


def _source_priority(path: Path) -> int:
    """Prefer finalized parts over partial trees when IDs overlap."""
    if any(re.fullmatch(r"part-\d+-of-\d+", item) for item in path.parts):
        return 0
    return 1


def _valid_shard(path: Path) -> tuple[np.ndarray, int] | None:
    required = {
        "x_coeff.npy",
        "x2_coeff.npy",
        "reconstruction_error.npy",
        "deviation_envelope.npy",
        "map_ids.npy",
    }
    if not required.issubset({item.name for item in path.iterdir()}):
        return None
    try:
        map_ids = np.asarray(np.load(path / "map_ids.npy"), dtype=np.int64)
        if map_ids.ndim != 1 or not len(map_ids) or len(set(map_ids.tolist())) != len(
            map_ids
        ):
            return None
        for name in required - {"map_ids.npy"}:
            values = np.load(path / name, mmap_mode="r")
            if values.shape[0] != len(map_ids):
                return None
        return map_ids, _source_priority(path)
    except (OSError, ValueError, TypeError):
        return None


def index_existing_shards(
    source_root: str | Path,
    expected_ids: Iterable[int],
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    """Index durable old-run rows without copying their large arrays."""
    source_root = Path(source_root).resolve()
    expected = set(int(value) for value in expected_ids)
    selected: dict[int, dict[str, object]] = {}
    duplicates: list[dict[str, object]] = []
    for shard in sorted(source_root.rglob("shard_*")):
        if not shard.is_dir():
            continue
        valid = _valid_shard(shard)
        if valid is None:
            continue
        map_ids, priority = valid
        bucket = shard.parent
        radial_nodes = bucket / "radial_nodes.npy"
        if not radial_nodes.is_file():
            continue
        for row, map_id_value in enumerate(map_ids.tolist()):
            map_id = int(map_id_value)
            if map_id not in expected:
                continue
            record = {
                "map_id": map_id,
                "bucket": bucket.name,
                "shard": str(shard),
                "row": row,
                "radial_nodes": str(radial_nodes),
                "priority": priority,
            }
            previous = selected.get(map_id)
            if previous is None or priority < int(previous["priority"]):
                if previous is not None:
                    duplicates.append(
                        {"map_id": map_id, "kept": record, "discarded": previous}
                    )
                selected[map_id] = record
            else:
                duplicates.append(
                    {"map_id": map_id, "kept": previous, "discarded": record}
                )
    return selected, duplicates


def initialize_production_map_run(
    root: str | Path,
    grid: ParameterGrid,
    config: MapConfig,
    *,
    seed_root: str | Path | None = None,
) -> dict[str, int]:
    """Create a fresh map-level run and seed durable rows from an old run."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in ("layouts", "results", "seed"):
        (root / name).mkdir()
    tasks, buckets = production_tasks(grid, config)
    queue = MapQueue.initialize(
        root / "queue",
        (task.map_id for task in tasks),
        metadata={"n_maps": len(tasks), "scheduler": "map-level-v1"},
    )
    (root / "tasks.json").write_text(
        json.dumps([task.to_dict() for task in tasks], indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "buckets.json").write_text(
        json.dumps(buckets, indent=2) + "\n", encoding="utf-8"
    )
    (root / "grid.json").write_text(
        json.dumps(grid.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    (root / "maps-config.json").write_text(
        json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8"
    )

    seeded: dict[int, dict[str, object]] = {}
    duplicates: list[dict[str, object]] = []
    if seed_root is not None:
        seeded, duplicates = index_existing_shards(
            seed_root, (task.map_id for task in tasks)
        )
        (root / "seed" / "index.json").write_text(
            json.dumps(
                {"source_root": str(Path(seed_root).resolve()), "maps": seeded},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (root / "seed" / "duplicates.json").write_text(
            json.dumps(duplicates, indent=2) + "\n", encoding="utf-8"
        )

    by_bucket: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for record in seeded.values():
        by_bucket[str(record["bucket"])].append(record)
    layout_count = 0
    for bucket_name in buckets:
        layout_dir = root / "layouts" / bucket_name
        layout_dir.mkdir()
        candidates = sorted(
            by_bucket.get(bucket_name, []),
            key=lambda record: (int(record["priority"]), int(record["map_id"])),
        )
        if candidates:
            source = Path(str(candidates[0]["radial_nodes"]))
            shutil.copy2(source, layout_dir / "radial_nodes.npy")
            layout_count += 1

    pending_layouts = [
        bucket_name
        for bucket_name in sorted(buckets)
        if not (root / "layouts" / bucket_name / "radial_nodes.npy").is_file()
    ]
    (root / "layout-pending-buckets.json").write_text(
        json.dumps(pending_layouts, indent=2) + "\n", encoding="utf-8"
    )

    for map_id, record in seeded.items():
        queue.seed_done(map_id, source=str(record["shard"]))
    return {
        "total_maps": len(tasks),
        "seeded_maps": len(seeded),
        "duplicate_rows": len(duplicates),
        "seeded_layouts": layout_count,
        "missing_layouts": len(buckets) - layout_count,
    }
