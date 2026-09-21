#!/usr/bin/env python3
"""Prepare and run a small Grid Engine benchmark of production map tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from hammlet._core.atlas_builder import MapBuildSpec
from hammlet._core.direct_vbm import DirectVBMPolarAtlasBuilder, VBMBinaryLensEvaluator
from hammlet._core.map_scheduler import index_existing_shards, production_tasks
from hammlet.build import _spectrum_config
from hammlet.cli import _config


def _prepare(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    seed_root = Path(args.seed_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    if manifest_path.exists():
        raise FileExistsError(f"benchmark manifest already exists: {manifest_path}")

    grid, config = _config(config_path)
    tasks, _ = production_tasks(grid, config)
    if len(tasks) != 33993:
        raise ValueError(f"expected 33993 production maps, got {len(tasks)}")
    indexed, _ = index_existing_shards(seed_root, (task.map_id for task in tasks))

    # Pick one durable layout per bucket, then spread the test over the
    # available bucket range.  The source rows are read-only; no old map is
    # copied or modified by the benchmark.
    by_bucket: dict[str, list[dict[str, object]]] = {}
    for task in tasks:
        record = indexed.get(task.map_id)
        if record is not None:
            by_bucket.setdefault(task.bucket, []).append(
                {
                    "map_id": task.map_id,
                    "logs": task.logs,
                    "logq": task.logq,
                    "logrho": task.logrho,
                    "bucket": task.bucket,
                    "radial_nodes": record["radial_nodes"],
                }
            )
    buckets = sorted(by_bucket)
    if len(buckets) < args.count:
        raise ValueError(
            f"only {len(buckets)} buckets have durable layouts; need {args.count}"
        )
    positions = np.linspace(0, len(buckets) - 1, args.count, dtype=np.int64)
    selected = [by_bucket[buckets[int(position)]][0] for position in positions]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "format": "hammlet-ge-map-benchmark",
                "version": 1,
                "config": str(config_path),
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "source_root": str(seed_root),
                "tasks": selected,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"count": len(selected), "map_ids": [item["map_id"] for item in selected]}))


def _run(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    root = Path(args.output).resolve()
    task_number = int(os.environ.get("SGE_TASK_ID", args.task_number))
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    tasks = manifest["tasks"]
    if not 1 <= task_number <= len(tasks):
        raise ValueError(f"SGE_TASK_ID={task_number} is outside 1..{len(tasks)}")
    task = tasks[task_number - 1]
    map_id = int(task["map_id"])
    destination = root / "results" / f"map-{map_id:08d}"
    if destination.is_dir():
        print(f"skip existing map_id={map_id} destination={destination}")
        return

    root.joinpath("results").mkdir(parents=True, exist_ok=True)
    temporary = root / "results" / f".map-{map_id:08d}.partial-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    temporary.mkdir()
    started = time.monotonic()
    try:
        config_path = Path(manifest["config"])
        _, config = _config(config_path)
        radial_nodes = np.load(task["radial_nodes"])
        evaluator = VBMBinaryLensEvaluator(
            10.0 ** float(task["logs"]),
            10.0 ** float(task["logq"]),
            10.0 ** float(task["logrho"]),
            tolerance=config.vbm_tolerance,
            relative_tolerance=config.vbm_relative_tolerance,
            coordinate_frame=config.coordinate_frame,
        )
        builder = DirectVBMPolarAtlasBuilder(
            radial_nodes,
            spectrum_config=_spectrum_config(config),
            shard_size=1,
        )
        builder.build(
            temporary,
            [
                MapBuildSpec(
                    map_id=map_id,
                    logs=float(task["logs"]),
                    logq=float(task["logq"]),
                    logrho=float(task["logrho"]),
                    evaluator=evaluator,
                )
            ],
            progress_every=1,
        )
        temporary.replace(destination)
    except BaseException:
        print(f"failed map_id={map_id} temporary={temporary}")
        raise
    elapsed = time.monotonic() - started
    print(
        f"completed map_id={map_id} bucket={task['bucket']} "
        f"elapsed_seconds={elapsed:.1f} destination={destination}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("config", type=Path)
    prepare.add_argument("seed_root", type=Path)
    prepare.add_argument("manifest", type=Path)
    prepare.add_argument("--count", type=int, default=9)
    run = subparsers.add_parser("run")
    run.add_argument("manifest", type=Path)
    run.add_argument("output", type=Path)
    run.add_argument("--task-number", type=int, default=1)
    args = parser.parse_args()
    if args.command == "prepare":
        _prepare(args)
    else:
        _run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
