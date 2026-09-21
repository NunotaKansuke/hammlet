#!/usr/bin/env python3
"""Build exactly one pending production map from a shared map queue."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import threading
import time

import numpy as np

from hammlet._core.atlas_builder import MapBuildSpec
from hammlet._core.direct_vbm import DirectVBMPolarAtlasBuilder, VBMBinaryLensEvaluator
from hammlet._core.map_queue import MapQueue
from hammlet.build import _spectrum_config
from hammlet.config import MapConfig


def _worker_id() -> str:
    host = os.environ.get("HOSTNAME", "unknown-host")
    job = os.environ.get("JOB_ID", "manual")
    task = os.environ.get("SGE_TASK_ID", "1")
    return f"{host}/job-{job}/task-{task}/pid-{os.getpid()}"


def _run_one(run_root: Path) -> int:
    queue = MapQueue.open(run_root / "queue")
    claim = queue.claim_next(_worker_id())
    if claim is None:
        print("queue empty")
        return 0

    tasks = json.loads((run_root / "tasks.json").read_text(encoding="utf-8"))
    task_by_id = {int(task["map_id"]): task for task in tasks}
    task = task_by_id[claim.map_id]
    config = MapConfig(
        **json.loads((run_root / "maps-config.json").read_text(encoding="utf-8"))
    )
    radial_nodes_path = (
        run_root / "layouts" / str(task["bucket"]) / "radial_nodes.npy"
    )
    destination = run_root / "results" / f"map-{claim.map_id:08d}"
    temporary = run_root / "results" / (
        f".map-{claim.map_id:08d}.partial-{os.getpid()}-{time.time_ns()}"
    )
    stop = threading.Event()

    def refresh() -> None:
        while not stop.wait(30.0):
            try:
                claim.heartbeat(stage="building")
            except (FileNotFoundError, RuntimeError):
                return

    heartbeat = threading.Thread(target=refresh, daemon=True)
    heartbeat.start()
    try:
        if destination.is_dir():
            claim.heartbeat(stage="already-published", destination=str(destination))
            print(f"skip existing map_id={claim.map_id} destination={destination}")
            claim.complete()
            return 0
        if not radial_nodes_path.is_file():
            raise FileNotFoundError(
                f"missing radial layout for {task['bucket']}: {radial_nodes_path}"
            )
        temporary.mkdir(parents=True)
        radial_nodes = np.load(radial_nodes_path)
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
            shard_size=config.shard_size,
        )
        started = time.monotonic()
        builder.build(
            temporary,
            [
                MapBuildSpec(
                    map_id=claim.map_id,
                    logs=float(task["logs"]),
                    logq=float(task["logq"]),
                    logrho=float(task["logrho"]),
                    evaluator=evaluator,
                )
            ],
            progress_every=1,
        )
        try:
            temporary.rename(destination)
        except FileExistsError:
            shutil.rmtree(temporary, ignore_errors=True)
            temporary = None
        claim.heartbeat(
            stage="published",
            destination=str(destination),
            elapsed_seconds=round(time.monotonic() - started, 1),
        )
        print(
            f"completed map_id={claim.map_id} bucket={task['bucket']} "
            f"elapsed_seconds={time.monotonic() - started:.1f} "
            f"destination={destination}"
        )
        claim.complete()
        return 0
    except BaseException as error:
        claim.fail(f"{type(error).__name__}: {error}")
        raise
    finally:
        stop.set()
        heartbeat.join(timeout=1.0)
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        if not claim._closed:
            claim.fail("worker exited before publishing map")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    return _run_one(args.run_root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
