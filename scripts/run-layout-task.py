#!/usr/bin/env python3
"""Build one missing production radial layout.

The layout is shared by all rho values in one (s, q) bucket.  This script is
intended to be one Grid Engine array task per missing bucket; it never builds a
map and it never overwrites an already published layout.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from hammlet._core.adaptive_grid import (
    build_adaptive_radial_nodes,
    sharp_qs_pilot_indices,
)
from hammlet._core.direct_vbm import VBMBinaryLensEvaluator
from hammlet.config import MapConfig


def _load_config(run_root: Path) -> MapConfig:
    return MapConfig(
        **json.loads((run_root / "maps-config.json").read_text(encoding="utf-8"))
    )


def _task_bucket(run_root: Path, task_number: int) -> str:
    pending_path = run_root / "layout-pending-buckets.json"
    if pending_path.is_file():
        raw_pending = json.loads(pending_path.read_text(encoding="utf-8"))
        if not isinstance(raw_pending, list):
            raise ValueError(f"layout pending list is not an array: {pending_path}")
        pending = [
            str(item["bucket"]) if isinstance(item, dict) else str(item)
            for item in raw_pending
        ]
    else:
        # Backward-compatible fallback for runs prepared before the fixed
        # pending-list manifest was introduced.  New runs persist the list so
        # that array task IDs remain stable while other layout workers finish.
        tasks = json.loads((run_root / "tasks.json").read_text(encoding="utf-8"))
        all_buckets = sorted({str(task["bucket"]) for task in tasks})
        pending = [
            bucket
            for bucket in all_buckets
            if not (run_root / "layouts" / bucket / "radial_nodes.npy").is_file()
        ]
    if not 1 <= task_number <= len(pending):
        raise ValueError(
            f"SGE_TASK_ID={task_number} is outside 1..{len(pending)}"
        )
    return str(pending[task_number - 1])


def _build(run_root: Path, bucket: str) -> None:
    layout_dir = run_root / "layouts" / bucket
    destination = layout_dir / "radial_nodes.npy"
    if destination.is_file():
        print(json.dumps({"bucket": bucket, "status": "already-ready"}))
        return
    layout_dir.mkdir(parents=True, exist_ok=True)

    tasks = json.loads((run_root / "tasks.json").read_text(encoding="utf-8"))
    bucket_rows = [
        [
            int(task["map_id"]),
            float(task["logs"]),
            float(task["logq"]),
            float(task["logrho"]),
        ]
        for task in tasks
        if task["bucket"] == bucket
    ]
    if not bucket_rows:
        raise ValueError(f"bucket has no production maps: {bucket}")

    config = _load_config(run_root)
    rows = np.asarray(bucket_rows, dtype=np.float64)
    pilot_indices = sharp_qs_pilot_indices(
        rows[:, 1:], min(int(config.radial_pilot_maps), len(rows))
    )
    pilot_rows = rows[pilot_indices]
    evaluators = [
        VBMBinaryLensEvaluator(
            10.0**logs,
            10.0**logq,
            10.0**logrho,
            tolerance=config.vbm_tolerance,
            relative_tolerance=config.vbm_relative_tolerance,
            coordinate_frame=config.coordinate_frame,
        )
        for _, logs, logq, logrho in pilot_rows
    ]

    started = time.monotonic()
    radial_nodes, pilot_radii, difficulty = build_adaptive_radial_nodes(
        evaluators,
        max_radius=config.radial_max,
        n_nodes=config.radial_nodes,
        pilot_bins=config.radial_pilot_bins,
        n_phi=config.radial_pilot_phi,
        m_max=config.radial_pilot_m_max,
        map_quantile=1.0,
        adaptive_fraction=config.radial_adaptive_fraction,
    )

    # Diagnostics are published first; radial_nodes.npy is the completion
    # marker.  Temporary names keep an interrupted task from looking ready.
    token = f"{os.getpid()}-{time.time_ns()}"
    files = {
        "radial_pilot_map_ids.npy": pilot_rows[:, 0].astype(np.int64),
        "radial_pilot_radii.npy": pilot_radii,
        "radial_pilot_difficulty.npy": difficulty,
    }
    temporary_paths: list[Path] = []
    try:
        for name, values in files.items():
            temporary = layout_dir / f".{name}.partial-{token}.npy"
            np.save(temporary, values)
            temporary_paths.append(temporary)
            temporary.replace(layout_dir / name)
        temporary = layout_dir / f".radial_nodes.npy.partial-{token}.npy"
        np.save(temporary, radial_nodes)
        temporary_paths.append(temporary)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            pass
        else:
            temporary.unlink(missing_ok=True)
        if not destination.is_file():
            raise RuntimeError(f"failed to publish radial layout: {destination}")
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "bucket": bucket,
                "status": "completed",
                "map_count": len(rows),
                "pilot_map_ids": pilot_rows[:, 0].astype(np.int64).tolist(),
                "elapsed_seconds": round(time.monotonic() - started, 1),
                "destination": str(destination),
            }
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--task-number", type=int)
    parser.add_argument("--bucket")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    if args.task_number is not None and args.bucket is not None:
        raise ValueError("specify at most one of --task-number or --bucket")
    task_number = int(os.environ.get("SGE_TASK_ID", "1"))
    bucket = args.bucket or _task_bucket(run_root, args.task_number or task_number)
    _build(run_root, bucket)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
