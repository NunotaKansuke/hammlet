#!/usr/bin/env python3
"""Report durable progress for a production map-level run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hammlet._core.map_queue import MapQueue


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--stale-after", type=float, default=1800.0)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    queue = MapQueue.open(run_root / "queue")
    status = queue.status(stale_after=args.stale_after).to_dict()
    tasks = json.loads((run_root / "tasks.json").read_text(encoding="utf-8"))
    layouts = {
        task["bucket"]
        for task in tasks
        if (run_root / "layouts" / task["bucket"] / "radial_nodes.npy").is_file()
    }
    all_buckets = sorted({task["bucket"] for task in tasks})
    pending = [
        task
        for task in tasks
        if not queue.is_done(int(task["map_id"]))
    ]
    status.update(
        {
            "layout_ready_buckets": len(layouts),
            "layout_pending_buckets": len(all_buckets) - len(layouts),
            "pending_maps_with_layout": sum(
                task["bucket"] in layouts for task in pending
            ),
            "pending_maps_waiting_for_layout": sum(
                task["bucket"] not in layouts for task in pending
            ),
            "failed_records": len(list((run_root / "queue" / "failed").glob("*.json"))),
        }
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
