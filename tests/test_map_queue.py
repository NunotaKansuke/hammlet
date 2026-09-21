from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import json
import os
from pathlib import Path
import time

from hammlet._core.map_queue import MapQueue, run_worker


def _claim_in_process(arguments):
    root, worker_id = arguments
    queue = MapQueue.open(root)
    claim = queue.claim_next(f"process-{worker_id}")
    if claim is None:
        return None
    time.sleep(0.01)
    return claim.map_id


def test_atomic_claim_allows_one_owner_per_map(tmp_path: Path):
    queue = MapQueue.initialize(tmp_path / "queue", range(8))

    def claim(worker: int):
        item = queue.claim_next(f"worker-{worker}")
        if item is None:
            return None
        item.heartbeat(test=True)
        return item.map_id

    with ThreadPoolExecutor(max_workers=32) as pool:
        claimed = list(pool.map(claim, range(32)))

    owned = [map_id for map_id in claimed if map_id is not None]
    assert sorted(owned) == list(range(8))
    assert len(owned) == len(set(owned))
    assert queue.status(stale_after=60).running == 8


def test_separate_processes_cannot_claim_same_map(tmp_path: Path):
    queue = MapQueue.initialize(tmp_path / "queue", range(12))
    arguments = [(str(queue.root), worker_id) for worker_id in range(32)]
    with ProcessPoolExecutor(
        max_workers=16, mp_context=mp.get_context("spawn")
    ) as pool:
        claimed = list(pool.map(_claim_in_process, arguments))

    owned = [map_id for map_id in claimed if map_id is not None]
    assert sorted(owned) == list(range(12))
    assert len(owned) == len(set(owned))


def test_workers_finish_every_map_once(tmp_path: Path):
    queue = MapQueue.initialize(tmp_path / "queue", range(24))
    seen: list[int] = []

    def task(map_id: int, claim):
        claim.heartbeat(stage="computing")
        time.sleep(0.002)
        seen.append(map_id)

    def worker(worker_id: int):
        return run_worker(queue, f"worker-{worker_id}", task, heartbeat_interval=0.01)

    with ThreadPoolExecutor(max_workers=8) as pool:
        counts = list(pool.map(worker, range(8)))

    status = queue.status(stale_after=60)
    assert sum(counts) == 24
    assert sorted(seen) == list(range(24))
    assert len(seen) == len(set(seen))
    assert status.done == 24
    assert status.running == 0
    assert status.pending == 0


def test_stale_claim_can_be_reclaimed(tmp_path: Path):
    queue = MapQueue.initialize(tmp_path / "queue", [7])
    claim = queue.claim_next("dead-worker")
    assert claim is not None
    heartbeat = queue._heartbeat_path(7)
    old = time.time() - 120
    payload = json.loads(heartbeat.read_text(encoding="utf-8"))
    payload["updated_at"] = old
    heartbeat.write_text(json.dumps(payload), encoding="utf-8")
    os.utime(heartbeat, (old, old))
    assert queue.status(stale_after=60).stale == 1
    assert queue.reclaim_stale(stale_after=60) == [7]
    replacement = queue.claim_next("replacement")
    assert replacement is not None
    assert replacement.map_id == 7
