"""Filesystem-backed map-level scheduling primitives.

The production map builder uses parameter-space buckets because radial layouts
are shared by ``(s, q)`` groups.  This module deliberately separates that
storage concern from scheduling: one worker claims one global map ID at a
time.

Claims use a canonical directory and ``mkdir``.  Directory creation is atomic
on the shared filesystems used by the multi-machine runner, so two workers
cannot own the same map.  A task callback must publish its result atomically
before calling :meth:`MapClaim.complete`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Callable, Iterable
import uuid


class QueueFormatError(ValueError):
    """Raised when a map queue is missing or has an invalid manifest."""


def _atomic_json(path: Path, payload: object) -> None:
    """Write JSON and make the completed file visible with one rename."""
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


@dataclass(frozen=True)
class QueueStatus:
    """A point-in-time queue summary suitable for human-readable monitoring."""

    total: int
    done: int
    running: int
    pending: int
    stale: int
    workers: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "done": self.done,
            "running": self.running,
            "pending": self.pending,
            "stale": self.stale,
            "workers": dict(sorted(self.workers.items())),
        }


class MapQueue:
    """A durable map-ID queue backed by atomic shared-filesystem operations."""

    manifest_name = "queue.json"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.claims = self.root / "claims"
        self.done = self.root / "done"
        self.failed = self.root / "failed"
        self.heartbeats = self.root / "heartbeats"
        self._map_ids: tuple[int, ...] | None = None

    @classmethod
    def initialize(
        cls,
        root: str | Path,
        map_ids: Iterable[int],
        *,
        metadata: dict[str, object] | None = None,
    ) -> "MapQueue":
        """Create a queue, refusing to overwrite an existing queue."""
        queue = cls(root)
        queue.root.mkdir(parents=True, exist_ok=False)
        for directory in (
            queue.claims,
            queue.done,
            queue.failed,
            queue.heartbeats,
        ):
            directory.mkdir()
        normalized = sorted({int(map_id) for map_id in map_ids})
        if not normalized:
            raise ValueError("map_ids must not be empty")
        if normalized[0] < 0:
            raise ValueError("map_ids must be non-negative")
        queue._map_ids = tuple(normalized)
        _atomic_json(
            queue.root / cls.manifest_name,
            {
                "format": "hammlet-map-queue",
                "version": 1,
                "map_ids": normalized,
                "metadata": metadata or {},
            },
        )
        return queue

    @classmethod
    def open(cls, root: str | Path) -> "MapQueue":
        """Open and validate an existing queue."""
        queue = cls(root)
        try:
            payload = json.loads(
                (queue.root / cls.manifest_name).read_text(encoding="utf-8")
            )
        except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
            raise QueueFormatError(f"invalid map queue: {queue.root}") from error
        if payload.get("format") != "hammlet-map-queue" or payload.get("version") != 1:
            raise QueueFormatError(f"unsupported map queue: {queue.root}")
        raw_ids = payload.get("map_ids")
        if not isinstance(raw_ids, list) or not raw_ids or not all(
            isinstance(map_id, int) and not isinstance(map_id, bool) and map_id >= 0
            for map_id in raw_ids
        ):
            raise QueueFormatError("queue map_ids must be non-negative integers")
        if raw_ids != sorted(set(raw_ids)):
            raise QueueFormatError("queue map_ids must be sorted and unique")
        for directory in (
            queue.claims,
            queue.done,
            queue.failed,
            queue.heartbeats,
        ):
            if not directory.is_dir():
                raise QueueFormatError(f"queue directory is missing: {directory}")
        queue._map_ids = tuple(raw_ids)
        return queue

    @property
    def map_ids(self) -> tuple[int, ...]:
        if self._map_ids is None:
            raise QueueFormatError("queue is not initialized")
        return self._map_ids

    @staticmethod
    def _name(map_id: int) -> str:
        return f"map-{int(map_id):08d}"

    def _done_path(self, map_id: int) -> Path:
        return self.done / self._name(map_id)

    def _claim_path(self, map_id: int) -> Path:
        return self.claims / self._name(map_id)

    def _heartbeat_path(self, map_id: int) -> Path:
        return self.heartbeats / f"{self._name(map_id)}.json"

    def is_done(self, map_id: int) -> bool:
        return self._done_path(map_id).is_dir()

    def seed_done(self, map_id: int, *, source: str | None = None) -> bool:
        """Mark a previously durable map as done without claiming it."""
        if int(map_id) not in self.map_ids:
            raise KeyError(f"map ID is not present in the queue: {map_id}")
        done_path = self._done_path(map_id)
        try:
            done_path.mkdir()
        except FileExistsError:
            return False
        if source is not None:
            _atomic_json(
                done_path / "seed.json",
                {"map_id": int(map_id), "source": str(source)},
            )
        return True

    def claim_next(self, worker_id: str) -> "MapClaim | None":
        """Atomically claim the first currently available map ID."""
        worker_id = str(worker_id).strip()
        if not worker_id:
            raise ValueError("worker_id must not be empty")
        for map_id in self.map_ids:
            if self.is_done(map_id):
                continue
            claim_path = self._claim_path(map_id)
            try:
                claim_path.mkdir()
            except FileExistsError:
                continue
            try:
                owner = {
                    "map_id": map_id,
                    "worker_id": worker_id,
                    "pid": os.getpid(),
                    "claimed_at": time.time(),
                    "token": uuid.uuid4().hex,
                }
                _atomic_json(claim_path / "owner.json", owner)
                # A completed task may have won a race immediately before the
                # claim directory was created.  Never run it in that case.
                if self.is_done(map_id):
                    shutil.rmtree(claim_path)
                    return self.claim_next(worker_id)
                claim = MapClaim(self, map_id, worker_id, claim_path, owner["token"])
                claim.heartbeat()
                return claim
            except BaseException:
                shutil.rmtree(claim_path, ignore_errors=True)
                raise
        return None

    def status(self, *, stale_after: float = 600.0) -> QueueStatus:
        """Return queue counts and worker ownership from current heartbeats."""
        if stale_after <= 0:
            raise ValueError("stale_after must be positive")
        now = time.time()
        done = sum(self.is_done(map_id) for map_id in self.map_ids)
        running = 0
        stale = 0
        workers: Counter[str] = Counter()
        for map_id in self.map_ids:
            claim = self._claim_path(map_id)
            if not claim.is_dir() or self.is_done(map_id):
                continue
            running += 1
            heartbeat = self._heartbeat_path(map_id)
            try:
                payload = json.loads(heartbeat.read_text(encoding="utf-8"))
                worker = str(payload["worker_id"])
                updated = float(payload["updated_at"])
            except (FileNotFoundError, OSError, KeyError, TypeError, ValueError):
                worker = "unknown"
                updated = claim.stat().st_mtime
            workers[worker] += 1
            if now - updated > stale_after:
                stale += 1
        return QueueStatus(
            total=len(self.map_ids),
            done=done,
            running=running,
            pending=len(self.map_ids) - done - running,
            stale=stale,
            workers=dict(workers),
        )

    def reclaim_stale(self, *, stale_after: float = 3600.0) -> list[int]:
        """Quarantine stale claims so a replacement worker can retry them.

        This operation is intentionally separate from :meth:`status`; a
        monitor may report stale work without killing a worker.  The claim is
        first atomically renamed away from its canonical name, so a new claim
        cannot overlap the old claim directory.
        """
        if stale_after <= 0:
            raise ValueError("stale_after must be positive")
        now = time.time()
        reclaimed: list[int] = []
        for map_id in self.map_ids:
            if self.is_done(map_id):
                continue
            claim = self._claim_path(map_id)
            if not claim.is_dir():
                continue
            heartbeat = self._heartbeat_path(map_id)
            try:
                updated = float(
                    json.loads(heartbeat.read_text(encoding="utf-8"))["updated_at"]
                )
            except (FileNotFoundError, OSError, KeyError, TypeError, ValueError):
                updated = claim.stat().st_mtime
            if now - updated <= stale_after:
                continue
            quarantine = self.claims / (
                f".reclaimed-{self._name(map_id)}-{os.getpid()}-{uuid.uuid4().hex}"
            )
            try:
                claim.rename(quarantine)
            except FileNotFoundError:
                continue
            shutil.rmtree(quarantine, ignore_errors=True)
            self._heartbeat_path(map_id).unlink(missing_ok=True)
            reclaimed.append(map_id)
        return reclaimed


@dataclass
class MapClaim:
    """An exclusive lease for one map ID."""

    queue: MapQueue
    map_id: int
    worker_id: str
    path: Path
    token: str
    _closed: bool = False

    def heartbeat(self, **extra: object) -> None:
        """Refresh the lease heartbeat atomically."""
        if self._closed:
            raise RuntimeError("claim is already closed")
        payload = {
            "map_id": self.map_id,
            "worker_id": self.worker_id,
            "pid": os.getpid(),
            "token": self.token,
            "updated_at": time.time(),
            **extra,
        }
        _atomic_json(self.queue._heartbeat_path(self.map_id), payload)

    def complete(self) -> bool:
        """Publish the done marker, winning safely against stale retries."""
        if self._closed:
            return False
        try:
            self.queue._done_path(self.map_id).mkdir()
        except FileExistsError:
            completed = False
        else:
            completed = True
        self._close()
        return completed

    def publish_file(self, temporary: str | Path, destination: str | Path) -> Path:
        """Publish one result file without overwriting another worker's file.

        ``temporary`` and ``destination`` must be on the same filesystem.
        ``link`` is used instead of ``replace`` so a stale retry cannot
        overwrite a result that another worker has already published.
        """
        if self._closed:
            raise RuntimeError("claim is already closed")
        temporary = Path(temporary)
        destination = Path(destination)
        if not temporary.is_file():
            raise FileNotFoundError(temporary)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if not destination.is_file():
                raise
        else:
            temporary.unlink()
        return destination

    def fail(self, message: str) -> None:
        """Record a failure and release the claim for later inspection."""
        _atomic_json(
            self.queue.failed / f"{self._name()}.json",
            {
                "map_id": self.map_id,
                "worker_id": self.worker_id,
                "message": str(message),
                "updated_at": time.time(),
            },
        )
        self._close()

    def _name(self) -> str:
        return self.queue._name(self.map_id)

    def _close(self) -> None:
        self.queue._heartbeat_path(self.map_id).unlink(missing_ok=True)
        shutil.rmtree(self.path, ignore_errors=True)
        self._closed = True


def run_worker(
    queue: MapQueue,
    worker_id: str,
    task: Callable[[int, MapClaim], None],
    *,
    heartbeat_interval: float = 30.0,
) -> int:
    """Run a persistent worker until the queue is empty.

    The callback must write its map output to a private temporary location and
    atomically publish it before returning.  ``run_worker`` only marks the map
    done after the callback succeeds.
    """
    if heartbeat_interval <= 0:
        raise ValueError("heartbeat_interval must be positive")
    completed = 0
    while True:
        claim = queue.claim_next(worker_id)
        if claim is None:
            return completed
        stop = threading.Event()

        def refresh() -> None:
            while not stop.wait(heartbeat_interval):
                try:
                    claim.heartbeat()
                except (FileNotFoundError, RuntimeError):
                    return

        thread = threading.Thread(target=refresh, daemon=True)
        thread.start()
        try:
            task(claim.map_id, claim)
        except BaseException as error:
            stop.set()
            thread.join(timeout=heartbeat_interval)
            claim.fail(f"{type(error).__name__}: {error}")
            raise
        else:
            stop.set()
            thread.join(timeout=heartbeat_interval)
            if claim.complete():
                completed += 1
