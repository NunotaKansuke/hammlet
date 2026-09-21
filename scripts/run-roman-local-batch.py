#!/usr/bin/env python3
"""Run a local Roman/GULLS batch with one isolated result per event.

This is intentionally a local diagnostic driver.  It validates the packed
atlas read-only, preflights the input rows, then invokes the existing packed
parallel runner once per event.  Event output, logs, and resumable bucket
results stay below one separate batch directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--packed-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cores", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--m-max", type=int, default=128)
    parser.add_argument("--n-alpha", type=int, default=540)
    parser.add_argument("--radial-order", type=int, choices=(1, 3), default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--candidate-count", type=int, default=100)
    parser.add_argument(
        "--event-id",
        action="append",
        dest="event_ids",
        help="run only this event ID; repeat to select several events",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed event results and continue partial event results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="preflight inputs and atlas, write no event scan output",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)

    input_root = args.input_root.expanduser().resolve()
    truth_path = args.truth.expanduser().resolve()
    atlas_path = args.atlas.expanduser().resolve()
    packed_path = args.packed_cache.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not input_root.is_dir():
        raise SystemExit(f"--input-root is not a directory: {input_root}")
    if not truth_path.is_file():
        raise SystemExit(f"--truth does not exist: {truth_path}")
    _ensure_separate(output_root, atlas_path, packed_path)
    if output_root.exists() and not args.resume:
        raise SystemExit(
            f"output already exists; use --resume or choose another path: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    jax_cache_dir = output_root / "jax-compilation-cache"
    jax_cache_dir.mkdir(exist_ok=True)

    events = _read_and_preflight_events(input_root, truth_path)
    events = _select_events(events, args.event_ids, args.limit)
    if not events:
        raise SystemExit("no events selected")

    _validate_atlas(atlas_path, packed_path)
    batch_config = _batch_config(
        args,
        input_root=input_root,
        truth_path=truth_path,
        atlas_path=atlas_path,
        packed_path=packed_path,
        output_root=output_root,
        jax_cache_dir=jax_cache_dir,
        events=events,
    )
    batch_path = output_root / "batch.json"
    if args.resume and batch_path.is_file():
        old = _read_json(batch_path)
        _validate_resume_config(old, batch_config)
        old_events = {
            str(event.get("event_id")): event
            for event in old.get("events", [])
            if isinstance(event, dict) and event.get("event_id") is not None
        }
        new_events = {
            str(event.get("event_id")): event
            for event in batch_config["events"]
        }
        for event_id in set(old_events) & set(new_events):
            if old_events[event_id] != new_events[event_id]:
                raise SystemExit(
                    f"batch resume event definition changed: {event_id}"
                )
        merged = dict(old_events)
        merged.update(new_events)
        ordered_ids = [
            str(event.get("event_id"))
            for event in old.get("events", [])
            if isinstance(event, dict) and event.get("event_id") is not None
        ]
        ordered_ids.extend(
            event["event_id"] for event in events if event["event_id"] not in old_events
        )
        batch_config["events"] = [merged[event_id] for event_id in ordered_ids]
    _write_json(batch_path, batch_config)

    status_path = output_root / "status.json"
    statuses = _load_status(status_path) if args.resume else {}
    if args.dry_run:
        _write_json(
            status_path,
            {
                "schema": "hammlet-roman-local-batch-status-v1",
                "state": "dry-run",
                "updated_at": _now(),
                "events": {
                    event["event_id"]: {
                        "state": "preflight-ok",
                        "points": event["points"],
                        "window": event["window"],
                    }
                    for event in events
                },
            },
        )
        print(
            json.dumps(
                {
                    "output_root": str(output_root),
                    "event_count": len(events),
                    "points_per_event": sorted({event["points"] for event in events}),
                    "total_points": sum(event["points"] for event in events),
                    "dry_run": True,
                },
                indent=2,
            )
        )
        return 0

    logs_dir = output_root / "logs"
    events_dir = output_root / "events"
    logs_dir.mkdir(exist_ok=True)
    events_dir.mkdir(exist_ok=True)
    selected_ids = {event["event_id"] for event in events}
    started = time.perf_counter()
    completed = 0
    failed = 0
    for position, event in enumerate(events, start=1):
        event_id = event["event_id"]
        event_output = events_dir / event_id
        report_path = event_output / "report.json"
        if _completed_report_matches(report_path, event, args):
            statuses[event_id] = _completed_status_from_report(
                event, report_path, skipped=True
            )
            completed += 1
            _write_batch_status(status_path, statuses, selected_ids, "running")
            print(
                f"[roman-local-batch] {position}/{len(events)} "
                f"{event_id} already complete; skipping",
                flush=True,
            )
            continue

        command = _event_command(args, event, atlas_path, packed_path, event_output)
        log_path = logs_dir / f"{event_id}.log"
        event_output.parent.mkdir(parents=True, exist_ok=True)
        event_started_at = _now()
        statuses[event_id] = {
            "event_id": event_id,
            "state": "running",
            "points_expected": event["points"],
            "window_expected": event["window"],
            "started_at": event_started_at,
            "output": str(event_output),
            "log": str(log_path),
            "command": command,
        }
        _write_batch_status(status_path, statuses, selected_ids, "running")
        print(
            f"[roman-local-batch] {position}/{len(events)} start {event_id} "
            f"points={event['points']}",
            flush=True,
        )
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n===== batch invocation {event_started_at} =====\n")
            log_handle.flush()
            process_started = time.perf_counter()
            try:
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=_child_environment(jax_cache_dir),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                returncode = process.wait()
            except OSError as error:
                returncode = 127
                log_handle.write(f"could not start runner: {error}\n")
            process_seconds = time.perf_counter() - process_started

        if returncode == 0 and _completed_report_matches(report_path, event, args):
            status = _completed_status_from_report(event, report_path, skipped=False)
            status["process_wall_seconds"] = process_seconds
            completed += 1
            print(
                f"[roman-local-batch] {position}/{len(events)} done {event_id} "
                f"wall={process_seconds:.1f}s",
                flush=True,
            )
        else:
            failed += 1
            status = dict(statuses[event_id])
            status.update(
                {
                    "state": "failed",
                    "returncode": int(returncode),
                    "finished_at": _now(),
                    "process_wall_seconds": process_seconds,
                    "report_valid": report_path.is_file(),
                }
            )
            print(
                f"[roman-local-batch] {position}/{len(events)} FAILED {event_id} "
                f"returncode={returncode}; see {log_path}",
                flush=True,
            )
        statuses[event_id] = status
        _write_batch_status(status_path, statuses, selected_ids, "running")

    state = "complete" if failed == 0 else "failed"
    summary = {
        "schema": "hammlet-roman-local-batch-summary-v1",
        "state": state,
        "finished_at": _now(),
        "selected_events": len(events),
        "completed_events": completed,
        "failed_events": failed,
        "total_points_expected": sum(event["points"] for event in events),
        "wall_seconds": time.perf_counter() - started,
        "output_root": str(output_root),
        "event_reports": [
            str((events_dir / event["event_id"] / "report.json").resolve())
            for event in events
        ],
    }
    _write_json(output_root / "summary.json", summary)
    _write_batch_status(status_path, statuses, selected_ids, state)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if failed == 0 else 1


def _validate_args(args: argparse.Namespace) -> None:
    if args.cores < 1 or args.workers < 1 or args.threads_per_worker < 1:
        raise SystemExit("cores, workers, and threads-per-worker must be positive")
    if args.workers * args.threads_per_worker > args.cores:
        raise SystemExit("workers * threads-per-worker must not exceed cores")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")


def _read_and_preflight_events(input_root: Path, truth_path: Path) -> list[dict[str, Any]]:
    from tools.roman_local.roman_event import RomanReference, load_data_event

    with truth_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"truth file has no rows: {truth_path}")
    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        event_id = str(row.get("event_id", "")).strip()
        if not event_id or event_id in seen:
            raise SystemExit(f"truth file has a missing or duplicate event_id: {event_id!r}")
        seen.add(event_id)
        try:
            data_path = (input_root / "hammlet_input" / row["hammlet_lightcurve"]).resolve()
            t0 = float(row["t0_day"])
            u0 = float(row["u0"])
            tE = float(row["tE_days"])
            s = float(row["planet_s"])
            q = float(row["planet_q"])
            rho = float(row["rho"])
            expected_points = int(row["n_points"])
        except (KeyError, TypeError, ValueError) as error:
            raise SystemExit(f"invalid truth row for event {event_id!r}") from error
        if not data_path.is_file():
            raise SystemExit(f"missing data file for event {event_id}: {data_path}")
        event = load_data_event(
            [data_path],
            t0=t0,
            u0=u0,
            tE=tE,
            data_format="time-flux-error",
            reference=RomanReference(t0, u0, tE, s=s, q=q, rho=rho),
            name=event_id,
        )
        actual_points = event.total_points
        if actual_points != expected_points:
            raise SystemExit(
                f"event {event_id} has {actual_points} valid points, "
                f"truth.csv declares {expected_points}: {data_path}"
            )
        events.append(
            {
                "event_id": event_id,
                "data_path": str(data_path),
                "t0": t0,
                "u0": u0,
                "tE": tE,
                "s": s,
                "q": q,
                "rho": rho,
                "points": actual_points,
                "window": [float(event.window[0]), float(event.window[1])],
                "truth_row": row,
            }
        )
    return events


def _select_events(
    events: list[dict[str, Any]], event_ids: list[str] | None, limit: int | None
) -> list[dict[str, Any]]:
    if event_ids is not None:
        wanted = {str(value) for value in event_ids}
        unknown = sorted(wanted.difference(event["event_id"] for event in events))
        if unknown:
            raise SystemExit(f"requested event IDs are not in truth.csv: {unknown}")
        events = [event for event in events if event["event_id"] in wanted]
    if limit is not None:
        events = events[: int(limit)]
    return events


def _validate_atlas(atlas_path: Path, packed_path: Path) -> None:
    from tools.roman_local.atlas import open_readonly_atlas
    from tools.roman_local.packed import open_packed_atlas, validate_packed_source

    source = open_readonly_atlas(atlas_path)
    packed = open_packed_atlas(packed_path)
    try:
        validate_packed_source(packed, source)
    except ValueError as error:
        raise SystemExit(f"packed-cache/source validation failed: {error}") from error
    print(
        f"[roman-local-batch] atlas snapshot maps={source.total_maps}/"
        f"{source.expected_maps}; packed={packed.total_maps}; read-only check OK",
        flush=True,
    )


def _batch_config(
    args: argparse.Namespace,
    *,
    input_root: Path,
    truth_path: Path,
    atlas_path: Path,
    packed_path: Path,
    output_root: Path,
    jax_cache_dir: Path,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "hammlet-roman-local-batch-v1",
        "profile": "roman-local-parallel-packed-batch",
        "local_only": True,
        "atlas_read_only": True,
        "created_at": _now(),
        "input_root": str(input_root),
        "truth": str(truth_path),
        "atlas": str(atlas_path),
        "packed_cache": str(packed_path),
        "output_root": str(output_root),
        "jax_compilation_cache": str(jax_cache_dir),
        "search_settings": {
            "m_max": int(args.m_max),
            "n_alpha": int(args.n_alpha),
            "radial_order": int(args.radial_order),
            "batch_size": int(args.batch_size),
            "progress_every": int(args.progress_every),
            "candidate_count": int(args.candidate_count),
            "tail_mode": "auto",
            "window": "all-valid-input-rows",
            "chi2_guarantee": False,
        },
        "execution": {
            "cores": int(args.cores),
            "workers": int(args.workers),
            "threads_per_worker": int(args.threads_per_worker),
        },
        "events": [
            {
                "event_id": event["event_id"],
                "data_path": event["data_path"],
                "points": event["points"],
                "window": event["window"],
                "truth": {
                    key: event[key]
                    for key in ("t0", "u0", "tE", "s", "q", "rho")
                },
            }
            for event in events
        ],
    }


def _event_command(
    args: argparse.Namespace,
    event: dict[str, Any],
    atlas_path: Path,
    packed_path: Path,
    output: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "tools.roman_local.parallel_run",
        "--data-file",
        event["data_path"],
        "--data-format",
        "time-flux-error",
        "--t0",
        str(event["t0"]),
        "--u0",
        str(event["u0"]),
        "--tE",
        str(event["tE"]),
        "--s",
        str(event["s"]),
        "--q",
        str(event["q"]),
        "--rho",
        str(event["rho"]),
        "--atlas",
        str(atlas_path),
        "--packed-cache",
        str(packed_path),
        "--output",
        str(output),
        "--event-name",
        event["event_id"],
        "--m-max",
        str(args.m_max),
        "--n-alpha",
        str(args.n_alpha),
        "--radial-order",
        str(args.radial_order),
        "--tail-mode",
        "auto",
        "--batch-size",
        str(args.batch_size),
        "--progress-every",
        str(args.progress_every),
        "--candidate-count",
        str(args.candidate_count),
        "--cores",
        str(args.cores),
        "--workers",
        str(args.workers),
        "--threads-per-worker",
        str(args.threads_per_worker),
        "--resume",
    ]
    return command


def _child_environment(jax_cache_dir: Path) -> dict[str, str]:
    environment = os.environ.copy()
    roots = [str(SRC_ROOT), str(REPO_ROOT)]
    existing = environment.get("PYTHONPATH")
    if existing:
        roots.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(roots)
    environment["JAX_COMPILATION_CACHE_DIR"] = str(jax_cache_dir)
    return environment


def _completed_report_matches(
    report_path: Path, event: dict[str, Any], args: argparse.Namespace
) -> bool:
    if not report_path.is_file():
        return False
    try:
        report = _read_json(report_path)
        settings = report["search_settings"]
        report_event = report["event"]
        timing = report["timing"]
        return (
            report_event["name"] == event["event_id"]
            and int(report_event["total_points"]) == int(event["points"])
            and report_event["window"] == event["window"]
            and int(settings["m_max"]) == int(args.m_max)
            and int(settings["n_alpha"]) == int(args.n_alpha)
            and int(settings["radial_order"]) == int(args.radial_order)
            and float(timing.get("total_seconds", 0.0)) > 0.0
            and (report_path.parent / "minima.npz").is_file()
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def _completed_status_from_report(
    event: dict[str, Any], report_path: Path, *, skipped: bool
) -> dict[str, Any]:
    report = _read_json(report_path)
    timing = report.get("timing", {})
    return {
        "event_id": event["event_id"],
        "state": "complete",
        "points_expected": event["points"],
        "points_actual": report.get("event", {}).get("total_points"),
        "window": report.get("event", {}).get("window"),
        "output": str(report_path.parent),
        "report": str(report_path),
        "best_fft": report.get("best_fft"),
        "total_seconds": timing.get("total_seconds"),
        "parallel_wall_seconds": timing.get("parallel_wall_seconds"),
        "skipped_existing": bool(skipped),
        "finished_at": _now(),
    }


def _load_status(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    payload = _read_json(path)
    events = payload.get("events", {})
    return dict(events) if isinstance(events, dict) else {}


def _write_batch_status(
    path: Path,
    statuses: dict[str, dict[str, Any]],
    selected_ids: set[str],
    state: str,
) -> None:
    selected = {event_id: statuses[event_id] for event_id in selected_ids if event_id in statuses}
    _write_json(
        path,
        {
            "schema": "hammlet-roman-local-batch-status-v1",
            "state": state,
            "updated_at": _now(),
            "events": selected,
        },
    )


def _validate_resume_config(old: dict[str, Any], new: dict[str, Any]) -> None:
    for key in (
        "input_root",
        "truth",
        "atlas",
        "packed_cache",
        "output_root",
        "jax_compilation_cache",
    ):
        if key == "jax_compilation_cache" and key not in old:
            # Batch manifests written before the persistent JAX cache was
            # added can be resumed; the new cache path is recorded below.
            continue
        if old.get(key) != new.get(key):
            raise SystemExit(f"batch resume configuration changed: {key}")
    if old.get("search_settings") != new.get("search_settings"):
        raise SystemExit("batch resume search settings changed")


def _ensure_separate(output: Path, *protected: Path) -> None:
    output = output.resolve()
    for path in protected:
        path = path.resolve()
        if output == path or output in path.parents or path in output.parents:
            raise SystemExit(
                "--output-root must be separate from the atlas and packed cache"
            )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
