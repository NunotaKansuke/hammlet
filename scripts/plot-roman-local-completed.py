#!/usr/bin/env python3
"""Make one horizontal light-curve/map image for every completed event."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PLOT_SCRIPT = REPO_ROOT / "scripts" / "plot-roman-local.sh"
EVENT_SCRIPT = REPO_ROOT / "scripts" / "plot-roman-local-event.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_root", type=Path, help="batch result directory containing status.json")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of plotting subprocesses (default: 4)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="regenerate the two source plots even when they already exist",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="override the directory containing the data files",
    )
    parser.add_argument(
        "--min-zoom-half-width",
        type=float,
        default=None,
        help="minimum light-curve half-width in days",
    )
    parser.add_argument(
        "--lightcurve-only",
        action="store_true",
        help="refresh only lightcurve.png and reuse each existing chi2_map.png",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    batch_root = args.batch_root.expanduser().resolve()
    status_path = batch_root / "status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"could not read {status_path}") from error
    if not isinstance(status, dict) or not isinstance(status.get("events"), dict):
        raise SystemExit(f"invalid batch status: {status_path}")

    event_ids = sorted(
        str(event_id)
        for event_id, entry in status["events"].items()
        if isinstance(entry, dict) and entry.get("state") == "complete"
    )
    if not event_ids:
        raise SystemExit("no completed events in status.json")

    def process(event_id: str) -> tuple[str, str]:
        event = batch_root / "events" / event_id
        if not event.is_dir():
            raise RuntimeError(f"missing event directory: {event}")
        lightcurve = event / "figures" / "lightcurve.png"
        chi2_map = event / "figures" / "chi2_map.png"
        if args.refresh or not (lightcurve.is_file() and chi2_map.is_file()):
            plot_command = ["bash", str(PLOT_SCRIPT), str(event)]
            if args.data_root is not None:
                plot_command.extend(["--data-root", str(args.data_root)])
            if args.min_zoom_half_width is not None:
                plot_command.extend(
                    ["--min-zoom-half-width", str(args.min_zoom_half_width)]
                )
            if args.lightcurve_only:
                plot_command.append("--skip-map")
            result = subprocess.run(
                plot_command,
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(f"plot failed for {event_id}: {detail}")
        result = subprocess.run(
            [sys.executable, str(EVENT_SCRIPT), str(event)],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"summary failed for {event_id}: {detail}")
        return event_id, result.stdout.strip().splitlines()[-1]

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, event_id): event_id for event_id in event_ids}
        for future in as_completed(futures):
            event_id = futures[future]
            try:
                _, output = future.result()
            except Exception as error:  # noqa: BLE001 - report all event failures
                failures.append(f"{event_id}: {error}")
                print(f"[FAIL] {failures[-1]}", file=sys.stderr)
            else:
                print(f"[OK] {event_id}: {output}")
    if failures:
        raise SystemExit(f"{len(failures)} event(s) failed")
    print(f"created one summary image for {len(event_ids)} completed events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
