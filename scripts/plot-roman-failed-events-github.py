#!/usr/bin/env python3
"""Render and index the normal-LM Roman events above a chi-square threshold."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = (
    REPO_ROOT
    / "results/roman_local/roman_alpha_direct_short_long_2048_vbmfix_dchi2gt100_batch176/summary.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "assets/roman-failure-events-chi2gt1p1"
PLOT_SCRIPT = REPO_ROOT / "tools/roman_local/plot_paper_event.py"

RAW_ROOTS = {
    "991": (
        Path(
            "/rogue1_8/nunota/gulls/runs/examples/"
            "roman_static_bound_planet_hammlet_batch100_vbmfix/gulls/raw/"
            "roman_static_bound_planet_hammlet_batch100_vbmfix"
        ),
        "roman_static_bound_planet_hammlet_batch100_vbmfix_0_0_",
    ),
    "992": (
        Path(
            "/rogue1_8/nunota/gulls/runs/examples/"
            "roman_static_binary_hammlet_signal_batch100_vbmfix/gulls/raw/"
            "roman_static_binary_hammlet_signal_batch100_vbmfix"
        ),
        "roman_static_binary_hammlet_signal_batch100_vbmfix_0_0_",
    ),
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=1.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _raw_lightcurve(event_id: str) -> Path:
    prefix = event_id[:3]
    try:
        root, stem = RAW_ROOTS[prefix]
    except KeyError as exc:
        raise ValueError(f"no raw-lightcurve mapping for event {event_id}") from exc
    return root / f"{stem}{event_id}.all.lc"


def _event_spec(row: dict, output_root: Path) -> dict:
    event_id = str(row["event_id"])
    event_json = Path(row["output"]).expanduser().resolve()
    with event_json.open(encoding="utf-8") as stream:
        record = json.load(stream)
    report_path = Path(record["report_path"]).expanduser().resolve()
    event_dir = report_path.parent
    with report_path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    reference = report["event"]["reference"]
    return {
        "event_id": event_id,
        "chi2_dof": float(row["best_chi2_dof"]),
        "s_true": float(reference["s"]),
        "q_true": float(reference["q"]),
        "event_json": event_json,
        "event_dir": event_dir,
        "raw_lightcurve": _raw_lightcurve(event_id),
        "full_output": output_root / f"{event_id}-full.png",
        "zoom_output": output_root / f"{event_id}-zoom.png",
    }


def _render_one(spec: dict, force: bool) -> dict:
    outputs_exist = spec["full_output"].is_file() and spec["zoom_output"].is_file()
    if outputs_exist and not force:
        return {**spec, "status": "cached"}
    command = [
        sys.executable,
        str(PLOT_SCRIPT),
        str(spec["event_dir"]),
        "--raw-lightcurve",
        str(spec["raw_lightcurve"]),
        "--lm-result",
        str(spec["event_json"]),
        "--q-log-max",
        "4",
        "--time-window",
        "-5",
        "25",
        "--show-all-caustics",
        "--no-inset",
        "--zoom-output",
        str(spec["zoom_output"]),
        "--output",
        str(spec["full_output"]),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{spec['event_id']} failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return {**spec, "status": "rendered"}


def _markdown(rows: list[dict], threshold: float, output_root: Path) -> str:
    rows = sorted(rows, key=lambda row: row["chi2_dof"], reverse=True)
    relative_root = output_root.relative_to(REPO_ROOT)
    lines = [
        "# Roman events with poor normal-LM fits",
        "",
        f"Normal-LM events with `chi2/dof > {threshold:g}`: **{len(rows)} of 176**.",
        "",
        "The figures use the same renderer and settings for every event:",
        "the full light curve and `(s, q)` map are in `*-full.png`, while the",
        "zoomed light curve and caustic geometry are in `*-zoom.png`. The zoom",
        "window is centered automatically on the strongest modeled feature, so",
        "it remains valid even when the event peak is near an observing-window edge.",
        "The normal FFT seed is dashed gray and the ordinary LM-polished model is",
        "solid orange. No fallback result or truth parameter is used as a fit.",
        "",
    ]
    for rank, row in enumerate(rows, 1):
        event_id = row["event_id"]
        lines.append(
            f"## {rank}. `{event_id}` — `chi2/dof = {row['chi2_dof']:.4f}`"
        )
        lines.extend(
            [
                f"truth: `s={row['s_true']:.5g}`, `q={row['q_true']:.5g}`",
                "",
                "<p>",
                f'<img src="./{event_id}-full.png" alt="{event_id} full figure" width="49%">',
                f'<img src="./{event_id}-zoom.png" alt="{event_id} zoom and caustics" width="49%">',
                "</p>",
                "",
            ]
        )
    lines.extend(
        [
            "## Provenance",
            "",
            f"- Result summary: `{DEFAULT_SUMMARY.relative_to(REPO_ROOT)}`",
            f"- Output directory: `{relative_root}`",
            "- Search: FFT with `n_alpha=2048`, direct VBM screening, short LM, then long LM.",
            "- Display: `log10(q)` from -5.8 to 4 and `t-t0` from -5 to 25 d; the zoom window is event-dependent.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _args()
    if args.threshold <= 0.0:
        raise SystemExit("--threshold must be positive")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    summary_path = args.summary.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    with summary_path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    rows = [
        row
        for row in summary
        if float(row["best_chi2_dof"]) > float(args.threshold)
    ]
    specs = [_event_spec(row, output_root) for row in rows]
    output_root.mkdir(parents=True, exist_ok=True)

    rendered: list[dict] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_render_one, spec, args.force): spec["event_id"]
            for spec in specs
        }
        for future in as_completed(futures):
            event_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - batch failure report
                failures.append(f"{event_id}: {exc}")
                print(f"FAILED {event_id}: {exc}", file=sys.stderr)
            else:
                rendered.append(result)
                print(
                    f"{result['status'].upper()} {event_id} "
                    f"chi2/dof={result['chi2_dof']:.4f}"
                )
    if failures:
        raise SystemExit("\n".join(failures))

    rendered.sort(key=lambda row: row["chi2_dof"], reverse=True)
    manifest = [
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in row.items()
            if key not in {"event_dir", "event_json", "raw_lightcurve"}
        }
        for row in rendered
    ]
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "threshold": args.threshold,
                "event_count": len(rendered),
                "events": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_root / "README.md").write_text(
        _markdown(rendered, args.threshold, output_root),
        encoding="utf-8",
    )
    print(f"WROTE {output_root / 'README.md'}")
    print(f"EVENTS {len(rendered)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
