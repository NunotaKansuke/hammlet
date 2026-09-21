#!/usr/bin/env python3
"""Plot truth versus LM-recovered ``q`` and ``s`` for the Roman batch.

The PSPL--truth ``dchi2`` value is used only as an event-strength label.  The
LM fit itself is never selected by this script: all events in the supplied LM
summary are joined by event ID, and the optional cut is applied deterministically.
The q panel colors points by truth s; the s panel colors points by truth q.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="roman-lm-refine summary.json")
    parser.add_argument(
        "--pspl-truth",
        type=Path,
        required=True,
        help="all-point PSPL--truth dchi2 JSON indexed by event_id",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dchi2-min",
        type=float,
        default=100.0,
        help="retain events with dchi2 strictly above this value (default: 100)",
    )
    parser.add_argument(
        "--q-max",
        type=float,
        default=1.0e3,
        help="upper display limit for both q axes (default: 1e3)",
    )
    parser.add_argument(
        "--s-min",
        type=float,
        default=1.0e-2,
        help="lower display limit for both s axes (default: 1e-2)",
    )
    parser.add_argument(
        "--s-max",
        type=float,
        default=1.0e2,
        help="upper display limit for both s axes (default: 1e2)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.dchi2_min < 0.0:
        raise SystemExit("--dchi2-min must be non-negative")
    if args.q_max <= 0.0 or args.s_min <= 0.0 or args.s_max <= args.s_min:
        raise SystemExit("invalid positive plot limits")

    summary = _read(args.summary.expanduser().resolve())
    pspl = _read(args.pspl_truth.expanduser().resolve())
    lm_events = summary.get("events", [])
    pspl_events = pspl.get("events", [])
    if not isinstance(lm_events, list) or not isinstance(pspl_events, list):
        raise SystemExit("both inputs must contain an events array")
    pspl_by_id = {str(row["event_id"]): row for row in pspl_events}

    rows = []
    skipped = {"missing_pspl": 0, "invalid": 0, "below_cut": 0}
    for event in lm_events:
        event_id = str(event.get("event_id"))
        strength = pspl_by_id.get(event_id)
        truth = event.get("truth")
        optimized = event.get("optimized", {}).get("parameters", {})
        if strength is None:
            skipped["missing_pspl"] += 1
            continue
        try:
            dchi2 = float(strength["dchi2"])
            truth_q = float(truth["q"])
            truth_s = float(truth["s"])
            recovered_q = float(optimized["q"])
            recovered_s = float(optimized["s"])
        except (KeyError, TypeError, ValueError):
            skipped["invalid"] += 1
            continue
        if not all(value > 0.0 for value in (truth_q, truth_s, recovered_q, recovered_s)):
            skipped["invalid"] += 1
            continue
        if dchi2 <= args.dchi2_min:
            skipped["below_cut"] += 1
            continue
        rows.append(
            {
                "event_id": event_id,
                "dchi2": dchi2,
                "truth_q": truth_q,
                "truth_s": truth_s,
                "recovered_q": recovered_q,
                "recovered_s": recovered_s,
            }
        )
    if not rows:
        raise SystemExit("no events remain after the requested cut")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm

    truth_q = np.asarray([row["truth_q"] for row in rows])
    truth_s = np.asarray([row["truth_s"] for row in rows])
    recovered_q = np.minimum(
        np.asarray([row["recovered_q"] for row in rows]), args.q_max
    )
    recovered_s = np.asarray([row["recovered_s"] for row in rows])

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)
    cmap = "viridis"
    q_color_min = max(float(np.min(truth_s)), args.s_min)
    q_color_max = max(float(np.max(truth_s)), q_color_min * 1.01)
    s_color_min = max(float(np.min(truth_q)), 1.0e-6)
    s_color_max = max(float(np.max(truth_q)), s_color_min * 1.01)
    q_norm = LogNorm(vmin=q_color_min, vmax=q_color_max)
    s_norm = LogNorm(vmin=s_color_min, vmax=s_color_max)

    q_artist = axes[0].scatter(
        np.minimum(truth_q, args.q_max),
        recovered_q,
        c=truth_s,
        cmap=cmap,
        norm=q_norm,
        s=30,
        edgecolors="none",
        alpha=0.85,
    )
    axes[0].plot([1.0e-6, args.q_max], [1.0e-6, args.q_max], color="0.35", lw=1.0)
    axes[0].set(
        xscale="log",
        yscale="log",
        xlim=(1.0e-6, args.q_max),
        ylim=(1.0e-6, args.q_max),
        xlabel="truth q",
        ylabel="LM-recovered q",
        title=f"q ({len(rows)} events)",
    )
    figure.colorbar(q_artist, ax=axes[0], label="truth s")

    s_artist = axes[1].scatter(
        truth_s,
        np.clip(recovered_s, args.s_min, args.s_max),
        c=truth_q,
        cmap=cmap,
        norm=s_norm,
        s=30,
        edgecolors="none",
        alpha=0.85,
    )
    axes[1].plot(
        [args.s_min, args.s_max],
        [args.s_min, args.s_max],
        color="0.35",
        lw=1.0,
    )
    axes[1].set(
        xscale="log",
        yscale="log",
        xlim=(args.s_min, args.s_max),
        ylim=(args.s_min, args.s_max),
        xlabel="truth s",
        ylabel="LM-recovered s",
        title=f"s ({len(rows)} events)",
    )
    figure.colorbar(s_artist, ax=axes[1], label="truth q")
    for axis in axes:
        axis.grid(alpha=0.22, which="both")
        axis.set_aspect("equal", adjustable="box")
    figure.suptitle(
        rf"Roman injection--recovery: PSPL--truth $Deltachi^2>{args.dchi2_min:g}$"
    )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)

    sidecar = output.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "summary": str(args.summary.expanduser().resolve()),
                "pspl_truth": str(args.pspl_truth.expanduser().resolve()),
                "dchi2_min_exclusive": args.dchi2_min,
                "q_max": args.q_max,
                "s_limits": [args.s_min, args.s_max],
                "n_plotted": len(rows),
                "n_input_lm": len(lm_events),
                "skipped": skipped,
                "event_ids": [row["event_id"] for row in rows],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "sidecar": str(sidecar), "n_plotted": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
