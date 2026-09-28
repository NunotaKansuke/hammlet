#!/usr/bin/env python3
"""Compare recovered and injected fit quality for the Roman benchmark.

The script uses the model column written by GULLS for the injected truth, then
writes two paper-style figures:

* a distribution of ``chi2_recovered - chi2_truth`` and an ECDF of reduced
  chi-square for recovered and truth models;
* folded-q and s recovery panels coloured by the recovered reduced chi-square.

The input summary may be either the original object-style LM summary or the
list-style summary written by the alpha-bank direct/short/long-LM runner.  In
the latter case the per-event records and their FFT reports are loaded and
normalised before plotting.

The GULLS model column is used deliberately here. Re-evaluating the truth
with an independent direct-VBM implementation can mix model-coordinate
conventions for a subset of wide, low-q events; the raw GULLS model is the
actual injected model against which the benchmark data were generated.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
from pathlib import Path
import numpy as np

def _read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    return payload


def _read(path: Path) -> dict:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON object expected: {path}")
    return payload


def _numeric_event_id(value: str) -> tuple[int, str]:
    try:
        return 0, f"{int(value):020d}"
    except ValueError:
        return 1, value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="main roman-lm summary.json")
    parser.add_argument("--extra-event", action="append", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="quality figure PNG")
    parser.add_argument("--qs-output", type=Path, required=True, help="q/s figure PNG")
    parser.add_argument(
        "--pspl-truth",
        type=Path,
        help="PSPL--truth dchi2 JSON used to select the paper q/s sample",
    )
    parser.add_argument(
        "--dchi2-min",
        type=float,
        default=100.0,
        help="strict lower cut for the paper q/s sample (default: 100)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, 200),
        help="workers for GULLS truth evaluations (default: 8)",
    )
    parser.add_argument(
        "--gulls-root",
        type=Path,
        default=Path("/rogue1_8/nunota/gulls/runs/examples"),
        help="GULLS examples root containing the two benchmark runs",
    )
    parser.add_argument(
        "--gulls-dataset-suffix",
        default="",
        help="suffix appended to the configured GULLS dataset names",
    )
    parser.add_argument("--q-min", type=float, default=1.0e-5)
    parser.add_argument("--q-max", type=float, default=1.0e1)
    parser.add_argument("--s-min", type=float, default=1.0e-1)
    parser.add_argument("--s-max", type=float, default=1.0e1)
    parser.add_argument(
        "--quality-vmax-percentile",
        type=float,
        default=95.0,
        help="upper percentile for the q/s colour scale (default: 95)",
    )
    return parser.parse_args()


_GULLS_DATASETS = {
    "991": "roman_static_bound_planet_hammlet_batch100",
    "992": "roman_static_binary_hammlet_signal_batch100",
}


def _resolve_gulls_truth_path(
    event_id: str, gulls_root: Path, dataset_suffix: str = ""
) -> Path:
    event_id = str(event_id)
    try:
        dataset = _GULLS_DATASETS[event_id[:3]]
    except KeyError as error:
        raise FileNotFoundError(
            f"no GULLS raw root is configured for event {event_id}"
        ) from error
    dataset = f"{dataset}{dataset_suffix}"
    root = gulls_root.expanduser().resolve() / dataset / "gulls" / "raw" / dataset
    matches = sorted(root.glob(f"*_{event_id}.all.lc"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one GULLS raw truth file for {event_id}, found {matches}"
        )
    return matches[0]


def _truth_chi2(event: dict) -> dict:
    import numpy as local_np

    event_id = str(event["event_id"])
    raw_path = Path(event["truth_raw_path"]).expanduser().resolve()
    raw = local_np.loadtxt(raw_path, comments="#")
    if raw.ndim != 2 or raw.shape[1] < 4:
        raise ValueError(f"unexpected GULLS raw format: {raw_path}")
    observed = raw[:, 1]
    error = raw[:, 2]
    injected_model = raw[:, 3]
    if not (
        local_np.all(local_np.isfinite(observed))
        and local_np.all(local_np.isfinite(error))
        and local_np.all(local_np.isfinite(injected_model))
        and local_np.all(error > 0.0)
    ):
        raise ValueError(f"invalid GULLS truth columns: {raw_path}")
    chi2 = float(local_np.sum(((observed - injected_model) / error) ** 2))
    return {
        "event_id": event_id,
        "truth_chi2": chi2,
        "n_data_truth": int(raw.shape[0]),
        "truth_source": str(raw_path),
    }


def _evaluate_truth(events: list[dict], workers: int) -> dict[str, dict]:
    if workers < 1:
        raise SystemExit("--workers must be positive")
    results: dict[str, dict] = {}
    if workers == 1:
        for index, event in enumerate(events, 1):
            result = _truth_chi2(event)
            results[result["event_id"]] = result
            print(f"[truth-chi2] {index}/{len(events)} {result['event_id']}", flush=True)
        return results
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(_truth_chi2, event): event for event in events}
        completed = 0
        for future in as_completed(pending):
            result = future.result()
            results[result["event_id"]] = result
            completed += 1
            if completed % 10 == 0 or completed == len(events):
                print(
                    f"[truth-chi2] {completed}/{len(events)} {result['event_id']}",
                    flush=True,
                )
    return results


def _normalise_event(event: dict, source: Path) -> dict:
    """Return the common event shape used by the plotting code.

    The older LM runner stores ``truth``, ``n_data`` and ``optimized`` in
    each event record.  The 2048 alpha-bank runner stores the recovered
    parameters in ``long_lm`` and keeps the injected reference in the FFT
    report, so join those two records here.
    """
    if {"event_id", "n_data", "truth", "optimized"}.issubset(event):
        return event

    long_lm = event.get("long_lm")
    report_name = event.get("report_path")
    if not isinstance(long_lm, dict) or not isinstance(report_name, str):
        raise SystemExit(f"not a supported LM event record: {source}")
    report_path = Path(report_name).expanduser().resolve()
    report = _read(report_path)
    report_event = report.get("event")
    if not isinstance(report_event, dict):
        raise SystemExit(f"FFT report has no event section: {report_path}")
    reference = report_event.get("reference")
    recovered_parameters = long_lm.get("optimized_parameters")
    recovered_chi2 = long_lm.get("direct_vbm_chi2")
    if not isinstance(reference, dict) or not isinstance(recovered_parameters, dict):
        raise SystemExit(f"incomplete alpha-bank event record: {source}")
    if recovered_chi2 is None:
        raise SystemExit(f"alpha-bank event has no long-LM chi2: {source}")
    try:
        n_data = int(report_event["total_points"])
        float(recovered_chi2)
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"invalid alpha-bank event metadata: {source}") from error
    return {
        "event_id": str(event.get("event_id")),
        "n_data": n_data,
        "truth": reference,
        "optimized": {
            "parameters": recovered_parameters,
            "direct_vbm_chi2": float(recovered_chi2),
        },
    }


def _events_from_summary(summary_path: Path) -> list[dict]:
    payload = _read_json(summary_path)
    if isinstance(payload, list):
        events = []
        for entry in payload:
            if not isinstance(entry, dict) or not isinstance(entry.get("output"), str):
                raise SystemExit(
                    f"list-style summary has an invalid event entry: {summary_path}"
                )
            event_path = Path(entry["output"]).expanduser().resolve()
            events.append(_normalise_event(_read(event_path), event_path))
        return events
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise SystemExit("summary must contain an events array or be a summary list")
    return [
        _normalise_event(event, summary_path)
        for event in payload["events"]
        if isinstance(event, dict)
    ]


def _merge_events(summary_path: Path, extra_paths: list[Path]) -> tuple[list[dict], list[str]]:
    merged = _events_from_summary(summary_path)
    resolved_extra = []
    seen = {str(event.get("event_id")) for event in merged}
    for path in extra_paths:
        resolved = path.expanduser().resolve()
        event = _normalise_event(_read(resolved), resolved)
        event_id = str(event.get("event_id"))
        if event_id in seen:
            raise SystemExit(f"duplicate event ID in extra input: {event_id}")
        merged.append(event)
        seen.add(event_id)
        resolved_extra.append(str(resolved))
    merged.sort(key=lambda event: _numeric_event_id(str(event["event_id"])))
    return merged, resolved_extra


def _attach_truth_paths(
    events: list[dict], gulls_root: Path, dataset_suffix: str = ""
) -> None:
    for event in events:
        event["truth_raw_path"] = str(
            _resolve_gulls_truth_path(
                str(event["event_id"]), gulls_root, dataset_suffix
            )
        )


def _build_rows(events: list[dict], truth_results: dict[str, dict]) -> list[dict]:
    rows = []
    for event in events:
        event_id = str(event["event_id"])
        truth = truth_results.get(event_id)
        if truth is None:
            raise SystemExit(f"missing truth chi2 for event {event_id}")
        n_data = int(event["n_data"])
        dof = n_data - 9
        if int(truth["n_data_truth"]) != n_data:
            raise SystemExit(
                f"truth/data length mismatch for {event_id}: "
                f"{truth['n_data_truth']} != {n_data}"
            )
        recovered_chi2 = float(event["optimized"]["direct_vbm_chi2"])
        truth_chi2 = float(truth["truth_chi2"])
        truth_parameters = event["truth"]
        recovered_parameters = event["optimized"]["parameters"]
        rows.append(
            {
                "event_id": event_id,
                "n_data": n_data,
                "dof": dof,
                "recovered_chi2": recovered_chi2,
                "truth_chi2": truth_chi2,
                "delta_chi2_recovered_minus_truth": recovered_chi2 - truth_chi2,
                "recovered_chi2_dof": recovered_chi2 / dof,
                "truth_chi2_dof": truth_chi2 / dof,
                "truth_q": float(truth_parameters["q"]),
                "recovered_q": float(recovered_parameters["q"]),
                "truth_s": float(truth_parameters["s"]),
                "recovered_s": float(recovered_parameters["s"]),
            }
        )
    return rows


def _select_paper_qs_rows(
    rows: list[dict],
    pspl_by_id: dict[str, dict] | None,
    dchi2_min: float,
) -> tuple[list[dict], dict[str, int]]:
    if pspl_by_id is None:
        return list(rows), {"missing_pspl": 0, "below_cut": 0}
    selected = []
    skipped = {"missing_pspl": 0, "below_cut": 0}
    for row in rows:
        strength = pspl_by_id.get(row["event_id"])
        if strength is None:
            skipped["missing_pspl"] += 1
            continue
        try:
            dchi2 = float(strength["dchi2"])
        except (KeyError, TypeError, ValueError):
            skipped["missing_pspl"] += 1
            continue
        if dchi2 <= dchi2_min:
            skipped["below_cut"] += 1
            continue
        selected.append(row)
    return selected, skipped


def _style_axis(axis) -> None:
    axis.xaxis.label.set_size(15)
    axis.yaxis.label.set_size(15)
    axis.tick_params(axis="both", which="major", labelsize=12.5)
    axis.tick_params(axis="both", which="minor", length=2.5)


def _plot_quality(
    rows: list[dict],
    output: Path,
    extra_paths: list[str],
    gulls_root: Path,
    gulls_dataset_suffix: str,
    selection_metadata: dict | None = None,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    delta = np.asarray(
        [row["delta_chi2_recovered_minus_truth"] for row in rows], dtype=float
    )
    recovered = np.asarray([row["recovered_chi2_dof"] for row in rows], dtype=float)

    sorted_values = np.sort(recovered)
    fraction = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    figure, ecdf_axis = plt.subplots(figsize=(7.2, 4.8))
    figure.subplots_adjust(left=0.18, right=0.98, bottom=0.22, top=0.96)
    ecdf_axis.step(
        sorted_values,
        fraction,
        where="post",
        color="#d95f02",
        linewidth=2.0,
    )
    ecdf_axis.set_xscale("log")
    ecdf_axis.set_xlim(
        max(0.8, float(np.min(recovered)) * 0.9),
        float(np.max(recovered)) * 1.1,
    )
    ecdf_axis.set_ylim(0.0, 1.02)
    ecdf_axis.set_xlabel(r"$\chi^2/\nu$")
    ecdf_axis.set_ylabel("Cumulative fraction")
    _style_axis(ecdf_axis)

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        dpi=240,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.10,
    )
    plt.close(figure)

    thresholds = {}
    for threshold in (2.0, 5.0, 10.0):
        thresholds[str(threshold)] = {
            "recovered": float(np.mean(recovered < threshold)),
        }
    metadata = {
        "figure": str(output),
        "extra_event_files": extra_paths,
        "gulls_root": str(gulls_root.expanduser().resolve()),
        "gulls_dataset_suffix": gulls_dataset_suffix,
        "n_events": len(rows),
        "selection": selection_metadata,
        "degrees_of_freedom": "n_data - 9",
        "truth_chi2_definition": (
            "sum((GULLS raw observed - GULLS raw model column 4)^2 / "
            "GULLS raw error^2)"
        ),
        "quality_threshold_fractions": thresholds,
        "delta_chi2_summary": {
            "min": float(np.min(delta)),
            "median": float(np.median(delta)),
            "max": float(np.max(delta)),
            "negative_fraction": float(np.mean(delta < 0.0)),
        },
        "rows": rows,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def _plot_qs(
    rows: list[dict],
    output: Path,
    *,
    q_min: float,
    q_max: float,
    s_min: float,
    s_max: float,
    extra_paths: list[str],
    gulls_root: Path,
    gulls_dataset_suffix: str,
    selection_metadata: dict,
    quality_vmax_percentile: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    truth_q = np.minimum(
        np.asarray([row["truth_q"] for row in rows], dtype=float),
        1.0 / np.asarray([row["truth_q"] for row in rows], dtype=float),
    )
    recovered_q_raw = np.asarray(
        [row["recovered_q"] for row in rows], dtype=float
    )
    recovered_q = np.minimum(recovered_q_raw, 1.0 / recovered_q_raw)
    truth_s = np.asarray([row["truth_s"] for row in rows], dtype=float)
    recovered_s = np.asarray([row["recovered_s"] for row in rows], dtype=float)
    quality = np.asarray([row["recovered_chi2_dof"] for row in rows], dtype=float)

    colour_min = max(0.9, float(np.min(quality)))
    colour_max = max(
        float(np.percentile(quality, quality_vmax_percentile)),
        colour_min * 1.01,
    )
    norm = LogNorm(vmin=colour_min, vmax=colour_max, clip=True)
    figure = plt.figure(figsize=(13.8, 6.4))
    grid = figure.add_gridspec(
        1,
        2,
        left=0.08,
        right=0.90,
        bottom=0.13,
        top=0.96,
        wspace=0.18,
    )
    q_axis = figure.add_subplot(grid[0, 0])
    s_axis = figure.add_subplot(grid[0, 1])
    scatter = q_axis.scatter(
        np.clip(truth_q, q_min, q_max),
        np.clip(recovered_q, q_min, q_max),
        c=quality,
        cmap="viridis",
        norm=norm,
        s=42,
        alpha=0.72,
        edgecolors="none",
        rasterized=True,
        zorder=3,
    )
    q_axis.plot(
        [q_min, q_max],
        [q_min, q_max],
        color="#333333",
        linewidth=1.25,
        zorder=2,
    )
    q_axis.set(
        xscale="log",
        yscale="log",
        xlim=(q_min, q_max),
        ylim=(q_min, q_max),
        xlabel=r"True $q$",
        ylabel=r"Recovered $q$",
    )
    s_axis.scatter(
        np.clip(truth_s, s_min, s_max),
        np.clip(recovered_s, s_min, s_max),
        c=quality,
        cmap="viridis",
        norm=norm,
        s=42,
        alpha=0.72,
        edgecolors="none",
        rasterized=True,
        zorder=3,
    )
    s_axis.plot(
        [s_min, s_max],
        [s_min, s_max],
        color="#333333",
        linewidth=1.25,
        zorder=2,
    )
    s_axis.plot(
        [s_min, s_max],
        [1.0 / s_min, 1.0 / s_max],
        color="#666666",
        linewidth=1.15,
        linestyle=":",
        zorder=2,
    )
    s_axis.set(
        xscale="log",
        yscale="log",
        xlim=(s_min, s_max),
        ylim=(s_min, s_max),
        xlabel=r"True $s$",
        ylabel=r"Recovered $s$",
    )
    for axis in (q_axis, s_axis):
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.18, which="both", linewidth=0.55)
        _style_axis(axis)
    # Place the colour bar beside the settled equal-aspect panel so that its
    # height matches the panel exactly and the gap stays compact.
    figure.canvas.draw()
    s_position = s_axis.get_position()
    colour_axis = figure.add_axes(
        [s_position.x1 + 0.012, s_position.y0, 0.018, s_position.height]
    )
    colourbar = figure.colorbar(scatter, cax=colour_axis)
    colourbar.set_label(r"Recovered $\chi^2/\nu$", fontsize=15)
    colourbar.ax.tick_params(labelsize=11)

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, facecolor="white")
    plt.close(figure)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "figure": str(output),
                "extra_event_files": extra_paths,
                "gulls_root": str(gulls_root.expanduser().resolve()),
                "gulls_dataset_suffix": gulls_dataset_suffix,
                "n_events": len(rows),
                "selection": selection_metadata,
                "colour": "recovered_chi2_dof",
                "colour_vmax_percentile": quality_vmax_percentile,
                "colour_vmax": colour_max,
                "q_limits": [q_min, q_max],
                "q_folded_at_one": True,
                "s_limits": [s_min, s_max],
                "inverse_s_line": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    if args.dchi2_min < 0.0:
        raise SystemExit("--dchi2-min must be non-negative")
    if not 0.0 < args.quality_vmax_percentile <= 100.0:
        raise SystemExit("--quality-vmax-percentile must be in (0, 100]")
    if not (
        0.0 < args.q_min < args.q_max
        and 0.0 < args.s_min < args.s_max
    ):
        raise SystemExit("invalid q/s limits")
    summary_path = args.summary.expanduser().resolve()
    events, extra_paths = _merge_events(
        summary_path,
        [path for path in (args.extra_event or [])],
    )
    gulls_root = args.gulls_root.expanduser().resolve()
    _attach_truth_paths(events, gulls_root, args.gulls_dataset_suffix)
    truth_results = _evaluate_truth(events, args.workers)
    rows = _build_rows(events, truth_results)
    pspl_by_id = None
    pspl_truth_path = None
    if args.pspl_truth is not None:
        pspl_truth_path = args.pspl_truth.expanduser().resolve()
        pspl_payload = _read(pspl_truth_path)
        pspl_events = pspl_payload.get("events")
        if not isinstance(pspl_events, list):
            raise SystemExit("--pspl-truth must contain an events array")
        pspl_by_id = {str(row["event_id"]): row for row in pspl_events}
    qs_rows, qs_skipped = _select_paper_qs_rows(
        rows,
        pspl_by_id,
        args.dchi2_min,
    )
    if not qs_rows:
        raise SystemExit("no events remain for the q/s paper sample")
    selection_metadata = {
        "pspl_truth": str(pspl_truth_path) if pspl_truth_path else None,
        "dchi2_min_exclusive": args.dchi2_min if pspl_by_id is not None else None,
        "n_input_events": len(rows),
        "n_plotted": len(qs_rows),
        "skipped": qs_skipped,
    }
    quality_rows = qs_rows if pspl_by_id is not None else rows
    metadata = _plot_quality(
        quality_rows,
        args.output,
        extra_paths,
        gulls_root,
        args.gulls_dataset_suffix,
        selection_metadata,
    )
    _plot_qs(
        qs_rows,
        args.qs_output,
        q_min=args.q_min,
        q_max=args.q_max,
        s_min=args.s_min,
        s_max=args.s_max,
        extra_paths=extra_paths,
        gulls_root=gulls_root,
        gulls_dataset_suffix=args.gulls_dataset_suffix,
        selection_metadata=selection_metadata,
        quality_vmax_percentile=args.quality_vmax_percentile,
    )
    print(
        json.dumps(
            {
                "quality_output": str(args.output.expanduser().resolve()),
                "qs_output": str(args.qs_output.expanduser().resolve()),
                "n_events": len(quality_rows),
                "n_qs_events": len(qs_rows),
                "thresholds": metadata["quality_threshold_fractions"],
                "delta_summary": metadata["delta_chi2_summary"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
