#!/usr/bin/env python3
"""Run the local Roman FFT scan with a packed-cache process pool.

This command is deliberately local-only.  ``--atlas`` is opened for a
read-only snapshot check, while all coefficient reads come from
``--packed-cache``.  The cache and the result directory must be separate from
the production map tree.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--catalog", type=Path)
    input_group.add_argument("--data-file", action="append")
    input_group.add_argument("--gulls-lightcurve", type=Path)
    parser.add_argument("--gulls-manifest", type=Path)
    parser.add_argument("--event-id")
    parser.add_argument("--simulation-root", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--anomaly-product", type=Path)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--packed-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--event-name", default="local-event")
    parser.add_argument("--band", action="append", dest="bands")
    parser.add_argument("--band-name", action="append", dest="band_names")
    parser.add_argument(
        "--data-format", choices=("roman-mag", "time-flux-error"), default="roman-mag"
    )
    parser.add_argument("--time-offset", type=float, default=8234.0)
    parser.add_argument("--window-tE", type=float)
    parser.add_argument("--window-start", type=float)
    parser.add_argument("--window-end", type=float)
    parser.add_argument("--t0", type=float)
    parser.add_argument("--u0", type=float)
    parser.add_argument("--tE", type=float)
    parser.add_argument("--s", type=float)
    parser.add_argument("--q", type=float)
    parser.add_argument("--rho", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument(
        "--geometry-center",
        choices=("baseline", "truth", "map-truth"),
        default="baseline",
    )
    parser.add_argument(
        "--geometry-stencil", choices=("full", "te-only", "center"), default="full"
    )
    parser.add_argument("--geometry-dt0", type=float)
    parser.add_argument("--geometry-du0", type=float)
    parser.add_argument("--geometry-dlog-te", type=float)
    parser.add_argument("--geometry-te-steps", type=int, default=3)
    parser.add_argument("--no-extra-geometry", action="store_true")
    parser.add_argument("--m-max", type=int, default=128)
    parser.add_argument("--n-alpha", type=int, default=540)
    parser.add_argument("--radial-order", type=int, choices=(1, 3), default=1)
    parser.add_argument("--tail-mode", choices=("auto", "off"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--compute-dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--logq-max", type=float)
    parser.add_argument("--max-maps", type=int)
    parser.add_argument("--candidate-count", type=int, default=100)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument(
        "--workers",
        type=int,
        help="number of spawned bucket workers (default: cores)",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=1,
        help="thread budget and CPU affinity width per worker (default: 1)",
    )
    parser.add_argument(
        "--reference-result",
        type=Path,
        help="serial or previous minima result used for numerical comparison",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse validated bucket-results already present below --output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate event, source snapshot, and cache without scanning",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from tools.roman_local.atlas import open_readonly_atlas
    from tools.roman_local.parallel import (
        configure_parallel_execution,
        scan_packed_atlas,
    )
    from tools.roman_local.packed import open_packed_atlas, validate_packed_source
    from tools.roman_local.roman_event import (
        RomanReference,
        geometry_stencil,
        load_catalog_event,
        load_data_event,
        load_gulls_event,
    )
    from tools.roman_local.run import (
        _build_map_filter,
        _explicit_window,
        _filter_count,
        _load_event,
        _report,
        _required_radial_radius,
        _geometry_center,
    )

    _validate_args(args)
    explicit_window = _explicit_window(args)
    event = _load_event(
        args,
        RomanReference,
        load_catalog_event,
        load_data_event,
        load_gulls_event,
        explicit_window,
    )
    source = open_readonly_atlas(args.atlas)
    packed_full = open_packed_atlas(args.packed_cache)
    try:
        validate_packed_source(packed_full, source)
    except ValueError as error:
        raise SystemExit(f"packed-cache/source validation failed: {error}") from error
    packed = (
        packed_full
        if args.max_maps is None
        else open_packed_atlas(args.packed_cache, max_maps=args.max_maps)
    )
    if args.m_max > packed.m_max:
        raise SystemExit(
            f"requested --m-max={args.m_max}, but packed cache stores only M={packed.m_max}"
        )
    if args.candidate_count < 1:
        raise SystemExit("--candidate-count must be positive")

    output = _result_path(args, event.name)
    _ensure_separate(output, source.path, packed.path)
    center, center_metadata = _geometry_center(event, args.geometry_center)
    extra = None if args.no_extra_geometry else (-0.5, -0.5, 0.5)
    geometries, stencil_report = geometry_stencil(
        center,
        preset=args.geometry_stencil,
        dt0=args.geometry_dt0,
        du0=args.geometry_du0,
        dlog_te=args.geometry_dlog_te,
        te_steps=args.geometry_te_steps,
        extra=extra,
    )
    stencil_report.update(center_metadata)
    required_radius = None
    if args.tail_mode == "auto":
        required_radius = _required_radial_radius(event, geometries)
        packed = packed.with_point_lens_tail(
            required_radius, radial_order=args.radial_order
        )
    map_filter = _build_map_filter(args.logq_max)
    filter_kept = _filter_count(packed.map_parameters, map_filter)
    execution = configure_parallel_execution(
        args.cores,
        args.workers,
        args.threads_per_worker,
        bucket_count=len(packed.buckets),
    )
    plan = {
        "profile": "roman-local-parallel-packed",
        "local_only": True,
        "atlas_read_only": True,
        "source_atlas": str(source.path),
        "packed_cache": str(packed.path),
        "packed_snapshot": packed.metadata,
        "event": event.as_dict(),
        "atlas": {
            "path": str(packed.path),
            "kind": packed.kind,
            "stored_m_max": packed.m_max,
            "selected_maps": packed.total_maps,
            "completed_maps": packed.completed_maps,
            "expected_maps": packed.expected_maps,
        },
        "execution": execution,
        "search_settings": {
            "m_max": args.m_max,
            "n_alpha": args.n_alpha,
            "radial_order": args.radial_order,
            "batch_size": args.batch_size,
            "compute_dtype": args.compute_dtype,
            "map_filter_logq_max": args.logq_max,
            "tail_mode": args.tail_mode,
            "maps_after_filter": filter_kept,
            "error_certificate": {
                "enabled": False,
                "mode": "single_resolution",
                "chi2_guarantee": False,
            },
        },
        "geometry_stencil": stencil_report,
        "geometry_count": len(geometries),
        "output": str(output),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    previous_report = None
    if args.resume and (output / "report.json").is_file():
        try:
            previous_report = json.loads(
                (output / "report.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            previous_report = None
    _create_output(output, resume=args.resume)
    scan_started = time.perf_counter()
    scan, parallel_timing, execution = scan_packed_atlas(
        packed,
        event.datasets,
        geometries,
        m_max=args.m_max,
        n_alpha=args.n_alpha,
        batch_size=args.batch_size,
        compute_dtype=args.compute_dtype,
        radial_order=args.radial_order,
        cores=args.cores,
        workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        max_maps=args.max_maps,
        required_radius=required_radius,
        map_filter_logq_max=args.logq_max,
        bucket_result_dir=output / "bucket-results",
        resume=args.resume,
        progress_every=args.progress_every,
    )
    total_scan_wall = time.perf_counter() - scan_started

    import numpy as np

    np.savez_compressed(
        output / "minima.npz",
        map_ids=scan.map_ids,
        parameters=scan.parameters,
        chi2=scan.chi2,
        geometry_index=scan.geometry_index,
        alpha_index=scan.alpha_index,
        scanned=scan.scanned,
        geometries=np.asarray(
            [[geometry.t0, geometry.u0, geometry.tE] for geometry in geometries],
            dtype=np.float64,
        ),
    )
    candidates = scan.candidates(
        n_alpha=args.n_alpha,
        count=args.candidate_count,
        rescanned_only=True,
    )
    (output / "candidates.json").write_text(
        json.dumps([asdict(candidate) for candidate in candidates], indent=2) + "\n",
        encoding="utf-8",
    )
    factory_info = SimpleNamespace(
        basis_seconds=parallel_timing["kernel_factory_basis_seconds"],
        build_seconds=parallel_timing["kernel_factory_binding_seconds"],
    )
    report = _report(
        args,
        event,
        packed,
        geometries,
        stencil_report,
        execution,
        scan,
        filter_kept,
        factory_wall_seconds=float(parallel_timing["kernel_factory_basis_seconds"]),
        kernel_factory=factory_info,
        scan_wall_seconds=float(parallel_timing["parallel_wall_seconds"]),
        scan_timing=parallel_timing,
        output=output,
    )
    report["profile"] = "roman-local-parallel-packed"
    report["atlas"]["path"] = str(source.path)
    report["atlas"]["kind"] = packed.kind
    report["atlas"]["source_path"] = str(source.path)
    report["atlas"]["packed_cache"] = str(packed.path)
    report["atlas"]["metadata"]["completion_count_exact"] = bool(
        source.metadata.get("completion_count_exact", True)
    )
    report["maps"]["completed_count_exact"] = bool(
        source.metadata.get("completion_count_exact", True)
    )
    report["source_atlas"] = {
        "path": str(source.path),
        "kind": source.kind,
        "selected_maps": source.total_maps,
        "completed_maps": source.completed_maps,
        "expected_maps": source.expected_maps,
    }
    report["execution"] = execution
    report["parallel"] = parallel_timing
    report["timing"]["parallel_wall_seconds"] = float(
        parallel_timing["parallel_wall_seconds"]
    )
    report["timing"]["outer_scan_wall_seconds"] = float(total_scan_wall)
    report["timing"]["total_seconds"] = float(total_scan_wall)
    if isinstance(previous_report, dict):
        previous_parallel = previous_report.get("parallel")
        if isinstance(previous_parallel, dict):
            report["previous_run"] = {
                "report": str((output / "report.json").resolve()),
                "parallel": previous_parallel,
                "timing": previous_report.get("timing"),
                "best_fft": previous_report.get("best_fft"),
                "note": "Preserved before --resume re-combined bucket results.",
            }
    if args.reference_result is not None:
        report["reference_comparison"] = _compare_reference(
            args.reference_result, scan
        )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "best": report["best_fft"],
                "parallel_wall_seconds": parallel_timing["parallel_wall_seconds"],
                "cpu_utilization": parallel_timing["cpu_utilization"],
                "reference_comparison": report.get("reference_comparison"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    from tools.roman_local.run import _validate_search_args

    _validate_search_args(args)
    if args.workers is not None and args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.threads_per_worker < 1:
        raise SystemExit("--threads-per-worker must be positive")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be non-negative")
def _result_path(args: argparse.Namespace, event_name: str) -> Path:
    if args.output is not None:
        return args.output.expanduser().resolve()
    safe = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in event_name
    )
    workers = args.workers if args.workers is not None else args.cores
    return (
        REPO_ROOT
        / "results"
        / "roman_local"
        / f"roman_{safe}_m{args.m_max}_{workers}w{args.threads_per_worker}t_packed"
    ).resolve()


def _ensure_separate(output: Path, *protected: Path) -> None:
    output = output.resolve()
    for path in protected:
        path = path.resolve()
        if output == path or output in path.parents or path in output.parents:
            raise SystemExit(
                "--output must be separate from both the source atlas and packed cache"
            )


def _create_output(output: Path, *, resume: bool) -> None:
    if output.exists():
        if not resume:
            raise SystemExit(
                f"output already exists; use --resume or choose another path: {output}"
            )
        if not output.is_dir():
            raise SystemExit(f"--output is not a directory: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()


def _compare_reference(path: Path, scan) -> dict[str, object]:
    import numpy as np

    result = path.expanduser().resolve()
    minima_path = result / "minima.npz"
    if not minima_path.is_file():
        raise SystemExit(f"reference result has no minima.npz: {minima_path}")
    with np.load(minima_path, allow_pickle=False) as arrays:
        reference_ids = np.asarray(arrays["map_ids"], dtype=np.int64)
        reference_parameters = np.asarray(arrays["parameters"], dtype=np.float64)
        reference_chi2 = np.asarray(arrays["chi2"], dtype=np.float64)
        reference_geometry = np.asarray(arrays["geometry_index"], dtype=np.int64)
        reference_alpha = np.asarray(arrays["alpha_index"], dtype=np.int64)
        reference_scanned = np.asarray(arrays["scanned"], dtype=bool)
    current_ids = np.asarray(scan.map_ids, dtype=np.int64)
    current_chi2 = np.asarray(scan.chi2, dtype=np.float64)
    current_scanned = np.asarray(scan.scanned, dtype=bool)
    same_set = set(map(int, reference_ids)) == set(map(int, current_ids))
    payload: dict[str, object] = {
        "reference_result": str(result),
        "same_map_id_set": bool(same_set),
        "same_map_order": bool(np.array_equal(reference_ids, current_ids)),
    }
    if not same_set:
        payload["note"] = "map-ID sets differ; numerical row comparison was skipped"
        return payload
    reference_rows = {int(map_id): index for index, map_id in enumerate(reference_ids)}
    current_rows = {int(map_id): index for index, map_id in enumerate(current_ids)}
    common_ids = np.asarray(sorted(reference_rows), dtype=np.int64)
    rr = np.asarray([reference_rows[int(map_id)] for map_id in common_ids])
    cr = np.asarray([current_rows[int(map_id)] for map_id in common_ids])
    finite = reference_scanned[rr] & current_scanned[cr]
    differences = current_chi2[cr] - reference_chi2[rr]
    finite_differences = differences[finite]
    payload.update(
        {
            "map_count": int(len(common_ids)),
            "max_abs_chi2_difference": (
                None
                if not finite_differences.size
                else float(np.max(np.abs(finite_differences)))
            ),
            "max_relative_chi2_difference": (
                None
                if not finite_differences.size
                else float(
                    np.max(
                        np.abs(finite_differences)
                        / np.maximum(np.abs(reference_chi2[rr][finite]), 1.0)
                    )
                )
            ),
            "same_scanned_mask": bool(
                np.array_equal(reference_scanned[rr], current_scanned[cr])
            ),
            "same_geometry_index": bool(
                np.array_equal(reference_geometry[rr], scan.geometry_index[cr])
            ),
            "same_alpha_index": bool(
                np.array_equal(reference_alpha[rr], scan.alpha_index[cr])
            ),
            "same_parameters": bool(
                np.array_equal(reference_parameters[rr], scan.parameters[cr])
            ),
        }
    )
    reference_finite = reference_scanned & np.isfinite(reference_chi2)
    current_finite = current_scanned & np.isfinite(current_chi2)
    if np.any(reference_finite) and np.any(current_finite):
        payload["reference_best_map_id"] = int(
            reference_ids[np.flatnonzero(reference_finite)[np.argmin(reference_chi2[reference_finite])]]
        )
        payload["current_best_map_id"] = int(
            current_ids[np.flatnonzero(current_finite)[np.argmin(current_chi2[current_finite])]]
        )
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
