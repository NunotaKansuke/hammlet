#!/usr/bin/env python3
"""Run a local Roman event against an existing Hammlet atlas.

The default profile is deliberately explicit and small enough to compare
locally:

    M=128, n_alpha=540, radial_order=1, 4 CPU cores, no chi2 guarantee.

The atlas is read-only.  Results go to a separate directory and an existing
result directory is never overwritten.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

DEFAULT_CORES = 4
DEFAULT_M_MAX = 128
DEFAULT_N_ALPHA = 540
DEFAULT_RADIAL_ORDER = 1
_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--catalog",
        type=Path,
        help="Roman simulation catalog CSV (use with --simulation-root)",
    )
    input_group.add_argument(
        "--data-file",
        action="append",
        help="user-provided three-column data file; repeat for multiple bands",
    )
    input_group.add_argument(
        "--gulls-lightcurve",
        type=Path,
        help="GULLS .all.lc file (use with --gulls-manifest)",
    )
    parser.add_argument(
        "--gulls-manifest",
        type=Path,
        help="manifest.csv corresponding to --gulls-lightcurve",
    )
    parser.add_argument(
        "--event-id",
        help="GULLS manifest event_id when the manifest contains multiple rows",
    )
    parser.add_argument("--simulation-root", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--anomaly-product", type=Path)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--event-name", default="local-event")
    parser.add_argument(
        "--band",
        action="append",
        dest="bands",
        help="Roman catalog band to load; repeatable (default: W146)",
    )
    parser.add_argument(
        "--band-name",
        action="append",
        dest="band_names",
        help="name for a corresponding --data-file; repeatable",
    )
    parser.add_argument(
        "--data-format",
        choices=("roman-mag", "time-flux-error"),
        default="roman-mag",
        help="input columns: magnitude/mag_error/time or time/flux/error",
    )
    parser.add_argument("--time-offset", type=float, default=8234.0)
    parser.add_argument(
        "--window-tE",
        type=float,
        default=None,
        help="optionally limit data to center +/- this many tE (default: all valid rows)",
    )
    parser.add_argument(
        "--window-start",
        type=float,
        help="optional explicit lower data-time bound (use with --window-end)",
    )
    parser.add_argument(
        "--window-end",
        type=float,
        help="optional explicit upper data-time bound (use with --window-start)",
    )

    parser.add_argument("--t0", type=float, help="user-data geometry t0")
    parser.add_argument("--u0", type=float, help="user-data geometry u0")
    parser.add_argument("--tE", type=float, help="user-data geometry tE")
    parser.add_argument("--s", type=float, help="optional reference separation")
    parser.add_argument("--q", type=float, help="optional reference mass ratio")
    parser.add_argument("--rho", type=float, help="optional reference source radius")
    parser.add_argument(
        "--alpha",
        type=float,
        help="optional reference alpha in radians for user-data truth geometry",
    )

    parser.add_argument(
        "--geometry-center",
        choices=("baseline", "truth", "map-truth"),
        default="baseline",
        help=(
            "center the seed stencil on the anomaly baseline, native truth, or "
            "truth transformed into the atlas map frame"
        ),
    )
    parser.add_argument(
        "--geometry-stencil",
        choices=("full", "te-only", "center"),
        default="full",
        help="PSPL seed stencil (default full: 16 geometry seeds)",
    )
    parser.add_argument("--geometry-dt0", type=float)
    parser.add_argument("--geometry-du0", type=float)
    parser.add_argument("--geometry-dlog-te", type=float)
    parser.add_argument("--geometry-te-steps", type=int, default=3)
    parser.add_argument(
        "--no-extra-geometry",
        action="store_true",
        help="omit the historical off-lattice full-stencil point",
    )

    parser.add_argument(
        "--m-max",
        type=int,
        default=DEFAULT_M_MAX,
        help="linear Fourier modes (default: 128; local profile setting)",
    )
    parser.add_argument(
        "--n-alpha",
        type=int,
        default=DEFAULT_N_ALPHA,
        help="alpha samples (default: 540)",
    )
    parser.add_argument(
        "--radial-order",
        type=int,
        default=DEFAULT_RADIAL_ORDER,
        choices=(1, 3),
        help="fixed radial interpolation order (default: 1)",
    )
    parser.add_argument(
        "--tail-mode",
        choices=("auto", "off"),
        default="auto",
        help=(
            "GULLS radial-support tail: auto adds the in-memory point-lens "
            "fallback, off disables it for controlled diagnostics"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="approximate coefficient batch size before JAX padding",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="write a progress line every N radial buckets (0 disables it)",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=("float64", "float32"),
        default="float64",
    )
    parser.add_argument(
        "--logq-max",
        type=float,
        help="optional map filter, keeping only log10(q) <= this value",
    )
    parser.add_argument(
        "--max-maps",
        type=int,
        help="read only the first map IDs for a local smoke test",
    )
    parser.add_argument(
        "--candidate-count", type=int, default=100, help="saved candidate count"
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=DEFAULT_CORES,
        help="CPU affinity/thread budget (default: 4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="load and validate event/atlas metadata, then print the run plan",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    execution = _configure_cpu(args.cores)
    _validate_search_args(args)
    explicit_window = _explicit_window(args)
    from tools.roman_local.atlas import open_readonly_atlas
    from tools.roman_local.roman_event import (
        RomanReference,
        geometry_stencil,
        load_catalog_event,
        load_data_event,
        load_gulls_event,
    )

    event = _load_event(
        args,
        RomanReference,
        load_catalog_event,
        load_data_event,
        load_gulls_event,
        explicit_window,
    )
    atlas = open_readonly_atlas(args.atlas, max_maps=args.max_maps)
    output = _result_path(args, event.name)
    _ensure_separate_output(output, atlas.path)
    if args.m_max > atlas.m_max:
        raise SystemExit(
            f"requested --m-max={args.m_max}, but atlas stores only M={atlas.m_max}"
        )
    if args.candidate_count < 1:
        raise SystemExit("--candidate-count must be positive")
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
    if args.tail_mode == "auto":
        required_radius = _required_radial_radius(event, geometries)
        atlas = atlas.with_point_lens_tail(
            required_radius,
            radial_order=args.radial_order,
        )
    map_filter = _build_map_filter(args.logq_max)
    filter_kept = _filter_count(atlas.map_parameters, map_filter)
    plan = _plan(
        args,
        event,
        atlas,
        geometries,
        stencil_report,
        execution,
        filter_kept,
        output,
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    from hammlet._core.trajectory import ConsistentKernelFactory
    from tools.roman_local.scan import scan_single_resolution

    _create_output(output)
    factory_started = time.perf_counter()
    kernel_factory = ConsistentKernelFactory(
        event.datasets,
        geometries,
        args.m_max,
        radial_order=args.radial_order,
    )
    factory_wall_seconds = time.perf_counter() - factory_started
    scan_started = time.perf_counter()
    scan_timing: dict[str, object] = {}
    scan = scan_single_resolution(
        atlas,
        kernel_factory,
        m_max=args.m_max,
        n_alpha=args.n_alpha,
        batch_size=args.batch_size,
        map_filter=map_filter,
        compute_dtype=args.compute_dtype,
        timing=scan_timing,
        progress_every=args.progress_every,
    )
    scan_wall_seconds = time.perf_counter() - scan_started

    import numpy as np

    npz_values = {
        "map_ids": scan.map_ids,
        "parameters": scan.parameters,
        "chi2": scan.chi2,
        "geometry_index": scan.geometry_index,
        "alpha_index": scan.alpha_index,
        "scanned": scan.scanned,
        "geometries": _geometry_array(geometries),
    }
    np.savez_compressed(output / "minima.npz", **npz_values)
    candidates = scan.candidates(
        n_alpha=args.n_alpha,
        count=args.candidate_count,
        rescanned_only=True,
    )
    (output / "candidates.json").write_text(
        json.dumps([asdict(candidate) for candidate in candidates], indent=2) + "\n",
        encoding="utf-8",
    )
    report = _report(
        args,
        event,
        atlas,
        geometries,
        stencil_report,
        execution,
        scan,
        filter_kept,
        factory_wall_seconds=factory_wall_seconds,
        kernel_factory=kernel_factory,
        scan_wall_seconds=scan_wall_seconds,
        scan_timing=scan_timing,
        output=output,
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "best": report["best_fft"]}, indent=2))
    return 0


def _load_event(
    args,
    reference_type,
    catalog_loader,
    data_loader,
    gulls_loader,
    explicit_window,
):
    if args.catalog is not None:
        if args.simulation_root is None:
            raise SystemExit("--catalog requires --simulation-root")
        if args.data_file is not None:
            raise SystemExit("--data-file cannot be combined with --catalog")
        bands = tuple(args.bands or ("W146",))
        return catalog_loader(
            args.index,
            catalog=args.catalog,
            simulation_root=args.simulation_root,
            anomaly_path=args.anomaly_product,
            bands=bands,
            window_tE=args.window_tE,
            window=explicit_window,
            time_offset=args.time_offset,
            data_format=args.data_format,
        )
    if args.gulls_lightcurve is not None:
        if args.gulls_manifest is None:
            raise SystemExit("--gulls-lightcurve requires --gulls-manifest")
        if args.event_id is not None and not str(args.event_id):
            raise SystemExit("--event-id must not be empty")
        if args.anomaly_product is not None:
            raise SystemExit("--anomaly-product is not available with --gulls-lightcurve")
        if args.simulation_root is not None:
            raise SystemExit("--simulation-root is not available with --gulls-lightcurve")
        if args.bands or args.band_names:
            raise SystemExit("--band/--band-name are not available with --gulls-lightcurve")
        if any(value is not None for value in (args.t0, args.u0, args.tE, args.s, args.q, args.rho)):
            raise SystemExit("--t0/--u0/--tE/--s/--q/--rho are not available with --gulls-lightcurve")
        event_name = None if args.event_name == "local-event" else args.event_name
        return gulls_loader(
            args.gulls_lightcurve,
            manifest=args.gulls_manifest,
            event_id=args.event_id,
            window_tE=args.window_tE,
            window=explicit_window,
            name=event_name,
        )
    if args.anomaly_product is not None:
        raise SystemExit("--anomaly-product is available only with --catalog")
    if args.simulation_root is not None:
        raise SystemExit("--simulation-root is available only with --catalog")
    if args.t0 is None or args.u0 is None or args.tE is None:
        raise SystemExit("--data-file requires --t0, --u0, and --tE")
    if args.bands:
        raise SystemExit("--band is available only with --catalog; use --band-name")
    if args.s is not None or args.q is not None or args.rho is not None:
        if not all(value is not None for value in (args.s, args.q, args.rho)):
            raise SystemExit("--s, --q, and --rho must be supplied together")
        reference = reference_type(
            args.t0,
            args.u0,
            args.tE,
            s=args.s,
            q=args.q,
            rho=args.rho,
            catalog_alpha=args.alpha,
        )
    else:
        reference = None
    return data_loader(
        args.data_file,
        t0=args.t0,
        u0=args.u0,
        tE=args.tE,
        names=args.band_names,
        window_tE=args.window_tE,
        window=explicit_window,
        data_format=args.data_format,
        reference=reference,
        name=args.event_name,
    )


def _configure_cpu(cores: int) -> dict[str, object]:
    cores = int(cores)
    if cores < 1:
        raise SystemExit("--cores must be positive")
    affinity: list[int] = []
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        available = sorted(os.sched_getaffinity(0))
        if cores > len(available):
            raise SystemExit(
                f"--cores={cores} exceeds the {len(available)} CPUs available to this process"
            )
        affinity = available[:cores]
        os.sched_setaffinity(0, affinity)
    for name in _THREAD_ENV:
        os.environ[name] = str(cores)
    return {
        "cores_requested": cores,
        "cpu_affinity": affinity,
        "thread_env": {name: os.environ[name] for name in _THREAD_ENV},
    }


def _required_radial_radius(event, geometries) -> float:
    import numpy as np

    maximum = 0.0
    for dataset in event.datasets:
        for geometry in geometries:
            radius = np.hypot(
                (dataset.time - geometry.t0) / geometry.tE,
                geometry.u0,
            )
            maximum = max(maximum, float(np.max(radius)))
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise SystemExit("could not determine a finite GULLS radial support")
    return maximum


def _validate_search_args(args) -> None:
    if args.m_max < 0:
        raise SystemExit("--m-max must be non-negative")
    if args.n_alpha < 4 or args.n_alpha % 2 or args.n_alpha < 4 * args.m_max:
        raise SystemExit("--n-alpha must be even and at least 4*--m-max")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.max_maps is not None and args.max_maps < 1:
        raise SystemExit("--max-maps must be positive")


def _explicit_window(args) -> tuple[float, float] | None:
    if (args.window_start is None) != (args.window_end is None):
        raise SystemExit("--window-start and --window-end must be supplied together")
    if args.window_start is None:
        return None
    if args.window_start >= args.window_end:
        raise SystemExit("--window-start must be smaller than --window-end")
    return float(args.window_start), float(args.window_end)


def _build_map_filter(logq_max: float | None):
    if logq_max is None:
        return None
    logq_max = float(logq_max)

    def keep(parameters):
        import numpy as np

        parameters = np.asarray(parameters)
        return parameters[:, 1] <= logq_max + 1.0e-12

    return keep


def _filter_count(parameters, map_filter) -> int:
    if map_filter is None:
        return int(len(parameters))
    return int(np_count(map_filter(parameters)))


def np_count(values) -> int:
    import numpy as np

    return int(np.count_nonzero(values))


def _truth_geometry(event):
    from hammlet._core.config import PSPLGeometry

    return PSPLGeometry(event.reference.t0, event.reference.u0, event.reference.tE)


def _map_truth_geometry(event):
    """Return the injected truth expressed in the atlas map frame.

    ``RomanReference.catalog_alpha`` is the GULLS/native convention. The FFT
    trajectory helper uses the equivalent line with alpha shifted by pi, so
    the transformed geometry is computed with that search alpha.
    """
    import numpy as np

    from hammlet._core.caustics import (
        adamgrid_map_origin_shift,
        map_frame_pspl_parameters,
    )
    from hammlet._core.config import PSPLGeometry

    reference = event.reference
    if any(
        value is None
        for value in (reference.s, reference.q, reference.catalog_alpha)
    ):
        raise ValueError(
            "map-truth geometry requires truth s, q, and catalog alpha"
        )
    search_alpha = (float(reference.catalog_alpha) + np.pi) % (2.0 * np.pi)
    t0_map, u0_map = map_frame_pspl_parameters(
        reference.t0,
        reference.u0,
        reference.tE,
        reference.s,
        reference.q,
        search_alpha,
    )
    return PSPLGeometry(t0_map, u0_map, reference.tE), {
        "mode": "map-truth",
        "geometry_frame": "map",
        "native_alpha": float(reference.catalog_alpha),
        "search_alpha": float(search_alpha),
        "shift_x": float(
            adamgrid_map_origin_shift(reference.s, reference.q)
        ),
        "native_center": {
            "t0": float(reference.t0),
            "u0": float(reference.u0),
            "tE": float(reference.tE),
        },
        "map_center": {
            "t0": float(t0_map),
            "u0": float(u0_map),
            "tE": float(reference.tE),
        },
    }


def _geometry_center(event, mode: str):
    if mode == "baseline":
        return event.geometry_center, {
            "mode": "baseline",
            "geometry_frame": "native",
        }
    if mode == "truth":
        return _truth_geometry(event), {
            "mode": "truth",
            "geometry_frame": "native",
        }
    if mode == "map-truth":
        return _map_truth_geometry(event)
    raise ValueError(f"unsupported geometry center mode: {mode!r}")


def _geometry_array(geometries):
    import numpy as np

    return np.asarray(
        [[float(geometry.t0), float(geometry.u0), float(geometry.tE)] for geometry in geometries],
        dtype=np.float64,
    )


def _plan(
    args,
    event,
    atlas,
    geometries,
    stencil_report,
    execution,
    filter_kept,
    output,
):
    return {
        "profile": "roman-local-single-resolution",
        "local_only": True,
        "atlas_read_only": True,
        "event": event.as_dict(),
        "atlas": {
            "path": str(atlas.path),
            "kind": atlas.kind,
            "stored_m_max": atlas.m_max,
            "selected_maps": atlas.total_maps,
            "completed_maps": atlas.completed_maps,
            "expected_maps": atlas.expected_maps,
            "completed_count_exact": bool(
                atlas.metadata.get("completion_count_exact", True)
            ),
            "metadata": atlas.metadata,
        },
        "execution": execution,
        "search_settings": {
            "m_max": args.m_max,
            "n_alpha": args.n_alpha,
            "radial_order": args.radial_order,
            "batch_size": args.batch_size,
            "progress_every": args.progress_every,
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


def _report(
    args,
    event,
    atlas,
    geometries,
    stencil_report,
    execution,
    scan,
    filter_kept,
    *,
    factory_wall_seconds,
    kernel_factory,
    scan_wall_seconds,
    scan_timing,
    output,
):
    import numpy as np

    scanned = np.asarray(scan.scanned, dtype=bool)
    finite = scanned & np.isfinite(scan.chi2)
    best_row = None if not np.any(finite) else int(np.flatnonzero(finite)[np.argmin(scan.chi2[finite])])
    best = None
    if best_row is not None:
        best = {
            "map_id": int(scan.map_ids[best_row]),
            "parameters": [float(value) for value in scan.parameters[best_row]],
            "chi2": float(scan.chi2[best_row]),
            "geometry_index": int(scan.geometry_index[best_row]),
            "alpha_index": int(scan.alpha_index[best_row]),
            "alpha": float(2.0 * np.pi * scan.alpha_index[best_row] / args.n_alpha),
        }
    nearest = _nearest_reference(event, scan, scanned)
    selected_maps = int(len(scan.map_ids))
    scanned_maps = int(np.count_nonzero(scanned))
    limitations = [
        "No reconstruction-error or certified-error sidecars are read.",
        "The reported chi2 is not an error-bounded interval.",
        "This is a static binary-lens seed search; final fitting remains direct VBM work.",
    ]
    if atlas.metadata.get("point_lens_tail", {}).get("enabled", False):
        limitations.append(
            "Rows outside the stored radial support use the in-memory map-specific "
            "point-lens far-field fallback; no map files are changed."
        )
    return {
        "schema": "hammlet-roman-local-v1",
        "profile": "roman-local-single-resolution",
        "local_only": True,
        "atlas_read_only": True,
        "output": str(output.resolve()),
        "event": event.as_dict(),
        "atlas": {
            "path": str(atlas.path),
            "kind": atlas.kind,
            "metadata": atlas.metadata,
            "stored_m_max": atlas.m_max,
            "radial_support": {
                "maximum_used": _required_radial_radius(event, geometries),
                "stored_maximum": max(
                    float(bucket.radial_nodes[-1]) for bucket in atlas.buckets
                ),
            },
        },
        "maps": {
            "expected": int(atlas.expected_maps),
            "completed": int(atlas.completed_maps),
            "completed_count_exact": bool(
                atlas.metadata.get("completion_count_exact", True)
            ),
            "selected": selected_maps,
            "scanned": scanned_maps,
            "skipped_by_filter": selected_maps - scanned_maps,
            "missing_or_partial": max(int(atlas.expected_maps) - int(atlas.completed_maps), 0),
            "limited_by_max_maps": max(int(atlas.completed_maps) - selected_maps, 0),
            "filter_logq_max": args.logq_max,
            "filter_kept_before_scan": int(filter_kept),
        },
        "geometries": _geometry_array(geometries).tolist(),
        "geometry_stencil": stencil_report,
        "geometry_frame": str(stencil_report.get("geometry_frame", "native")),
        "best_fft": best,
        "nearest_reference_map": nearest,
        "error_certificate": {
            "enabled": False,
            "mode": "single_resolution",
            "chi2_guarantee": False,
            "sidecars_loaded": False,
            "note": "FFT chi2 is a seed-search diagnostic; re-evaluate retained seeds with direct VBM.",
        },
        "search_settings": {
            "m_max": int(args.m_max),
            "n_alpha": int(args.n_alpha),
            "radial_order": int(args.radial_order),
            "batch_size": int(args.batch_size),
            "compute_dtype": args.compute_dtype,
            "map_filter_logq_max": args.logq_max,
            "tail_mode": args.tail_mode,
        },
        "execution": execution,
        "timing": {
            "kernel_factory_wall_seconds": float(factory_wall_seconds),
            "kernel_basis_seconds": float(kernel_factory.basis_seconds),
            "kernel_binding_seconds": float(kernel_factory.build_seconds),
            "scan_wall_seconds": float(scan_wall_seconds),
            "scan_seconds": float(scan.full_seconds),
            "scan_breakdown": scan_timing,
            "total_seconds": float(factory_wall_seconds + scan_wall_seconds),
        },
        "artifacts": {
            "minima": str((output / "minima.npz").resolve()),
            "candidates": str((output / "candidates.json").resolve()),
            "report": str((output / "report.json").resolve()),
        },
        "limitations": limitations,
    }


def _nearest_reference(event, scan, scanned):
    import numpy as np

    reference = event.reference
    if reference.s is None or reference.q is None or reference.rho is None:
        return None
    rows = np.flatnonzero(np.asarray(scanned, dtype=bool))
    if not rows.size:
        return None
    target = np.log10([reference.s, reference.q, reference.rho])
    scale = np.asarray([0.05, 0.1, 0.3])
    local = rows[np.argmin(np.sum(((scan.parameters[rows] - target) / scale) ** 2, axis=1))]
    rank = np.argsort(scan.chi2[rows], kind="stable")
    rank_position = int(np.flatnonzero(rank == np.flatnonzero(rows == local)[0])[0]) + 1
    return {
        "map_id": int(scan.map_ids[local]),
        "parameters": [float(value) for value in scan.parameters[local]],
        "distance_squared": float(np.sum(((scan.parameters[local] - target) / scale) ** 2)),
        "chi2": float(scan.chi2[local]),
        "rank_among_scanned": rank_position,
    }


def _result_path(args, event_name: str) -> Path:
    if args.output is not None:
        return args.output.expanduser().resolve()
    index = f"{args.index}" if args.catalog is not None else event_name
    safe_index = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in index
    )
    return (
        REPO_ROOT
        / "results"
        / "roman_local"
        / f"roman_{safe_index}_m{args.m_max}_{args.cores}c"
    ).resolve()


def _create_output(output: Path) -> None:
    if output.exists():
        raise SystemExit(
            f"output already exists; refusing to overwrite local results: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()


def _ensure_separate_output(output: Path, atlas_path: Path) -> None:
    candidate = output.expanduser().resolve()
    atlas_path = atlas_path.resolve()
    if candidate == atlas_path or atlas_path in candidate.parents or candidate in atlas_path.parents:
        raise SystemExit(
            "--output must be separate from the atlas path; map data is read-only"
        )


if __name__ == "__main__":
    raise SystemExit(main())
