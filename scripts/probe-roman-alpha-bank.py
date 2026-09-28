#!/usr/bin/env python3
"""Blindly retain several alpha/geometry seeds from the best FFT maps.

The normal Roman result stores only the minimum over ``(geometry, alpha)`` for
each map.  This probe keeps the initial truth-centred geometry stencil, but
does not use the injected binary parameters to choose alpha.  It reopens the
packed atlas, rescans only the best FFT maps with the full
``(map, geometry, alpha)`` cube, and hands the resulting bank to the exact
direct-VBM objective before one LM polish.

The intended production pattern is::

    all-map FFT minima -> top map bank -> top-K alpha/geometry bank
                       -> direct VBM ranking -> one LM polish

The truth is deliberately not read by the candidate-generation or ranking
path.  Truth comparison, if desired, belongs in a separate diagnostic.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT / "src", REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


REPORT_ROOTS = {
    "9920084": REPO_ROOT
    / "results/roman_local/roman_maptruth_fft_batch200_m128_16w1t/events/9920084",
    "9920099": REPO_ROOT
    / "results/roman_local/roman_maptruth_fft_event9920099_m128_16w1t/events/9920099",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", action="append", choices=sorted(REPORT_ROOTS))
    parser.add_argument("--map-count", type=int, default=100)
    parser.add_argument("--alpha-count", type=int, default=16)
    parser.add_argument("--n-alpha", type=int, default=540)
    parser.add_argument("--geometry-index", type=int)
    parser.add_argument(
        "--alpha-window",
        action="store_true",
        help="retain a contiguous alpha window around each map's FFT minimum",
    )
    parser.add_argument("--direct-count", type=int)
    parser.add_argument("--polish-max-nfev", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results/roman_local/roman_alpha_bank_probe",
    )
    return parser.parse_args(argv)


def candidate_parameters(report: dict, candidate: dict) -> dict[str, float]:
    """Convert one map-frame FFT seed to the native VBM parameter frame."""
    from hammlet._core.caustics import native_pspl_parameters_from_map

    logs = float(candidate["logs"])
    logq = float(candidate["logq"])
    logrho = float(candidate["logrho"])
    s = 10.0**logs
    q = 10.0**logq
    rho = 10.0**logrho
    geometry = report["geometries"][int(candidate["geometry_index"])]
    t0_geometry, u0_geometry, tE = (float(value) for value in geometry)
    alpha = float(candidate["alpha"])
    frame = str(report.get("geometry_frame", "map"))
    if frame == "map":
        t0, u0 = native_pspl_parameters_from_map(
            t0_geometry, u0_geometry, tE, s, q, alpha
        )
    elif frame == "native":
        t0, u0 = t0_geometry, u0_geometry
    else:
        raise ValueError(f"unsupported geometry frame: {frame}")
    return {
        "t0": float(t0),
        "u0": float(u0),
        "tE": float(tE),
        "s": float(s),
        "q": float(q),
        "rho": float(rho),
        "alpha": alpha,
    }


def _selected_bucket(bucket, selected_ids: set[int]):
    """Make a read-only bucket view containing only selected map rows."""
    from tools.roman_local.packed import PackedBucket

    rows = np.asarray(
        [index for index, map_id in enumerate(bucket.map_ids) if int(map_id) in selected_ids],
        dtype=np.int64,
    )
    if not len(rows):
        return None
    storage_rows = rows if bucket.storage_rows is None else bucket.storage_rows[rows]
    return PackedBucket(
        index=bucket.index,
        name=bucket.name,
        path=bucket.path,
        radial_nodes=bucket.radial_nodes,
        map_ids=bucket.map_ids[rows],
        parameters=bucket.parameters[rows],
        m_max=bucket.m_max,
        point_lens_tail_nodes=bucket.point_lens_tail_nodes,
        storage_rows=storage_rows,
    )


def _pad_coefficients(coefficients: np.ndarray, quantum: int = 128) -> np.ndarray:
    count = len(coefficients)
    if count == 0:
        raise ValueError("cannot pad an empty coefficient batch")
    remainder = count % quantum
    if not remainder:
        return coefficients
    return np.concatenate(
        (coefficients, np.repeat(coefficients[-1:], quantum - remainder, axis=0)),
        axis=0,
    )


def _full_alpha_bank(
    report: dict,
    report_path: Path,
    candidates: list[dict],
    *,
    map_count: int,
    alpha_count: int,
    geometry_index: int | None = None,
    n_alpha: int = 540,
    alpha_window: bool = False,
    per_geometry: bool = False,
) -> tuple[list[dict], dict[str, object]]:
    """Re-expand top map minima into a blind alpha/geometry candidate bank."""
    from hammlet._core.config import PSPLGeometry
    from hammlet._core.jax_backend import JAXConsistentGeometryBatchScanner
    from hammlet._core.trajectory import ConsistentKernelFactory
    from tools.roman_local.packed import open_packed_atlas

    if map_count < 1 or alpha_count < 1:
        raise ValueError("map_count and alpha_count must be positive")
    if n_alpha < 4 or n_alpha % 2:
        raise ValueError("n_alpha must be an even integer >= 4")
    if geometry_index is not None and per_geometry:
        raise ValueError("geometry_index and per_geometry are mutually exclusive")
    if geometry_index is not None and not 0 <= int(geometry_index) < len(report["geometries"]):
        raise ValueError(f"geometry_index is unavailable: {geometry_index}")
    if len(candidates) < map_count:
        map_count = len(candidates)
    selected_candidates = candidates[:map_count]
    selected_ids = {int(candidate["map_id"]) for candidate in selected_candidates}
    by_id = {int(candidate["map_id"]): candidate for candidate in selected_candidates}

    packed_path = Path(report["atlas"]["packed_cache"]).expanduser().resolve()
    packed = open_packed_atlas(packed_path)
    radial_support = report.get("atlas", {}).get("radial_support", {})
    maximum_used = radial_support.get("maximum_used")
    if maximum_used is not None:
        packed = packed.with_point_lens_tail(float(maximum_used), radial_order=1)

    datasets = load_module(
        "roman_lm_alpha_bank_data", REPO_ROOT / "scripts/roman-lm-refine.py"
    )._load_datasets(report, report_path)
    geometries = [PSPLGeometry(*map(float, values)) for values in report["geometries"]]
    factory = ConsistentKernelFactory(datasets, geometries, 128, radial_order=1)
    scanner = None
    selected_map_rows: dict[int, dict[str, object]] = {}
    started = time.perf_counter()
    scan_calls = 0

    for bucket in packed.iter_buckets():
        view = _selected_bucket(bucket, selected_ids)
        if view is None:
            continue
        kernels = factory(np.asarray(view.radial_nodes), 128)
        if not kernels or not kernels[0]:
            raise ValueError("kernel factory returned no kernels")
        shape_key = (
            int(view.n_r),
            len(kernels),
            len(kernels[0]),
            128,
        )
        if scanner is None:
            scanner = JAXConsistentGeometryBatchScanner(
                kernels, n_alpha=int(n_alpha), compute_dtype="float64"
            )
        else:
            # All Roman buckets in this cache share the same radial shape.  If
            # that ever changes, constructing a second scanner is safer than
            # silently binding incompatible kernels.
            if (
                scanner.n_radius,
                scanner.n_geometry,
                scanner.n_mode - 1,
            ) != shape_key[:1] + shape_key[1:3]:
                scanner = JAXConsistentGeometryBatchScanner(
                    kernels, n_alpha=int(n_alpha), compute_dtype="float64"
                )
            else:
                scanner.set_kernels(kernels)

        coefficients = view.load_coefficients(128)
        padded = _pad_coefficients(coefficients)
        scan_started = time.perf_counter()
        cube = scanner.scan(padded)[: len(coefficients)]
        scan_seconds = time.perf_counter() - scan_started
        scan_calls += 1
        for row, map_id in enumerate(view.map_ids):
            map_id = int(map_id)
            flat_cube = np.asarray(cube[row], dtype=np.float64).reshape(
                scanner.n_geometry, scanner.n_alpha
            )
            selected_rows = []
            if per_geometry:
                for candidate_geometry_index in range(scanner.n_geometry):
                    flat = flat_cube[candidate_geometry_index]
                    if alpha_window:
                        center = int(np.argmin(flat))
                        left = int(alpha_count) // 2
                        offsets = np.arange(
                            -left, int(alpha_count) - left, dtype=np.int64
                        )
                        order = (center + offsets) % scanner.n_alpha
                        order = order[np.argsort(flat[order], kind="stable")]
                    else:
                        order = np.argsort(flat, kind="stable")[:alpha_count]
                    for rank, alpha_index in enumerate(order, start=1):
                        selected_rows.append(
                            (
                                int(candidate_geometry_index),
                                int(alpha_index),
                                float(flat[alpha_index]),
                                int(rank),
                            )
                        )
            elif geometry_index is None:
                flat = flat_cube.reshape(-1)
                order = np.argsort(flat, kind="stable")[:alpha_count]
                for rank, flat_index in enumerate(order, start=1):
                    candidate_geometry_index, alpha_index = divmod(
                        int(flat_index), scanner.n_alpha
                    )
                    selected_rows.append(
                        (
                            int(candidate_geometry_index),
                            int(alpha_index),
                            float(flat[flat_index]),
                            int(rank),
                        )
                    )
            else:
                flat = flat_cube[int(geometry_index)]
                if alpha_window:
                    center = int(np.argmin(flat))
                    left = int(alpha_count) // 2
                    offsets = np.arange(-left, int(alpha_count) - left, dtype=np.int64)
                    order = (center + offsets) % scanner.n_alpha
                    order = order[np.argsort(flat[order], kind="stable")]
                else:
                    order = np.argsort(flat, kind="stable")[:alpha_count]
                for rank, alpha_index in enumerate(order, start=1):
                    selected_rows.append(
                        (
                            int(geometry_index),
                            int(alpha_index),
                            float(flat[alpha_index]),
                            int(rank),
                        )
                    )
            map_candidates = []
            logs, logq, logrho = (float(value) for value in view.parameters[row])
            for candidate_geometry_index, alpha_index, value, rank in selected_rows:
                map_candidates.append(
                    {
                        "map_id": map_id,
                        "logs": logs,
                        "logq": logq,
                        "logrho": logrho,
                        "geometry_index": int(candidate_geometry_index),
                        "alpha_index": int(alpha_index),
                        "alpha": float(2.0 * np.pi * alpha_index / scanner.n_alpha),
                        "fft_chi2": float(value),
                        "rank_within_map": int(rank),
                    }
                )
            selected_map_rows[map_id] = {
                "map_minimum": by_id[map_id],
                "candidates": map_candidates,
                "scan_seconds": float(scan_seconds),
            }

    if scanner is None:
        raise RuntimeError("none of the selected map IDs were found in the packed atlas")

    bank = []
    for candidate in selected_candidates:
        rows = selected_map_rows[int(candidate["map_id"])]
        for row in rows["candidates"]:
            row = dict(row)
            row["parameters"] = candidate_parameters(report, row)
            bank.append(row)
    bank.sort(key=lambda item: item["fft_chi2"])
    return bank, {
        "map_count": int(map_count),
        "alpha_count_per_map": int(alpha_count),
        "geometry_index": None if geometry_index is None else int(geometry_index),
        "per_geometry": bool(per_geometry),
        "n_alpha": int(n_alpha),
        "alpha_window": bool(alpha_window),
        "candidate_count": int(len(bank)),
        "scan_calls": int(scan_calls),
        "wall_seconds": float(time.perf_counter() - started),
        "selection": "top map minima from saved FFT result; no truth parameters used",
    }


def _screen_direct(lm_module, report: dict, report_path: Path, bank: list[dict], direct_count: int | None):
    datasets = lm_module._load_datasets(report, report_path)
    frame = lm_module._coordinate_frame(report)
    selected = bank if direct_count is None else bank[: int(direct_count)]
    screened = []
    started = time.perf_counter()
    for index, candidate in enumerate(selected, start=1):
        objective = lm_module._LMObjective(
            datasets,
            candidate["parameters"],
            vbm_tolerance=1.0e-3,
            vbm_relative_tolerance=1.0e-4,
            coordinate_frame=frame,
        )
        residual = objective(np.zeros(7, dtype=np.float64))
        screened.append(
            {
                **candidate,
                "rank_direct": int(index),
                "direct_vbm_chi2": float(np.dot(residual, residual)),
            }
        )
        if index == 1 or index % 50 == 0 or index == len(selected):
            best = min(screened, key=lambda item: item["direct_vbm_chi2"])
            print(
                f"[alpha-bank-screen] {index}/{len(selected)} "
                f"best={best['direct_vbm_chi2']:.1f} "
                f"map={best['map_id']} alpha_index={best['alpha_index']} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    screened.sort(key=lambda item: item["direct_vbm_chi2"])
    return screened, time.perf_counter() - started


def run_event(event_id: str, args: argparse.Namespace) -> dict:
    report_path = REPORT_ROOTS[event_id] / "report.json"
    report = read_json(report_path)
    candidates = json.loads(
        (report_path.parent / "candidates.json").read_text(encoding="utf-8")
    )
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"no saved FFT candidates for {event_id}")
    lm_module = load_module(
        f"roman_lm_alpha_bank_{event_id}", REPO_ROOT / "scripts/roman-lm-refine.py"
    )
    bank, bank_config = _full_alpha_bank(
        report,
        report_path,
        candidates,
        map_count=int(args.map_count),
        alpha_count=int(args.alpha_count),
        geometry_index=(None if args.geometry_index is None else int(args.geometry_index)),
        n_alpha=int(args.n_alpha),
        alpha_window=bool(args.alpha_window),
    )
    screened, screen_wall = _screen_direct(
        lm_module, report, report_path, bank, args.direct_count
    )
    if not screened:
        raise RuntimeError("direct screen produced no candidates")
    best = screened[0]
    from scipy.optimize import least_squares

    objective = lm_module._LMObjective(
        lm_module._load_datasets(report, report_path),
        best["parameters"],
        vbm_tolerance=1.0e-3,
        vbm_relative_tolerance=1.0e-4,
        coordinate_frame=lm_module._coordinate_frame(report),
    )
    z0 = np.zeros(7, dtype=np.float64)
    start_residual = objective(z0)
    start_chi2 = float(np.dot(start_residual, start_residual))
    print(
        f"[alpha-bank-polish] {event_id} map={best['map_id']} "
        f"alpha_index={best['alpha_index']} chi2={start_chi2:.1f}",
        flush=True,
    )
    polish_started = time.perf_counter()
    fit = least_squares(
        objective,
        z0,
        method="lm",
        jac="2-point",
        diff_step=1.0e-3,
        max_nfev=int(args.polish_max_nfev),
        ftol=1.0e-8,
        xtol=1.0e-8,
        gtol=1.0e-8,
        x_scale="jac",
    )
    final_values = objective.unpack(fit.x)
    if final_values is None:
        raise ValueError("LM returned invalid parameters")
    final_residual = objective(fit.x)
    final_chi2 = float(np.dot(final_residual, final_residual))
    polish_wall = time.perf_counter() - polish_started
    payload = {
        "schema": "hammlet-roman-alpha-bank-probe-v1",
        "event_id": event_id,
        "report_path": str(report_path.resolve()),
        "truth_used_for_selection": False,
        "coordinate_frame": str(report.get("geometry_frame", "map")),
        "configuration": {
            **bank_config,
            "direct_count": len(screened),
            "polish_max_nfev": int(args.polish_max_nfev),
            "direct_screen_wall_seconds": float(screen_wall),
            "polish_wall_seconds": float(polish_wall),
        },
        "direct_screen": screened,
        "initial_seed": {
            key: best[key]
            for key in (
                "map_id",
                "geometry_index",
                "alpha_index",
                "alpha",
                "fft_chi2",
                "rank_within_map",
                "parameters",
                "direct_vbm_chi2",
            )
        },
        "optimized": {
            "parameters": lm_module._parameter_dict(final_values),
            "direct_vbm_chi2": final_chi2,
            "chi2_improvement": start_chi2 - final_chi2,
        },
        "lm": {
            "success": bool(fit.success),
            "status": int(fit.status),
            "message": str(fit.message),
            "nfev": int(fit.nfev),
            "optimality": float(fit.optimality),
            "residual_calls": int(objective.calls),
            "wall_seconds": float(polish_wall),
        },
    }
    output_dir = args.output.expanduser().resolve()
    event_path = output_dir / "events" / f"{event_id}.json"
    event_path.parent.mkdir(parents=True, exist_ok=True)
    event_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    dof = int(report["event"]["total_points"]) - 9
    return {
        "event_id": event_id,
        "output": str(event_path),
        "map_count": bank_config["map_count"],
        "alpha_count_per_map": bank_config["alpha_count_per_map"],
        "candidate_count": bank_config["candidate_count"],
        "screen_count": len(screened),
        "best_map_id": best["map_id"],
        "best_alpha_index": best["alpha_index"],
        "best_rank_within_map": best["rank_within_map"],
        "direct_chi2_dof": start_chi2 / dof,
        "polished_chi2_dof": final_chi2 / dof,
        "alpha_bank_wall_seconds": bank_config["wall_seconds"],
        "direct_screen_wall_seconds": screen_wall,
        "polish_wall_seconds": polish_wall,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.map_count < 1 or args.alpha_count < 1:
        raise SystemExit("--map-count and --alpha-count must be positive")
    if args.direct_count is not None and args.direct_count < 1:
        raise SystemExit("--direct-count must be positive")
    if args.polish_max_nfev < 1:
        raise SystemExit("--polish-max-nfev must be positive")
    events = args.event or ["9920099"]
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = [run_event(event_id, args) for event_id in events]
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
