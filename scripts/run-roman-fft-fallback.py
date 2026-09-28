#!/usr/bin/env python3
"""Robust fallback refinement inside the region identified by the FFT scan.

The normal fast LM path is unchanged.  This fallback is intended only for an
event whose normal fit remains poor:

    top FFT maps
      -> per-geometry alpha minima
      -> geometry-stratified elite + spatially diverse seeds
      -> short map-frame LM on every seed
      -> continue the best fits and the strongest improvers
      -> optional bounded 3-D Powell lens rescue
      -> long map-frame LM and observed-data chi-square selection

Candidate selection never reads injected binary parameters.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import importlib.util
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np


for _thread_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_thread_name, "1")


REPO_ROOT = Path(__file__).resolve().parents[1]
for _import_root in (REPO_ROOT / "src", REPO_ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from tools.roman_local.fft_fallback import (  # noqa: E402
    select_continuations,
    select_fft_seeds,
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _resolve_report(roots: list[Path], event_id: str) -> Path:
    tried = []
    for root_value in roots:
        root = root_value.expanduser().resolve()
        for path in (
            root / "events" / event_id / "report.json",
            root / event_id / "report.json",
            root / "report.json",
        ):
            tried.append(path)
            if path.is_file() and (
                path.parent.name == event_id or path == root / "report.json"
            ):
                return path
    raise FileNotFoundError(
        f"no report for {event_id}; tried: " + ", ".join(map(str, tried))
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, action="append", required=True)
    parser.add_argument("--event", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-count", type=int, default=1024)
    parser.add_argument("--elite-per-geometry", type=int, default=12)
    parser.add_argument("--diverse-per-geometry", type=int, default=4)
    parser.add_argument("--minimum-grid-distance", type=float, default=2.0)
    parser.add_argument("--short-max-nfev", type=int, default=80)
    parser.add_argument("--short-stop-chi2-dof", type=float, default=1.2)
    parser.add_argument("--short-stop-min-completed", type=int, default=16)
    parser.add_argument("--continue-best", type=int, default=8)
    parser.add_argument("--continue-improvement", type=int, default=8)
    parser.add_argument("--long-max-nfev", type=int, default=320)
    parser.add_argument("--powell-trigger-chi2-dof", type=float, default=2.0)
    parser.add_argument("--powell-s-half-width", type=float, default=0.25)
    parser.add_argument("--powell-q-half-width", type=float, default=0.60)
    parser.add_argument("--powell-rho-half-width", type=float, default=0.60)
    parser.add_argument("--powell-maxiter", type=int, default=60)
    parser.add_argument("--powell-lm-count", type=int, default=8)
    parser.add_argument(
        "--enable-powell",
        action="store_true",
        help="enable the optional slow 3-D lens-only rescue after LM",
    )
    parser.add_argument("--workers", type=int, default=min(18, os.cpu_count() or 1))
    parser.add_argument(
        "--event-workers",
        type=int,
        default=1,
        help="independent events processed concurrently; total workers are approximate",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


_STATE: dict[str, object] | None = None


def _worker_init(report_path: str, event_id: str) -> None:
    global _STATE
    report_path_obj = Path(report_path)
    report = _read_json(report_path_obj)
    lm_module = _load_module(
        f"roman_fft_fallback_lm_{event_id}",
        REPO_ROOT / "scripts/roman-lm-refine.py",
    )
    map_lm_module = _load_module(
        f"roman_fft_fallback_map_lm_{event_id}",
        REPO_ROOT / "scripts/probe-roman-alpha-lm-bank.py",
    )
    _STATE = {
        "report": report,
        "lm_module": lm_module,
        "map_lm_module": map_lm_module,
        "datasets": lm_module._load_datasets(report, report_path_obj),
    }


def _map_lm_objective(parameters: dict):
    if _STATE is None:
        raise RuntimeError("fallback worker is not initialized")
    lm_module = _STATE["lm_module"]
    report = _STATE["report"]
    return _STATE["map_lm_module"]._MapFrameLMObjective(
        lm_module,
        _STATE["datasets"],
        parameters,
        vbm_tolerance=1.0e-3,
        vbm_relative_tolerance=1.0e-4,
        coordinate_frame=lm_module._coordinate_frame(report),
    )


def _lm_worker(candidate: dict, max_nfev: int, stage: str) -> dict:
    if _STATE is None:
        raise RuntimeError("fallback worker is not initialized")
    from scipy.optimize import least_squares

    lm_module = _STATE["lm_module"]
    objective = _map_lm_objective(candidate["parameters"])
    z0 = np.zeros(7, dtype=np.float64)
    start_residual = objective(z0)
    start_chi2 = float(np.dot(start_residual, start_residual))
    started = time.perf_counter()
    fit = least_squares(
        objective,
        z0,
        method="lm",
        jac="2-point",
        diff_step=1.0e-3,
        max_nfev=int(max_nfev),
        ftol=1.0e-8,
        xtol=1.0e-8,
        gtol=1.0e-8,
        x_scale="jac",
    )
    values = objective.unpack(fit.x)
    if values is None:
        raise ValueError("map-frame LM returned invalid parameters")
    residual = objective(fit.x)
    final_chi2 = float(np.dot(residual, residual))
    return {
        **candidate,
        "lm_stage": stage,
        "initial_parameters": candidate["parameters"],
        "start_direct_vbm_chi2": start_chi2,
        "optimized_parameters": lm_module._parameter_dict(values),
        "direct_vbm_chi2": final_chi2,
        "chi2_improvement": start_chi2 - final_chi2,
        "lm": {
            "success": bool(fit.success),
            "status": int(fit.status),
            "message": str(fit.message),
            "nfev": int(fit.nfev),
            "optimality": float(fit.optimality),
            "residual_calls": int(objective.calls),
            "wall_seconds": float(time.perf_counter() - started),
        },
    }


def _direct_chi2(parameters: dict) -> float:
    if _STATE is None:
        raise RuntimeError("fallback worker is not initialized")
    from hammlet._core.caustics import adamgrid_map_origin_shift
    from hammlet._core.direct_vbm import VBMBinaryLensEvaluator
    from hammlet._core.map_adapter import trajectory_xy
    from hammlet._core.reference import profile_flux

    report = _STATE["report"]
    lm_module = _STATE["lm_module"]
    coordinate_frame = lm_module._coordinate_frame(report)
    evaluator = VBMBinaryLensEvaluator(
        parameters["s"],
        parameters["q"],
        parameters["rho"],
        tolerance=1.0e-3,
        relative_tolerance=1.0e-4,
        coordinate_frame=coordinate_frame,
    )
    total = 0.0
    for dataset in _STATE["datasets"]:
        x, y = trajectory_xy(
            dataset.time,
            parameters["t0"],
            parameters["u0"],
            parameters["tE"],
            parameters["alpha"],
        )
        if coordinate_frame == "map":
            x = x - adamgrid_map_origin_shift(parameters["s"], parameters["q"])
        magnification = evaluator.magnification(x, y)
        total += float(profile_flux(magnification, dataset).chi2)
    return total


def _powell_worker(
    candidate: dict,
    widths: tuple[float, float, float],
    maxiter: int,
) -> dict:
    if _STATE is None:
        raise RuntimeError("fallback worker is not initialized")
    from hammlet._core.caustics import (
        map_frame_pspl_parameters,
        native_pspl_parameters_from_map,
    )
    from scipy.optimize import minimize

    initial = candidate["parameters"]
    t0_map, u0_map = map_frame_pspl_parameters(
        initial["t0"],
        initial["u0"],
        initial["tE"],
        initial["s"],
        initial["q"],
        initial["alpha"],
    )
    center = np.log10([initial["s"], initial["q"], initial["rho"]])
    bounds = [
        (float(value - width), float(value + width))
        for value, width in zip(center, widths, strict=True)
    ]
    evaluations = 0

    def parameters_at(z) -> dict[str, float]:
        s, q, rho = np.power(10.0, np.asarray(z, dtype=np.float64))
        t0, u0 = native_pspl_parameters_from_map(
            t0_map, u0_map, initial["tE"], s, q, initial["alpha"]
        )
        return {
            "t0": float(t0),
            "u0": float(u0),
            "tE": float(initial["tE"]),
            "s": float(s),
            "q": float(q),
            "rho": float(rho),
            "alpha": float(initial["alpha"]),
        }

    def objective(z) -> float:
        nonlocal evaluations
        evaluations += 1
        try:
            value = _direct_chi2(parameters_at(z))
            return value if math.isfinite(value) else 1.0e100
        except Exception:
            return 1.0e100

    started = time.perf_counter()
    fit = minimize(
        objective,
        center,
        method="Powell",
        bounds=bounds,
        options={"maxiter": int(maxiter), "xtol": 1.0e-3, "ftol": 1.0e-4},
    )
    parameters = parameters_at(fit.x)
    return {
        **candidate,
        "parameters_before_powell": initial,
        "parameters": parameters,
        "powell_chi2": float(fit.fun),
        "powell": {
            "success": bool(fit.success),
            "message": str(fit.message),
            "nfev": int(fit.nfev),
            "evaluations": int(evaluations),
            "wall_seconds": float(time.perf_counter() - started),
            "log10_lens_bounds": bounds,
        },
    }


def _parallel(
    report_path: Path,
    event_id: str,
    workers: int,
    label: str,
    candidates: list[dict],
    function,
    *function_args,
    early_stop_score: float | None = None,
    minimum_completed: int = 0,
) -> list[dict]:
    if not candidates:
        return []
    context = mp.get_context("spawn")
    results = []
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=min(int(workers), len(candidates)),
        mp_context=context,
        initializer=_worker_init,
        initargs=(str(report_path), event_id),
    ) as pool:
        futures = {
            pool.submit(function, candidate, *function_args): candidate
            for candidate in candidates
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if completed == 1 or completed % 16 == 0 or completed == len(candidates):
                score_field = (
                    "powell_chi2" if label == "powell" else "direct_vbm_chi2"
                )
                best = min(results, key=lambda row: float(row[score_field]))
                print(
                    f"[{label}] {event_id} {completed}/{len(candidates)} "
                    f"best={float(best[score_field]):.3f} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
            if (
                early_stop_score is not None
                and completed >= int(minimum_completed)
                and float(best[score_field]) <= float(early_stop_score)
            ):
                cancelled = 0
                for pending in futures:
                    if not pending.done() and pending.cancel():
                        cancelled += 1
                print(
                    f"[{label}] {event_id} early-stop best="
                    f"{float(best[score_field]):.3f} cancelled={cancelled}",
                    flush=True,
                )
                break
    return results


def _top_map_candidates(report: dict, report_path: Path, map_count: int) -> list[dict]:
    minima_path = report_path.parent / "minima.npz"
    with np.load(minima_path) as saved:
        map_ids = np.asarray(saved["map_ids"], dtype=np.int64)
        parameters = np.asarray(saved["parameters"], dtype=np.float64)
        chi2 = np.asarray(saved["chi2"], dtype=np.float64)
        geometry = np.asarray(saved["geometry_index"], dtype=np.int64)
        alpha = np.asarray(saved["alpha_index"], dtype=np.int64)
    order = np.argsort(chi2, kind="stable")[: min(int(map_count), len(chi2))]
    n_alpha = int(report.get("search_settings", {}).get("n_alpha", 2048))
    return [
        {
            "map_id": int(map_ids[row]),
            "logs": float(parameters[row, 0]),
            "logq": float(parameters[row, 1]),
            "logrho": float(parameters[row, 2]),
            "geometry_index": int(geometry[row]),
            "alpha_index": int(alpha[row]),
            "alpha": float(2.0 * np.pi * alpha[row] / n_alpha),
            "fft_chi2": float(chi2[row]),
            "rank_fft_map": int(rank),
        }
        for rank, row in enumerate(order, start=1)
    ]


def _build_seed_bank(
    report: dict,
    report_path: Path,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    alpha_module = _load_module(
        f"roman_fft_fallback_alpha_{report['event']['name']}",
        REPO_ROOT / "scripts/probe-roman-alpha-bank.py",
    )
    top_maps = _top_map_candidates(report, report_path, int(args.map_count))
    n_alpha = int(report.get("search_settings", {}).get("n_alpha", 2048))
    bank, scan_config = alpha_module._full_alpha_bank(
        report,
        report_path,
        top_maps,
        map_count=len(top_maps),
        alpha_count=1,
        geometry_index=None,
        n_alpha=n_alpha,
        alpha_window=True,
        per_geometry=True,
    )
    selected = select_fft_seeds(
        bank,
        elite_per_geometry=int(args.elite_per_geometry),
        diverse_per_geometry=int(args.diverse_per_geometry),
        minimum_grid_distance=float(args.minimum_grid_distance),
    )
    return selected, {
        **scan_config,
        "top_map_count": len(top_maps),
        "full_per_geometry_bank_count": len(bank),
        "selected_seed_count": len(selected),
        "elite_per_geometry": int(args.elite_per_geometry),
        "diverse_per_geometry": int(args.diverse_per_geometry),
        "minimum_grid_distance": float(args.minimum_grid_distance),
    }


def _long_seeds(continuations: list[dict]) -> list[dict]:
    seeds = []
    for row in continuations:
        seed = dict(row)
        seed["parameters"] = row["optimized_parameters"]
        seed["continued_from_short"] = True
        seeds.append(seed)
    return seeds


def _stage_path(output: Path, event_id: str, stage: str) -> Path:
    return output / "checkpoints" / event_id / f"{stage}.json"


def _run_event(event_id: str, args: argparse.Namespace) -> dict:
    report_path = _resolve_report(args.report_root, event_id)
    report = _read_json(report_path)
    dof = int(report["event"]["total_points"]) - 9
    output = args.output.expanduser().resolve()
    event_path = output / "events" / f"{event_id}.json"
    if args.resume and event_path.is_file():
        existing = _read_json(event_path)
        configuration = existing["configuration"]
        print(f"[resume] {event_id} completed event", flush=True)
        return {
            "event_id": event_id,
            "output": str(event_path),
            "short_seed_count": int(configuration["selected_seed_count"]),
            "continuation_count": int(configuration["continuation_count"]),
            "powell_ran": bool(configuration["powell_ran"]),
            "best_chi2_dof": float(existing["best_chi2_dof"]),
            "wall_seconds": float(configuration["wall_seconds"]),
            "resumed": True,
        }
    started = time.perf_counter()

    short_path = _stage_path(output, event_id, "short-lm")
    if args.resume and short_path.is_file():
        short_payload = _read_json(short_path)
        short_results = short_payload["results"]
        bank_config = short_payload["bank_configuration"]
        print(f"[resume] {event_id} short LM", flush=True)
    else:
        seeds, bank_config = _build_seed_bank(report, report_path, args)
        short_results = _parallel(
            report_path,
            event_id,
            int(args.workers),
            "short-lm",
            seeds,
            _lm_worker,
            int(args.short_max_nfev),
            "short",
            early_stop_score=float(args.short_stop_chi2_dof) * dof,
            minimum_completed=int(args.short_stop_min_completed),
        )
        short_results.sort(key=lambda row: float(row["direct_vbm_chi2"]))
        _write_json(
            short_path,
            {
                "event_id": event_id,
                "truth_used_for_selection": False,
                "bank_configuration": bank_config,
                "short_max_nfev": int(args.short_max_nfev),
                "short_stop_chi2_dof": float(args.short_stop_chi2_dof),
                "short_stop_min_completed": int(args.short_stop_min_completed),
                "short_early_stopped": len(short_results) < len(seeds),
                "results": short_results,
            },
        )

    continuations = select_continuations(
        short_results,
        best_count=int(args.continue_best),
        improvement_count=int(args.continue_improvement),
        one_improvement_per_geometry=True,
    )
    long_path = _stage_path(output, event_id, "long-lm")
    if args.resume and long_path.is_file():
        long_results = _read_json(long_path)["results"]
        print(f"[resume] {event_id} long LM", flush=True)
    else:
        long_results = _parallel(
            report_path,
            event_id,
            int(args.workers),
            "long-lm",
            _long_seeds(continuations),
            _lm_worker,
            int(args.long_max_nfev),
            "long",
        )
        long_results.sort(key=lambda row: float(row["direct_vbm_chi2"]))
        _write_json(
            long_path,
            {
                "event_id": event_id,
                "truth_used_for_selection": False,
                "continuation_count": len(continuations),
                "results": long_results,
            },
        )

    all_final = list(long_results)
    best_before_powell = min(all_final, key=lambda row: float(row["direct_vbm_chi2"]))
    powell_results = []
    powell_lm_results = []
    if (
        args.enable_powell
        and float(best_before_powell["direct_vbm_chi2"]) / dof
        > float(args.powell_trigger_chi2_dof)
    ):
        powell_path = _stage_path(output, event_id, "powell")
        if args.resume and powell_path.is_file():
            powell_results = _read_json(powell_path)["results"]
            print(f"[resume] {event_id} Powell", flush=True)
        else:
            powell_results = _parallel(
                report_path,
                event_id,
                int(args.workers),
                "powell",
                _long_seeds(continuations),
                _powell_worker,
                (
                    float(args.powell_s_half_width),
                    float(args.powell_q_half_width),
                    float(args.powell_rho_half_width),
                ),
                int(args.powell_maxiter),
            )
            powell_results.sort(key=lambda row: float(row["powell_chi2"]))
            _write_json(
                powell_path,
                {
                    "event_id": event_id,
                    "truth_used_for_selection": False,
                    "results": powell_results,
                },
            )

        powell_lm_path = _stage_path(output, event_id, "powell-long-lm")
        if args.resume and powell_lm_path.is_file():
            powell_lm_results = _read_json(powell_lm_path)["results"]
            print(f"[resume] {event_id} Powell LM", flush=True)
        else:
            powell_seeds = powell_results[: int(args.powell_lm_count)]
            powell_lm_results = _parallel(
                report_path,
                event_id,
                int(args.workers),
                "powell-long-lm",
                powell_seeds,
                _lm_worker,
                int(args.long_max_nfev),
                "powell-long",
            )
            powell_lm_results.sort(
                key=lambda row: float(row["direct_vbm_chi2"])
            )
            _write_json(
                powell_lm_path,
                {
                    "event_id": event_id,
                    "truth_used_for_selection": False,
                    "results": powell_lm_results,
                },
            )
        all_final.extend(powell_lm_results)

    best = min(all_final, key=lambda row: float(row["direct_vbm_chi2"]))
    payload = {
        "schema": "hammlet-roman-fft-fallback-v1",
        "event_id": event_id,
        "report_path": str(report_path),
        "truth_used_for_selection": False,
        "configuration": {
            **bank_config,
            "short_max_nfev": int(args.short_max_nfev),
            "short_stop_chi2_dof": float(args.short_stop_chi2_dof),
            "short_stop_min_completed": int(args.short_stop_min_completed),
            "continue_best": int(args.continue_best),
            "continue_improvement": int(args.continue_improvement),
            "continuation_count": len(continuations),
            "long_max_nfev": int(args.long_max_nfev),
            "powell_trigger_chi2_dof": float(args.powell_trigger_chi2_dof),
            "powell_enabled": bool(args.enable_powell),
            "powell_ran": bool(powell_results),
            "workers": int(args.workers),
            "event_workers": int(args.event_workers),
            "wall_seconds": float(time.perf_counter() - started),
        },
        "continuations": continuations,
        "long_lm": long_results,
        "powell": powell_results,
        "powell_long_lm": powell_lm_results,
        "best": best,
        "best_chi2_dof": float(best["direct_vbm_chi2"] / dof),
    }
    _write_json(event_path, payload)
    return {
        "event_id": event_id,
        "output": str(event_path),
        "short_seed_count": int(bank_config["selected_seed_count"]),
        "continuation_count": len(continuations),
        "powell_ran": bool(powell_results),
        "best_chi2_dof": payload["best_chi2_dof"],
        "wall_seconds": payload["configuration"]["wall_seconds"],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    positive = (
        args.map_count,
        args.elite_per_geometry,
        args.short_max_nfev,
        args.long_max_nfev,
        args.workers,
        args.event_workers,
    )
    if any(int(value) < 1 for value in positive):
        raise SystemExit("map, elite, LM, and worker counts must be positive")
    if int(args.diverse_per_geometry) < 0:
        raise SystemExit("diverse-per-geometry must be non-negative")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = []
    if int(args.event_workers) == 1:
        for index, event_id in enumerate(args.event, start=1):
            print(f"[fallback] event {index}/{len(args.event)} {event_id}", flush=True)
            results.append(_run_event(str(event_id), args))
    else:
        with ThreadPoolExecutor(max_workers=int(args.event_workers)) as pool:
            futures = {
                pool.submit(_run_event, str(event_id), args): index
                for index, event_id in enumerate(args.event)
            }
            ordered = [None] * len(args.event)
            for future in as_completed(futures):
                ordered[futures[future]] = future.result()
            results = ordered
    summary = {
        "schema": "hammlet-roman-fft-fallback-summary-v1",
        "state": "complete",
        "truth_used_for_selection": False,
        "results": results,
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
