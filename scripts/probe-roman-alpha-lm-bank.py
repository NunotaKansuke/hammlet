#!/usr/bin/env python3
"""Run a bounded LM bank from the full FFT alpha candidates.

This is the deliberately small refinement probe:

    saved FFT map minima -> top alpha/geometry candidates -> parallel LM bank

The FFT scan has already evaluated the complete alpha grid.  We retain more
than one alpha per map and let the observed-data LM objective decide which
local basin wins.  There is no dense direct-VBM ``(s,q)`` profile and no
truth-based ranking in this path.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import json
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
    parser.add_argument("--map-count", type=int, default=4)
    parser.add_argument("--alpha-count", type=int, default=16)
    parser.add_argument(
        "--lm-count",
        type=int,
        help="LM at most this many FFT candidates; default is the full bank",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, os.cpu_count() or 1),
    )
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument(
        "--map-frame-lm",
        action="store_true",
        help="optimize map-frame t0,u0 and convert to native coordinates per trial",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results/roman_local/roman_alpha_lm_bank_probe",
    )
    return parser.parse_args(argv)


_LM_STATE: dict[str, object] | None = None


class _MapFrameLMObjective:
    """LM objective whose trajectory variables stay in the map frame.

    The direct evaluator consumes native ``t0,u0`` while the atlas geometry
    is stored in map coordinates.  Keeping map-frame trajectory coordinates
    as the optimization variables makes the dependence of the origin shift
    on ``s,q`` explicit instead of forcing LM to discover that coupling
    through two weakly scaled native variables.
    """

    def __init__(self, lm_module, datasets, initial: dict[str, float], **kwargs):
        from hammlet._core.caustics import map_frame_pspl_parameters

        self._lm_module = lm_module
        self._base = lm_module._LMObjective(datasets, initial, **kwargs)
        self.datasets = datasets
        self.initial = initial
        self.t0_map, self.u0_map = map_frame_pspl_parameters(
            initial["t0"],
            initial["u0"],
            initial["tE"],
            initial["s"],
            initial["q"],
            initial["alpha"],
        )
        self.t0_scale = max(abs(initial["tE"]), 1.0)
        self.u0_scale = max(abs(self.u0_map), initial["rho"], 0.01)
        self.vbm_tolerance = self._base.vbm_tolerance
        self.vbm_relative_tolerance = self._base.vbm_relative_tolerance
        self.coordinate_frame = self._base.coordinate_frame
        self.n_residuals = self._base.n_residuals
        self.calls = 0
        self.last_error = None
        self._cached_z = None
        self._cached_residual = None

    def unpack(self, z):
        from hammlet._core.caustics import native_pspl_parameters_from_map

        z = np.asarray(z, dtype=np.float64)
        if z.shape != (7,) or not np.all(np.isfinite(z)):
            return None
        if np.any(np.abs(z[2:6]) > 30.0) or abs(float(z[0])) > 1.0e6:
            return None
        tE = self.initial["tE"] * float(np.exp(z[2]))
        s = self.initial["s"] * float(np.exp(z[3]))
        q = self.initial["q"] * float(np.exp(z[4]))
        rho = self.initial["rho"] * float(np.exp(z[5]))
        alpha = self.initial["alpha"] + float(z[6])
        t0_map = self.t0_map + self.t0_scale * float(z[0])
        u0_map = self.u0_map + self.u0_scale * float(z[1])
        if not np.all(np.isfinite((t0_map, u0_map, tE, s, q, rho, alpha))):
            return None
        if any(value <= 0.0 for value in (tE, s, q, rho)):
            return None
        if not (
            1.0e-6 <= tE <= 1.0e4
            and 1.0e-4 <= s <= 1.0e2
            and 1.0e-8 <= q <= 1.0e4
            and 1.0e-8 <= rho <= 1.0
        ):
            return None
        t0, u0 = native_pspl_parameters_from_map(
            t0_map, u0_map, tE, s, q, alpha
        )
        values = (t0, u0, tE, s, q, rho, alpha)
        return values if np.all(np.isfinite(values)) else None

    def __call__(self, z):
        # Reuse the exact residual implementation from roman-lm-refine while
        # replacing only its transformed-parameter unpacking.
        z = np.asarray(z, dtype=np.float64)
        if self._cached_z is not None and np.array_equal(z, self._cached_z):
            return self._cached_residual.copy()
        self.calls += 1
        values = self.unpack(z)
        if values is None:
            self.last_error = "invalid trial parameter transform"
            residual = self._base._invalid_residual()
            self._cached_z = z.copy()
            self._cached_residual = residual.copy()
            return residual
        t0, u0, tE, s, q, rho, alpha = values
        try:
            evaluator = __import__(
                "hammlet._core.direct_vbm", fromlist=["VBMBinaryLensEvaluator"]
            ).VBMBinaryLensEvaluator(
                s,
                q,
                rho,
                tolerance=self.vbm_tolerance,
                relative_tolerance=self.vbm_relative_tolerance,
                coordinate_frame=self.coordinate_frame,
            )
            pieces = []
            for dataset in self.datasets:
                from hammlet._core.map_adapter import trajectory_xy
                from hammlet._core.reference import profile_flux

                x, y = trajectory_xy(dataset.time, t0, u0, tE, alpha)
                if self.coordinate_frame == "map":
                    from hammlet._core.caustics import adamgrid_map_origin_shift

                    x = x - adamgrid_map_origin_shift(s, q)
                magnification = evaluator.magnification(x, y)
                profile = profile_flux(magnification, dataset)
                if not np.isfinite(profile.chi2):
                    raise ValueError("profiled flux fit returned non-finite chi2")
                model = profile.source_flux * magnification + profile.blend_flux
                pieces.append((dataset.flux - model) / dataset.error)
            residual = np.concatenate(pieces)
            self.last_error = None
        except Exception as error:
            self.last_error = f"{type(error).__name__}: {error}"
            residual = self._base._invalid_residual()
        self._cached_z = z.copy()
        self._cached_residual = residual.copy()
        return residual


def _lm_worker_init(report_path: str, event_id: str, map_frame_lm: bool) -> None:
    global _LM_STATE
    report_path_obj = Path(report_path)
    report = read_json(report_path_obj)
    lm_module = load_module(
        f"roman_lm_alpha_bank_worker_{event_id}",
        REPO_ROOT / "scripts/roman-lm-refine.py",
    )
    datasets = lm_module._load_datasets(report, report_path_obj)
    _LM_STATE = {
        "report": report,
        "lm_module": lm_module,
        "datasets": datasets,
        "map_frame_lm": bool(map_frame_lm),
    }


def _lm_worker(candidate: dict, max_nfev: int) -> dict:
    if _LM_STATE is None:
        raise RuntimeError("LM worker was not initialized")
    from scipy.optimize import least_squares

    report = _LM_STATE["report"]
    lm_module = _LM_STATE["lm_module"]
    datasets = _LM_STATE["datasets"]
    objective_class = (
        _MapFrameLMObjective
        if _LM_STATE.get("map_frame_lm")
        else lm_module._LMObjective
    )
    objective = objective_class(
        lm_module,
        datasets,
        candidate["parameters"],
        vbm_tolerance=1.0e-3,
        vbm_relative_tolerance=1.0e-4,
        coordinate_frame=lm_module._coordinate_frame(report),
    ) if objective_class is _MapFrameLMObjective else objective_class(
        datasets,
        candidate["parameters"],
        vbm_tolerance=1.0e-3,
        vbm_relative_tolerance=1.0e-4,
        coordinate_frame=lm_module._coordinate_frame(report),
    )
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
    final_values = objective.unpack(fit.x)
    if final_values is None:
        raise ValueError("LM returned invalid parameters")
    final_residual = objective(fit.x)
    final_chi2 = float(np.dot(final_residual, final_residual))
    return {
        "rank_fft": int(candidate.get("rank_fft", 0)),
        "map_id": int(candidate["map_id"]),
        "geometry_index": int(candidate["geometry_index"]),
        "alpha_index": int(candidate["alpha_index"]),
        "rank_within_map": int(candidate["rank_within_map"]),
        "fft_chi2": float(candidate["fft_chi2"]),
        "initial_parameters": candidate["parameters"],
        "start_direct_vbm_chi2": start_chi2,
        "optimized_parameters": lm_module._parameter_dict(final_values),
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


def run_event(event_id: str, args: argparse.Namespace) -> dict:
    report_path = REPORT_ROOTS[event_id] / "report.json"
    report = read_json(report_path)
    candidates = json.loads(
        (report_path.parent / "candidates.json").read_text(encoding="utf-8")
    )
    alpha_module = load_module(
        f"roman_alpha_bank_for_lm_{event_id}",
        REPO_ROOT / "scripts/probe-roman-alpha-bank.py",
    )
    bank, bank_config = alpha_module._full_alpha_bank(
        report,
        report_path,
        candidates,
        map_count=int(args.map_count),
        alpha_count=int(args.alpha_count),
    )
    # Preserve FFT ordering for reproducibility.  If a limit is requested it
    # is applied after the blind FFT ranking, never after looking at truth.
    bank.sort(key=lambda item: item["fft_chi2"])
    for rank, candidate in enumerate(bank, start=1):
        candidate["rank_fft"] = int(rank)
    selected = bank if args.lm_count is None else bank[: int(args.lm_count)]
    if not selected:
        raise ValueError("empty alpha/geometry bank")

    results = []
    started = time.perf_counter()
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(int(args.workers), len(selected)),
        mp_context=context,
        initializer=_lm_worker_init,
        initargs=(str(report_path.resolve()), event_id, bool(args.map_frame_lm)),
    ) as pool:
        future_to_candidate = {
            pool.submit(_lm_worker, candidate, int(args.max_nfev)): candidate
            for candidate in selected
        }
        for index, future in enumerate(as_completed(future_to_candidate), start=1):
            result = future.result()
            results.append(result)
            best = min(results, key=lambda item: item["direct_vbm_chi2"])
            print(
                f"[alpha-lm-bank] {event_id} {index}/{len(selected)} "
                f"best={best['direct_vbm_chi2']:.1f} "
                f"map={best['map_id']} alpha_index={best['alpha_index']} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    results.sort(key=lambda item: item["direct_vbm_chi2"])
    best = results[0]
    dof = int(report["event"]["total_points"]) - 9
    output_path = (
        args.output.expanduser().resolve() / "events" / f"{event_id}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "hammlet-roman-alpha-lm-bank-v1",
        "event_id": event_id,
        "report_path": str(report_path.resolve()),
        "truth_used_for_selection": False,
        "configuration": {
            **bank_config,
            "lm_count": len(selected),
            "available_bank": len(bank),
            "workers": int(args.workers),
            "max_nfev": int(args.max_nfev),
            "lm_parameter_frame": "map" if args.map_frame_lm else "native",
            "wall_seconds": float(time.perf_counter() - started),
        },
        "lm_candidates": results,
        "best": best,
        "best_chi2_dof": float(best["direct_vbm_chi2"] / dof),
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "event_id": event_id,
        "output": str(output_path),
        "bank_count": len(bank),
        "lm_count": len(selected),
        "best_map_id": best["map_id"],
        "best_alpha_index": best["alpha_index"],
        "best_rank_fft": best["rank_fft"],
        "best_rank_within_map": best["rank_within_map"],
        "best_chi2_dof": float(best["direct_vbm_chi2"] / dof),
        "wall_seconds": payload["configuration"]["wall_seconds"],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.map_count < 1 or args.alpha_count < 1:
        raise SystemExit("map-count and alpha-count must be positive")
    if args.lm_count is not None and args.lm_count < 1:
        raise SystemExit("lm-count must be positive")
    if args.workers < 1 or args.max_nfev < 1:
        raise SystemExit("workers and max-nfev must be positive")
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
