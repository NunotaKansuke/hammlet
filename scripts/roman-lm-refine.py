#!/usr/bin/env python3
"""Refine Roman/GULLS Hammlet seeds with direct VBM and LM.

The Roman local search stores an FFT-grid seed in ``best_fft``.  This script
starts from that seed and performs a local least-squares fit using the direct
VBMicrolensing evaluator.  The source and blend fluxes are profiled
analytically at every residual evaluation, matching Hammlet's likelihood
convention.

The fit parameters are transformed to keep the positive parameters positive::

    (t0 / tE, u0, log(tE), log(s), log(q), log(rho), alpha)

``scipy.optimize.least_squares(method="lm")`` is deliberately used without
hard bounds.  The result is therefore a local refinement of the supplied seed,
not a global fit or a posterior interval.  The raw q convention is retained;
in particular, q > 1 is not folded to 1/q.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any


# Keep one BLAS/OpenMP thread per event worker unless the caller explicitly
# supplied a different setting.  The direct VBM calls are already scalar and
# event-level parallelism is the useful parallel dimension here.
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
SRC_ROOT = REPO_ROOT / "src"
for _import_root in (REPO_ROOT, SRC_ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))


DEFAULT_EVENT_ROOTS = (
    REPO_ROOT
    / "results/roman_local/roman_hammlet_batch100_full_m128_16w1t/events",
    REPO_ROOT
    / "results/roman_local/roman_binary_hammlet_signal_batch100_full_m128_16w1t/events",
)
DEFAULT_OUTPUT = REPO_ROOT / "results/roman_local/roman_lm_direct_vbm_batch200"


PARAMETER_NAMES = ("t0", "u0", "tE", "s", "q", "rho", "alpha")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        action="append",
        type=Path,
        help=(
            "event directory containing report.json files; repeatable "
            "(defaults to the two 100-event Roman result roots)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--event-id",
        action="append",
        help="run only this event ID; repeatable",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        help="optional deterministic limit after sorting event IDs",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(24, os.cpu_count() or 1),
        help="event-level worker count (default: min(24, CPU count))",
    )
    parser.add_argument(
        "--max-nfev",
        type=int,
        default=50,
        help="maximum LM function evaluations per event (default: 50)",
    )
    parser.add_argument(
        "--diff-step",
        type=float,
        default=1.0e-3,
        help="relative finite-difference step used by LM (default: 1e-3)",
    )
    parser.add_argument(
        "--vbm-tolerance",
        type=float,
        default=1.0e-3,
        help="VBMicrolensing absolute tolerance (default: 1e-3)",
    )
    parser.add_argument(
        "--vbm-relative-tolerance",
        type=float,
        default=1.0e-4,
        help="VBMicrolensing relative tolerance (default: 1e-4)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse per-event JSON files already present in the output directory",
    )
    return parser.parse_args(argv)


def _numeric_event_id(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):020d}")
    except ValueError:
        return (1, value)


def _collect_reports(
    roots: list[Path], requested_ids: set[str] | None, max_events: int | None
) -> list[Path]:
    by_event_id: dict[str, Path] = {}
    for root in roots:
        root = root.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"input event root does not exist: {root}")
        for report in root.glob("*/report.json"):
            event_id = report.parent.name
            if requested_ids is not None and event_id not in requested_ids:
                continue
            previous = by_event_id.get(event_id)
            if previous is not None and previous != report:
                raise ValueError(
                    f"event ID {event_id} occurs in both {previous} and {report}"
                )
            by_event_id[event_id] = report
    reports = sorted(by_event_id.values(), key=lambda path: _numeric_event_id(path.parent.name))
    if max_events is not None:
        if max_events <= 0:
            raise ValueError("--max-events must be positive")
        reports = reports[:max_events]
    if not reports:
        raise ValueError("no event reports matched the requested inputs")
    return reports


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def _load_datasets(report: dict[str, Any], report_path: Path):
    from tools.roman_local.roman_event import load_data_file, load_gulls_data_file

    event = report["event"]
    paths = []
    batch_root = report_path.resolve().parents[2]
    for value in event["data_paths"]:
        path = Path(value).expanduser()
        if not path.is_file():
            # Reports may have been produced on a different host whose
            # absolute input prefix is not mounted on the LM worker.  The
            # batch result keeps the immutable input snapshot beside the
            # event reports, so resolve by basename within that snapshot.
            local_copy = batch_root / "input" / "hammlet_input" / "lightcurves" / path.name
            if local_copy.is_file():
                path = local_copy
        paths.append(path)
    names = list(event.get("datasets", {}).keys())
    if len(names) != len(paths):
        names = [f"data-{index}" for index in range(len(paths))]
    window = tuple(float(value) for value in event["window"])
    data_format = event.get("data_format", "time-flux-error")
    datasets = []
    for name, path in zip(names, paths, strict=True):
        if data_format == "gulls-all-lc":
            datasets.append(load_gulls_data_file(path, name=name, window=window))
        else:
            datasets.append(
                load_data_file(
                    path,
                    name=name,
                    window=window,
                    data_format=data_format,
                )
            )
    return tuple(datasets)


def _initial_parameters(report: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    best = report.get("best_fft")
    if not isinstance(best, dict):
        raise ValueError("report has no best_fft object")
    geometry_index = int(best["geometry_index"])
    geometry = report["geometries"][geometry_index]
    logs, logq, logrho = (float(value) for value in best["parameters"])
    t0_geometry = float(geometry[0])
    u0_geometry = float(geometry[1])
    tE = float(geometry[2])
    alpha = float(best["alpha"])
    geometry_frame = str(
        report.get("geometry_frame")
        or report.get("geometry_stencil", {}).get("geometry_frame", "native")
    )
    if geometry_frame not in ("native", "map"):
        raise ValueError(f"unsupported search geometry frame: {geometry_frame!r}")
    if geometry_frame == "map":
        from hammlet._core.caustics import native_pspl_parameters_from_map

        t0, u0 = native_pspl_parameters_from_map(
            t0_geometry,
            u0_geometry,
            tE,
            10.0**logs,
            10.0**logq,
            alpha,
        )
    else:
        t0, u0 = t0_geometry, u0_geometry
    values = {
        "t0": float(t0),
        "u0": float(u0),
        "tE": tE,
        "s": float(10.0**logs),
        "q": float(10.0**logq),
        "rho": float(10.0**logrho),
        "alpha": alpha,
    }
    for name in ("tE", "s", "q", "rho"):
        if not (values[name] > 0.0):
            raise ValueError(f"invalid positive initial parameter {name}: {values[name]}")
    metadata = {
        "map_id": int(best["map_id"]),
        "geometry_index": geometry_index,
        "alpha_index": int(best["alpha_index"]),
        "fft_chi2": float(best["chi2"]),
        "grid_parameters": [logs, logq, logrho],
        "geometry_frame": geometry_frame,
        "geometry_parameters": {
            "t0": t0_geometry,
            "u0": u0_geometry,
            "tE": tE,
        },
    }
    return values, metadata


def _parameter_dict(values: tuple[float, ...] | list[float] | Any) -> dict[str, float]:
    return {
        name: float(value) for name, value in zip(PARAMETER_NAMES, values, strict=True)
    }


def _coordinate_frame(report: dict[str, Any]) -> str:
    """Return the atlas frame, defaulting to the legacy map frame."""
    metadata = report.get("atlas", {}).get("metadata", {})
    if not isinstance(metadata, dict):
        return "map"
    frame = metadata.get("coordinate_frame")
    if frame is None and isinstance(metadata.get("source_metadata"), dict):
        frame = metadata["source_metadata"].get("coordinate_frame")
    if frame is None:
        frame = "map"
    if frame not in ("map", "native"):
        raise ValueError(f"unsupported atlas coordinate frame: {frame!r}")
    return str(frame)


class _LMObjective:
    """Profiled direct-VBM residual for one event."""

    def __init__(
        self,
        datasets,
        initial: dict[str, float],
        *,
        vbm_tolerance: float,
        vbm_relative_tolerance: float,
        coordinate_frame: str,
    ) -> None:
        self.datasets = datasets
        self.initial = initial
        self.t0_scale = max(abs(initial["tE"]), 1.0)
        self.u0_scale = max(abs(initial["u0"]), initial["rho"], 0.01)
        self.vbm_tolerance = float(vbm_tolerance)
        self.vbm_relative_tolerance = float(vbm_relative_tolerance)
        if coordinate_frame not in ("map", "native"):
            raise ValueError("coordinate_frame must be 'map' or 'native'")
        self.coordinate_frame = coordinate_frame
        self.n_residuals = int(sum(len(dataset.time) for dataset in datasets))
        self.calls = 0
        self.last_error: str | None = None
        self._cached_z = None
        self._cached_residual = None

    def unpack(self, z) -> tuple[float, ...] | None:
        import numpy as np

        z = np.asarray(z, dtype=np.float64)
        if z.shape != (7,) or not np.all(np.isfinite(z)):
            return None
        # Reject runaway trial points before exp() or VBM sees them.  These
        # limits are only numerical guards; they are far outside the useful
        # neighborhood of any sensible local fit.
        if np.any(np.abs(z[2:6]) > 30.0) or abs(float(z[0])) > 1.0e6:
            return None
        t0 = self.initial["t0"] + self.t0_scale * float(z[0])
        u0 = self.initial["u0"] + self.u0_scale * float(z[1])
        tE = self.initial["tE"] * float(np.exp(z[2]))
        s = self.initial["s"] * float(np.exp(z[3]))
        q = self.initial["q"] * float(np.exp(z[4]))
        rho = self.initial["rho"] * float(np.exp(z[5]))
        alpha = self.initial["alpha"] + float(z[6])
        values = (t0, u0, tE, s, q, rho, alpha)
        if not np.all(np.isfinite(values)) or any(value <= 0.0 for value in values[2:6]):
            return None
        # Direct VBMicrolensing is very fast in the intended local
        # neighbourhood but can spend an unbounded amount of time on
        # numerically meaningless trial binaries.  These are deliberately
        # broad physical guards, not inference priors; they only turn such
        # runaway LM probes into the invalid residual used above.
        if not (
            1.0e-6 <= tE <= 1.0e4
            and 1.0e-4 <= s <= 1.0e2
            and 1.0e-8 <= q <= 1.0e4
            and 1.0e-8 <= rho <= 1.0
        ):
            return None
        return values

    def _invalid_residual(self):
        import numpy as np

        return np.full(self.n_residuals, 1.0e6, dtype=np.float64)

    def __call__(self, z):
        import numpy as np

        z = np.asarray(z, dtype=np.float64)
        if self._cached_z is not None and np.array_equal(z, self._cached_z):
            return self._cached_residual.copy()
        self.calls += 1
        values = self.unpack(z)
        if values is None:
            self.last_error = "invalid trial parameter transform"
            residual = self._invalid_residual()
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
                piece = (dataset.flux - model) / dataset.error
                if not np.all(np.isfinite(piece)):
                    raise ValueError("profiled direct-VBM residual is non-finite")
                pieces.append(piece)
            residual = np.concatenate(pieces)
            self.last_error = None
        except Exception as error:  # keep one bad LM trial from killing the batch
            self.last_error = f"{type(error).__name__}: {error}"
            residual = self._invalid_residual()
        self._cached_z = z.copy()
        self._cached_residual = residual.copy()
        return residual


def _zero_state() -> list[float]:
    return [0.0] * 7


def _run_one(payload: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from scipy.optimize import least_squares

    report_path = Path(payload["report_path"])
    event_id = report_path.parent.name
    started = time.perf_counter()
    try:
        report = _load_json(report_path)
        datasets = _load_datasets(report, report_path)
        initial, seed_metadata = _initial_parameters(report)
        objective = _LMObjective(
            datasets,
            initial,
            vbm_tolerance=float(payload["vbm_tolerance"]),
            vbm_relative_tolerance=float(payload["vbm_relative_tolerance"]),
            coordinate_frame=_coordinate_frame(report),
        )
        z0 = np.asarray(_zero_state(), dtype=np.float64)
        start_residual = objective(z0)
        start_chi2 = float(np.dot(start_residual, start_residual))
        optimization_error = None
        try:
            fit = least_squares(
                objective,
                z0,
                method="lm",
                jac="2-point",
                diff_step=float(payload["diff_step"]),
                max_nfev=int(payload["max_nfev"]),
                ftol=1.0e-8,
                xtol=1.0e-8,
                gtol=1.0e-8,
                x_scale="jac",
            )
            final_values = objective.unpack(fit.x)
            if final_values is None:
                raise ValueError("LM returned an invalid transformed parameter vector")
            final_residual = objective(fit.x)
            final_chi2 = float(np.dot(final_residual, final_residual))
            lm = {
                "success": bool(fit.success),
                "status": int(fit.status),
                "message": str(fit.message),
                "nfev": int(fit.nfev),
                "njev": None if fit.njev is None else int(fit.njev),
                "optimality": float(fit.optimality),
                "residual_calls": int(objective.calls),
            }
        except Exception as error:
            optimization_error = f"{type(error).__name__}: {error}"
            final_values = tuple(initial[name] for name in PARAMETER_NAMES)
            final_chi2 = start_chi2
            lm = {
                "success": False,
                "status": -1,
                "message": optimization_error,
                "nfev": None,
                "njev": None,
                "optimality": None,
                "residual_calls": int(objective.calls),
            }
        result = {
            "schema": "hammlet-roman-lm-direct-vbm-v1",
            "event_id": event_id,
            "report_path": str(report_path.resolve()),
            "n_data": int(objective.n_residuals),
            "initial_seed": {
                **seed_metadata,
                "parameters": initial,
                "direct_vbm_chi2": start_chi2,
            },
            "optimized": {
                "parameters": _parameter_dict(final_values),
                "direct_vbm_chi2": final_chi2,
                "chi2_improvement": start_chi2 - final_chi2,
            },
            "truth": report.get("event", {}).get("reference"),
            "coordinate_frame": objective.coordinate_frame,
            "lm": lm,
            "error": optimization_error,
            "last_objective_error": objective.last_error,
            "wall_seconds": time.perf_counter() - started,
        }
        return result
    except Exception as error:
        return {
            "schema": "hammlet-roman-lm-direct-vbm-v1",
            "event_id": event_id,
            "report_path": str(report_path.resolve()),
            "status": "worker-error",
            "error": f"{type(error).__name__}: {error}",
            "wall_seconds": time.perf_counter() - started,
        }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _summary(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    ordered = sorted(results, key=lambda row: _numeric_event_id(str(row["event_id"])))
    successful = [row for row in ordered if row.get("lm", {}).get("success")]
    finite_improvements = [
        float(row["optimized"]["chi2_improvement"])
        for row in ordered
        if isinstance(row.get("optimized"), dict)
        and isinstance(row["optimized"].get("chi2_improvement"), (int, float))
    ]
    return {
        "schema": "hammlet-roman-lm-direct-vbm-batch-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "max_nfev": int(args.max_nfev),
            "diff_step": float(args.diff_step),
            "vbm_tolerance": float(args.vbm_tolerance),
            "vbm_relative_tolerance": float(args.vbm_relative_tolerance),
            "workers": int(args.workers),
        },
        "counts": {
            "total": len(ordered),
            "lm_success": len(successful),
            "worker_errors": sum(row.get("status") == "worker-error" for row in ordered),
            "chi2_improvement_nonnegative": sum(value >= 0.0 for value in finite_improvements),
        },
        "events": ordered,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.max_nfev < 1:
        raise SystemExit("--max-nfev must be positive")
    if args.diff_step <= 0.0:
        raise SystemExit("--diff-step must be positive")
    roots = [path.expanduser().resolve() for path in (args.input_root or DEFAULT_EVENT_ROOTS)]
    requested = None if not args.event_id else {str(value) for value in args.event_id}
    reports = _collect_reports(roots, requested, args.max_events)
    output = args.output.expanduser().resolve()
    event_output = output / "events"
    event_output.mkdir(parents=True, exist_ok=True)

    pending: list[Path] = []
    results: list[dict[str, Any]] = []
    for report in reports:
        result_path = event_output / f"{report.parent.name}.json"
        if args.resume and result_path.is_file():
            existing = _load_json(result_path)
            # A worker error is resumable: the source path or host may have
            # been repaired since the failed attempt.  Keep successful
            # results immutable while retrying only these failed records.
            if existing.get("status") == "worker-error":
                pending.append(report)
            else:
                results.append(existing)
        else:
            pending.append(report)

    print(
        f"[roman-lm] selected={len(reports)} pending={len(pending)} "
        f"workers={args.workers} max_nfev={args.max_nfev}",
        flush=True,
    )
    options = {
        "max_nfev": int(args.max_nfev),
        "diff_step": float(args.diff_step),
        "vbm_tolerance": float(args.vbm_tolerance),
        "vbm_relative_tolerance": float(args.vbm_relative_tolerance),
    }
    if pending:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            futures = {
                pool.submit(
                    _run_one,
                    {"report_path": str(report), **options},
                ): report
                for report in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                report = futures[future]
                try:
                    result = future.result()
                except Exception as error:  # defensive: worker should return JSON
                    result = {
                        "schema": "hammlet-roman-lm-direct-vbm-v1",
                        "event_id": report.parent.name,
                        "report_path": str(report.resolve()),
                        "status": "worker-error",
                        "error": f"{type(error).__name__}: {error}",
                    }
                results.append(result)
                _write_json(event_output / f"{report.parent.name}.json", result)
                optimized = result.get("optimized", {})
                start_chi2 = result.get("initial_seed", {}).get("direct_vbm_chi2")
                final_chi2 = optimized.get("direct_vbm_chi2")
                if isinstance(start_chi2, (int, float)) and isinstance(
                    final_chi2, (int, float)
                ):
                    message = f"chi2 {start_chi2:.1f}->{final_chi2:.1f}"
                else:
                    message = str(result.get("error", "no chi2"))
                print(f"[roman-lm] {index}/{len(pending)} {report.parent.name} {message}", flush=True)

    _write_json(output / "summary.json", _summary(results, args))
    print(f"[roman-lm] wrote {output / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
