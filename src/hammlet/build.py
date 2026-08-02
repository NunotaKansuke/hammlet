"""Map construction, partitioning and merge operations."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np

from .config import MapConfig, ParameterGrid, Partition
from ._core.adaptive_grid import (
    build_adaptive_radial_nodes,
    sharp_qs_pilot_indices,
)
from ._core.atlas_builder import MapBuildSpec
from ._core.bucketed_atlas import BucketedPolarAtlas, write_bucket_manifest
from ._core.direct_vbm import (
    DirectSpectrumConfig,
    DirectVBMPolarAtlasBuilder,
    VBMBinaryLensEvaluator,
)


def _spectrum_config(config: MapConfig) -> DirectSpectrumConfig:
    return DirectSpectrumConfig(
        m_max=config.m_max,
        core_m_max=config.core_m_max,
        min_n_phi=config.min_n_phi,
        caustic_base_n_phi=config.caustic_base_n_phi,
        max_n_phi=config.max_n_phi,
        coefficient_rtol=config.coefficient_rtol,
        coefficient_atol=config.coefficient_atol,
        caustic_guard_rho=config.caustic_guard_rho,
        caustic_local_levels=config.caustic_local_levels,
        caustic_points_per_side=config.caustic_points_per_side,
        diagnostic_m_max=config.diagnostic_m_max,
        radial_certificate_levels=config.radial_certificate_levels,
    )


def _specs(rows: np.ndarray, config: MapConfig) -> Iterable[MapBuildSpec]:
    for map_id_value, logs, logq, logrho in rows:
        yield MapBuildSpec(
            map_id=int(map_id_value),
            logs=float(logs),
            logq=float(logq),
            logrho=float(logrho),
            evaluator=VBMBinaryLensEvaluator(
                10.0**logs,
                10.0**logq,
                10.0**logrho,
                tolerance=config.vbm_tolerance,
                relative_tolerance=config.vbm_relative_tolerance,
            ),
        )


def _parameter_buckets(table: np.ndarray, config: MapConfig) -> list[dict[str, object]]:
    """Group complete rho axes into deterministic `(s,q)` radial-layout bins."""
    logs_values = np.unique(table[:, 1])
    logq_values = np.unique(table[:, 2])
    logs_groups = np.array_split(
        logs_values, min(int(config.radial_s_buckets), len(logs_values))
    )
    logq_groups = np.array_split(
        logq_values, min(int(config.radial_q_buckets), len(logq_values))
    )
    buckets: list[dict[str, object]] = []
    for logs_index, logs_group in enumerate(logs_groups):
        for logq_index, logq_group in enumerate(logq_groups):
            mask = np.isin(table[:, 1], logs_group) & np.isin(table[:, 2], logq_group)
            rows = table[mask]
            if not len(rows):
                continue
            buckets.append(
                {
                    "name": f"s{logs_index:03d}_q{logq_index:03d}",
                    "rows": rows,
                    "logs": [float(logs_group[0]), float(logs_group[-1])],
                    "logq": [float(logq_group[0]), float(logq_group[-1])],
                }
            )
    return buckets


def _pilot_evaluators(rows: np.ndarray, config: MapConfig):
    local = sharp_qs_pilot_indices(
        rows[:, 1:], min(int(config.radial_pilot_maps), len(rows))
    )
    pilot_rows = rows[local]
    evaluators = [
        VBMBinaryLensEvaluator(
            10.0**logs,
            10.0**logq,
            10.0**logrho,
            tolerance=config.vbm_tolerance,
            relative_tolerance=config.vbm_relative_tolerance,
        )
        for _, logs, logq, logrho in pilot_rows
    ]
    return pilot_rows, evaluators


def build_maps(
    output: str | Path,
    grid: ParameterGrid,
    *,
    config: MapConfig | None = None,
    partition: Partition | None = None,
    progress_every: int = 1,
) -> Path:
    """Generate one complete map collection or one schedulable part.

    For ``Partition(index=i, count=n)``, every machine receives the same grid
    and config and writes ``output/parts/part-iiiii-of-nnnnn``.  A private
    temporary directory is atomically renamed only after the part succeeds.
    """
    config = config or MapConfig()
    partition = partition or Partition()
    output = Path(output).resolve()
    table = grid.table()
    all_buckets = _parameter_buckets(table, config)
    buckets = all_buckets[partition.rows(len(all_buckets))]
    if not buckets:
        raise ValueError("this partition contains no radial-layout buckets")
    destination = (
        output
        if partition.count == 1
        else output / "parts" / f"part-{partition.index:05d}-of-{partition.count:05d}"
    )
    if destination.exists():
        raise FileExistsError(f"map destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"stale partial output exists: {temporary}")

    try:
        temporary.mkdir(parents=True)
        entries = []
        selected_ids: list[int] = []
        for bucket in buckets:
            rows = np.asarray(bucket["rows"], dtype=np.float64)
            pilot_rows, evaluators = _pilot_evaluators(rows, config)
            radial_nodes, pilot_radii, difficulty = build_adaptive_radial_nodes(
                evaluators,
                max_radius=config.radial_max,
                n_nodes=config.radial_nodes,
                pilot_bins=config.radial_pilot_bins,
                n_phi=config.radial_pilot_phi,
                m_max=config.radial_pilot_m_max,
                map_quantile=1.0,
                adaptive_fraction=config.radial_adaptive_fraction,
            )
            child = temporary / str(bucket["name"])
            builder = DirectVBMPolarAtlasBuilder(
                radial_nodes,
                spectrum_config=_spectrum_config(config),
                shard_size=config.shard_size,
            )
            builder.build(child, _specs(rows, config), progress_every=progress_every)
            np.save(child / "radial_pilot_map_ids.npy", pilot_rows[:, 0].astype(np.int64))
            np.save(child / "radial_pilot_radii.npy", pilot_radii)
            np.save(child / "radial_pilot_difficulty.npy", difficulty)
            map_ids = rows[:, 0].astype(int).tolist()
            selected_ids.extend(map_ids)
            entries.append(
                {
                    "name": bucket["name"],
                    "path": bucket["name"],
                    "count": len(rows),
                    "logs": bucket["logs"],
                    "logq": bucket["logq"],
                    "map_ids": map_ids,
                }
            )
        write_bucket_manifest(
            temporary,
            entries,
            {
                "builder": "direct-vbm-rho-aware-adaptive-radial-fourier",
                "radial_layout": {
                    "s_buckets": config.radial_s_buckets,
                    "q_buckets": config.radial_q_buckets,
                    "pilot_maps": config.radial_pilot_maps,
                    "pilot_bins": config.radial_pilot_bins,
                    "pilot_phi": config.radial_pilot_phi,
                    "pilot_m_max": config.radial_pilot_m_max,
                    "adaptive_fraction": config.radial_adaptive_fraction,
                },
            },
        )
        metadata = {
            "grid": grid.to_dict(),
            "config": config.to_dict(),
            "partition": asdict(partition),
            "selected_map_ids": sorted(selected_ids),
        }
        (temporary / "hammlet-build.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def merge_maps(output: str | Path, *, destination: str | Path | None = None) -> Path:
    """Validate and merge all generated parts without recomputing coefficients."""
    output = Path(output).resolve()
    parts = sorted((output / "parts").glob("part-*-of-*"))
    if not parts:
        raise FileNotFoundError(f"no parts found below {output / 'parts'}")
    records = [json.loads((part / "hammlet-build.json").read_text()) for part in parts]
    expected = int(records[0]["partition"]["count"])
    indices = [int(record["partition"]["index"]) for record in records]
    if len(parts) != expected or indices != list(range(expected)):
        raise ValueError(f"expected all partition indices 0..{expected - 1}; got {indices}")
    if any(record["grid"] != records[0]["grid"] for record in records[1:]):
        raise ValueError("parts were generated from different parameter grids")
    if any(record["config"] != records[0]["config"] for record in records[1:]):
        raise ValueError("parts were generated with different map configs")

    destination = Path(destination or output / "maps").resolve()
    if destination.exists():
        raise FileExistsError(f"merge destination already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    temporary.mkdir(parents=True)
    try:
        entries = []
        selected_ids: list[int] = []
        m_max: int | None = None
        for part in parts:
            collection = BucketedPolarAtlas(part)
            if m_max is None:
                m_max = collection.m_max
            elif collection.m_max != m_max:
                raise ValueError("parts use different stored mode budgets")
            for bucket in collection:
                target_name = bucket.name
                target = temporary / target_name
                if target.exists():
                    raise ValueError(f"duplicate radial bucket: {target_name}")
                shutil.copytree(bucket.path, target)
                ids = [int(value) for value in bucket.map_ids]
                selected_ids.extend(ids)
                entries.append(
                    {
                        "name": target_name,
                        "path": target_name,
                        "count": len(ids),
                        "logs": list(bucket.logs),
                        "logq": list(bucket.logq),
                        "map_ids": ids,
                    }
                )
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("parts contain duplicate map IDs")
        entries.sort(key=lambda item: str(item["name"]))
        write_bucket_manifest(
            temporary,
            entries,
            {
                "builder": "merged-direct-vbm-rho-aware-adaptive-radial-fourier",
                "radial_layout": json.loads(
                    (parts[0] / "manifest.json").read_text()
                ).get("radial_layout"),
            },
        )
        (temporary / "hammlet-build.json").write_text(
            json.dumps(
                {
                    **records[0],
                    "partition": {"index": 0, "count": 1},
                    "selected_map_ids": sorted(selected_ids),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination
