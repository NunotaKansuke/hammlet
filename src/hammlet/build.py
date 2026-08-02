"""Atlas construction, partitioning and merge operations."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np

from .config import AtlasConfig, ParameterGrid, Partition, default_radial_nodes
from ._core.atlas_builder import MapBuildSpec
from ._core.direct_vbm import (
    DirectSpectrumConfig,
    DirectVBMPolarAtlasBuilder,
    VBMBinaryLensEvaluator,
)


def _spectrum_config(config: AtlasConfig) -> DirectSpectrumConfig:
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
    )


def _specs(rows: np.ndarray, config: AtlasConfig) -> Iterable[MapBuildSpec]:
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


def build_atlas(
    output: str | Path,
    grid: ParameterGrid,
    *,
    config: AtlasConfig | None = None,
    partition: Partition | None = None,
    progress_every: int = 1,
) -> Path:
    """Generate one complete atlas or one independently schedulable part.

    For ``Partition(index=i, count=n)``, every machine receives the same grid
    and config and writes ``output/parts/part-iiiii-of-nnnnn``.  A private
    temporary directory is atomically renamed only after the part succeeds.
    """
    config = config or AtlasConfig()
    partition = partition or Partition()
    output = Path(output).resolve()
    rows = grid.table()[partition.rows(len(grid.table()))]
    if not len(rows):
        raise ValueError("this partition contains no parameter rows")
    destination = (
        output
        if partition.count == 1
        else output / "parts" / f"part-{partition.index:05d}-of-{partition.count:05d}"
    )
    if destination.exists():
        raise FileExistsError(f"atlas destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"stale partial output exists: {temporary}")

    builder = DirectVBMPolarAtlasBuilder(
        default_radial_nodes(config.radial_nodes, config.radial_max),
        spectrum_config=_spectrum_config(config),
        shard_size=config.shard_size,
    )
    try:
        builder.build(temporary, _specs(rows, config), progress_every=progress_every)
        metadata = {
            "grid": grid.to_dict(),
            "config": config.to_dict(),
            "partition": asdict(partition),
            "selected_map_ids": rows[:, 0].astype(int).tolist(),
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


def merge_parts(output: str | Path, *, destination: str | Path | None = None) -> Path:
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
        raise ValueError("parts were generated with different atlas configs")

    destination = Path(destination or output / "atlas").resolve()
    if destination.exists():
        raise FileExistsError(f"merge destination already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    temporary.mkdir(parents=True)
    try:
        first_manifest = json.loads((parts[0] / "manifest.json").read_text())
        np.save(temporary / "radial_nodes.npy", np.load(parts[0] / "radial_nodes.npy"))
        all_ids, all_parameters, shards, diagnostic_parts = [], [], [], []
        shard_index = 0
        for part in parts:
            manifest = json.loads((part / "manifest.json").read_text())
            if any(
                manifest[key] != first_manifest[key]
                for key in ("m_max", "core_m_max", "coefficient_dtype", "n_r")
            ):
                raise ValueError("part atlas formats are incompatible")
            all_ids.append(np.load(part / "map_ids.npy"))
            all_parameters.append(np.load(part / "map_parameters.npy"))
            diagnostic_path = part / "direct_diagnostics.npz"
            if diagnostic_path.exists():
                with np.load(diagnostic_path) as diagnostic:
                    diagnostic_parts.append(
                        {name: np.asarray(diagnostic[name]) for name in diagnostic.files}
                    )
            for entry in manifest["shards"]:
                target_name = f"shard_{shard_index:05d}"
                shutil.copytree(part / entry["path"], temporary / target_name)
                shards.append({**entry, "path": target_name})
                shard_index += 1
        ids = np.concatenate(all_ids).astype(np.int64)
        parameters = np.concatenate(all_parameters).astype(np.float64)
        order = np.argsort(ids, kind="stable")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("parts contain duplicate map IDs")
        np.save(temporary / "map_ids.npy", ids[order])
        np.save(temporary / "map_parameters.npy", parameters[order])
        manifest = {**first_manifest, "n_maps": len(ids), "shards": shards}
        if diagnostic_parts:
            names = diagnostic_parts[0]
            if any(set(item) != set(names) for item in diagnostic_parts):
                raise ValueError("part diagnostic formats are incompatible")
            diagnostics = {
                name: np.concatenate([item[name] for item in diagnostic_parts])
                for name in names
            }
            np.savez_compressed(
                temporary / "direct_diagnostics.npz", **diagnostics
            )
            manifest["direct_diagnostics"] = {
                "total_vbm_evaluations": int(np.sum(diagnostics["evaluations"])),
                "unconverged_rings": int(np.sum(~diagnostics["converged"])),
                "caustic_guarded_rings": int(
                    np.sum(diagnostics["caustic_guarded"])
                ),
                "max_normalized_coefficient_error": float(
                    np.max(diagnostics["normalized_coefficient_error"])
                ),
                "per_ring_diagnostics": "direct_diagnostics.npz",
            }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        (temporary / "hammlet-build.json").write_text(
            json.dumps(
                {
                    **records[0],
                    "partition": {"index": 0, "count": 1},
                    "selected_map_ids": ids[order].tolist(),
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
