from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .map_adapter import MagnificationEvaluator, point_lens_magnification


@dataclass(frozen=True)
class MapBuildSpec:
    map_id: int
    logs: float
    logq: float
    logrho: float
    evaluator: MagnificationEvaluator


class PolarAtlasBuilder:
    """Build low-pass angular spectra from adaptive magnification maps."""

    def __init__(
        self,
        radial_nodes: np.ndarray,
        *,
        n_phi_build: int = 512,
        m_max: int = 48,
        core_m_max: int | None = None,
        shard_size: int = 256,
        coefficient_dtype: str = "complex64",
    ) -> None:
        nodes = np.asarray(radial_nodes, dtype=np.float64)
        if nodes.ndim != 1 or nodes.size < 2 or np.any(np.diff(nodes) <= 0):
            raise ValueError("radial_nodes must be a strictly increasing 1-D array")
        if nodes[0] < 0:
            raise ValueError("radial_nodes cannot contain negative radii")
        if n_phi_build < 2 * (m_max + 1):
            raise ValueError("n_phi_build is too small for m_max")
        if core_m_max is not None and not 0 <= core_m_max <= m_max:
            raise ValueError("core_m_max must be between zero and m_max")
        self.radial_nodes = nodes
        self.n_phi_build = int(n_phi_build)
        self.m_max = int(m_max)
        self.core_m_max = self.m_max if core_m_max is None else int(core_m_max)
        self.shard_size = int(shard_size)
        self.coefficient_dtype = np.dtype(coefficient_dtype)

    def sample(self, evaluator: MagnificationEvaluator) -> dict[str, np.ndarray]:
        phi = np.arange(self.n_phi_build, dtype=np.float64) * (
            2.0 * np.pi / self.n_phi_build
        )
        x = self.radial_nodes[:, None] * np.cos(phi)[None, :]
        y = self.radial_nodes[:, None] * np.sin(phi)[None, :]
        magnification = np.asarray(evaluator.magnification(x, y), dtype=np.float64)
        if magnification.shape != x.shape or not np.all(np.isfinite(magnification)):
            raise ValueError("map evaluator returned invalid magnifications")
        excess = magnification - 1.0
        modes = self.m_max + 1
        x_coeff = np.fft.rfft(excess, axis=-1)[..., :modes] / self.n_phi_build
        x2_coeff = (
            np.fft.rfft(excess * excess, axis=-1)[..., :modes] / self.n_phi_build
        )
        reconstructed = np.fft.irfft(
            self._to_dft(x_coeff, self.n_phi_build),
            n=self.n_phi_build,
            axis=-1,
        )
        error = np.max(np.abs(excess - reconstructed), axis=-1)
        if self.core_m_max < self.m_max:
            core_reconstructed = np.fft.irfft(
                self._to_dft(
                    x_coeff[..., : self.core_m_max + 1], self.n_phi_build
                ),
                n=self.n_phi_build,
                axis=-1,
            )
            core_error = np.max(np.abs(excess - core_reconstructed), axis=-1)
        else:
            core_error = error
        pspl = point_lens_magnification(self.radial_nodes**2)
        envelope = np.max(np.abs(magnification - pspl[:, None]), axis=-1)
        return {
            "x_coeff": x_coeff.astype(self.coefficient_dtype),
            "x2_coeff": x2_coeff.astype(self.coefficient_dtype),
            "reconstruction_error": error.astype(np.float32),
            "reconstruction_error_core": core_error.astype(np.float32),
            "deviation_envelope": envelope.astype(np.float32),
        }

    @staticmethod
    def _to_dft(coefficients: np.ndarray, n: int) -> np.ndarray:
        spectrum = np.zeros(coefficients.shape[:-1] + (n // 2 + 1,), dtype=np.complex128)
        spectrum[..., : coefficients.shape[-1]] = coefficients * n
        return spectrum

    def _existing_shards(
        self, output: Path
    ) -> tuple[list[dict[str, object]], set[int], bool, int]:
        """Return valid checkpoint shards already present in ``output``.

        A shard is considered durable only when all of its arrays are present
        and have the same leading dimension as ``map_ids.npy``.  This lets a
        killed process leave an incomplete shard behind without making the
        next resume trust corrupted data.
        """
        entries: list[dict[str, object]] = []
        map_ids: set[int] = set()
        has_certificates = False
        next_shard = 0
        required = {
            "x_coeff.npy",
            "x2_coeff.npy",
            "reconstruction_error.npy",
            "deviation_envelope.npy",
            "map_ids.npy",
        }
        if self.core_m_max < self.m_max:
            required.update(
                {
                    "x_coeff_extension.npy",
                    "x2_coeff_extension.npy",
                    "reconstruction_error_full.npy",
                }
            )

        for shard in sorted(output.glob("shard_*")):
            if not shard.is_dir():
                continue
            match = re.fullmatch(r"shard_(\d+)", shard.name)
            if match:
                next_shard = max(next_shard, int(match.group(1)) + 1)
            if not required.issubset({path.name for path in shard.iterdir()}):
                continue
            try:
                ids = np.asarray(np.load(shard / "map_ids.npy"), dtype=np.int64)
                if ids.ndim != 1 or not len(ids):
                    continue
                for name in required - {"map_ids.npy"}:
                    values = np.load(shard / name, mmap_mode="r")
                    if values.shape[0] != len(ids):
                        raise ValueError("checkpoint array length mismatch")
                if any(int(value) in map_ids for value in ids):
                    raise ValueError("duplicate map ID in checkpoint shards")
            except (OSError, ValueError, TypeError):
                continue
            map_ids.update(int(value) for value in ids)
            has_certificates = has_certificates or (shard / "certified_error.npy").is_file()
            entries.append(
                {
                    "path": shard.name,
                    "count": len(ids),
                    "map_ids": [int(value) for value in ids],
                }
            )
        return entries, map_ids, has_certificates, next_shard

    def build(
        self,
        output_path: str | Path,
        specs: Iterable[MapBuildSpec],
        *,
        progress_every: int = 0,
        resume: bool = False,
    ) -> Path:
        output = Path(output_path)
        if output.exists() and any(output.iterdir()) and not resume:
            raise FileExistsError(f"atlas output is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / "radial_nodes.npy", self.radial_nodes)

        all_parameters: list[tuple[float, float, float]] = []
        all_map_ids: list[int] = []
        if resume:
            shard_entries, existing_ids, has_certificates, shard_index = (
                self._existing_shards(output)
            )
        else:
            shard_entries = []
            existing_ids = set()
            has_certificates = False
            shard_index = 0
        pending: list[tuple[MapBuildSpec, dict[str, np.ndarray]]] = []
        started = time.monotonic()

        def flush(shard_index: int) -> None:
            nonlocal has_certificates
            if not pending:
                return
            shard_dir = output / f"shard_{shard_index:04d}"
            temporary = output / f".shard_{shard_index:04d}.partial-{time.time_ns()}"
            temporary.mkdir()
            core_modes = self.core_m_max + 1
            for name in ("x_coeff", "x2_coeff"):
                values = np.stack([item[1][name] for item in pending])
                np.save(temporary / f"{name}.npy", values[..., :core_modes])
                if self.core_m_max < self.m_max:
                    np.save(
                        temporary / f"{name}_extension.npy",
                        values[..., core_modes:],
                    )
            np.save(
                temporary / "reconstruction_error.npy",
                np.stack(
                    [item[1]["reconstruction_error_core"] for item in pending]
                ),
            )
            if self.core_m_max < self.m_max:
                np.save(
                    temporary / "reconstruction_error_full.npy",
                    np.stack(
                        [item[1]["reconstruction_error"] for item in pending]
                    ),
                )
            if "certified_error" in pending[0][1]:
                has_certificates = True
                np.save(
                    temporary / "certified_error.npy",
                    np.stack(
                        [item[1]["certified_error_core"] for item in pending]
                    ),
                )
                if self.core_m_max < self.m_max:
                    np.save(
                        temporary / "certified_error_full.npy",
                        np.stack(
                            [item[1]["certified_error"] for item in pending]
                        ),
                    )
            np.save(
                temporary / "deviation_envelope.npy",
                np.stack([item[1]["deviation_envelope"] for item in pending]),
            )
            map_ids = np.asarray([item[0].map_id for item in pending], dtype=np.int64)
            np.save(temporary / "map_ids.npy", map_ids)
            temporary.replace(shard_dir)
            shard_entries.append(
                {"path": shard_dir.name, "count": len(pending), "map_ids": map_ids.tolist()}
            )
            pending.clear()

        seen_ids: set[int] = set()
        for spec in specs:
            map_id = int(spec.map_id)
            if map_id in seen_ids:
                raise ValueError(f"duplicate map ID in build specs: {map_id}")
            seen_ids.add(map_id)
            all_map_ids.append(spec.map_id)
            all_parameters.append((spec.logs, spec.logq, spec.logrho))
            if map_id in existing_ids:
                continue
            pending.append((spec, self.sample(spec.evaluator)))
            if progress_every and len(all_map_ids) % progress_every == 0:
                elapsed = time.monotonic() - started
                print(
                    f"built {len(all_map_ids)} maps in {elapsed:.1f}s "
                    f"({len(all_map_ids) / elapsed:.2f} maps/s)",
                    flush=True,
                )
            if len(pending) == self.shard_size:
                flush(shard_index)
                shard_index += 1
        flush(shard_index)
        missing_ids = existing_ids.difference(seen_ids)
        if missing_ids:
            raise ValueError(
                "checkpoint contains map IDs absent from build specs: "
                f"{sorted(missing_ids)[:5]}"
            )
        if not all_map_ids:
            raise ValueError("cannot build an empty atlas")

        np.save(output / "map_ids.npy", np.asarray(all_map_ids, dtype=np.int64))
        np.save(output / "map_parameters.npy", np.asarray(all_parameters, dtype=np.float64))
        manifest = {
            "format": "hammlet-fourier-maps",
            "version": 1,
            "n_maps": len(all_map_ids),
            "n_r": self.radial_nodes.size,
            "n_phi_build": self.n_phi_build,
            "m_max": self.m_max,
            "core_m_max": self.core_m_max,
            "coefficient_dtype": self.coefficient_dtype.name,
            "normalization": "fourier-series",
            "angle_convention": "hammlet-map-frame-radians",
            "error_certificate": (
                "periodic-piecewise-linear-vbm-reference"
                if has_certificates
                else None
            ),
            "shards": shard_entries,
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return output


def specs_from_adamgrid(
    map_directory: str | Path,
    parameters: np.ndarray,
    map_factory: Callable[[Path, float, float], MagnificationEvaluator],
    map_ids: Sequence[int] | None = None,
) -> Iterable[MapBuildSpec]:
    directory = Path(map_directory)
    parameters = np.asarray(parameters, dtype=np.float64)
    ids = range(len(parameters)) if map_ids is None else map_ids
    for map_id in ids:
        path = directory / f"{map_id}.npz"
        if not path.is_file():
            continue
        logs, logq, logrho = parameters[map_id]
        yield MapBuildSpec(
            map_id=int(map_id),
            logs=float(logs),
            logq=float(logq),
            logrho=float(logrho),
            evaluator=map_factory(path, 10.0**logs, 10.0**logq),
        )
