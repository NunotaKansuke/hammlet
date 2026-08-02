from __future__ import annotations

import json
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

    def build(
        self,
        output_path: str | Path,
        specs: Iterable[MapBuildSpec],
        *,
        progress_every: int = 0,
    ) -> Path:
        output = Path(output_path)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"atlas output is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / "radial_nodes.npy", self.radial_nodes)

        all_parameters: list[tuple[float, float, float]] = []
        all_map_ids: list[int] = []
        shard_entries: list[dict[str, object]] = []
        pending: list[tuple[MapBuildSpec, dict[str, np.ndarray]]] = []
        has_certificates = False
        started = time.monotonic()

        def flush(shard_index: int) -> None:
            nonlocal has_certificates
            if not pending:
                return
            shard_dir = output / f"shard_{shard_index:04d}"
            shard_dir.mkdir()
            core_modes = self.core_m_max + 1
            for name in ("x_coeff", "x2_coeff"):
                values = np.stack([item[1][name] for item in pending])
                np.save(shard_dir / f"{name}.npy", values[..., :core_modes])
                if self.core_m_max < self.m_max:
                    np.save(
                        shard_dir / f"{name}_extension.npy",
                        values[..., core_modes:],
                    )
            np.save(
                shard_dir / "reconstruction_error.npy",
                np.stack(
                    [item[1]["reconstruction_error_core"] for item in pending]
                ),
            )
            if self.core_m_max < self.m_max:
                np.save(
                    shard_dir / "reconstruction_error_full.npy",
                    np.stack(
                        [item[1]["reconstruction_error"] for item in pending]
                    ),
                )
            if "certified_error" in pending[0][1]:
                has_certificates = True
                np.save(
                    shard_dir / "certified_error.npy",
                    np.stack(
                        [item[1]["certified_error_core"] for item in pending]
                    ),
                )
                if self.core_m_max < self.m_max:
                    np.save(
                        shard_dir / "certified_error_full.npy",
                        np.stack(
                            [item[1]["certified_error"] for item in pending]
                        ),
                    )
            np.save(
                shard_dir / "deviation_envelope.npy",
                np.stack([item[1]["deviation_envelope"] for item in pending]),
            )
            map_ids = np.asarray([item[0].map_id for item in pending], dtype=np.int64)
            np.save(shard_dir / "map_ids.npy", map_ids)
            shard_entries.append(
                {"path": shard_dir.name, "count": len(pending), "map_ids": map_ids.tolist()}
            )
            pending.clear()

        shard_index = 0
        for spec in specs:
            pending.append((spec, self.sample(spec.evaluator)))
            all_map_ids.append(spec.map_id)
            all_parameters.append((spec.logs, spec.logq, spec.logrho))
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
        if not all_map_ids:
            raise ValueError("cannot build an empty atlas")

        np.save(output / "map_ids.npy", np.asarray(all_map_ids, dtype=np.int64))
        np.save(output / "map_parameters.npy", np.asarray(all_parameters, dtype=np.float64))
        manifest = {
            "format": "hammlet-fourier-atlas",
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
