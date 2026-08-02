from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class AtlasShard:
    map_ids: np.ndarray
    parameters: np.ndarray
    x_coeff: np.ndarray
    x2_coeff: np.ndarray
    reconstruction_error: np.ndarray
    deviation_envelope: np.ndarray
    certified_error: np.ndarray | None = None


class PolarAtlas:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.manifest = json.loads(
            (self.path / "manifest.json").read_text(encoding="utf-8")
        )
        if (
            self.manifest.get("format")
            not in {"hammlet-fourier-maps", "hammlet-fourier-atlas"}
            or self.manifest.get("version") != 1
        ):
            raise ValueError("unsupported polar atlas format")
        self.radial_nodes = np.load(self.path / "radial_nodes.npy", mmap_mode="r")
        self.map_ids = np.load(self.path / "map_ids.npy", mmap_mode="r")
        self.map_parameters = np.load(self.path / "map_parameters.npy", mmap_mode="r")
        self._parameter_index = {
            int(map_id): index for index, map_id in enumerate(self.map_ids)
        }

    @property
    def m_max(self) -> int:
        return int(self.manifest["m_max"])

    @property
    def core_m_max(self) -> int:
        return int(self.manifest.get("core_m_max", self.m_max))

    def iter_shards(self, m_max: int | None = None) -> Iterator[AtlasShard]:
        requested_m_max = self.m_max if m_max is None else int(m_max)
        if not 0 <= requested_m_max <= self.m_max:
            raise ValueError("requested m_max is outside the stored range")
        for entry in self.manifest["shards"]:
            shard_path = self.path / entry["path"]
            ids = np.load(shard_path / "map_ids.npy", mmap_mode="r")
            parameter_rows = np.asarray(
                [self._parameter_index[int(map_id)] for map_id in ids], dtype=np.int64
            )
            core_modes = min(requested_m_max, self.core_m_max) + 1
            x_coeff = np.load(shard_path / "x_coeff.npy", mmap_mode="r")[
                ..., :core_modes
            ]
            x2_coeff = np.load(shard_path / "x2_coeff.npy", mmap_mode="r")[
                ..., :core_modes
            ]
            if requested_m_max > self.core_m_max:
                extension_modes = requested_m_max - self.core_m_max
                x_coeff = np.concatenate(
                    (
                        x_coeff,
                        np.load(shard_path / "x_coeff_extension.npy", mmap_mode="r")[
                            ..., :extension_modes
                        ],
                    ),
                    axis=-1,
                )
                x2_coeff = np.concatenate(
                    (
                        x2_coeff,
                        np.load(shard_path / "x2_coeff_extension.npy", mmap_mode="r")[
                            ..., :extension_modes
                        ],
                    ),
                    axis=-1,
                )
            error_name = (
                "reconstruction_error_full.npy"
                if requested_m_max > self.core_m_max
                else "reconstruction_error.npy"
            )
            certificate_name = (
                "certified_error_full.npy"
                if requested_m_max > self.core_m_max
                else "certified_error.npy"
            )
            certificate_path = shard_path / certificate_name
            yield AtlasShard(
                map_ids=ids,
                parameters=self.map_parameters[parameter_rows],
                x_coeff=x_coeff,
                x2_coeff=x2_coeff,
                reconstruction_error=np.load(shard_path / error_name, mmap_mode="r"),
                deviation_envelope=np.load(
                    shard_path / "deviation_envelope.npy", mmap_mode="r"
                ),
                certified_error=(
                    np.load(certificate_path, mmap_mode="r")
                    if certificate_path.exists()
                    else None
                ),
            )

    def parameters_for(self, map_id: int) -> np.ndarray:
        try:
            return np.asarray(self.map_parameters[self._parameter_index[int(map_id)]])
        except KeyError as error:
            raise KeyError(f"map id is not present in the atlas: {map_id}") from error

    def coefficient_rows(
        self, map_ids: list[int] | np.ndarray, *, m_max: int
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Bulk-load ``(parameters, x_coeff)`` while touching each shard once.

        Unlike :meth:`iter_shards`, this focused path never loads ``x2_coeff``.
        It is intended for high-mode candidate racing, where repeatedly loading
        a complete extension shard for individual map IDs is prohibitively
        expensive.
        """
        requested = [int(map_id) for map_id in map_ids]
        if not requested:
            return {}
        requested_set = set(requested)
        missing = requested_set.difference(self._parameter_index)
        if missing:
            raise KeyError(f"map ids are not present in the atlas: {sorted(missing)}")
        requested_m_max = int(m_max)
        if not 0 <= requested_m_max <= self.m_max:
            raise ValueError("requested m_max is outside the stored range")
        output: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for entry in self.manifest["shards"]:
            shard_path = self.path / entry["path"]
            ids = np.load(shard_path / "map_ids.npy", mmap_mode="r")
            rows = np.asarray(
                [
                    index
                    for index, map_id in enumerate(ids)
                    if int(map_id) in requested_set
                ],
                dtype=np.int64,
            )
            if not rows.size:
                continue
            core_modes = min(requested_m_max, self.core_m_max) + 1
            coefficients = np.asarray(
                np.load(shard_path / "x_coeff.npy", mmap_mode="r")[rows, :, :core_modes]
            )
            if requested_m_max > self.core_m_max:
                extension_modes = requested_m_max - self.core_m_max
                extension = np.asarray(
                    np.load(shard_path / "x_coeff_extension.npy", mmap_mode="r")[
                        rows, :, :extension_modes
                    ]
                )
                coefficients = np.concatenate((coefficients, extension), axis=-1)
            for local_row, shard_row in enumerate(rows):
                map_id = int(ids[int(shard_row)])
                parameter_row = self._parameter_index[map_id]
                output[map_id] = (
                    np.asarray(self.map_parameters[parameter_row], dtype=np.float64),
                    coefficients[local_row],
                )
        if output.keys() != requested_set:
            raise RuntimeError("requested map disappeared while bulk-loading atlas")
        return output

    def neighboring_maps(
        self, map_id: int, limit: int = 26
    ) -> list[tuple[int, np.ndarray]]:
        """Return available immediate neighbors in the three map-grid dimensions."""
        center = self.parameters_for(map_id)
        allowed: list[np.ndarray] = []
        for axis in range(3):
            values = np.unique(np.asarray(self.map_parameters[:, axis]))
            position = int(np.searchsorted(values, center[axis]))
            choices = [values[position]]
            if position:
                choices.append(values[position - 1])
            if position + 1 < len(values):
                choices.append(values[position + 1])
            allowed.append(np.asarray(choices))
        parameters = np.asarray(self.map_parameters)
        mask = np.ones(len(parameters), dtype=bool)
        for axis in range(3):
            mask &= np.isin(parameters[:, axis], allowed[axis])
        mask[self._parameter_index[int(map_id)]] = False
        rows = np.flatnonzero(mask)
        if len(rows) > limit:
            scales = []
            for axis in range(3):
                differences = np.diff(np.unique(parameters[:, axis]))
                scales.append(
                    max(float(np.min(differences)), 1e-12) if differences.size else 1.0
                )
            scale = np.asarray(scales)
            distance = np.sum(((parameters[rows] - center) / scale) ** 2, axis=1)
            rows = rows[np.argsort(distance)[:limit]]
        return [(int(self.map_ids[row]), np.asarray(parameters[row])) for row in rows]
