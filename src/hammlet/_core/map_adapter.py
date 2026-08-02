from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class MagnificationEvaluator(Protocol):
    def magnification(self, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...


def point_lens_magnification(radius_squared: np.ndarray) -> np.ndarray:
    radius_squared = np.asarray(radius_squared, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (radius_squared + 2.0) / np.sqrt(radius_squared * (radius_squared + 4.0))


@dataclass
class AdaMGridMap:
    """Pure NumPy reader for an AdaMGrid adaptive-map ``.npz`` file."""

    corner_magnification: np.ndarray
    refined: np.ndarray
    next_offset: np.ndarray
    layer_offsets: np.ndarray
    box_size: float
    s: float
    q: float

    @classmethod
    def from_npz(cls, path: str | Path, s: float, q: float) -> "AdaMGridMap":
        with np.load(path, allow_pickle=True) as content:
            layer_length = np.asarray(content["layer_length"], dtype=np.int64)
            offsets = np.concatenate(([0], np.cumsum(layer_length[:-1])))
            raw_next = content["all_layer_sequence_number_in_next_layer_file"]
            next_offset = np.fromiter(
                (-1 if value is None else int(float(value) + 0.5) for value in raw_next),
                dtype=np.int64,
                count=raw_next.size,
            )
            return cls(
                corner_magnification=np.asarray(
                    content["all_layer_corner_mag"], dtype=np.float64
                ).reshape(-1, 4),
                refined=np.asarray(content["all_layer_whether_densed"], dtype=bool),
                next_offset=next_offset,
                layer_offsets=offsets,
                box_size=float(np.asarray(content["box_size"]).flat[0]),
                s=float(s),
                q=float(q),
            )

    def magnification(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x, y = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        )
        shape = x.shape
        x_flat = x.ravel()
        y_flat = y.ravel()
        result = np.empty(x_flat.size, dtype=np.float64)
        inside = (
            (x_flat > -self.box_size)
            & (x_flat < self.box_size)
            & (y_flat > -self.box_size)
            & (y_flat < self.box_size)
        )
        outside_r2 = x_flat[~inside] ** 2 + y_flat[~inside] ** 2
        if self.s > 4.25:
            outside_r2 = outside_r2 * (1.0 + self.q)
        result[~inside] = point_lens_magnification(outside_r2)
        if np.any(inside):
            result[inside] = self._interpolate_inside(x_flat[inside], y_flat[inside])
        return result.reshape(shape)

    def _interpolate_inside(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        local_x = 0.5 * (x + self.box_size) / self.box_size
        local_y = 0.5 * (y + self.box_size) / self.box_size
        cell = np.zeros(x.size, dtype=np.int64)
        active = np.ones(x.size, dtype=bool)
        final_cell = np.empty(x.size, dtype=np.int64)
        final_x = np.empty(x.size, dtype=np.float64)
        final_y = np.empty(x.size, dtype=np.float64)

        for layer, layer_offset in enumerate(self.layer_offsets):
            indices = np.flatnonzero(active)
            if not indices.size:
                break
            absolute = layer_offset + cell[indices]
            descend = self.refined[absolute]
            done_indices = indices[~descend]
            done_absolute = absolute[~descend]
            final_cell[done_indices] = done_absolute
            final_x[done_indices] = local_x[done_indices]
            final_y[done_indices] = local_y[done_indices]
            active[done_indices] = False

            down_indices = indices[descend]
            if not down_indices.size:
                continue
            down_absolute = absolute[descend]
            child_x = np.minimum((2.0 * local_x[down_indices]).astype(np.int64), 1)
            child_y = np.minimum((2.0 * local_y[down_indices]).astype(np.int64), 1)
            local_x[down_indices] = 2.0 * local_x[down_indices] - child_x
            local_y[down_indices] = 2.0 * local_y[down_indices] - child_y
            cell[down_indices] = (
                4 * self.next_offset[down_absolute] + 2 * child_y + child_x
            )
        if np.any(active):
            raise ValueError("adaptive map ends before a terminal cell is reached")

        corners = self.corner_magnification[final_cell]
        return (
            (1 - final_x) * (1 - final_y) * corners[:, 0]
            + final_x * (1 - final_y) * corners[:, 1]
            + (1 - final_x) * final_y * corners[:, 2]
            + final_x * final_y * corners[:, 3]
        )


def trajectory_xy(
    time: np.ndarray, t0: float, u0: float, tE: float, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates matching ``chi2_calculator_pipeline_struct.c``."""
    tau = (np.asarray(time, dtype=np.float64) - t0) / tE
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    return (
        u0 * sin_alpha - tau * cos_alpha,
        -u0 * cos_alpha - tau * sin_alpha,
    )
