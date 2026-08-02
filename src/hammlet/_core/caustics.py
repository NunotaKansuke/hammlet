from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .config import PSPLGeometry


def adamgrid_map_origin_shift(separation: float, mass_ratio: float) -> float:
    """Return native-VBM x added to an AdaMGrid map coordinate.

    The released map builder evaluates ``BinaryMag2(x_map + shift_x, ...)``.
    Caustics returned by VBMicrolensing are in that native coordinate frame,
    so callers must use ``x_map = x_native - shift_x`` before intersecting an
    atlas trajectory.
    """
    separation = float(separation)
    mass_ratio = float(mass_ratio)
    if (
        not np.isfinite(separation)
        or separation <= 0.0
        or not np.isfinite(mass_ratio)
        or mass_ratio <= 0.0
    ):
        raise ValueError("separation and mass_ratio must be finite and positive")
    if separation <= 1.0:
        return 0.0
    return -(separation - 1.0 / separation) * mass_ratio / (1.0 + mass_ratio)


def caustics_to_adamgrid_map_frame(
    components: Iterable[np.ndarray],
    *,
    separation: float,
    mass_ratio: float,
) -> tuple[np.ndarray, ...]:
    """Transform native VBMicrolensing caustics into AdaMGrid map coordinates."""
    shift_x = adamgrid_map_origin_shift(separation, mass_ratio)
    output = []
    for component in components:
        points = np.asarray(component, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("each caustic component must have shape (n, 2)")
        transformed = points.copy()
        transformed[:, 0] -= shift_x
        output.append(transformed)
    return tuple(output)


@dataclass(frozen=True)
class CausticClockSeed:
    alpha: float
    tau_entry: float
    tau_exit: float
    geometry: PSPLGeometry
    component_index: int = -1
    entry_segment_index: int = -1
    exit_segment_index: int = -1


@dataclass(frozen=True)
class CausticIntersection:
    """One trajectory/caustic intersection with its topology identity."""

    component_index: int
    segment_index: int
    segment_fraction: float
    tau: float
    x: float
    y: float


@dataclass(frozen=True)
class CausticCrossingPair:
    """Two consecutive boundary crossings belonging to one caustic component."""

    component_index: int
    alpha: float
    u0: float
    entry: CausticIntersection
    exit: CausticIntersection


def caustic_intersections_by_component(
    components: Iterable[np.ndarray],
    *,
    alpha: float,
    u0: float,
    dedup_atol: float = 1e-10,
) -> tuple[tuple[CausticIntersection, ...], ...]:
    """Intersect a trajectory with each caustic component independently.

    The returned outer tuple follows input component order.  Intersections are
    ordered by increasing trajectory coordinate ``tau`` inside each component.
    A caustic vertex touched by two adjacent polyline segments is retained only
    once, so it cannot create a zero-width crossing pair.
    """
    if not np.isfinite(alpha) or not np.isfinite(u0):
        raise ValueError("alpha and u0 must be finite")
    if not np.isfinite(dedup_atol) or dedup_atol < 0.0:
        raise ValueError("dedup_atol must be finite and non-negative")

    cos_alpha = float(np.cos(alpha))
    sin_alpha = float(np.sin(alpha))
    output: list[tuple[CausticIntersection, ...]] = []
    for component_index, component in enumerate(components):
        points = np.asarray(component, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("each caustic component must have shape (n, 2)")
        if not np.all(np.isfinite(points)):
            raise ValueError("caustic component coordinates must be finite")

        # Rotate lens coordinates back to the alpha=0 trajectory frame:
        # x'=-tau and y'=-u0.
        xp = cos_alpha * points[:, 0] + sin_alpha * points[:, 1]
        yp = -sin_alpha * points[:, 0] + cos_alpha * points[:, 1]
        target = -float(u0)
        local: list[CausticIntersection] = []
        for segment_index in range(len(points) - 1):
            y1 = float(yp[segment_index])
            y2 = float(yp[segment_index + 1])
            # A segment collinear with the trajectory represents a degenerate
            # continuum of contacts, not a uniquely timed crossing.
            if y1 == y2 or (y1 - target) * (y2 - target) > 0.0:
                continue
            fraction = (target - y1) / (y2 - y1)
            if not 0.0 <= fraction <= 1.0:
                continue
            point = (
                points[segment_index]
                + fraction * (points[segment_index + 1] - points[segment_index])
            )
            x_prime = float(
                xp[segment_index]
                + fraction * (xp[segment_index + 1] - xp[segment_index])
            )
            local.append(
                CausticIntersection(
                    component_index=component_index,
                    segment_index=segment_index,
                    segment_fraction=float(fraction),
                    tau=-x_prime,
                    x=float(point[0]),
                    y=float(point[1]),
                )
            )

        local.sort(key=lambda item: (item.tau, item.segment_index))
        unique: list[CausticIntersection] = []
        for intersection in local:
            if unique and abs(intersection.tau - unique[-1].tau) <= dedup_atol:
                continue
            unique.append(intersection)
        output.append(tuple(unique))
    return tuple(output)


def caustic_crossing_pairs(
    components: Iterable[np.ndarray],
    *,
    alpha: float,
    u0: float,
    dedup_atol: float = 1e-10,
) -> tuple[CausticCrossingPair, ...]:
    """Return non-overlapping entry/exit pairs without crossing components."""
    intersections = caustic_intersections_by_component(
        components,
        alpha=alpha,
        u0=u0,
        dedup_atol=dedup_atol,
    )
    pairs: list[CausticCrossingPair] = []
    for component_index, local in enumerate(intersections):
        # A closed caustic alternates outside/inside at its ordered
        # intersections.  Pair (0,1), (2,3), ... rather than constructing
        # sliding or cross-component pairs.
        for index in range(0, len(local) - 1, 2):
            entry = local[index]
            exit = local[index + 1]
            if exit.tau <= entry.tau:
                continue
            pairs.append(
                CausticCrossingPair(
                    component_index=component_index,
                    alpha=float(alpha),
                    u0=float(u0),
                    entry=entry,
                    exit=exit,
                )
            )
    return tuple(pairs)


def crossing_pair_geometry(
    pair: CausticCrossingPair,
    *,
    entry_time: float,
    exit_time: float,
) -> PSPLGeometry:
    """Derive positive ``tE`` and ``t0`` from two observed time anchors."""
    if not np.isfinite(entry_time) or not np.isfinite(exit_time):
        raise ValueError("entry_time and exit_time must be finite")
    if exit_time <= entry_time:
        raise ValueError("exit_time must be later than entry_time")
    delta_tau = pair.exit.tau - pair.entry.tau
    if not np.isfinite(delta_tau) or delta_tau <= 0.0:
        raise ValueError("crossing pair must have increasing finite tau")
    tE = (float(exit_time) - float(entry_time)) / delta_tau
    t0 = float(entry_time) - tE * pair.entry.tau
    return PSPLGeometry(
        t0=float(t0),
        u0=float(pair.u0),
        tE=float(tE),
        reliable=False,
        origin="caustic_clock",
    )


def caustic_clock_seeds(
    components: Iterable[np.ndarray],
    *,
    alpha: float,
    u0: float,
    entry_time: float,
    exit_time: float,
) -> list[CausticClockSeed]:
    """Intersect AdaMGrid-map-frame caustic polylines and derive seeds."""
    output: list[CausticClockSeed] = []
    for pair in caustic_crossing_pairs(
        components,
        alpha=alpha,
        u0=u0,
    ):
        geometry = crossing_pair_geometry(
            pair,
            entry_time=entry_time,
            exit_time=exit_time,
        )
        output.append(
            CausticClockSeed(
                alpha=float(alpha),
                tau_entry=float(pair.entry.tau),
                tau_exit=float(pair.exit.tau),
                geometry=geometry,
                component_index=pair.component_index,
                entry_segment_index=pair.entry.segment_index,
                exit_segment_index=pair.exit.segment_index,
            )
        )
    return output


class PolarCausticDistanceAtlas:
    """Caustic distance sampled on the same polar grid as the spectral atlas."""

    def __init__(
        self,
        radial_nodes: np.ndarray,
        n_alpha: int,
        distances: Mapping[tuple[float, float], np.ndarray],
    ) -> None:
        self.radial_nodes = np.asarray(radial_nodes, dtype=np.float64)
        self.n_alpha = int(n_alpha)
        self.distances = {
            (round(float(key[0]), 12), round(float(key[1]), 12)): np.asarray(value)
            for key, value in distances.items()
        }

    @classmethod
    def from_polylines(
        cls,
        radial_nodes: np.ndarray,
        n_alpha: int,
        polylines: Mapping[tuple[float, float], Iterable[np.ndarray]],
    ) -> "PolarCausticDistanceAtlas":
        radial_nodes = np.asarray(radial_nodes, dtype=np.float64)
        phi = np.arange(n_alpha) * 2.0 * np.pi / n_alpha
        query = np.stack(
            (
                (radial_nodes[:, None] * np.cos(phi)).ravel(),
                (radial_nodes[:, None] * np.sin(phi)).ravel(),
            ),
            axis=1,
        )
        distances = {}
        for key, components in polylines.items():
            points = np.concatenate([np.asarray(item, dtype=np.float64) for item in components])
            # Builder-time operation: chunk to avoid an unbounded query-by-point matrix.
            minimum = np.full(len(query), np.inf)
            for start in range(0, len(points), 4096):
                delta = query[:, None, :] - points[None, start : start + 4096, :]
                minimum = np.minimum(minimum, np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1)))
            distances[key] = minimum.reshape(len(radial_nodes), n_alpha).astype(np.float32)
        return cls(radial_nodes, n_alpha, distances)

    def at_radius(self, logs: float, logq: float, radius: float) -> np.ndarray:
        key = (round(float(logs), 12), round(float(logq), 12))
        values = self.distances[key]
        if radius < self.radial_nodes[0] or radius > self.radial_nodes[-1]:
            return np.full(self.n_alpha, np.inf)
        upper = np.clip(
            np.searchsorted(self.radial_nodes, radius, side="right"),
            1,
            len(self.radial_nodes) - 1,
        )
        lower = upper - 1
        fraction = (radius - self.radial_nodes[lower]) / (
            self.radial_nodes[upper] - self.radial_nodes[lower]
        )
        return (1.0 - fraction) * values[lower] + fraction * values[upper]
