"""Independent caustic-crossing-clock candidate generation and scoring.

Unlike the angular FFT scanner, a crossing clock derives a different
``(t0, tE)`` for every map, alpha, u0, and caustic crossing pair.  This module
therefore exposes a separate fixed-alpha lane and does not pretend those
map-specific geometries can share one FFT event kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .caustics import (
    CausticCrossingPair,
    caustic_crossing_pairs,
    crossing_pair_geometry,
)
from .config import PSPLGeometry
from .reference import profile_flux
from .radial import DEFAULT_RADIAL_ORDER, interpolate_coefficients
from .trajectory import PhotometricDataset, radial_coordinates


@dataclass(frozen=True)
class CrossingClockCandidate:
    """Auditable score for one map-specific caustic-clock hypothesis."""

    map_id: int
    logs: float
    logq: float
    logrho: float
    component_index: int
    entry_segment_index: int
    exit_segment_index: int
    entry_segment_fraction: float
    exit_segment_fraction: float
    entry_x: float
    entry_y: float
    exit_x: float
    exit_y: float
    alpha: float
    u0: float
    tau_entry: float
    tau_exit: float
    entry_time: float
    exit_time: float
    t0: float
    tE: float
    chi2: float
    anchor_semantics: str
    anchor_provenance: str
    caustic_coordinate_frame: str = "adamgrid_map"

    @property
    def geometry(self) -> PSPLGeometry:
        return PSPLGeometry(
            t0=self.t0,
            u0=self.u0,
            tE=self.tE,
            reliable=False,
            origin="caustic_clock",
        )


def evaluate_fixed_alpha_lowpass(
    x_coeff: np.ndarray,
    radial_nodes: np.ndarray,
    time: np.ndarray,
    geometry: PSPLGeometry,
    alpha: float,
    *,
    radial_order: int = DEFAULT_RADIAL_ORDER,
) -> np.ndarray:
    """Evaluate one radially interpolated low-pass ``x = A - 1`` trajectory.

    Radial interpolation is applied to the complex Fourier coefficients first.
    The angular series is then evaluated once at each observation.  Consumers
    must square these returned values, rather than interpolate or truncate an
    independently stored ``x**2`` spectrum.

    ``radial_order`` must match the order used to build the event kernels, so
    that this reference evaluation stays the exact scalar counterpart of the
    batched scan.
    """
    coefficients = np.asarray(x_coeff)
    nodes = np.asarray(radial_nodes, dtype=np.float64)
    if coefficients.ndim != 2:
        raise ValueError("x_coeff must have shape (radial_node, angular_mode)")
    if nodes.ndim != 1 or len(nodes) < 2 or not np.all(np.diff(nodes) > 0.0):
        raise ValueError("radial_nodes must be a strictly increasing 1-D array")
    if coefficients.shape[0] != len(nodes) or coefficients.shape[1] < 1:
        raise ValueError("x_coeff and radial_nodes shapes do not match")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("x_coeff must be finite")
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")

    radius, phase = radial_coordinates(time, geometry)
    interpolated = interpolate_coefficients(
        coefficients.astype(np.complex128, copy=False),
        radius,
        nodes,
        radial_order,
    )
    angle = phase + float(alpha)
    values = np.real(interpolated[:, 0]).astype(np.float64, copy=True)
    if coefficients.shape[1] > 1:
        modes = np.arange(1, coefficients.shape[1], dtype=np.float64)
        values += 2.0 * np.sum(
            np.real(
                interpolated[:, 1:]
                * np.exp(1j * angle[:, None] * modes[None, :])
            ),
            axis=1,
        )
    return values


def fixed_alpha_coefficient_chi2(
    x_coeff: np.ndarray,
    radial_nodes: np.ndarray,
    datasets: Sequence[PhotometricDataset],
    geometry: PSPLGeometry,
    alpha: float,
    *,
    radial_order: int = DEFAULT_RADIAL_ORDER,
) -> float:
    """Profile source/blend flux using one consistently reconstructed curve."""
    if not datasets:
        raise ValueError("at least one photometric dataset is required")
    total = 0.0
    for dataset in datasets:
        excess = evaluate_fixed_alpha_lowpass(
            x_coeff,
            radial_nodes,
            dataset.time,
            geometry,
            alpha,
            radial_order=radial_order,
        )
        # profile_flux forms design=A-1 and its square from this exact array.
        total += profile_flux(1.0 + excess, dataset).chi2
    return float(total)


def crossing_clock_map_candidates(
    x_coeff: np.ndarray,
    radial_nodes: np.ndarray,
    datasets: Sequence[PhotometricDataset],
    components: Iterable[np.ndarray],
    *,
    map_id: int,
    parameters: Sequence[float],
    alphas: Sequence[float],
    u0_values: Sequence[float],
    entry_time: float,
    exit_time: float,
    anchor_semantics: str = "source_center_caustic_crossings",
    anchor_provenance: str = "explicit",
    caustic_coordinate_frame: str = "adamgrid_map",
) -> list[CrossingClockCandidate]:
    """Enumerate and score map/alpha/u0-specific crossing-clock hypotheses."""
    parameters_array = np.asarray(parameters, dtype=np.float64)
    if parameters_array.shape != (3,) or not np.all(np.isfinite(parameters_array)):
        raise ValueError("parameters must contain finite (logs, logq, logrho)")
    alpha_array = np.asarray(alphas, dtype=np.float64)
    u0_array = np.asarray(u0_values, dtype=np.float64)
    if alpha_array.ndim != 1 or not alpha_array.size or not np.all(np.isfinite(alpha_array)):
        raise ValueError("alphas must be a nonempty finite 1-D sequence")
    if u0_array.ndim != 1 or not u0_array.size or not np.all(np.isfinite(u0_array)):
        raise ValueError("u0_values must be a nonempty finite 1-D sequence")
    if anchor_semantics not in {
        "source_center_caustic_crossings",
        "inferred_residual_feature_anchors",
    }:
        raise ValueError(
            "anchor_semantics must be source-center crossings or inferred features; "
            "finite-source limb contacts are not yet supported"
        )
    if not anchor_provenance:
        raise ValueError("anchor_provenance is required")
    if caustic_coordinate_frame != "adamgrid_map":
        raise ValueError(
            "caustic components must be transformed into the adamgrid_map frame"
        )
    # Materialize once because callers commonly supply generators.
    component_tuple = tuple(np.asarray(component, dtype=np.float64) for component in components)
    output: list[CrossingClockCandidate] = []
    logs, logq, logrho = map(float, parameters_array)
    for alpha in alpha_array:
        for u0 in u0_array:
            pairs = caustic_crossing_pairs(
                component_tuple,
                alpha=float(alpha),
                u0=float(u0),
            )
            for pair in pairs:
                geometry = crossing_pair_geometry(
                    pair,
                    entry_time=entry_time,
                    exit_time=exit_time,
                )
                try:
                    chi2 = fixed_alpha_coefficient_chi2(
                        x_coeff,
                        radial_nodes,
                        datasets,
                        geometry,
                        float(alpha),
                    )
                except ValueError as error:
                    if "outside the atlas radial range" not in str(error):
                        raise
                    continue
                output.append(
                    _candidate_from_pair(
                        pair,
                        geometry,
                        map_id=int(map_id),
                        logs=logs,
                        logq=logq,
                        logrho=logrho,
                        entry_time=float(entry_time),
                        exit_time=float(exit_time),
                        chi2=chi2,
                        anchor_semantics=anchor_semantics,
                        anchor_provenance=anchor_provenance,
                        caustic_coordinate_frame=caustic_coordinate_frame,
                    )
                )
    output.sort(key=lambda candidate: candidate.chi2)
    return output


def _candidate_from_pair(
    pair: CausticCrossingPair,
    geometry: PSPLGeometry,
    *,
    map_id: int,
    logs: float,
    logq: float,
    logrho: float,
    entry_time: float,
    exit_time: float,
    chi2: float,
    anchor_semantics: str,
    anchor_provenance: str,
    caustic_coordinate_frame: str,
) -> CrossingClockCandidate:
    return CrossingClockCandidate(
        map_id=map_id,
        logs=logs,
        logq=logq,
        logrho=logrho,
        component_index=pair.component_index,
        entry_segment_index=pair.entry.segment_index,
        exit_segment_index=pair.exit.segment_index,
        entry_segment_fraction=pair.entry.segment_fraction,
        exit_segment_fraction=pair.exit.segment_fraction,
        entry_x=pair.entry.x,
        entry_y=pair.entry.y,
        exit_x=pair.exit.x,
        exit_y=pair.exit.y,
        alpha=pair.alpha,
        u0=pair.u0,
        tau_entry=pair.entry.tau,
        tau_exit=pair.exit.tau,
        entry_time=entry_time,
        exit_time=exit_time,
        t0=float(geometry.t0),
        tE=float(geometry.tE),
        chi2=float(chi2),
        anchor_semantics=anchor_semantics,
        anchor_provenance=anchor_provenance,
        caustic_coordinate_frame=caustic_coordinate_frame,
    )
