from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import PSPLGeometry
from .map_adapter import MagnificationEvaluator, trajectory_xy
from .trajectory import PhotometricDataset


@dataclass(frozen=True)
class FluxProfile:
    chi2: float
    source_flux: float
    blend_flux: float


def profile_flux(magnification: np.ndarray, dataset: PhotometricDataset) -> FluxProfile:
    magnification = np.asarray(magnification, dtype=np.float64)
    if magnification.shape != dataset.flux.shape:
        raise ValueError("magnification and dataset shapes differ")
    weight = 1.0 / dataset.error**2
    design = magnification - 1.0
    weight_sum = np.sum(weight)
    x_sum = np.dot(weight, design)
    xx_sum = np.dot(weight, design * design)
    y_sum = np.dot(weight, dataset.flux)
    xy_sum = np.dot(weight, design * dataset.flux)
    s_xx = xx_sum - x_sum * x_sum / weight_sum
    if s_xx <= np.finfo(np.float64).eps * max(abs(xx_sum), 1.0):
        return FluxProfile(np.inf, np.nan, np.nan)
    source_flux = (xy_sum - x_sum * y_sum / weight_sum) / s_xx
    baseline = (y_sum - source_flux * x_sum) / weight_sum
    model = source_flux * design + baseline
    chi2 = float(np.dot(weight, (dataset.flux - model) ** 2))
    return FluxProfile(
        chi2=chi2,
        source_flux=float(source_flux),
        blend_flux=float(baseline - source_flux),
    )


def direct_chi2(
    evaluator: MagnificationEvaluator,
    datasets: Sequence[PhotometricDataset],
    geometry: PSPLGeometry,
    alpha: float,
) -> float:
    geometry.validate()
    total = 0.0
    for dataset in datasets:
        x, y = trajectory_xy(
            dataset.time, geometry.t0, geometry.u0, geometry.tE, alpha
        )
        total += profile_flux(evaluator.magnification(x, y), dataset).chi2
    return total


def direct_alpha_scan(
    evaluator: MagnificationEvaluator,
    datasets: Sequence[PhotometricDataset],
    geometry: PSPLGeometry,
    alphas: np.ndarray,
) -> np.ndarray:
    geometry.validate()
    alphas = np.asarray(alphas, dtype=np.float64)
    if alphas.ndim != 1:
        raise ValueError("alphas must be a 1-D array")
    total = np.zeros(len(alphas), dtype=np.float64)
    sin_alpha = np.sin(alphas)[:, None]
    cos_alpha = np.cos(alphas)[:, None]
    for dataset in datasets:
        tau = (dataset.time - geometry.t0) / geometry.tE
        x_position = geometry.u0 * sin_alpha - tau[None, :] * cos_alpha
        y_position = -geometry.u0 * cos_alpha - tau[None, :] * sin_alpha
        design = evaluator.magnification(x_position, y_position) - 1.0
        weight = 1.0 / dataset.error**2
        weight_sum = np.sum(weight)
        y_sum = np.dot(weight, dataset.flux)
        yy_centered = (
            np.dot(weight, dataset.flux**2) - y_sum * y_sum / weight_sum
        )
        x_sum = design @ weight
        xx_sum = (design * design) @ weight
        xy_sum = design @ (weight * dataset.flux)
        s_xx = xx_sum - x_sum * x_sum / weight_sum
        s_xy = xy_sum - x_sum * y_sum / weight_sum
        valid = s_xx > np.finfo(np.float64).eps * np.maximum(
            np.abs(xx_sum), 1.0
        )
        local = np.full(len(alphas), np.inf)
        local[valid] = yy_centered - s_xy[valid] ** 2 / s_xx[valid]
        total += np.maximum(local, 0.0)
    return total
