from __future__ import annotations

import numpy as np
import pytest

from hammlet._core.config import PSPLGeometry
from hammlet._core.crossing_clock import (
    evaluate_fixed_alpha_lowpass,
    fixed_alpha_coefficient_chi2,
)
from hammlet._core.jax_seed_refine import (
    JAXFixedAlphaBatchEvaluator,
    batched_pairwise_refine,
    batched_pattern_refine,
    evaluate_in_fixed_batches,
)
from hammlet._core.trajectory import PhotometricDataset


def _problem():
    pytest.importorskip("jax")
    nodes = np.linspace(0.0, 2.0, 33)
    coefficients = np.zeros((len(nodes), 3), dtype=np.complex64)
    coefficients[:, 0] = 0.3 * np.exp(-0.5 * nodes)
    coefficients[:, 1] = (0.07 + 0.02j) * nodes
    coefficients[:, 2] = (-0.02 + 0.01j) * nodes**2
    truth = PSPLGeometry(t0=10.1, u0=0.08, tE=4.2)
    alpha = 0.7
    time = np.linspace(4.0, 16.0, 81)
    excess = evaluate_fixed_alpha_lowpass(coefficients, nodes, time, truth, alpha)
    dataset = PhotometricDataset(
        time, 3.0 * excess + 12.0, np.full_like(time, 0.02), "test"
    )
    return nodes, coefficients, truth, alpha, (dataset,)


def test_jax_batch_objective_matches_scalar_coefficient_path() -> None:
    nodes, coefficients, truth, alpha, datasets = _problem()
    evaluator = JAXFixedAlphaBatchEvaluator(datasets, m_max=2)
    parameters = np.asarray(
        [
            [truth.t0, truth.u0, np.log(truth.tE), alpha],
            [truth.t0 + 0.1, truth.u0, np.log(truth.tE), alpha + 0.05],
        ]
    )
    batched = evaluator.evaluate(
        np.stack((coefficients, coefficients)),
        np.stack((nodes, nodes)),
        parameters,
    )
    scalar = np.asarray(
        [
            fixed_alpha_coefficient_chi2(
                coefficients,
                nodes,
                datasets,
                PSPLGeometry(t0=row[0], u0=row[1], tE=np.exp(row[2])),
                row[3],
                radial_order=3,
            )
            for row in parameters
        ]
    )

    np.testing.assert_allclose(batched, scalar, rtol=2.0e-10, atol=2.0e-8)


def test_fixed_batch_padding_and_pattern_search_preserve_incumbents() -> None:
    nodes, coefficients, truth, alpha, datasets = _problem()
    evaluator = JAXFixedAlphaBatchEvaluator(datasets, m_max=2)
    count = 3
    coefficient_batch = np.repeat(coefficients[None, ...], count, axis=0)
    node_batch = np.repeat(nodes[None, ...], count, axis=0)
    initial = np.repeat(
        np.asarray([[10.0, 0.1, np.log(4.0), alpha - 0.1]]), count, axis=0
    )
    direct = evaluator.evaluate(coefficient_batch, node_batch, initial)
    padded = evaluate_in_fixed_batches(
        evaluator,
        coefficient_batch,
        node_batch,
        initial,
        batch_size=4,
    )
    np.testing.assert_allclose(padded, direct)

    half_width = np.asarray([0.2, 0.05, 0.1, 0.2])
    bounds = np.stack((initial - half_width, initial + half_width), axis=-1)
    refined = batched_pattern_refine(
        evaluator,
        coefficient_batch,
        node_batch,
        initial,
        bounds,
        levels=2,
        batch_size=4,
    )

    assert refined.evaluations_per_candidate == 17
    assert np.all(refined.chi2 <= refined.initial_chi2)
    assert np.all(refined.chi2 < direct)

    pairwise = batched_pairwise_refine(
        evaluator,
        coefficient_batch,
        node_batch,
        initial,
        bounds,
        levels=1,
        batch_size=4,
    )
    assert pairwise.evaluations_per_candidate == 25
    assert np.all(pairwise.chi2 <= pairwise.initial_chi2)
