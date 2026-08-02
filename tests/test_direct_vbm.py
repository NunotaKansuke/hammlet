from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np

from hammlet._core.atlas import PolarAtlas
from hammlet._core.atlas_builder import MapBuildSpec
from hammlet._core.direct_vbm import (
    DirectSpectrumConfig,
    DirectVBMPolarAtlasBuilder,
    adaptive_ring_spectrum,
)


@dataclass
class SmoothRing:
    source_radius: float = 0.01
    caustic_components: tuple[np.ndarray, ...] = ()

    def magnification(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        angle = np.arctan2(y, x)
        return 1.4 + 0.2 * np.cos(3.0 * angle) - 0.1 * np.sin(2.0 * angle)


@dataclass
class NarrowCausticRing:
    source_radius: float = 0.002

    @property
    def caustic_components(self) -> tuple[np.ndarray, ...]:
        angle = 0.37
        direction = np.asarray([np.cos(angle), np.sin(angle)])
        tangent = np.asarray([-direction[1], direction[0]])
        return (
            np.stack(
                (
                    0.96 * direction - 0.02 * tangent,
                    1.04 * direction + 0.02 * tangent,
                )
            ),
        )

    def magnification(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        angle = np.arctan2(y, x)
        distance = np.angle(np.exp(1j * (angle - 0.37)))
        return 1.2 + 3.0 * np.exp(-0.5 * (distance / self.source_radius) ** 2)


def _config(**updates) -> DirectSpectrumConfig:
    values = {
        "m_max": 8,
        "core_m_max": 4,
        "min_n_phi": 32,
        "caustic_base_n_phi": 64,
        "max_n_phi": 128,
        "coefficient_rtol": 1.0e-5,
        "coefficient_atol": 1.0e-10,
        "diagnostic_m_max": 16,
        "caustic_local_levels": 6,
    }
    values.update(updates)
    return DirectSpectrumConfig(**values)


def test_nested_fft_recovers_smooth_low_modes() -> None:
    result = adaptive_ring_spectrum(SmoothRing(), 1.0, _config())

    assert result.converged
    assert not result.caustic_guarded
    assert result.uniform_n_phi == 64
    np.testing.assert_allclose(result.x_coeff[0], 0.4, atol=1.0e-14)
    np.testing.assert_allclose(result.x_coeff[2], 0.05j, atol=1.0e-14)
    np.testing.assert_allclose(result.x_coeff[3], 0.1, atol=1.0e-14)
    assert result.reconstruction_error < 1.0e-12


def test_caustic_local_refinement_recovers_narrow_feature() -> None:
    evaluator = NarrowCausticRing()
    config = _config(coefficient_rtol=2.0e-3)
    result = adaptive_ring_spectrum(evaluator, 1.0, config)
    dense_angles = np.arange(131072) * (2.0 * np.pi / 131072)
    dense_values = (
        evaluator.magnification(np.cos(dense_angles), np.sin(dense_angles)) - 1.0
    )
    reference = np.fft.rfft(dense_values)[: config.m_max + 1] / len(dense_angles)

    assert result.caustic_guarded
    assert result.caustic_angles
    assert result.evaluations < 512
    np.testing.assert_allclose(result.x_coeff, reference, rtol=6.0e-3, atol=2.0e-4)
    assert result.certified_error <= result.certified_error_core
    reconstructed = np.real(
        result.x_coeff[0]
        + 2.0
        * np.sum(
            result.x_coeff[None, 1:]
            * np.exp(
                1j
                * dense_angles[:, None]
                * np.arange(len(result.x_coeff))[None, 1:]
            ),
            axis=1,
        )
    )
    assert np.max(np.abs(dense_values - reconstructed)) <= result.certified_error


def test_direct_builder_writes_compatible_atlas_and_provenance(tmp_path) -> None:
    config = _config(m_max=4, core_m_max=2, diagnostic_m_max=8)
    builder = DirectVBMPolarAtlasBuilder(
        np.asarray([0.1, 0.5, 1.0]),
        spectrum_config=config,
        shard_size=2,
    )
    output = builder.build(
        tmp_path / "atlas",
        [MapBuildSpec(7, 0.0, -3.0, -2.0, SmoothRing())],
    )
    atlas = PolarAtlas(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert atlas.map_ids.tolist() == [7]
    assert atlas.m_max == 4
    assert manifest["builder"] == "direct-vbm-adaptive-fourier"
    assert manifest["direct_diagnostics"]["unconverged_rings"] == 0
    assert manifest["error_certificate"] == (
        "periodic-piecewise-linear-vbm-reference"
    )
    shard = next(atlas.iter_shards(m_max=4))
    assert shard.certified_error is not None
    assert np.all(shard.certified_error >= shard.reconstruction_error)
