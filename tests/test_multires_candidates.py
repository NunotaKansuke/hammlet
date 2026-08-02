from __future__ import annotations

import numpy as np
import pytest

from hammlet._core.multires import MultiResolutionScanResult


def test_multires_candidates_are_ranked_and_ready_for_direct_handoff() -> None:
    result = MultiResolutionScanResult(
        map_ids=np.asarray([4, 7, 9]),
        parameters=np.asarray([[0.0, -3.0, -4.0], [0.1, -3.1, -3.7], [0.2, -3.2, -3.4]]),
        chi2=np.asarray([30.0, 10.0, 20.0]),
        geometry_index=np.asarray([2, 1, 0]),
        alpha_index=np.asarray([8, 4, 2]),
        spectral_risk=np.asarray([0.1, 0.3, 0.2]),
        rescanned=np.asarray([True, False, True]),
        base_seconds=1.0,
        full_seconds=0.2,
        chi2_lower=np.asarray([28.0, 8.0, 17.0]),
        chi2_upper=np.asarray([35.0, 14.0, 24.0]),
    )
    candidates = result.candidates(n_alpha=16)
    assert [item.map_id for item in candidates] == [9, 4]
    assert candidates[0].geometry_index == 0
    assert candidates[0].alpha == pytest.approx(np.pi / 4.0)
    assert candidates[0].spectral_risk == pytest.approx(0.2)
    assert candidates[0].chi2_lower == pytest.approx(17.0)
    assert candidates[0].chi2_upper == pytest.approx(24.0)
    assert candidates[0].anomaly_score is None
    assert candidates[0].anomaly_rescued is False
