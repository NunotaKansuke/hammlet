from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.roman_local.atlas import (
    LocalAtlasBucket,
    LocalMapAtlas,
    _combine_continuation_atlases,
    open_readonly_atlas,
)
from tools.roman_local.roman_event import (
    geometry_stencil,
    load_catalog_event,
    load_data_event,
    load_gulls_event,
)
from tools.roman_local.scan import scan_single_resolution
from tools.roman_local.packed import (
    open_packed_atlas,
    pack_atlas,
    validate_packed_source,
)


def _write_standalone_map(root: Path) -> None:
    result = root / "results" / "map-00000007"
    shard = result / "shard_0000"
    shard.mkdir(parents=True)
    (root / "results").mkdir(exist_ok=True)
    (root / "tasks.json").write_text(
        json.dumps(
            [
                {
                    "map_id": 7,
                    "logs": 0.0,
                    "logq": -3.0,
                    "logrho": -2.0,
                    "bucket": "s000_q000",
                }
            ]
        ),
        encoding="utf-8",
    )
    np.save(result / "radial_nodes.npy", np.asarray([0.0, 2.0]))
    np.save(result / "map_ids.npy", np.asarray([7], dtype=np.int64))
    np.save(result / "map_parameters.npy", np.asarray([[0.0, -3.0, -2.0]]))
    np.save(
        shard / "map_ids.npy",
        np.asarray([7], dtype=np.int64),
    )
    np.save(
        shard / "x_coeff.npy",
        np.zeros((1, 2, 3), dtype=np.complex64),
    )
    (result / "manifest.json").write_text(
        json.dumps(
            {
                "format": "hammlet-fourier-maps",
                "version": 1,
                "n_maps": 1,
                "n_r": 2,
                "m_max": 2,
                "core_m_max": 2,
                "shards": [{"path": "shard_0000", "count": 1}],
            }
        ),
        encoding="utf-8",
    )


def test_standalone_reader_does_not_need_error_sidecars(tmp_path: Path) -> None:
    _write_standalone_map(tmp_path)

    atlas = open_readonly_atlas(tmp_path)
    assert atlas.total_maps == 1
    assert atlas.expected_maps == 1
    shard = next(atlas.buckets[0].iter_shards(2))
    assert shard.map_ids.tolist() == [7]
    assert shard.x_coeff.shape == (1, 2, 3)
    nodes, parameters, coefficients = atlas.coefficient_row(7, m_max=2)
    np.testing.assert_array_equal(nodes, [0.0, 2.0])
    np.testing.assert_array_equal(parameters, [0.0, -3.0, -2.0])
    assert coefficients.shape == (2, 3)


def test_packed_cache_preserves_a_read_only_snapshot(tmp_path: Path) -> None:
    _write_standalone_map(tmp_path)
    source = open_readonly_atlas(tmp_path)
    packed_path = tmp_path.parent / f"{tmp_path.name}-packed"

    pack_atlas(source, packed_path, m_max=2)
    packed = open_packed_atlas(packed_path)
    validate_packed_source(packed, source)

    assert packed.total_maps == source.total_maps == 1
    assert packed.expected_maps == 1
    np.testing.assert_array_equal(packed.map_ids, source.map_ids)
    np.testing.assert_array_equal(packed.map_parameters, source.map_parameters)
    np.testing.assert_array_equal(
        packed.buckets[0].load_coefficients(2),
        next(source.buckets[0].iter_shards(2)).x_coeff,
    )
    assert not (packed_path / "bucket-00000.npy").is_symlink()


def test_continuation_uses_restart_task_universe(tmp_path: Path) -> None:
    def fake_atlas(map_id: int, *, kind: str, expected: int, metadata=None):
        bucket = LocalAtlasBucket(
            name=kind,
            radial_nodes=np.asarray([0.0, 1.0]),
            map_ids=np.asarray([map_id], dtype=np.int64),
            parameters=np.asarray([[0.0, -3.0, -2.0]], dtype=np.float64),
            m_max=2,
            core_m_max=2,
            sources=(),
        )
        return LocalMapAtlas(
            tmp_path / kind,
            [bucket],
            expected_maps=expected,
            completed_maps=1,
            kind=kind,
            metadata=metadata,
        )

    previous = fake_atlas(1, kind="distributed-parts", expected=1)
    restart = fake_atlas(
        2,
        kind="standalone-results",
        expected=3,
        metadata={"task_order_maps": 3, "tasks_path": str(tmp_path / "tasks.json")},
    )

    atlas = _combine_continuation_atlases(
        tmp_path, [previous, restart], kind="combined-generations"
    )

    assert atlas.total_maps == 2
    assert atlas.completed_maps == 2
    assert atlas.expected_maps == 3
    assert atlas.metadata["missing_task_maps"] == 1


def test_roman_catalog_adapter_and_full_stencil(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.csv"
    catalog.write_text(
        "SubRun,Field,EventID,t0lens1,u0lens1,tE_ref,Planet_q,Planet_s,rho,alpha\n"
        "0,12,4,100.0,0.1,5.0,0.001,1.2,0.01,90.0\n",
        encoding="utf-8",
    )
    data = tmp_path / "RomanW146sat1.dat"
    data.write_text(
        "20.0 0.01 8330.0\n20.1 0.01 8335.0\n20.2 0.01 8340.0\n",
        encoding="utf-8",
    )
    event_dir = tmp_path / "events" / "event_0_12_4" / "Data"
    event_dir.mkdir(parents=True)
    data.rename(event_dir / data.name)
    event = load_catalog_event(
        0,
        catalog=catalog,
        simulation_root=tmp_path / "events",
        window_tE=4.0,
    )
    assert event.name == "0_12_4"
    assert event.total_points == 3
    assert event.geometry_source == "catalog-truth"
    geometries, report = geometry_stencil(event.geometry_center)
    assert len(geometries) == 16
    assert report["count"] == 16


def test_user_data_adapter_reads_all_rows_by_default(tmp_path: Path) -> None:
    data = tmp_path / "event.dat"
    data.write_text(
        "\n".join(
            f"{time_value} 1.0 0.1"
            for time_value in (0.0, 1.0, 4.0, 5.0, 6.0, 9.0, 20.0)
        )
        + "\n",
        encoding="utf-8",
    )

    event = load_data_event(
        [data],
        t0=5.0,
        u0=0.1,
        tE=1.0,
        data_format="time-flux-error",
    )
    assert event.total_points == 7
    assert event.window == (0.0, 20.0)

    windowed = load_data_event(
        [data],
        t0=5.0,
        u0=0.1,
        tE=1.0,
        window_tE=4.0,
        data_format="time-flux-error",
    )
    assert windowed.total_points == 5
    assert windowed.window == (1.0, 9.0)


def test_gulls_adapter_uses_relative_time_and_observed_columns(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "event_id,tE_days,rho,u0,alpha_deg,t0_day,t0_jd,planet_q,planet_s\n"
        "9900402,40.14,0.00048,0.006,0.01,50.0,2461497.0,0.001,1.2\n",
        encoding="utf-8",
    )
    lightcurve = tmp_path / "event.all.lc"
    rows = [
        "#Event: 0.006 0.01 50.0 40.14 0.00048",
        "49.0 1.2 0.1 1.2 0.1 0 0 0 0 0 2461496 0 0 0 0 0 0",
        "50.0 2.0 0.2 2.0 0.2 0 0 0 0 0 2461497 0 0 0 0 0 0",
        "51.0 1.1 0.1 1.1 0.1 0 1 0 0 0 2461498 0 0 0 0 0 0",
        "52.0 1.3 0.1 1.3 0.1 0 0 0 0 0 2461499 0 0 0 0 0 0",
    ]
    lightcurve.write_text("\n".join(rows) + "\n", encoding="utf-8")

    event = load_gulls_event(lightcurve, manifest=manifest, window_tE=1.0)

    assert event.name == "gulls-9900402"
    assert event.data_format == "gulls-all-lc"
    assert event.geometry_source == "gulls-manifest-truth"
    assert event.total_points == 3
    np.testing.assert_allclose(event.datasets[0].time, [49.0, 50.0, 52.0])
    np.testing.assert_allclose(event.datasets[0].flux, [1.2, 2.0, 1.3])
    np.testing.assert_allclose(event.reference.t0, 50.0)
    np.testing.assert_allclose(event.reference.tE, 40.14)


def test_single_resolution_scan_marks_filtered_rows_without_certificates(
    tmp_path: Path, monkeypatch
) -> None:
    _write_standalone_map(tmp_path)
    atlas = open_readonly_atlas(tmp_path)

    class FakeScanner:
        def __init__(self, kernels, *, n_alpha, compute_dtype):
            self.n_alpha = n_alpha
            self.n_geometry = len(kernels)

        def set_kernels(self, kernels):
            return None

        def scan_minima(self, coefficients):
            count = len(coefficients)
            return (
                np.arange(count, dtype=np.float64),
                np.zeros(count, dtype=np.int64),
                np.zeros(count, dtype=np.int64),
            )

    monkeypatch.setattr(
        "tools.roman_local.scan.JAXConsistentGeometryBatchScanner", FakeScanner
    )
    result = scan_single_resolution(
        atlas,
        lambda nodes, m_max: [[object()]],
        m_max=2,
        n_alpha=8,
        map_filter=lambda parameters: np.asarray([False]),
    )
    assert result.chi2_lower is None
    assert result.chi2_upper is None
    assert result.scanned.tolist() == [False]
    assert np.isinf(result.chi2[0])
