"""Small adapter for the Roman simulation files used by the local test."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from hammlet._core.config import PSPLGeometry
from hammlet._core.trajectory import PhotometricDataset


TIME_OFFSET = 8234.0
ROMAN_BAND_FILES = {
    "W146": "RomanW146sat1.dat",
    "Z087": "RomanZ087sat2.dat",
    "K213": "RomanK213sat3.dat",
}
_STENCIL_PRESETS = ("full", "te-only", "center")


@dataclass(frozen=True)
class RomanReference:
    """Injected map parameters and the catalog trajectory, when available."""

    t0: float
    u0: float
    tE: float
    s: float | None = None
    q: float | None = None
    rho: float | None = None
    catalog_alpha: float | None = None

    def __post_init__(self) -> None:
        for name in ("t0", "u0", "tE"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"reference {name} must be finite")
        if self.tE <= 0.0:
            raise ValueError("reference tE must be positive")
        for name in ("s", "q", "rho"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"reference {name} must be finite and positive")

    def as_dict(self) -> dict[str, float | None]:
        return {
            "t0": self.t0,
            "u0": self.u0,
            "tE": self.tE,
            "s": self.s,
            "q": self.q,
            "rho": self.rho,
            "catalog_alpha": self.catalog_alpha,
        }


@dataclass(frozen=True)
class RomanEvent:
    index: int | None
    name: str
    event_directory: Path | None
    data_paths: tuple[Path, ...]
    datasets: tuple[PhotometricDataset, ...]
    reference: RomanReference
    geometry_center: PSPLGeometry
    geometry_source: str
    window: tuple[float, float]
    anomaly_path: Path | None = None
    catalog_path: Path | None = None
    data_format: str = "roman-mag"

    @property
    def total_points(self) -> int:
        return int(sum(len(dataset.time) for dataset in self.datasets))

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "event_directory": (
                None
                if self.event_directory is None
                else str(self.event_directory.resolve())
            ),
            "data_paths": [str(path.resolve()) for path in self.data_paths],
            "datasets": {
                dataset.name: int(len(dataset.time)) for dataset in self.datasets
            },
            "total_points": self.total_points,
            "reference": self.reference.as_dict(),
            "geometry_center": {
                "t0": float(self.geometry_center.t0),
                "u0": float(self.geometry_center.u0),
                "tE": float(self.geometry_center.tE),
            },
            "geometry_source": self.geometry_source,
            "window": [float(value) for value in self.window],
            "data_format": self.data_format,
            "anomaly_path": (
                None if self.anomaly_path is None else str(self.anomaly_path.resolve())
            ),
            "catalog_path": (
                None if self.catalog_path is None else str(self.catalog_path.resolve())
            ),
        }


def load_catalog_event(
    index: int,
    *,
    catalog: str | Path,
    simulation_root: str | Path,
    anomaly_path: str | Path | None = None,
    bands: Sequence[str] = ("W146",),
    window_tE: float | None = None,
    window: tuple[float, float] | None = None,
    time_offset: float = TIME_OFFSET,
    data_format: str = "roman-mag",
) -> RomanEvent:
    """Load one catalog event and its Roman band files.

    The default ``roman-mag`` reader matches the sample files: magnitude,
    magnitude error, and time.  An anomaly JSON is optional; when absent, the
    catalog truth trajectory is used as the geometry center.
    """
    catalog = Path(catalog).expanduser().resolve()
    simulation_root = Path(simulation_root).expanduser().resolve()
    rows = _read_catalog(catalog)
    index = int(index)
    if not 0 <= index < len(rows):
        raise IndexError("Roman event index is outside the simulation catalog")
    row = rows[index]
    name = _event_name(row)
    reference = _reference_from_row(row, time_offset=float(time_offset))
    anomaly = None if anomaly_path is None else Path(anomaly_path).expanduser().resolve()
    if anomaly is None:
        center = PSPLGeometry(reference.t0, reference.u0, reference.tE)
        source = "catalog-truth"
    else:
        center = load_anomaly_geometry(anomaly)
        source = "anomaly-product"
    selected_window = (
        _validate_window(window)
        if window is not None
        else (None if window_tE is None else _window(center, window_tE))
    )
    event_directory = simulation_root / f"event_{name}"
    datasets, data_paths = load_band_datasets(
        event_directory / "Data",
        bands=bands,
        window=selected_window,
        data_format=data_format,
    )
    if selected_window is None:
        selected_window = _dataset_window(datasets)
    return RomanEvent(
        index=index,
        name=name,
        event_directory=event_directory,
        data_paths=tuple(data_paths),
        datasets=tuple(datasets),
        reference=reference,
        geometry_center=center,
        geometry_source=source,
        window=selected_window,
        anomaly_path=anomaly,
        catalog_path=catalog,
        data_format=data_format,
    )


def load_data_event(
    data_paths: Sequence[str | Path],
    *,
    t0: float,
    u0: float,
    tE: float,
    names: Sequence[str] | None = None,
    window_tE: float | None = None,
    window: tuple[float, float] | None = None,
    data_format: str = "roman-mag",
    reference: RomanReference | None = None,
    name: str = "local-event",
) -> RomanEvent:
    """Load user-provided Roman files without requiring a simulation catalog."""
    center = PSPLGeometry(float(t0), float(u0), float(tE))
    center.validate()
    selected_window = (
        _validate_window(window)
        if window is not None
        else (None if window_tE is None else _window(center, window_tE))
    )
    paths = tuple(Path(path).expanduser().resolve() for path in data_paths)
    if not paths:
        raise ValueError("at least one --data-file is required")
    if names is None:
        dataset_names = tuple(f"data-{index}" for index in range(len(paths)))
    else:
        dataset_names = tuple(str(value) for value in names)
    if len(dataset_names) != len(paths):
        raise ValueError("the number of --band-name values must match --data-file")
    datasets = []
    for path, dataset_name in zip(paths, dataset_names):
        datasets.append(
            load_data_file(
                path,
                name=dataset_name,
                window=selected_window,
                data_format=data_format,
            )
        )
    if selected_window is None:
        selected_window = _dataset_window(datasets)
    if reference is None:
        reference = RomanReference(float(t0), float(u0), float(tE))
    return RomanEvent(
        index=None,
        name=name,
        event_directory=None,
        data_paths=paths,
        datasets=tuple(datasets),
        reference=reference,
        geometry_center=center,
        geometry_source="user-geometry",
        window=selected_window,
        data_format=data_format,
    )


def load_gulls_event(
    lightcurve: str | Path,
    *,
    manifest: str | Path,
    event_id: str | int | None = None,
    window_tE: float | None = None,
    window: tuple[float, float] | None = None,
    name: str | None = None,
) -> RomanEvent:
    """Load one GULLS ``.all.lc`` file with its manifest truth row.

    GULLS stores the observed magnification and its uncertainty in columns
    1--3 (zero-based columns 0--2), the time in the simulation's relative
    day coordinate, and the saturation flag in column 6.  The manifest's
    ``t0_day`` is therefore the matching time origin; ``t0_jd`` is retained
    by GULLS for absolute-time bookkeeping and is not substituted here.
    """
    lightcurve_path = Path(lightcurve).expanduser().resolve()
    manifest_path = Path(manifest).expanduser().resolve()
    row = _read_manifest_row(manifest_path, event_id=event_id)
    reference = _reference_from_gulls_row(row)
    center = PSPLGeometry(reference.t0, reference.u0, reference.tE)
    center.validate()
    selected_window = (
        _validate_window(window)
        if window is not None
        else (None if window_tE is None else _window(center, window_tE))
    )
    dataset = load_gulls_data_file(
        lightcurve_path,
        name="Roman-GULLS",
        window=selected_window,
    )
    if selected_window is None:
        selected_window = (
            float(np.min(dataset.time)),
            float(np.max(dataset.time)),
        )
    event_name = str(name) if name is not None else f"gulls-{row['event_id']}"
    return RomanEvent(
        index=None,
        name=event_name,
        event_directory=None,
        data_paths=(lightcurve_path,),
        datasets=(dataset,),
        reference=reference,
        geometry_center=center,
        geometry_source="gulls-manifest-truth",
        window=selected_window,
        catalog_path=manifest_path,
        data_format="gulls-all-lc",
    )


def load_band_datasets(
    data_directory: str | Path,
    *,
    bands: Sequence[str],
    window: tuple[float, float] | None = None,
    data_format: str = "roman-mag",
) -> tuple[list[PhotometricDataset], list[Path]]:
    data_directory = Path(data_directory).expanduser().resolve()
    datasets = []
    paths = []
    for band in bands:
        try:
            filename = ROMAN_BAND_FILES[str(band)]
        except KeyError as error:
            raise ValueError(
                f"unsupported Roman band {band!r}; choose from {sorted(ROMAN_BAND_FILES)}"
            ) from error
        path = data_directory / filename
        datasets.append(
            load_data_file(
                path,
                name=str(band),
                window=window,
                data_format=data_format,
            )
        )
        paths.append(path)
    return datasets, paths


def load_data_file(
    path: str | Path,
    *,
    name: str,
    window: tuple[float, float] | None = None,
    data_format: str = "roman-mag",
) -> PhotometricDataset:
    """Read three-column Roman data and convert it to flux space.

    ``window=None`` intentionally means the whole file: only non-finite rows
    and rows with non-positive uncertainties are rejected.  A time window is
    an opt-in diagnostic filter.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Roman data file does not exist: {path}")
    if data_format not in {"roman-mag", "time-flux-error"}:
        raise ValueError("data_format must be 'roman-mag' or 'time-flux-error'")
    try:
        values = np.genfromtxt(
            path,
            comments="#",
            usecols=(0, 1, 2),
            ndmin=2,
            invalid_raise=False,
        )
    except (OSError, ValueError) as error:
        raise ValueError(f"could not parse Roman data file: {path}") from error
    if values.ndim != 2 or values.shape[1] != 3 or values.size == 0:
        raise ValueError(f"Roman data file must contain three columns: {path}")
    if data_format == "roman-mag":
        magnitude, magnitude_error, observation_time = values.T
        flux = np.power(10.0, -magnitude / 2.5)
        error = flux * (np.log(10.0) / 2.5) * magnitude_error
    else:
        observation_time, flux, error = values.T
    if window is None:
        in_window = np.ones(observation_time.shape, dtype=bool)
    else:
        lower, upper = _validate_window(window)
        in_window = (observation_time >= lower) & (observation_time <= upper)
    valid = (
        np.isfinite(observation_time)
        & np.isfinite(flux)
        & np.isfinite(error)
        & (error > 0.0)
        & in_window
    )
    if np.count_nonzero(valid) < 3:
        raise ValueError(
            f"Roman data file has fewer than three valid points in the window: {path}"
        )
    return PhotometricDataset(
        observation_time[valid], flux[valid], error[valid], str(name)
    )


def load_gulls_data_file(
    path: str | Path,
    *,
    name: str,
    window: tuple[float, float] | None,
) -> PhotometricDataset:
    """Read the observed stream from a GULLS ``.all.lc`` file.

    The local scanner works in a generic weighted-flux space.  GULLS's
    ``Aobs``/``Aerr`` pair is already an observed quantity with an uncertainty,
    so it is passed through as the data/error pair.  Rows marked saturated are
    excluded; the model truth columns are deliberately never used as data.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"GULLS lightcurve does not exist: {path}")
    try:
        values = np.genfromtxt(
            path,
            comments="#",
            usecols=(0, 1, 2, 6),
            ndmin=2,
            invalid_raise=False,
        )
    except (OSError, ValueError) as error:
        raise ValueError(f"could not parse GULLS lightcurve: {path}") from error
    if values.ndim != 2 or values.shape[1] != 4 or values.size == 0:
        raise ValueError(
            "GULLS lightcurve must contain at least columns time, Aobs, Aerr, and satflag: "
            f"{path}"
        )
    observation_time, observed, error, saturation = values.T
    if window is None:
        in_window = np.ones(observation_time.shape, dtype=bool)
    else:
        lower, upper = _validate_window(window)
        in_window = (observation_time >= lower) & (observation_time <= upper)
    valid = (
        np.isfinite(observation_time)
        & np.isfinite(observed)
        & np.isfinite(error)
        & np.isfinite(saturation)
        & (error > 0.0)
        & (saturation == 0.0)
        & in_window
    )
    if np.count_nonzero(valid) < 3:
        raise ValueError(
            f"GULLS lightcurve has fewer than three valid unsaturated points in the window: {path}"
        )
    return PhotometricDataset(
        observation_time[valid], observed[valid], error[valid], str(name)
    )


def load_anomaly_geometry(path: str | Path) -> PSPLGeometry:
    """Extract ``t0``, ``u0``, and ``tE`` from common anomaly JSON layouts."""
    path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid anomaly product: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"anomaly product must be a JSON object: {path}")
    containers: list[Mapping[str, object]] = [payload]
    for key in ("fit", "params", "best_fit", "baseline"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
            params = value.get("params")
            if isinstance(params, Mapping):
                containers.append(params)
    values = {
        "t0": _find_parameter(containers, ("t0", "t_0", "t0_pspl"), path),
        "u0": _find_parameter(containers, ("u0", "u_0", "u0_pspl"), path),
        "tE": _find_parameter(containers, ("tE", "t_E", "te", "t_e"), path),
    }
    geometry = PSPLGeometry(**values)
    geometry.validate()
    return geometry


def geometry_stencil(
    center: PSPLGeometry,
    *,
    preset: str = "full",
    dt0: float | None = None,
    du0: float | None = None,
    dlog_te: float | None = None,
    te_steps: int = 3,
    extra: tuple[float, float, float] | None = (-0.5, -0.5, 0.5),
) -> tuple[list[PSPLGeometry], dict[str, object]]:
    """Expand an event-scaled PSPL seed stencil."""
    if preset not in _STENCIL_PRESETS:
        raise ValueError(f"stencil must be one of {_STENCIL_PRESETS}")
    if int(te_steps) < 1:
        raise ValueError("te_steps must be positive")
    center.validate()
    steps = {
        "dt0": max(2.0, 0.1 * float(center.tE)) if dt0 is None else float(dt0),
        "du0": max(0.03, 0.1 * abs(float(center.u0))) if du0 is None else float(du0),
        "dlog_te": 0.15 if dlog_te is None else float(dlog_te),
    }
    if any(not np.isfinite(value) or value < 0.0 for value in steps.values()):
        raise ValueError("geometry stencil steps must be finite and non-negative")
    offsets = _preset_offsets(preset, int(te_steps), extra)
    geometries = [
        PSPLGeometry(
            t0=float(center.t0) + offset[0] * steps["dt0"],
            u0=float(center.u0) + offset[1] * steps["du0"],
            tE=float(center.tE) * np.exp(offset[2] * steps["dlog_te"]),
        )
        for offset in offsets
    ]
    return geometries, {
        "preset": preset,
        "te_steps": int(te_steps),
        "offsets": [list(offset) for offset in offsets],
        "steps": steps,
        "extra": None if extra is None else list(extra),
        "count": len(geometries),
    }


def _read_catalog(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Roman catalog does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Roman catalog has no header: {path}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Roman catalog is empty: {path}")
    return rows


def _read_manifest_row(
    path: Path, *, event_id: str | int | None
) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"GULLS manifest does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"GULLS manifest has no header: {path}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"GULLS manifest is empty: {path}")
    if event_id is None:
        if len(rows) != 1:
            raise ValueError(
                "GULLS manifest contains multiple events; supply --event-id"
            )
        return rows[0]
    target = str(event_id)
    matches = [row for row in rows if str(row.get("event_id", "")) == target]
    if len(matches) != 1:
        raise ValueError(
            f"GULLS manifest event_id={target!r} matched {len(matches)} rows: {path}"
        )
    return matches[0]


def _reference_from_row(row: Mapping[str, str], *, time_offset: float) -> RomanReference:
    try:
        catalog_alpha = float(row["alpha"])
        return RomanReference(
            t0=float(row["t0lens1"]) + time_offset,
            u0=float(row["u0lens1"]),
            tE=float(row["tE_ref"]),
            s=float(row["Planet_s"]),
            q=float(row["Planet_q"]),
            rho=float(row["rho"]),
            catalog_alpha=float(np.deg2rad(catalog_alpha) % (2.0 * np.pi)),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Roman catalog row is missing a required parameter") from error


def _reference_from_gulls_row(row: Mapping[str, str]) -> RomanReference:
    try:
        alpha_deg = float(row["alpha_deg"])
        return RomanReference(
            t0=float(row["t0_day"]),
            u0=float(row["u0"]),
            tE=float(row["tE_days"]),
            s=float(row["planet_s"]),
            q=float(row["planet_q"]),
            rho=float(row["rho"]),
            catalog_alpha=float(np.deg2rad(alpha_deg) % (2.0 * np.pi)),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "GULLS manifest row is missing a required t0_day/u0/tE_days/planet_s/"
            "planet_q/rho/alpha_deg parameter"
        ) from error


def _event_name(row: Mapping[str, str]) -> str:
    try:
        return f"{int(float(row['SubRun']))}_{int(float(row['Field']))}_{int(float(row['EventID']))}"
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Roman catalog row has invalid SubRun/Field/EventID") from error


def _window(center: PSPLGeometry, window_tE: float) -> tuple[float, float]:
    center.validate()
    window_tE = float(window_tE)
    if not np.isfinite(window_tE) or window_tE <= 0.0:
        raise ValueError("window_tE must be finite and positive")
    half_width = window_tE * float(center.tE)
    return _validate_window((float(center.t0) - half_width, float(center.t0) + half_width))


def _validate_window(window: tuple[float, float]) -> tuple[float, float]:
    lower, upper = (float(value) for value in window)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("Roman data window must be finite and increasing")
    return lower, upper


def _dataset_window(datasets: Sequence[PhotometricDataset]) -> tuple[float, float]:
    """Return the actual time span of the valid rows loaded from the files."""
    if not datasets:
        raise ValueError("at least one dataset is required")
    lower = min(float(np.min(dataset.time)) for dataset in datasets)
    upper = max(float(np.max(dataset.time)) for dataset in datasets)
    return _validate_window((lower, upper))


def _preset_offsets(
    preset: str,
    te_steps: int,
    extra: tuple[float, float, float] | None,
) -> tuple[tuple[float, float, float], ...]:
    reach = (te_steps - 1) // 2
    te_arm: list[float] = []
    for step in range(1, reach + 1):
        te_arm.extend((-float(step), float(step)))
    if te_steps % 2 == 0:
        te_arm.append(float(reach + 1))
    if preset == "center":
        return ((0.0, 0.0, 0.0),)
    if preset == "te-only":
        return ((0.0, 0.0, 0.0),) + tuple(
            (0.0, 0.0, value) for value in te_arm
        )
    offsets: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    offsets.extend((sign, 0.0, 0.0) for sign in (-1.0, 1.0))
    offsets.extend((0.0, sign, 0.0) for sign in (-1.0, 1.0))
    offsets.extend((0.0, 0.0, value) for value in te_arm)
    offsets.extend(
        (a, b, c)
        for a in (-1.0, 1.0)
        for b in (-1.0, 1.0)
        for c in (-1.0, 1.0)
    )
    if extra is not None:
        if len(extra) != 3 or not np.all(np.isfinite(extra)):
            raise ValueError("extra geometry offset must contain three finite values")
        offsets.append(tuple(float(value) for value in extra))
    return tuple(offsets)


def _find_parameter(
    containers: Sequence[Mapping[str, object]],
    names: Sequence[str],
    path: Path,
) -> float:
    for container in containers:
        for name in names:
            if name in container:
                try:
                    value = float(container[name])  # type: ignore[arg-type]
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"anomaly parameter {name!r} is not numeric: {path}"
                    ) from error
                if np.isfinite(value):
                    return value
    raise ValueError(
        f"anomaly product has no parameter matching {tuple(names)}: {path}"
    )
