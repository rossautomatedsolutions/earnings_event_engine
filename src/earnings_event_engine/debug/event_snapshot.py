"""Utilities for writing event-level debug snapshots to disk."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

DEFAULT_DEBUG_SNAPSHOT_ROOT = Path("data/outputs/debug_snapshots")

_PRICE_SLICE_KEYS = (
    "price_slice",
    "price_window",
    "prices_around_event",
    "price_data",
)
_OPTION_SNAPSHOT_KEYS = (
    "option_snapshot",
    "options_snapshot",
    "option_chain_snapshot",
    "option_data",
)
_COMPUTED_FEATURE_KEYS = (
    "computed_features",
    "features",
    "feature_values",
)


def _json_default(value: Any) -> Any:
    """Serialize common non-JSON-native types used in pipeline data."""
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return str(value)


def _to_jsonable(payload: Any) -> Any:
    """Convert payload to a JSON-friendly representation."""
    if isinstance(payload, pd.DataFrame):
        return payload.to_dict(orient="records")
    if isinstance(payload, pd.Series):
        return payload.to_dict()
    if isinstance(payload, Mapping):
        return dict(payload)
    if isinstance(payload, np.ndarray):
        return payload.tolist()
    if isinstance(payload, (list, tuple, set)):
        return list(payload)
    return payload


def _to_dataframe(payload: Any) -> pd.DataFrame | None:
    """Best-effort conversion of payload into DataFrame for CSV output."""
    if payload is None:
        return None
    if isinstance(payload, pd.DataFrame):
        return payload.copy()
    if isinstance(payload, pd.Series):
        return payload.to_frame().T
    if isinstance(payload, Mapping):
        return pd.DataFrame([dict(payload)])
    if isinstance(payload, (list, tuple)):
        if not payload:
            return pd.DataFrame()
        if all(isinstance(item, Mapping) for item in payload):
            return pd.DataFrame([dict(item) for item in payload])
    return None


def _extract_candidate(row_dict: Mapping[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in row_dict and row_dict[key] is not None:
            return row_dict[key]
    return None


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(payload), handle, indent=2, default=_json_default)


def _value_for_csv(value: Any) -> Any:
    if isinstance(value, (list, tuple, set, dict, np.ndarray, pd.DataFrame, pd.Series)):
        return json.dumps(_to_jsonable(value), default=_json_default)
    if pd.isna(value):
        return None
    return _json_default(value)


def _write_csv(path: Path, payload: Any) -> bool:
    frame = _to_dataframe(payload)
    if frame is None:
        return False

    csv_frame = frame.copy()
    for column in csv_frame.columns:
        csv_frame[column] = csv_frame[column].map(_value_for_csv)
    csv_frame.to_csv(path, index=False)
    return True


def _coerce_full_event_row(full_event_row: Any) -> dict[str, Any]:
    if isinstance(full_event_row, pd.DataFrame):
        if full_event_row.empty:
            return {}
        return full_event_row.iloc[0].to_dict()
    if isinstance(full_event_row, pd.Series):
        return full_event_row.to_dict()
    if isinstance(full_event_row, Mapping):
        return dict(full_event_row)
    raise TypeError("full_event_row must be a pandas Series/DataFrame or a mapping.")


def dump_event_snapshot(event_id: Any, full_event_row: Any, output_dir: str | Path | None = None) -> Path:
    """Dump all debug artifacts for a single event.

    Parameters
    ----------
    event_id
        Unique event identifier used for the output directory name.
    full_event_row
        Mapping-like event record. If it includes nested structures for
        ``price_slice``, ``option_snapshot``, and ``computed_features`` they are
        exported as dedicated files.
    output_dir
        Optional parent directory for snapshots. Defaults to
        ``data/outputs/debug_snapshots``.

    Returns
    -------
    pathlib.Path
        Directory containing the snapshot artifacts for the event.
    """
    event_row_dict = _coerce_full_event_row(full_event_row)

    snapshot_root = Path(output_dir) if output_dir else DEFAULT_DEBUG_SNAPSHOT_ROOT
    event_dir = snapshot_root / str(event_id)
    event_dir.mkdir(parents=True, exist_ok=True)

    files_created: list[str] = []

    full_row_json = event_dir / "full_event_row.json"
    _write_json(full_row_json, event_row_dict)
    files_created.append(full_row_json.name)

    full_row_csv = event_dir / "full_event_row.csv"
    if _write_csv(full_row_csv, event_row_dict):
        files_created.append(full_row_csv.name)

    payloads = {
        "price_slice": _extract_candidate(event_row_dict, _PRICE_SLICE_KEYS),
        "option_snapshot": _extract_candidate(event_row_dict, _OPTION_SNAPSHOT_KEYS),
        "computed_features": _extract_candidate(event_row_dict, _COMPUTED_FEATURE_KEYS),
    }

    for artifact_name, payload in payloads.items():
        if payload is None:
            continue

        json_path = event_dir / f"{artifact_name}.json"
        _write_json(json_path, payload)
        files_created.append(json_path.name)

        csv_path = event_dir / f"{artifact_name}.csv"
        if _write_csv(csv_path, payload):
            files_created.append(csv_path.name)

    manifest_path = event_dir / "manifest.json"
    manifest = {
        "event_id": str(event_id),
        "snapshot_dir": str(event_dir),
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "files": files_created,
    }
    _write_json(manifest_path, manifest)

    return event_dir
