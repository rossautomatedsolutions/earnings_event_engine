"""Lightweight structured debug logging for the earnings event pipeline.

This module provides dependency-light helper functions for emitting
structured logs to stdout and (optionally) to JSONL files.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

_VALID_STAGES = {"events", "prices", "options", "features", "validation"}
_WRITE_LOCK = Lock()


def _is_truthy(value: str | None) -> bool:
    """Return True when an environment-style string represents truth."""
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _default_log_path() -> Path:
    """Return the default JSONL log file path for the current day."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return Path("data/outputs/logs") / f"pipeline_debug_{timestamp}.jsonl"


def _resolve_log_path() -> Path:
    """Resolve optional JSONL log path from environment or defaults."""
    env_path = os.getenv("EEE_DEBUG_LOG_PATH")
    return Path(env_path) if env_path else _default_log_path()


def _should_write_file() -> bool:
    """Whether JSONL logging is enabled via environment flag."""
    return _is_truthy(os.getenv("EEE_DEBUG_LOG_TO_FILE"))


def _normalize_stage(stage: str) -> str:
    normalized = (stage or "").strip().lower()
    return normalized if normalized in _VALID_STAGES else "validation"


def _build_record(
    *,
    level: str,
    stage: str,
    message: str,
    metadata_dict: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = dict(metadata_dict or {})
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "stage": _normalize_stage(stage),
        "message": message,
        "ticker": metadata.get("ticker"),
        "event_id": metadata.get("event_id"),
        "metadata": metadata,
    }


def _print_record(record: Mapping[str, Any]) -> None:
    """Print a compact structured log line to stdout."""
    timestamp = record.get("timestamp")
    level = record.get("level")
    stage = record.get("stage")
    ticker = record.get("ticker") or "-"
    event_id = record.get("event_id") or "-"
    message = record.get("message")
    print(
        f"[{timestamp}] [{level}] [stage={stage}] [ticker={ticker}] "
        f"[event_id={event_id}] {message}"
    )


def _append_jsonl(record: Mapping[str, Any]) -> None:
    """Append one JSON-serialized record to the configured JSONL file."""
    path = _resolve_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")


def _emit(level: str, stage: str, message: str, metadata_dict: Mapping[str, Any] | None = None) -> None:
    """Emit log record to console and optionally JSONL file."""
    record = _build_record(
        level=level,
        stage=stage,
        message=message,
        metadata_dict=metadata_dict,
    )
    _print_record(record)
    if _should_write_file():
        _append_jsonl(record)


def log_event(stage: str, message: str, metadata_dict: Mapping[str, Any] | None = None) -> None:
    """Log a standard pipeline event."""
    _emit(level="INFO", stage=stage, message=message, metadata_dict=metadata_dict)


def log_warning(stage: str, message: str, metadata_dict: Mapping[str, Any] | None = None) -> None:
    """Log a warning-level pipeline event."""
    _emit(level="WARNING", stage=stage, message=message, metadata_dict=metadata_dict)


def log_error(stage: str, error_message: str, metadata_dict: Mapping[str, Any] | None = None) -> None:
    """Log an error-level pipeline event."""
    _emit(level="ERROR", stage=stage, message=error_message, metadata_dict=metadata_dict)
