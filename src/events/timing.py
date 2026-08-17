"""Shared earnings announcement timing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Any

import pandas as pd


TIMING_COLUMN_CANDIDATES = (
    "announcement_timing",
    "timing",
    "earnings_timing",
    "earnings_time",
    "time_of_day",
    "session",
)

_TIMING_ALIASES = {
    "amc": "AMC",
    "after_close": "AMC",
    "after_market_close": "AMC",
    "after_market": "AMC",
    "after_market_close_": "AMC",
    "post_close": "AMC",
    "post_market": "AMC",
    "bmo": "BMO",
    "before_open": "BMO",
    "before_market_open": "BMO",
    "before_market": "BMO",
    "pre_market": "BMO",
    "pre_open": "BMO",
}
_PREMARKET_CUTOFF = time(hour=9, minute=30)
_POSTCLOSE_CUTOFF = time(hour=16, minute=0)


@dataclass(frozen=True)
class AnnouncementTimingResolution:
    timing: str | None
    status: str


def _has_text(value: object) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    return bool(str(value).strip())


def strip_timezone_preserving_wall_clock(value: Any) -> pd.Timestamp:
    """Drop timezone metadata while preserving the original clock reading."""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT

    timestamp = pd.Timestamp(parsed)
    if timestamp.tzinfo is not None:
        return timestamp.tz_localize(None)
    return timestamp


def strip_series_timezone_preserving_wall_clock(values: pd.Series) -> pd.Series:
    """Vector-safe wall-clock-preserving timezone stripping."""
    return pd.Series(
        [strip_timezone_preserving_wall_clock(value) for value in values],
        index=values.index,
        dtype="datetime64[ns]",
    )


def normalize_announcement_timing(value: object) -> str | None:
    if not _has_text(value):
        return None

    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return _TIMING_ALIASES.get(normalized)


def infer_announcement_timing_from_timestamp(earnings_date: Any) -> str | None:
    event_ts = strip_timezone_preserving_wall_clock(earnings_date)
    if pd.isna(event_ts):
        return None
    if event_ts == event_ts.normalize():
        return None

    event_time = event_ts.time()
    if event_time < _PREMARKET_CUTOFF:
        return "BMO"
    if event_time >= _POSTCLOSE_CUTOFF:
        return "AMC"
    return None


def resolve_announcement_timing(raw_timing: object, earnings_date: Any) -> AnnouncementTimingResolution:
    normalized = normalize_announcement_timing(raw_timing)
    if normalized is not None:
        return AnnouncementTimingResolution(normalized, "explicit")

    if _has_text(raw_timing):
        return AnnouncementTimingResolution(None, "unsupported")

    event_ts = strip_timezone_preserving_wall_clock(earnings_date)
    if pd.isna(event_ts):
        return AnnouncementTimingResolution(None, "missing")

    inferred = infer_announcement_timing_from_timestamp(event_ts)
    if inferred is not None:
        return AnnouncementTimingResolution(inferred, "inferred")

    if event_ts == event_ts.normalize():
        return AnnouncementTimingResolution(None, "missing")

    return AnnouncementTimingResolution(None, "ambiguous")


__all__ = [
    "AnnouncementTimingResolution",
    "TIMING_COLUMN_CANDIDATES",
    "infer_announcement_timing_from_timestamp",
    "normalize_announcement_timing",
    "resolve_announcement_timing",
    "strip_series_timezone_preserving_wall_clock",
    "strip_timezone_preserving_wall_clock",
]
