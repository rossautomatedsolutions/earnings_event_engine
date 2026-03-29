"""Shared datetime normalization utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd


def to_naive_datetime(dt: Any) -> pd.Timestamp | None:
    """Convert any datetime-like object to tz-naive pandas ``Timestamp``."""
    if dt is None:
        return None
    parsed = pd.to_datetime(dt)
    if hasattr(parsed, "tzinfo") and parsed.tzinfo is not None:
        return parsed.tz_convert(None)
    return parsed


def normalize_datetime_index(idx: pd.Index | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Ensure pandas ``DatetimeIndex`` is timezone-naive."""
    normalized = pd.to_datetime(idx)
    if not isinstance(normalized, pd.DatetimeIndex):
        normalized = pd.DatetimeIndex(normalized)
    if hasattr(normalized, "tz") and normalized.tz is not None:
        return normalized.tz_convert(None)
    return normalized


def assert_same_timezone(idx: pd.DatetimeIndex, dt: pd.Timestamp | None) -> None:
    """Hard fail when index and datetime are not both timezone-naive."""
    if hasattr(idx, "tz") and idx.tz is not None:
        raise ValueError("Index should be tz-naive")
    if dt is not None and hasattr(dt, "tzinfo") and dt.tzinfo is not None:
        raise ValueError("event_date should be tz-naive")


def validate_datetime(dt: Any) -> pd.Timestamp:
    """Validate datetime-like input and return tz-naive ``pd.Timestamp``."""
    if dt is None:
        raise ValueError("Datetime value cannot be None.")
    parsed = to_naive_datetime(dt)
    if parsed is None or pd.isna(parsed):
        raise ValueError(f"Invalid datetime value: {dt!r}.")
    return pd.Timestamp(parsed)


def to_naive(dt: Any) -> pd.Timestamp:
    """Backwards-compatible alias for ``to_naive_datetime``."""
    parsed = to_naive_datetime(dt)
    if parsed is None:
        raise ValueError("Datetime value cannot be None.")
    return pd.Timestamp(parsed)


def normalize_index(idx: pd.Index | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Backwards-compatible alias for ``normalize_datetime_index``."""
    normalized = normalize_datetime_index(idx)
    if normalized.hasnans:
        raise ValueError("DatetimeIndex contains null/NaT values.")
    return normalized


def align_to_trading_day(dt: Any, trading_index: pd.Index | pd.DatetimeIndex) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Find surrounding trading days for a datetime.

    Returns a tuple ``(nearest_before, nearest_after)`` where:
    - ``nearest_before`` is the closest trading day ``<= dt``
    - ``nearest_after`` is the closest trading day ``>= dt``

    If one side does not exist, that value is ``pd.NaT``.
    """

    target = to_naive(dt).normalize()
    dti = normalize_index(trading_index)
    assert_same_timezone(dti, target)

    if dti.empty:
        raise ValueError("Trading index is empty.")

    sorted_idx = dti.sort_values()
    left = sorted_idx.searchsorted(target, side="left")
    right = sorted_idx.searchsorted(target, side="right")

    nearest_before = sorted_idx[left - 1] if left > 0 else pd.NaT
    nearest_after = sorted_idx[right] if right < len(sorted_idx) else pd.NaT

    if left < len(sorted_idx) and sorted_idx[left] == target:
        nearest_before = target
        nearest_after = target

    return nearest_before, nearest_after
