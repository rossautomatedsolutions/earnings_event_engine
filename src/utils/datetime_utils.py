"""Shared datetime normalization utilities.

All helpers in this module enforce explicit behavior and raise clear exceptions
on invalid inputs instead of silently coercing data.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


_SUPPORTED_INPUT_TYPES = (str, datetime, pd.Timestamp)


def validate_datetime(dt: Any) -> pd.Timestamp:
    """Validate and parse a datetime-like input into ``pd.Timestamp``.

    Supported input types are ``str``, ``datetime.datetime``, and
    ``pandas.Timestamp``. Invalid strings, ``NaT`` values, and unsupported input
    types raise explicit exceptions.
    """

    if not isinstance(dt, _SUPPORTED_INPUT_TYPES):
        supported = ", ".join(t.__name__ for t in _SUPPORTED_INPUT_TYPES)
        raise TypeError(f"Unsupported datetime type: {type(dt).__name__}. Expected one of: {supported}.")

    try:
        parsed = pd.Timestamp(dt)
    except Exception as exc:  # pragma: no cover - defensive parsing branch
        raise ValueError(f"Invalid datetime value: {dt!r}.") from exc

    if pd.isna(parsed):
        raise ValueError(f"Invalid datetime value: {dt!r}. Parsed result is NaT.")

    return parsed


def to_naive(dt: Any) -> pd.Timestamp:
    """Convert a datetime-like value to a timezone-naive ``pd.Timestamp``.

    Behavior:
    - naive inputs are returned as timezone-naive timestamps
    - tz-aware inputs are converted to UTC, then made naive
    """

    parsed = validate_datetime(dt)

    if parsed.tz is not None:
        return parsed.tz_convert("UTC").tz_localize(None)

    return parsed.tz_localize(None)


def normalize_index(idx: pd.Index | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return a clean timezone-naive ``DatetimeIndex``.

    - Converts non-datetime indexes via ``pd.to_datetime(..., errors='raise')``
    - Converts tz-aware values to UTC and strips timezone
    - Raises explicit errors for unparseable or null datetime values
    """

    try:
        dti = idx if isinstance(idx, pd.DatetimeIndex) else pd.to_datetime(idx, errors="raise")
        dti = pd.DatetimeIndex(dti)
    except Exception as exc:
        raise ValueError("Index cannot be converted to a DatetimeIndex.") from exc

    if dti.hasnans:
        raise ValueError("DatetimeIndex contains null/NaT values.")

    if dti.tz is not None:
        dti = dti.tz_convert("UTC").tz_localize(None)

    return dti


def align_to_trading_day(dt: Any, trading_index: pd.Index | pd.DatetimeIndex) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Find surrounding trading days for a datetime.

    Returns a tuple ``(nearest_before, nearest_after)`` where:
    - ``nearest_before`` is the closest trading day ``<= dt``
    - ``nearest_after`` is the closest trading day ``>= dt``

    If one side does not exist, that value is ``pd.NaT``.
    """

    target = to_naive(dt).normalize()
    dti = normalize_index(trading_index)

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
