"""Map earnings events to pre/post trading sessions.

The alignment is intentionally explicit so downstream logic can audit exactly how
an event was mapped to market sessions.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from src.utils.datetime_utils import assert_same_timezone, normalize_datetime_index, to_naive_datetime

_BEFORE_OPEN_ALIASES = {"before_open", "bmo", "pre_market", "pre-open", "pre_open"}
_AFTER_CLOSE_ALIASES = {"after_close", "amc", "post_market", "post-close", "post_close"}


def _normalize_timing(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _normalize_price_index(price_index: Any) -> pd.DatetimeIndex:
    if isinstance(price_index, pd.DataFrame):
        trading_index = price_index.index
    elif isinstance(price_index, pd.Series):
        trading_index = price_index.index
    elif isinstance(price_index, pd.DatetimeIndex):
        trading_index = price_index
    elif isinstance(price_index, Iterable) and not isinstance(price_index, (str, bytes)):
        trading_index = pd.DatetimeIndex(price_index)
    else:
        raise TypeError("price_index must be a DatetimeIndex, DataFrame/Series with a DatetimeIndex, or iterable of dates")

    trading_index = normalize_datetime_index(trading_index)
    return trading_index.dropna().sort_values().unique()


def align_event_to_sessions(event_date: Any, announcement_timing: Any, price_index: Any) -> dict[str, Any]:
    """Align an earnings event date/timing to pre/post trading sessions.

    Rules implemented:
    1) after_close  -> pre: same-day close, post: next trading day open/close
    2) before_open  -> pre: previous trading day close, post: same-day open/close
    3) unknown      -> flagged, no silent assignment
    """

    trading_days = _normalize_price_index(price_index)
    event_ts = to_naive_datetime(event_date)
    if event_ts is None:
        raise ValueError("event_date is None")
    event_day = event_ts.normalize()
    assert_same_timezone(trading_days, event_day)

    timing = _normalize_timing(announcement_timing)

    result: dict[str, Any] = {
        "pre_event_date": pd.NaT,
        "post_event_date": pd.NaT,
        "alignment_type": "unknown",
        "alignment_flag": "unmapped_timing",
    }

    if timing in _AFTER_CLOSE_ALIASES:
        same_day_pos = trading_days.searchsorted(event_day, side="left")
        has_same_day = same_day_pos < len(trading_days) and trading_days[same_day_pos] == event_day

        next_day_pos = trading_days.searchsorted(event_day, side="right")
        has_next_day = next_day_pos < len(trading_days)

        if has_same_day and has_next_day:
            result.update(
                {
                    "pre_event_date": trading_days[same_day_pos],
                    "post_event_date": trading_days[next_day_pos],
                    "alignment_type": "after_close_to_next_open_close",
                    "alignment_flag": "ok",
                }
            )
        else:
            result["alignment_type"] = "after_close_to_next_open_close"
            result["alignment_flag"] = "missing_trading_day_context"
        return result

    if timing in _BEFORE_OPEN_ALIASES:
        same_day_pos = trading_days.searchsorted(event_day, side="left")
        has_same_day = same_day_pos < len(trading_days) and trading_days[same_day_pos] == event_day

        prev_day_pos = same_day_pos - 1 if has_same_day else -1
        has_prev_day = prev_day_pos >= 0

        if has_prev_day and has_same_day:
            result.update(
                {
                    "pre_event_date": trading_days[prev_day_pos],
                    "post_event_date": trading_days[same_day_pos],
                    "alignment_type": "before_open_to_same_open_close",
                    "alignment_flag": "ok",
                }
            )
        else:
            result["alignment_type"] = "before_open_to_same_open_close"
            result["alignment_flag"] = "missing_trading_day_context"
        return result

    return result
