"""Event-level data quality scoring for earnings-event datasets."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

_KNOWN_TIMING_VALUES = {
    "bmo",
    "pre_market",
    "before_open",
    "amc",
    "post_market",
    "after_close",
    "intraday",
    "during_market",
}

_SAME_DAY_TIMING = {"bmo", "pre_market", "before_open", "intraday", "during_market"}
_NEXT_DAY_TIMING = {"amc", "post_market", "after_close"}


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    as_str = series.astype(str).str.strip().str.lower()
    truthy = as_str.isin({"1", "true", "t", "yes", "y"})
    falsy = as_str.isin({"0", "false", "f", "no", "n", "nan", "none", ""})

    out = pd.Series(False, index=series.index)
    out.loc[truthy] = True

    remainder = ~truthy & ~falsy
    non_missing_remainder = remainder & series.notna()
    out.loc[non_missing_remainder] = series.loc[non_missing_remainder].astype(bool)
    return out


def _resolve_event_key(events_df: pd.DataFrame, event_key_col: str) -> tuple[pd.DataFrame, str]:
    if event_key_col in events_df.columns:
        return events_df.copy(), event_key_col

    working = events_df.copy().reset_index(drop=False)
    synthetic_key = "__event_row_id"
    working[synthetic_key] = np.arange(len(working), dtype=int)
    return working, synthetic_key


def _event_price_score(
    events_df: pd.DataFrame,
    *,
    pre_close_col: str,
    post_open_col: str,
    post_close_col: str,
) -> tuple[pd.Series, pd.DataFrame]:
    pre_close = pd.to_numeric(events_df.get(pre_close_col), errors="coerce")
    post_open = pd.to_numeric(events_df.get(post_open_col), errors="coerce")
    post_close = pd.to_numeric(events_df.get(post_close_col), errors="coerce")

    pre_valid = pre_close.notna() & (pre_close > 0)
    open_valid = post_open.notna() & (post_open > 0)
    close_valid = post_close.notna() & (post_close > 0)

    component = pd.DataFrame(index=events_df.index)
    component["price_component_pre_close"] = pre_valid.astype(float) * 10.0
    component["price_component_post_open"] = open_valid.astype(float) * 10.0
    component["price_component_post_close"] = close_valid.astype(float) * 10.0

    return component.sum(axis=1), component


def _event_options_score(
    events_df: pd.DataFrame,
    options_df: pd.DataFrame | None,
    *,
    event_key_col: str,
    options_valid_col: str,
) -> pd.Series:
    if options_valid_col in events_df.columns:
        return _coerce_bool(events_df[options_valid_col]).astype(float) * 30.0

    if "missing_options_data" in events_df.columns:
        return (~_coerce_bool(events_df["missing_options_data"]).astype(bool)).astype(float) * 30.0

    if "has_options_data" in events_df.columns:
        return _coerce_bool(events_df["has_options_data"]).astype(float) * 30.0

    if options_df is None or options_df.empty or event_key_col not in options_df.columns:
        return pd.Series(0.0, index=events_df.index)

    has_options = events_df[event_key_col].isin(options_df[event_key_col])
    return has_options.astype(float) * 30.0


def _event_timing_score(
    events_df: pd.DataFrame,
    *,
    timing_col: str,
    mapped_session_col: str,
    timing_valid_col: str,
) -> tuple[pd.Series, pd.DataFrame]:
    if timing_valid_col in events_df.columns:
        timing_valid = _coerce_bool(events_df[timing_valid_col])
        component = pd.DataFrame(
            {
                "timing_component_known_timing": timing_valid.astype(float) * 10.0,
                "timing_component_session_mapping": timing_valid.astype(float) * 10.0,
            },
            index=events_df.index,
        )
        return component.sum(axis=1), component

    timing = events_df.get(timing_col, pd.Series(index=events_df.index, dtype=object)).map(_normalize_text)
    mapped = events_df.get(mapped_session_col, pd.Series(index=events_df.index, dtype=object)).map(_normalize_text)
    mapped = mapped.replace({"next_session": "next_day", "same_session": "same_day"})

    known_timing = timing.isin(_KNOWN_TIMING_VALUES)

    expected_session = pd.Series("", index=events_df.index, dtype=object)
    expected_session.loc[timing.isin(_SAME_DAY_TIMING)] = "same_day"
    expected_session.loc[timing.isin(_NEXT_DAY_TIMING)] = "next_day"

    mapping_known = mapped != ""
    mapping_consistent = (~expected_session.eq("")) & mapping_known & expected_session.eq(mapped)

    component = pd.DataFrame(index=events_df.index)
    component["timing_component_known_timing"] = known_timing.astype(float) * 10.0
    component["timing_component_session_mapping"] = mapping_consistent.astype(float) * 10.0

    return component.sum(axis=1), component


def _event_spread_score(
    events_df: pd.DataFrame,
    options_df: pd.DataFrame | None,
    *,
    event_key_col: str,
    spread_sane_col: str,
    bid_col: str,
    ask_col: str,
    mid_col: str,
    max_spread_pct: float,
) -> pd.Series:
    if spread_sane_col in events_df.columns:
        return _coerce_bool(events_df[spread_sane_col]).astype(float) * 20.0

    if options_df is None or options_df.empty or event_key_col not in options_df.columns:
        return pd.Series(0.0, index=events_df.index)

    working = options_df.copy()
    bid = pd.to_numeric(working.get(bid_col), errors="coerce")
    ask = pd.to_numeric(working.get(ask_col), errors="coerce")

    if mid_col in working.columns:
        mid = pd.to_numeric(working[mid_col], errors="coerce")
    else:
        mid = (bid + ask) / 2.0

    spread = ask - bid
    spread_pct = spread / mid.replace(0, np.nan)

    row_is_sane = (
        (~bid.isna())
        & (~ask.isna())
        & (~mid.isna())
        & (bid >= 0)
        & (ask >= 0)
        & (ask >= bid)
        & (mid > 0)
        & (spread_pct >= 0)
        & (spread_pct <= max_spread_pct)
    )

    event_sanity = (
        pd.DataFrame({event_key_col: working[event_key_col], "row_is_sane": row_is_sane.astype(float)})
        .groupby(event_key_col, dropna=False)["row_is_sane"]
        .mean()
    )

    event_level = events_df[[event_key_col]].join(event_sanity, on=event_key_col)
    return event_level["row_is_sane"].fillna(0.0).clip(0.0, 1.0) * 20.0


def score_event_data_quality(
    events_df: pd.DataFrame,
    options_df: pd.DataFrame | None = None,
    *,
    event_key_col: str = "event_id",
    pre_close_col: str = "pre_close",
    post_open_col: str = "post_open",
    post_close_col: str = "post_close",
    options_valid_col: str = "valid_options_data",
    timing_col: str = "announcement_timing",
    mapped_session_col: str = "mapped_session",
    timing_valid_col: str = "valid_timing_classification",
    spread_sane_col: str = "bid_ask_spread_sane",
    bid_col: str = "bid",
    ask_col: str = "ask",
    mid_col: str = "mid",
    max_spread_pct: float = 1.00,
) -> pd.DataFrame:
    """Add event-level quality scores and quality buckets.

    Score composition (0-100):
    - price completeness: 30 points
    - options availability: 30 points
    - timing clarity: 20 points
    - spread sanity: 20 points

    Buckets:
    - ``high``: score >= 80
    - ``medium``: 50 <= score < 80
    - ``low``: score < 50
    """
    if events_df is None:
        raise ValueError("events_df must be provided.")

    events_with_key, resolved_event_key = _resolve_event_key(events_df, event_key_col)

    price_score, price_components = _event_price_score(
        events_with_key,
        pre_close_col=pre_close_col,
        post_open_col=post_open_col,
        post_close_col=post_close_col,
    )

    options_score = _event_options_score(
        events_with_key,
        options_df,
        event_key_col=resolved_event_key,
        options_valid_col=options_valid_col,
    )

    timing_score, timing_components = _event_timing_score(
        events_with_key,
        timing_col=timing_col,
        mapped_session_col=mapped_session_col,
        timing_valid_col=timing_valid_col,
    )

    spread_score = _event_spread_score(
        events_with_key,
        options_df,
        event_key_col=resolved_event_key,
        spread_sane_col=spread_sane_col,
        bid_col=bid_col,
        ask_col=ask_col,
        mid_col=mid_col,
        max_spread_pct=max_spread_pct,
    )

    scored = events_with_key.copy()
    scored["quality_price_completeness"] = price_score
    scored["quality_options_availability"] = options_score
    scored["quality_timing_clarity"] = timing_score
    scored["quality_spread_sanity"] = spread_score

    scored["event_quality_score"] = (
        scored[
            [
                "quality_price_completeness",
                "quality_options_availability",
                "quality_timing_clarity",
                "quality_spread_sanity",
            ]
        ]
        .sum(axis=1)
        .clip(0.0, 100.0)
        .round(2)
    )

    scored["bucket"] = pd.cut(
        scored["event_quality_score"],
        bins=[-np.inf, 50.0, 80.0, np.inf],
        labels=["low", "medium", "high"],
        right=False,
    ).astype(str)

    scored = scored.join(price_components).join(timing_components)

    if resolved_event_key == "__event_row_id":
        scored = scored.drop(columns=["__event_row_id"])

    return scored


__all__: Iterable[str] = ["score_event_data_quality"]
