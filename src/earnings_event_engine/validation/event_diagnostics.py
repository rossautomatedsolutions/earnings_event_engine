"""Validation and diagnostics utilities for earnings event pipelines.

This module is intentionally side-effect free and notebook-friendly. All functions are
pure transformations from input DataFrames to output DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Configuration thresholds used by diagnostics checks."""

    max_extreme_jump_pct: float = 0.50
    max_implied_move_pct: float = 0.30
    max_bid_ask_spread_pct: float = 1.00
    max_atm_strike_distance_pct: float = 0.05


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _ensure_event_key(df: pd.DataFrame, event_key_col: str) -> tuple[pd.DataFrame, str]:
    if event_key_col in df.columns:
        return df.copy(), event_key_col

    working = df.copy().reset_index(drop=False)
    synthetic_key = "__event_row_id"
    working[synthetic_key] = np.arange(len(working), dtype=int)
    return working, synthetic_key


def _validate_event_integrity(
    events_df: pd.DataFrame,
    *,
    pre_close_col: str,
    post_open_col: str,
    post_close_col: str,
    implied_move_col: str,
    config: DiagnosticsConfig,
) -> pd.DataFrame:
    out = pd.DataFrame(index=events_df.index)

    pre_close = pd.to_numeric(events_df.get(pre_close_col), errors="coerce")
    post_open = pd.to_numeric(events_df.get(post_open_col), errors="coerce")
    post_close = pd.to_numeric(events_df.get(post_close_col), errors="coerce")
    implied_move = pd.to_numeric(events_df.get(implied_move_col), errors="coerce")

    out["missing_pre_close"] = pre_close.isna()
    out["missing_post_open"] = post_open.isna()
    out["missing_post_close"] = post_close.isna()

    out["invalid_nonpositive_prices"] = (
        (~pre_close.isna() & (pre_close <= 0))
        | (~post_open.isna() & (post_open <= 0))
        | (~post_close.isna() & (post_close <= 0))
    )

    open_jump = (post_open / pre_close - 1.0).abs()
    close_jump = (post_close / pre_close - 1.0).abs()
    out["extreme_price_jump"] = (
        (~open_jump.isna() & (open_jump > config.max_extreme_jump_pct))
        | (~close_jump.isna() & (close_jump > config.max_extreme_jump_pct))
    )

    out["missing_implied_move"] = implied_move.isna()
    out["implied_move_out_of_bounds"] = (~implied_move.isna()) & (
        (implied_move < 0) | (implied_move.abs() > config.max_implied_move_pct)
    )

    return out


def _aggregate_options_flags(
    options_df: pd.DataFrame,
    *,
    event_key_col: str,
    option_type_col: str,
    strike_col: str,
    underlying_price_col: str,
    bid_col: str,
    ask_col: str,
    mid_col: str,
    config: DiagnosticsConfig,
) -> pd.DataFrame:
    if options_df.empty:
        return pd.DataFrame(columns=[event_key_col])

    working = options_df.copy()

    bid = pd.to_numeric(working.get(bid_col), errors="coerce")
    ask = pd.to_numeric(working.get(ask_col), errors="coerce")
    strike = pd.to_numeric(working.get(strike_col), errors="coerce")
    underlying = pd.to_numeric(working.get(underlying_price_col), errors="coerce")

    if mid_col in working.columns:
        mid = pd.to_numeric(working[mid_col], errors="coerce")
    else:
        mid = (bid + ask) / 2.0

    spread = ask - bid
    spread_pct = spread / mid.replace(0, np.nan)

    option_type = working.get(option_type_col, pd.Series(index=working.index, dtype=object)).map(
        _normalize_text
    )
    is_call = option_type.isin({"call", "c"})
    is_put = option_type.isin({"put", "p"})

    strike_distance_pct = ((strike - underlying).abs() / underlying.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )

    row_flags = pd.DataFrame(
        {
            event_key_col: working[event_key_col],
            "_is_call": is_call,
            "_is_put": is_put,
            "_bad_spread": (
                (~bid.isna() & (bid < 0))
                | (~ask.isna() & (ask < 0))
                | ((~bid.isna()) & (~ask.isna()) & (ask < bid))
                | (~spread_pct.isna() & (spread_pct > config.max_bid_ask_spread_pct))
            ),
            "_bad_mid": mid.isna() | (mid <= 0),
            "_strike_distance_pct": strike_distance_pct,
        }
    )

    grouped = row_flags.groupby(event_key_col, dropna=False)
    result = grouped.agg(
        has_call_leg=("_is_call", "max"),
        has_put_leg=("_is_put", "max"),
        bid_ask_spread_invalid=("_bad_spread", "max"),
        zero_or_nan_mid=("_bad_mid", "max"),
        min_atm_strike_distance_pct=("_strike_distance_pct", "min"),
    )

    result["atm_strike_too_far"] = (
        result["min_atm_strike_distance_pct"].isna()
        | (result["min_atm_strike_distance_pct"] > config.max_atm_strike_distance_pct)
    )
    result["missing_call_leg"] = ~result["has_call_leg"].astype(bool)
    result["missing_put_leg"] = ~result["has_put_leg"].astype(bool)

    return result.reset_index()


def _validate_timing(
    events_df: pd.DataFrame,
    *,
    timing_col: str,
    mapped_session_col: str,
) -> pd.DataFrame:
    out = pd.DataFrame(index=events_df.index)

    timing = events_df.get(timing_col, pd.Series(index=events_df.index, dtype=object)).map(
        _normalize_text
    )
    mapped = events_df.get(mapped_session_col, pd.Series(index=events_df.index, dtype=object)).map(
        _normalize_text
    )

    known_timing = {
        "bmo",
        "pre_market",
        "before_open",
        "amc",
        "post_market",
        "after_close",
        "intraday",
        "during_market",
    }
    out["missing_or_unknown_timing"] = (timing == "") | (~timing.isin(known_timing))

    expected_map = {
        "bmo": "same_day",
        "pre_market": "same_day",
        "before_open": "same_day",
        "intraday": "same_day",
        "during_market": "same_day",
        "amc": "next_day",
        "post_market": "next_day",
        "after_close": "next_day",
    }

    mapped_norm = mapped.replace({"next_session": "next_day", "same_session": "same_day"})
    expected = timing.map(expected_map)

    out["inconsistent_timing_session_mapping"] = (
        (~expected.isna()) & (mapped_norm != "") & (mapped_norm != expected)
    )

    return out


def run_event_diagnostics(
    events_df: pd.DataFrame,
    options_df: pd.DataFrame | None = None,
    *,
    event_key_col: str = "event_id",
    pre_close_col: str = "pre_close",
    post_open_col: str = "post_open",
    post_close_col: str = "post_close",
    implied_move_col: str = "implied_move",
    timing_col: str = "announcement_timing",
    mapped_session_col: str = "mapped_session",
    option_type_col: str = "option_type",
    strike_col: str = "strike",
    underlying_price_col: str = "underlying_price",
    bid_col: str = "bid",
    ask_col: str = "ask",
    mid_col: str = "mid",
    config: DiagnosticsConfig = DiagnosticsConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate earnings events and return clean rows plus diagnostics outputs.

    Returns
    -------
    clean_events_df
        Subset of ``events_df`` with only rows that pass all diagnostics.
    diagnostics_summary_df
        Metrics and reason-level breakdown with counts and percentages.
    diagnostics_flags_df
        Row-level boolean flags and a compact failure-reasons column.
    """
    if events_df is None:
        raise ValueError("events_df must be provided.")

    events_with_key, resolved_event_key = _ensure_event_key(events_df, event_key_col)

    event_flags = _validate_event_integrity(
        events_with_key,
        pre_close_col=pre_close_col,
        post_open_col=post_open_col,
        post_close_col=post_close_col,
        implied_move_col=implied_move_col,
        config=config,
    )

    timing_flags = _validate_timing(
        events_with_key,
        timing_col=timing_col,
        mapped_session_col=mapped_session_col,
    )

    if options_df is None:
        options_flags = pd.DataFrame(
            {
                resolved_event_key: events_with_key[resolved_event_key],
                "missing_options_data": True,
                "bid_ask_spread_invalid": False,
                "atm_strike_too_far": False,
                "missing_call_leg": False,
                "missing_put_leg": False,
                "zero_or_nan_mid": False,
            }
        )
    else:
        option_agg = _aggregate_options_flags(
            options_df,
            event_key_col=resolved_event_key,
            option_type_col=option_type_col,
            strike_col=strike_col,
            underlying_price_col=underlying_price_col,
            bid_col=bid_col,
            ask_col=ask_col,
            mid_col=mid_col,
            config=config,
        )
        options_flags = events_with_key[[resolved_event_key]].merge(option_agg, on=resolved_event_key, how="left")
        options_flags["missing_options_data"] = options_flags["has_call_leg"].isna() & options_flags[
            "has_put_leg"
        ].isna()

        for col in [
            "bid_ask_spread_invalid",
            "atm_strike_too_far",
            "missing_call_leg",
            "missing_put_leg",
            "zero_or_nan_mid",
        ]:
            options_flags[col] = options_flags[col].eq(True)

        options_flags = options_flags[
            [
                resolved_event_key,
                "missing_options_data",
                "bid_ask_spread_invalid",
                "atm_strike_too_far",
                "missing_call_leg",
                "missing_put_leg",
                "zero_or_nan_mid",
            ]
        ]

    flags_df = pd.concat(
        [
            events_with_key[[resolved_event_key]].reset_index(drop=True),
            event_flags.reset_index(drop=True),
            timing_flags.reset_index(drop=True),
            options_flags.drop(columns=[resolved_event_key]).reset_index(drop=True),
        ],
        axis=1,
    )

    flag_columns = [col for col in flags_df.columns if col != resolved_event_key]
    flags_df["invalid_event"] = flags_df[flag_columns].any(axis=1)
    flags_df["valid_event"] = ~flags_df["invalid_event"]

    flags_df["failure_reasons"] = flags_df[flag_columns].apply(
        lambda row: ",".join(row.index[row.values].tolist()), axis=1
    )

    clean_mask = flags_df["valid_event"].to_numpy()
    clean_events_df = events_df.loc[clean_mask].copy()

    total_events = len(flags_df)
    valid_events = int(flags_df["valid_event"].sum())
    invalid_events = total_events - valid_events

    summary_rows: list[dict] = [
        {"category": "overview", "metric": "total_events", "count": total_events, "pct_of_events": 1.0 if total_events else 0.0},
        {
            "category": "overview",
            "metric": "valid_events",
            "count": valid_events,
            "pct_of_events": valid_events / total_events if total_events else 0.0,
        },
        {
            "category": "overview",
            "metric": "invalid_events",
            "count": invalid_events,
            "pct_of_events": invalid_events / total_events if total_events else 0.0,
        },
    ]

    for reason in flag_columns:
        reason_count = int(flags_df[reason].sum())
        summary_rows.append(
            {
                "category": "failure_breakdown",
                "metric": reason,
                "count": reason_count,
                "pct_of_events": reason_count / total_events if total_events else 0.0,
            }
        )

    missing_options_count = int(flags_df["missing_options_data"].sum())
    missing_timing_count = int(flags_df["missing_or_unknown_timing"].sum())
    summary_rows.extend(
        [
            {
                "category": "coverage",
                "metric": "pct_missing_options",
                "count": missing_options_count,
                "pct_of_events": missing_options_count / total_events if total_events else 0.0,
            },
            {
                "category": "coverage",
                "metric": "pct_missing_timing",
                "count": missing_timing_count,
                "pct_of_events": missing_timing_count / total_events if total_events else 0.0,
            },
        ]
    )

    diagnostics_summary_df = pd.DataFrame(summary_rows)

    return clean_events_df, diagnostics_summary_df, flags_df


__all__: Iterable[str] = ["DiagnosticsConfig", "run_event_diagnostics"]
