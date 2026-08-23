"""Validation and diagnostics utilities for earnings event pipelines.

This module is intentionally side-effect free and notebook-friendly. All functions are
pure transformations from input DataFrames to output DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

EVENT_INTEGRITY_FLAG_COLUMNS = [
    "missing_pre_close",
    "missing_post_open",
    "missing_post_close",
    "invalid_nonpositive_prices",
    "extreme_price_jump",
    "missing_implied_move",
    "implied_move_out_of_bounds",
]
TIMING_FLAG_COLUMNS = [
    "missing_or_unknown_timing",
    "inconsistent_timing_session_mapping",
]
OPTION_FLAG_COLUMNS = [
    "missing_options_data",
    "bid_ask_spread_invalid",
    "atm_strike_too_far",
    "missing_call_leg",
    "missing_put_leg",
    "zero_or_nan_mid",
    "missing_ticker_directory",
    "missing_eod_history",
    "missing_option_snapshot_directory",
    "missing_required_snapshot_file",
    "snapshot_unparseable",
    "no_usable_expiration",
    "no_valid_atm_pair",
]
OPTION_METADATA_COLUMNS = [
    "has_call_leg",
    "has_put_leg",
    "option_coverage_status",
    "historical_option_selection_success",
    "option_snapshot_quote_date",
    "option_snapshot_source_path",
    "option_selected_expiration",
    "option_selected_strike",
]


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
        return pd.DataFrame(
            columns=[
                event_key_col,
                "has_call_leg",
                "has_put_leg",
                "bid_ask_spread_invalid",
                "zero_or_nan_mid",
                "min_atm_strike_distance_pct",
                "atm_strike_too_far",
                "missing_call_leg",
                "missing_put_leg",
            ]
        )

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


def _coerce_bool(series: pd.Series | object, *, default: bool = False) -> pd.Series:
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    if series.empty:
        return pd.Series(dtype=bool)
    return series.astype("boolean").fillna(default).astype(bool)


def _default_options_flags(events_with_key: pd.DataFrame, event_key_col: str) -> pd.DataFrame:
    out = pd.DataFrame({event_key_col: events_with_key[event_key_col]})
    out["has_call_leg"] = False
    out["has_put_leg"] = False
    out["bid_ask_spread_invalid"] = False
    out["atm_strike_too_far"] = False
    out["missing_call_leg"] = False
    out["missing_put_leg"] = False
    out["zero_or_nan_mid"] = False
    out["missing_options_data"] = False
    out["missing_ticker_directory"] = False
    out["missing_eod_history"] = False
    out["missing_option_snapshot_directory"] = False
    out["missing_required_snapshot_file"] = False
    out["snapshot_unparseable"] = False
    out["no_usable_expiration"] = False
    out["no_valid_atm_pair"] = False
    out["option_coverage_status"] = "unknown"
    out["historical_option_selection_success"] = False
    out["option_snapshot_quote_date"] = pd.NaT
    out["option_snapshot_source_path"] = pd.NA
    out["option_selected_expiration"] = pd.NaT
    out["option_selected_strike"] = pd.NA
    return out


def _merge_explicit_option_event_flags(
    events_with_key: pd.DataFrame,
    *,
    event_key_col: str,
    options_event_flags_df: pd.DataFrame | None,
    options_df: pd.DataFrame | None,
    option_type_col: str,
    strike_col: str,
    underlying_price_col: str,
    bid_col: str,
    ask_col: str,
    mid_col: str,
    config: DiagnosticsConfig,
) -> pd.DataFrame:
    out = _default_options_flags(events_with_key, event_key_col)

    raw_agg = _aggregate_options_flags(
        options_df if options_df is not None else pd.DataFrame(columns=[event_key_col]),
        event_key_col=event_key_col,
        option_type_col=option_type_col,
        strike_col=strike_col,
        underlying_price_col=underlying_price_col,
        bid_col=bid_col,
        ask_col=ask_col,
        mid_col=mid_col,
        config=config,
    )
    raw_merged = out[[event_key_col]].merge(raw_agg, on=event_key_col, how="left")

    explicit = pd.DataFrame(columns=[event_key_col])
    if options_event_flags_df is not None and not options_event_flags_df.empty:
        explicit = options_event_flags_df.copy()
        if event_key_col not in explicit.columns:
            raise ValueError(
                f"options_event_flags_df must include the event key column '{event_key_col}'."
            )
        explicit = out[[event_key_col]].merge(explicit, on=event_key_col, how="left")
    else:
        explicit = out[[event_key_col]].copy()

    for column in ["has_call_leg", "has_put_leg"]:
        raw_values = raw_merged[column] if column in raw_merged.columns else pd.Series(index=out.index, dtype=object)
        explicit_values = explicit[column] if column in explicit.columns else pd.Series(index=out.index, dtype=object)
        combined = raw_values.combine_first(explicit_values).fillna(False)
        out[column] = _coerce_bool(combined, default=False)

    for column in ["bid_ask_spread_invalid", "atm_strike_too_far", "missing_call_leg", "missing_put_leg", "zero_or_nan_mid"]:
        raw_values = raw_merged[column] if column in raw_merged.columns else pd.Series(index=out.index, dtype=object)
        explicit_values = explicit[column] if column in explicit.columns else pd.Series(index=out.index, dtype=object)
        combined = raw_values.combine_first(explicit_values).fillna(False)
        out[column] = _coerce_bool(combined, default=False)

    for column in [
        "missing_options_data",
        "missing_ticker_directory",
        "missing_eod_history",
        "missing_option_snapshot_directory",
        "missing_required_snapshot_file",
        "snapshot_unparseable",
        "no_usable_expiration",
        "no_valid_atm_pair",
        "historical_option_selection_success",
    ]:
        if column in explicit.columns:
            out[column] = _coerce_bool(explicit[column], default=False)

    if "option_coverage_status" in explicit.columns:
        out["option_coverage_status"] = explicit["option_coverage_status"].astype("string").fillna("unknown")

    for column in ["option_snapshot_quote_date", "option_selected_expiration"]:
        if column in explicit.columns:
            out[column] = pd.to_datetime(explicit[column], errors="coerce")

    for column in ["option_snapshot_source_path", "option_selected_strike"]:
        if column in explicit.columns:
            out[column] = explicit[column]

    if options_event_flags_df is None:
        missing_options_mask = raw_merged["has_call_leg"].isna() & raw_merged["has_put_leg"].isna()
        out["missing_options_data"] = _coerce_bool(missing_options_mask, default=False)
        out.loc[out["missing_options_data"], "option_coverage_status"] = "no_option_rows_provided"
        out.loc[
            (~out["missing_options_data"]) & out["has_call_leg"] & out["has_put_leg"],
            "option_coverage_status",
        ] = "successful_selection"
    else:
        raw_missing_mask = raw_merged["has_call_leg"].isna() & raw_merged["has_put_leg"].isna()
        out.loc[raw_missing_mask & (out["option_coverage_status"] == "unknown"), "missing_options_data"] = True

    return out


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
    options_event_flags_df: pd.DataFrame | None = None,
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

    if options_df is None and options_event_flags_df is None:
        options_flags = _default_options_flags(events_with_key, resolved_event_key)
        options_flags["missing_options_data"] = True
        options_flags["option_coverage_status"] = "no_option_rows_provided"
    else:
        options_flags = _merge_explicit_option_event_flags(
            events_with_key,
            event_key_col=resolved_event_key,
            options_event_flags_df=options_event_flags_df,
            options_df=options_df,
            option_type_col=option_type_col,
            strike_col=strike_col,
            underlying_price_col=underlying_price_col,
            bid_col=bid_col,
            ask_col=ask_col,
            mid_col=mid_col,
            config=config,
        )

    flags_df = pd.concat(
        [
            events_with_key[[resolved_event_key]].reset_index(drop=True),
            event_flags.reset_index(drop=True),
            timing_flags.reset_index(drop=True),
            options_flags.drop(columns=[resolved_event_key]).reset_index(drop=True),
        ],
        axis=1,
    )

    flag_columns = [
        *EVENT_INTEGRITY_FLAG_COLUMNS,
        *TIMING_FLAG_COLUMNS,
        *[column for column in OPTION_FLAG_COLUMNS if column in flags_df.columns],
    ]
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

    if "option_coverage_status" in options_flags.columns:
        status_counts = (
            options_flags["option_coverage_status"]
            .astype("string")
            .fillna("unknown")
            .value_counts(dropna=False)
            .sort_index()
        )
        for status, count in status_counts.items():
            summary_rows.append(
                {
                    "category": "option_coverage_status",
                    "metric": str(status),
                    "count": int(count),
                    "pct_of_events": int(count) / total_events if total_events else 0.0,
                }
            )

    diagnostics_summary_df = pd.DataFrame(summary_rows)

    return clean_events_df, diagnostics_summary_df, flags_df


__all__: Iterable[str] = ["DiagnosticsConfig", "run_event_diagnostics"]
