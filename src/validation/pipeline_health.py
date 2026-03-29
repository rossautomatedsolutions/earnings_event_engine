"""Pipeline health scoring utilities.

This module provides a compact, high-signal health check for event + feature
pipeline outputs so reliability can be tracked over time.
"""

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


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _first_present(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _safe_pct(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _missing_timing_pct(events_df: pd.DataFrame) -> tuple[float, pd.Series]:
    timing_col = _first_present(events_df, ["announcement_timing", "timing", "earnings_timing"])
    if timing_col is None:
        missing = pd.Series(True, index=events_df.index)
    else:
        normalized = events_df[timing_col].map(_normalize_text)
        missing = normalized.eq("") | ~normalized.isin(_KNOWN_TIMING_VALUES)

    return _safe_pct(float(missing.sum()), float(len(events_df))), missing


def _missing_pre_post_pct(events_df: pd.DataFrame, features_df: pd.DataFrame) -> float:
    source = events_df
    pre_col = _first_present(source, ["pre_close", "pre_price", "pre_market_close"])
    post_col = _first_present(source, ["post_open", "post_close", "post_price", "next_open"])

    if pre_col is None or post_col is None:
        source = features_df
        pre_col = _first_present(source, ["pre_close", "pre_price", "pre_market_close"])
        post_col = _first_present(source, ["post_open", "post_close", "post_price", "next_open"])

    if pre_col is None or post_col is None:
        return 1.0

    pre = pd.to_numeric(source[pre_col], errors="coerce")
    post = pd.to_numeric(source[post_col], errors="coerce")

    missing = pre.isna() | post.isna() | (pre <= 0) | (post <= 0)
    return _safe_pct(float(missing.sum()), float(len(source)))


def _missing_implied_move_pct(events_df: pd.DataFrame, features_df: pd.DataFrame) -> float:
    implied_col = _first_present(
        features_df,
        ["implied_move", "implied_move_pct", "atm_implied_move", "expected_move"],
    )
    source = features_df

    if implied_col is None:
        implied_col = _first_present(
            events_df,
            ["implied_move", "implied_move_pct", "atm_implied_move", "expected_move"],
        )
        source = events_df

    if implied_col is None:
        return 1.0

    implied = pd.to_numeric(source[implied_col], errors="coerce")
    missing = implied.isna() | ~np.isfinite(implied)
    return _safe_pct(float(missing.sum()), float(len(source)))


def _invalid_rows_pct(events_df: pd.DataFrame) -> tuple[float, pd.Series]:
    invalid = pd.Series(False, index=events_df.index)

    ticker_col = _first_present(events_df, ["ticker", "symbol"])
    if ticker_col is not None:
        invalid = invalid | events_df[ticker_col].isna() | events_df[ticker_col].astype(str).str.strip().eq("")

    date_col = _first_present(events_df, ["event_date", "earnings_date", "date"])
    if date_col is not None:
        invalid = invalid | pd.to_datetime(events_df[date_col], errors="coerce").isna()

    event_id_col = _first_present(events_df, ["event_id", "id"])
    if event_id_col is not None:
        invalid = invalid | events_df[event_id_col].duplicated(keep=False)

    for col in ["pre_close", "post_open", "post_close"]:
        if col in events_df.columns:
            values = pd.to_numeric(events_df[col], errors="coerce")
            invalid = invalid | ((~values.isna()) & (values <= 0))

    return _safe_pct(float(invalid.sum()), float(len(events_df))), invalid


def _extreme_values_pct(features_df: pd.DataFrame) -> tuple[float, pd.Series]:
    if features_df.empty:
        return 0.0, pd.Series(dtype=bool)

    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return 0.0, pd.Series(False, index=features_df.index)

    extreme = pd.Series(False, index=features_df.index)

    for col in numeric_cols:
        values = pd.to_numeric(features_df[col], errors="coerce")
        finite = values[np.isfinite(values)]
        if finite.empty:
            continue

        mean = finite.mean()
        std = finite.std(ddof=0)
        if std == 0 or np.isnan(std):
            continue

        z = (values - mean) / std
        extreme = extreme | z.abs().gt(6)

    return _safe_pct(float(extreme.sum()), float(len(features_df))), extreme


def compute_pipeline_health(events_df: pd.DataFrame, features_df: pd.DataFrame) -> tuple[dict[str, float], float, pd.DataFrame]:
    """Compute a compact reliability report for the event pipeline.

    Returns
    -------
    tuple
        (health_summary, health_score, breakdown_table)
    """

    total_events = int(len(events_df))

    missing_timing_pct, missing_timing_mask = _missing_timing_pct(events_df)
    invalid_rows_pct, invalid_rows_mask = _invalid_rows_pct(events_df)
    missing_pre_post_pct = _missing_pre_post_pct(events_df, features_df)
    missing_implied_move_pct = _missing_implied_move_pct(events_df, features_df)
    extreme_values_pct, _ = _extreme_values_pct(features_df)

    valid_events = int((~missing_timing_mask & ~invalid_rows_mask).sum()) if total_events else 0

    event_validity_score = _safe_pct(valid_events, total_events)
    price_coverage_score = 1.0 - missing_pre_post_pct
    options_coverage_score = 1.0 - missing_implied_move_pct
    data_quality_score = 1.0 - ((extreme_values_pct + invalid_rows_pct) / 2.0)

    weights = {
        "event_coverage": 0.40,
        "price_coverage": 0.25,
        "options_coverage": 0.20,
        "data_quality": 0.15,
    }

    health_score = 100.0 * (
        weights["event_coverage"] * event_validity_score
        + weights["price_coverage"] * price_coverage_score
        + weights["options_coverage"] * options_coverage_score
        + weights["data_quality"] * data_quality_score
    )
    health_score = float(np.clip(health_score, 0.0, 100.0))

    health_summary = {
        "total_events": total_events,
        "valid_events": valid_events,
        "missing_timing_pct": missing_timing_pct,
        "missing_pre_post_pct": missing_pre_post_pct,
        "missing_implied_move_pct": missing_implied_move_pct,
        "extreme_values_pct": extreme_values_pct,
        "invalid_rows_pct": invalid_rows_pct,
        "health_score": health_score,
    }

    breakdown_table = pd.DataFrame(
        [
            {
                "category": "event_coverage",
                "metric": "valid_events_pct",
                "value": event_validity_score,
                "weight": weights["event_coverage"],
                "weighted_points": 100.0 * event_validity_score * weights["event_coverage"],
            },
            {
                "category": "event_coverage",
                "metric": "missing_timing_pct",
                "value": missing_timing_pct,
                "weight": 0.0,
                "weighted_points": 0.0,
            },
            {
                "category": "price_coverage",
                "metric": "missing_pre_post_pct",
                "value": missing_pre_post_pct,
                "weight": weights["price_coverage"],
                "weighted_points": 100.0 * price_coverage_score * weights["price_coverage"],
            },
            {
                "category": "options_coverage",
                "metric": "missing_implied_move_pct",
                "value": missing_implied_move_pct,
                "weight": weights["options_coverage"],
                "weighted_points": 100.0 * options_coverage_score * weights["options_coverage"],
            },
            {
                "category": "data_quality",
                "metric": "extreme_values_pct",
                "value": extreme_values_pct,
                "weight": 0.0,
                "weighted_points": 0.0,
            },
            {
                "category": "data_quality",
                "metric": "invalid_rows_pct",
                "value": invalid_rows_pct,
                "weight": 0.0,
                "weighted_points": 0.0,
            },
            {
                "category": "data_quality",
                "metric": "data_quality_component",
                "value": data_quality_score,
                "weight": weights["data_quality"],
                "weighted_points": 100.0 * data_quality_score * weights["data_quality"],
            },
        ]
    )

    return health_summary, health_score, breakdown_table
