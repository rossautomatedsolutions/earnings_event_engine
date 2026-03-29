"""Analyze implied vs realized earnings moves.

This module answers whether options appear overpriced into earnings by comparing
implied moves to realized post-earnings moves at the overall and cohort level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_IMPLIED_COL_CANDIDATES = ("implied_move", "implied_move_pct", "expected_move")
_REALIZED_COL_CANDIDATES = ("realized_move", "realized_move_pct", "actual_move")
_TICKER_COL_CANDIDATES = ("ticker", "symbol")
_TIMING_COL_CANDIDATES = ("announcement_timing", "earnings_time", "timing")


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find a {label} column. Tried: {list(candidates)}")


def _normalize_timing(value: object) -> str:
    if pd.isna(value):
        return "unknown"

    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"amc", "after_close", "post_market", "after_market", "postclose"}:
        return "after_close"
    if normalized in {"bmo", "before_open", "pre_market", "preopen"}:
        return "before_open"
    if normalized in {"during_market", "intraday", "market_hours"}:
        return "during_market"
    return "unknown"


def _implied_move_buckets(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series("missing", index=series.index, dtype="object")

    labels = ["q1_low", "q2_mid_low", "q3_mid_high", "q4_high"]
    try:
        bucketed = pd.qcut(numeric, q=4, labels=labels, duplicates="drop")
    except ValueError:
        ranked = numeric.rank(method="first")
        try:
            bucketed = pd.qcut(ranked, q=4, labels=labels, duplicates="drop")
        except ValueError:
            return pd.Series("missing", index=series.index, dtype="object")

    return bucketed.astype("object").fillna("missing")


def _cohort_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        df.groupby(group_col, dropna=False)
        .agg(
            event_count=("difference_realized_minus_implied", "size"),
            avg_implied_move=("implied_move_numeric", "mean"),
            avg_realized_move=("realized_move_numeric", "mean"),
            avg_difference=("difference_realized_minus_implied", "mean"),
            pct_realized_gt_implied=("realized_gt_implied", "mean"),
            pct_implied_gt_realized=("implied_gt_realized", "mean"),
        )
        .reset_index()
        .sort_values(group_col)
        .reset_index(drop=True)
    )


def analyze_iv_vs_realized(event_features_df: pd.DataFrame) -> dict[str, pd.DataFrame | str | dict[str, pd.DataFrame]]:
    """Compare implied and realized earnings moves.

    Parameters
    ----------
    event_features_df
        DataFrame with one row per earnings event.

    Returns
    -------
    dict
        Contains summary tables and ready-to-plot DataFrames:
        - overall_summary
        - distribution_summary
        - by_ticker
        - by_implied_move_bucket
        - by_announcement_timing
        - plot_data (dict of the three cohort tables)
        - conclusion (short textual answer)
    """
    if event_features_df is None or event_features_df.empty:
        raise ValueError("event_features_df must be a non-empty DataFrame.")

    implied_col = _pick_column(event_features_df, _IMPLIED_COL_CANDIDATES, "implied move")
    realized_col = _pick_column(event_features_df, _REALIZED_COL_CANDIDATES, "realized move")
    ticker_col = _pick_column(event_features_df, _TICKER_COL_CANDIDATES, "ticker")
    timing_col = _pick_column(event_features_df, _TIMING_COL_CANDIDATES, "announcement timing")

    working = event_features_df.copy()
    working["implied_move_numeric"] = pd.to_numeric(working[implied_col], errors="coerce")
    working["realized_move_numeric"] = pd.to_numeric(working[realized_col], errors="coerce")

    valid = working["implied_move_numeric"].notna() & working["realized_move_numeric"].notna()
    working = working.loc[valid].copy()
    if working.empty:
        raise ValueError("No rows contain both implied and realized move values.")

    working["difference_realized_minus_implied"] = (
        working["realized_move_numeric"] - working["implied_move_numeric"]
    )
    working["realized_gt_implied"] = (
        working["realized_move_numeric"] > working["implied_move_numeric"]
    ).astype(float)
    working["implied_gt_realized"] = (
        working["implied_move_numeric"] > working["realized_move_numeric"]
    ).astype(float)

    working["ticker_group"] = working[ticker_col].astype("string").fillna("unknown")
    working["implied_move_bucket"] = _implied_move_buckets(working["implied_move_numeric"])
    working["announcement_timing_group"] = working[timing_col].map(_normalize_timing)

    overall_summary = pd.DataFrame(
        {
            "event_count": [len(working)],
            "avg_implied_move": [working["implied_move_numeric"].mean()],
            "avg_realized_move": [working["realized_move_numeric"].mean()],
            "avg_difference": [working["difference_realized_minus_implied"].mean()],
        }
    )

    distribution_summary = pd.DataFrame(
        {
            "pct_realized_gt_implied": [working["realized_gt_implied"].mean()],
            "pct_implied_gt_realized": [working["implied_gt_realized"].mean()],
            "pct_equal": [
                1.0
                - working["realized_gt_implied"].mean()
                - working["implied_gt_realized"].mean()
            ],
        }
    )

    by_ticker = _cohort_table(working, "ticker_group")
    by_implied_move_bucket = _cohort_table(working, "implied_move_bucket")
    by_announcement_timing = _cohort_table(working, "announcement_timing_group")

    avg_difference = float(overall_summary.loc[0, "avg_difference"])
    if np.isclose(avg_difference, 0.0):
        conclusion = "Implied and realized moves are roughly in line on average."
    elif avg_difference < 0:
        conclusion = "Options look overpriced into earnings on average (implied > realized)."
    else:
        conclusion = "Options look underpriced into earnings on average (realized > implied)."

    return {
        "overall_summary": overall_summary,
        "distribution_summary": distribution_summary,
        "by_ticker": by_ticker,
        "by_implied_move_bucket": by_implied_move_bucket,
        "by_announcement_timing": by_announcement_timing,
        "plot_data": {
            "ticker": by_ticker,
            "implied_move_bucket": by_implied_move_bucket,
            "announcement_timing": by_announcement_timing,
        },
        "conclusion": conclusion,
    }
