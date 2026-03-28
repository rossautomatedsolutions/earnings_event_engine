"""Cohort analysis utilities for earnings event feature datasets.

This module groups earnings events by timing and pre-event characteristics to surface
where options-implied moves systematically under/over-estimate realized moves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize_timing(value: object) -> str:
    if pd.isna(value):
        return "unknown"

    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"bmo", "before_open", "pre_market", "preopen"}:
        return "before_open"
    if normalized in {"amc", "after_close", "post_market", "after_market", "postclose"}:
        return "after_close"
    return "unknown"


def _bucket_series(
    series: pd.Series,
    *,
    bins: list[float] | None,
    labels: list[str] | None,
    quantile_labels: list[str],
    include_lowest: bool = True,
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    if bins is not None:
        if labels is not None and len(labels) != len(bins) - 1:
            raise ValueError("labels length must be len(bins) - 1 when bins are provided.")

        return pd.cut(
            numeric,
            bins=bins,
            labels=labels,
            include_lowest=include_lowest,
            duplicates="drop",
        ).astype("object").fillna("missing")

    # Default to quantile buckets; if values are too concentrated fallback to rank-based qcut.
    try:
        bucketed = pd.qcut(numeric, q=len(quantile_labels), labels=quantile_labels, duplicates="drop")
    except ValueError:
        ranked = numeric.rank(method="first")
        bucketed = pd.qcut(ranked, q=len(quantile_labels), labels=quantile_labels, duplicates="drop")

    return bucketed.astype("object").fillna("missing")


def build_cohort_analysis(
    event_features_df: pd.DataFrame,
    *,
    timing_col: str = "announcement_timing",
    implied_move_col: str = "implied_move",
    realized_move_col: str = "realized_move",
    volume_col: str = "volume",
    pre_earnings_return_col: str = "pre_earnings_return",
    implied_move_bins: list[float] | None = None,
    implied_move_labels: list[str] | None = None,
    pre_earnings_return_bins: list[float] | None = None,
    pre_earnings_return_labels: list[str] | None = None,
    volume_bins: list[float] | None = None,
    volume_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate earnings-event cohorts for option pricing regime analysis.

    Parameters
    ----------
    event_features_df
        DataFrame containing one row per earnings event and required columns.
    timing_col, implied_move_col, realized_move_col, volume_col, pre_earnings_return_col
        Column names used to compute cohorts and summary statistics.
    *_bins, *_labels
        Optional explicit bins/labels. When omitted, quantile buckets are used.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with one row per cohort containing:
        - announcement_timing_bucket
        - implied_move_bucket
        - volume_bucket
        - pre_earnings_return_bucket
        - avg_implied_move
        - avg_realized_move
        - avg_difference
        - hit_rate_realized_gt_implied
        - event_count
    """
    if event_features_df is None:
        raise ValueError("event_features_df must be provided.")

    required_columns = [
        timing_col,
        implied_move_col,
        realized_move_col,
        volume_col,
        pre_earnings_return_col,
    ]
    missing_cols = [col for col in required_columns if col not in event_features_df.columns]
    if missing_cols:
        raise ValueError(f"event_features_df is missing required columns: {missing_cols}")

    working = event_features_df.copy()

    working["announcement_timing_bucket"] = working[timing_col].map(_normalize_timing)
    working["implied_move_bucket"] = _bucket_series(
        working[implied_move_col],
        bins=implied_move_bins,
        labels=implied_move_labels,
        quantile_labels=["low_implied", "mid_implied", "high_implied"],
    )
    working["volume_bucket"] = _bucket_series(
        working[volume_col],
        bins=volume_bins,
        labels=volume_labels,
        quantile_labels=["low_volume", "mid_volume", "high_volume"],
    )
    working["pre_earnings_return_bucket"] = _bucket_series(
        working[pre_earnings_return_col],
        bins=pre_earnings_return_bins,
        labels=pre_earnings_return_labels,
        quantile_labels=["low_pre_return", "mid_pre_return", "high_pre_return"],
    )

    working["implied_move_numeric"] = pd.to_numeric(working[implied_move_col], errors="coerce")
    working["realized_move_numeric"] = pd.to_numeric(working[realized_move_col], errors="coerce")
    working["difference_realized_minus_implied"] = (
        working["realized_move_numeric"] - working["implied_move_numeric"]
    )
    working["hit_realized_gt_implied"] = (
        working["realized_move_numeric"] > working["implied_move_numeric"]
    ).astype(float)

    grouping_columns = [
        "announcement_timing_bucket",
        "implied_move_bucket",
        "volume_bucket",
        "pre_earnings_return_bucket",
    ]

    aggregated = (
        working.groupby(grouping_columns, dropna=False)
        .agg(
            avg_implied_move=("implied_move_numeric", "mean"),
            avg_realized_move=("realized_move_numeric", "mean"),
            avg_difference=("difference_realized_minus_implied", "mean"),
            hit_rate_realized_gt_implied=("hit_realized_gt_implied", "mean"),
            event_count=("implied_move_numeric", "count"),
        )
        .reset_index()
    )

    # Plot-ready quality-of-life ordering for downstream charting.
    aggregated = aggregated.sort_values(
        [
            "announcement_timing_bucket",
            "implied_move_bucket",
            "volume_bucket",
            "pre_earnings_return_bucket",
        ]
    ).reset_index(drop=True)

    return aggregated
