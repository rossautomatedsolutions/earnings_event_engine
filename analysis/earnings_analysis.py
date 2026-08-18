"""Analysis helpers for earnings-event feature outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.option_selection import EVENT_OPTION_IMPLIED_MOVE_SOURCE


REALIZED_COL = "realized_gap_move"
IMPLIED_COL = "implied_move"
SURPRISE_COL = "surprise_vs_implied"
TICKER_COL = "ticker"
TIMING_COL = "announcement_timing"
OPTION_AVAILABLE_COL = "implied_move_available"
OPTION_SOURCE_COL = "implied_move_source"


def load_features(path: str) -> pd.DataFrame:
    """Load a feature parquet file into a DataFrame."""
    return pd.read_parquet(path)


def _mean_or_nan(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.mean())


def _pct_or_nan(mask: pd.Series) -> float:
    if len(mask) == 0:
        return float("nan")
    return float(mask.mean() * 100.0)


def _prepare_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(df, [REALIZED_COL, TICKER_COL, TIMING_COL])

    working = df.copy()
    working[REALIZED_COL] = pd.to_numeric(working[REALIZED_COL], errors="coerce")
    if IMPLIED_COL in working.columns:
        working[IMPLIED_COL] = pd.to_numeric(working[IMPLIED_COL], errors="coerce")
    else:
        working[IMPLIED_COL] = np.nan
    if SURPRISE_COL in working.columns:
        working[SURPRISE_COL] = pd.to_numeric(working[SURPRISE_COL], errors="coerce")
    else:
        working[SURPRISE_COL] = np.nan

    eligible = working[IMPLIED_COL].notna()
    if OPTION_AVAILABLE_COL in working.columns:
        eligible &= working[OPTION_AVAILABLE_COL].astype("boolean").fillna(False).astype(bool)
    if OPTION_SOURCE_COL in working.columns:
        source = working[OPTION_SOURCE_COL].astype("string").fillna("")
        eligible &= source.eq(EVENT_OPTION_IMPLIED_MOVE_SOURCE)

    working["_option_implied_eligible"] = eligible.astype(bool)
    working.loc[~working["_option_implied_eligible"], [IMPLIED_COL, SURPRISE_COL]] = np.nan
    return working


def compute_basic_statistics(df: pd.DataFrame) -> dict[str, float]:
    """Compute price-only and option-implied statistics without proxy substitution."""
    working = _prepare_analysis_frame(df)
    option_rows = working.loc[working["_option_implied_eligible"]].copy()

    return {
        "event_count": float(len(working)),
        "option_implied_event_count": float(len(option_rows)),
        "mean_realized_move": _mean_or_nan(working[REALIZED_COL]),
        "mean_implied_move": _mean_or_nan(option_rows[IMPLIED_COL]),
        "mean_surprise": _mean_or_nan(option_rows[SURPRISE_COL]),
        "pct_overpriced": _pct_or_nan(option_rows[REALIZED_COL] < option_rows[IMPLIED_COL]),
        "pct_underpriced": _pct_or_nan(option_rows[REALIZED_COL] > option_rows[IMPLIED_COL]),
    }


def compute_distribution(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Return descriptive distributions for realized and option-implied surprise metrics."""
    working = _prepare_analysis_frame(df)
    option_rows = working.loc[working["_option_implied_eligible"]].copy()
    empty_option_series = pd.Series(dtype=float)

    return {
        "realized_gap_move": pd.to_numeric(working[REALIZED_COL], errors="coerce").describe(),
        "surprise_vs_implied": (
            pd.to_numeric(option_rows[SURPRISE_COL], errors="coerce").describe()
            if not option_rows.empty
            else empty_option_series.describe()
        ),
    }


def compute_by_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate earnings behavior by ticker without leaking proxy rows into option metrics."""
    working = _prepare_analysis_frame(df)

    return (
        working.groupby(TICKER_COL, dropna=False)
        .agg(
            event_count=(TICKER_COL, "size"),
            option_implied_event_count=("_option_implied_eligible", "sum"),
            avg_realized_move=(REALIZED_COL, "mean"),
            avg_implied_move=(IMPLIED_COL, "mean"),
            avg_surprise=(SURPRISE_COL, "mean"),
        )
        .sort_values("event_count", ascending=False)
    )


def compute_by_timing(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate earnings behavior by announcement timing."""
    working = _prepare_analysis_frame(df)

    return (
        working.groupby(TIMING_COL, dropna=False)
        .agg(
            event_count=(TIMING_COL, "size"),
            option_implied_event_count=("_option_implied_eligible", "sum"),
            avg_realized_move=(REALIZED_COL, "mean"),
            avg_implied_move=(IMPLIED_COL, "mean"),
            avg_surprise=(SURPRISE_COL, "mean"),
        )
        .sort_values("event_count", ascending=False)
    )


def run_full_analysis(path: str) -> dict[str, Any]:
    """Run the complete analysis and print clean, readable output."""
    df = load_features(path)
    prepared = _prepare_analysis_frame(df)
    option_rows = prepared.loc[prepared["_option_implied_eligible"]].copy()
    basic_stats = compute_basic_statistics(prepared)
    distribution = compute_distribution(prepared)
    by_ticker = compute_by_ticker(prepared)
    by_timing = compute_by_timing(prepared)

    print("\n=== Earnings Event Analysis ===")
    print(f"Rows: {len(prepared):,}")
    print(f"Rows with event-time option-implied move: {len(option_rows):,}")

    print("\n--- Basic Statistics ---")
    for key, value in basic_stats.items():
        if key in {"event_count", "option_implied_event_count"}:
            print(f"{key}: {int(value)}")
        elif key.startswith("pct_"):
            print(f"{key}: {value:.2f}%")
        else:
            print(f"{key}: {value:.4f}")

    print("\n--- Distribution: realized_gap_move ---")
    print(distribution["realized_gap_move"].to_string())

    print("\n--- Distribution: surprise_vs_implied ---")
    print(distribution["surprise_vs_implied"].to_string())

    if option_rows.empty:
        print(
            "\nOption-implied pricing metrics are unavailable because no rows contain "
            "event-time option-derived implied_move values."
        )

    print("\n--- By Ticker (top 20 by count) ---")
    print(by_ticker.head(20).to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n--- By Announcement Timing ---")
    print(by_timing.to_string(float_format=lambda x: f"{x:.4f}"))

    return {
        "basic_statistics": basic_stats,
        "distribution": distribution,
        "by_ticker": by_ticker,
        "by_timing": by_timing,
    }


def _validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Raise a clear error if expected columns are missing."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{missing}. Available columns: {list(df.columns)}"
        )
