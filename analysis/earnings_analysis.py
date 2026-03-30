"""Analysis helpers for earnings-event feature outputs."""

from __future__ import annotations

from typing import Any

import pandas as pd


REALIZED_COL = "realized_gap_move"
IMPLIED_COL = "implied_move"
SURPRISE_COL = "surprise_vs_implied"
TICKER_COL = "ticker"
TIMING_COL = "announcement_timing"


def load_features(path: str) -> pd.DataFrame:
    """Load a feature parquet file into a DataFrame."""
    return pd.read_parquet(path)


def compute_basic_statistics(df: pd.DataFrame) -> dict[str, float]:
    """Compute high-level statistics used to assess pricing vs. realized moves."""
    _validate_columns(df, [REALIZED_COL, IMPLIED_COL, SURPRISE_COL])

    realized = df[REALIZED_COL]
    implied = df[IMPLIED_COL]
    surprise = df[SURPRISE_COL]

    return {
        "mean_realized_move": float(realized.mean()),
        "mean_implied_move": float(implied.mean()),
        "mean_surprise": float(surprise.mean()),
        "pct_overpriced": float((realized < implied).mean() * 100),
        "pct_underpriced": float((realized > implied).mean() * 100),
    }


def compute_distribution(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Return descriptive distributions for realized and surprise metrics."""
    _validate_columns(df, [REALIZED_COL, SURPRISE_COL])

    return {
        "realized_gap_move": df[REALIZED_COL].describe(),
        "surprise_vs_implied": df[SURPRISE_COL].describe(),
    }


def compute_by_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate earnings behavior by ticker."""
    _validate_columns(df, [TICKER_COL, REALIZED_COL, IMPLIED_COL, SURPRISE_COL])

    return (
        df.groupby(TICKER_COL)
        .agg(
            avg_realized_move=(REALIZED_COL, "mean"),
            avg_implied_move=(IMPLIED_COL, "mean"),
            avg_surprise=(SURPRISE_COL, "mean"),
            count=(TICKER_COL, "size"),
        )
        .sort_values("count", ascending=False)
    )


def compute_by_timing(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate earnings behavior by announcement timing."""
    _validate_columns(df, [TIMING_COL, REALIZED_COL, IMPLIED_COL, SURPRISE_COL])

    return (
        df.groupby(TIMING_COL)
        .agg(
            avg_realized_move=(REALIZED_COL, "mean"),
            avg_implied_move=(IMPLIED_COL, "mean"),
            avg_surprise=(SURPRISE_COL, "mean"),
            count=(TIMING_COL, "size"),
        )
        .sort_values("count", ascending=False)
    )


def run_full_analysis(path: str) -> dict[str, Any]:
    """Run the complete analysis and print clean, readable output."""
    df = load_features(path)
    basic_stats = compute_basic_statistics(df)
    distribution = compute_distribution(df)
    by_ticker = compute_by_ticker(df)
    by_timing = compute_by_timing(df)

    print("\n=== Earnings Event Analysis ===")
    print(f"Rows: {len(df):,}")

    print("\n--- Basic Statistics ---")
    for key, value in basic_stats.items():
        if key.startswith("pct_"):
            print(f"{key}: {value:.2f}%")
        else:
            print(f"{key}: {value:.4f}")

    print("\n--- Distribution: realized_gap_move ---")
    print(distribution["realized_gap_move"].to_string())

    print("\n--- Distribution: surprise_vs_implied ---")
    print(distribution["surprise_vs_implied"].to_string())

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
