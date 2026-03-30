"""Conditional analysis utilities for identifying earnings-event edge by regime."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


_REQUIRED_COMPARE_COLUMNS = {"implied_move", "pnl"}
_REQUIRED_TIMING_COLUMNS = {"timing", "realized_move", "pnl"}


def _validate_columns(df: pd.DataFrame, required: set[str], fn_name: str) -> None:
    """Raise a helpful error when required columns are missing."""
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{fn_name} requires columns: {missing}")


def filter_high_iv(df: pd.DataFrame) -> pd.DataFrame:
    """Return events where implied move is above the sample median."""
    _validate_columns(df, {"implied_move"}, "filter_high_iv")
    median_implied_move = pd.to_numeric(df["implied_move"], errors="coerce").median()
    numeric_implied_move = pd.to_numeric(df["implied_move"], errors="coerce")
    return df.loc[numeric_implied_move > median_implied_move].copy()


def filter_low_iv(df: pd.DataFrame) -> pd.DataFrame:
    """Return events where implied move is below the sample median."""
    _validate_columns(df, {"implied_move"}, "filter_low_iv")
    median_implied_move = pd.to_numeric(df["implied_move"], errors="coerce").median()
    numeric_implied_move = pd.to_numeric(df["implied_move"], errors="coerce")
    return df.loc[numeric_implied_move < median_implied_move].copy()


def _safe_win_rate(pnl_series: pd.Series) -> float:
    pnl_numeric = pd.to_numeric(pnl_series, errors="coerce")
    return float((pnl_numeric > 0).mean())


def compare_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Compare high-IV vs low-IV groups by average PnL and win rate."""
    _validate_columns(df, _REQUIRED_COMPARE_COLUMNS, "compare_groups")

    high_iv = filter_high_iv(df)
    low_iv = filter_low_iv(df)

    comparison = pd.DataFrame(
        {
            "group": ["high_iv", "low_iv"],
            "event_count": [len(high_iv), len(low_iv)],
            "avg_pnl": [
                pd.to_numeric(high_iv["pnl"], errors="coerce").mean(),
                pd.to_numeric(low_iv["pnl"], errors="coerce").mean(),
            ],
            "win_rate": [_safe_win_rate(high_iv["pnl"]), _safe_win_rate(low_iv["pnl"])],
        }
    )
    return comparison


def compare_by_timing(df: pd.DataFrame) -> pd.DataFrame:
    """Compare AMC vs BMO by average realized move and average PnL."""
    _validate_columns(df, _REQUIRED_TIMING_COLUMNS, "compare_by_timing")

    working = df.copy()
    timing = (
        working["timing"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"AFTER_CLOSE": "AMC", "BEFORE_OPEN": "BMO"})
    )
    working["timing_normalized"] = timing

    filtered = working.loc[working["timing_normalized"].isin(["AMC", "BMO"])].copy()

    grouped = (
        filtered.groupby("timing_normalized", dropna=False)
        .agg(
            event_count=("timing_normalized", "size"),
            avg_realized_move=("realized_move", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            avg_pnl=("pnl", lambda s: pd.to_numeric(s, errors="coerce").mean()),
        )
        .reset_index()
        .rename(columns={"timing_normalized": "timing"})
    )

    return grouped.sort_values("timing").reset_index(drop=True)


def _load_analysis_frame(path: str | Path) -> pd.DataFrame:
    """Load a local analysis file from CSV or Parquet."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Data file not found: {input_path}")

    if input_path.suffix.lower() == ".parquet":
        return pd.read_parquet(input_path)

    return pd.read_csv(input_path)


def run_conditional_analysis(path: str | Path) -> dict[str, pd.DataFrame]:
    """Run all conditional comparisons to identify potential edge conditions."""
    df = _load_analysis_frame(path)

    return {
        "high_iv": filter_high_iv(df),
        "low_iv": filter_low_iv(df),
        "group_comparison": compare_groups(df),
        "timing_comparison": compare_by_timing(df),
    }
