"""Event filtering utilities for identifying strong and weak PnL events."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

_DEFAULT_TOP_BOTTOM_N = 20
_RESULT_REQUIRED_COLUMNS = {"pnl"}
_NUMERIC_COMPARE_COLUMNS = ["implied_move", "realized_gap_move"]
_CATEGORICAL_COMPARE_COLUMNS = ["ticker", "announcement_timing"]


def _validate_columns(df: pd.DataFrame, required: set[str], fn_name: str) -> None:
    """Raise a helpful error when required columns are missing."""
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{fn_name} requires columns: {missing}")


def _load_frame(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Data file not found: {input_path}")

    if input_path.suffix.lower() == ".parquet":
        return pd.read_parquet(input_path)

    return pd.read_csv(input_path)


def _coerce_results_frame(results: Any) -> pd.DataFrame:
    if isinstance(results, pd.DataFrame):
        frame = results.copy()
    elif isinstance(results, (str, Path)):
        frame = _load_frame(results)
    else:
        frame = pd.DataFrame(results)

    _validate_columns(frame, _RESULT_REQUIRED_COLUMNS, "results")
    frame["pnl"] = pd.to_numeric(frame["pnl"], errors="coerce")
    return frame.dropna(subset=["pnl"]).copy()


def get_top_events(results: Any, n: int = _DEFAULT_TOP_BOTTOM_N) -> pd.DataFrame:
    """Return top-N events ranked by PnL (largest first)."""
    frame = _coerce_results_frame(results)
    return frame.nlargest(n, "pnl").reset_index(drop=True)


def get_bottom_events(results: Any, n: int = _DEFAULT_TOP_BOTTOM_N) -> pd.DataFrame:
    """Return bottom-N events ranked by PnL (most negative first)."""
    frame = _coerce_results_frame(results)
    return frame.nsmallest(n, "pnl").reset_index(drop=True)


def _infer_merge_keys(features_df: pd.DataFrame, results_df: pd.DataFrame) -> list[str]:
    preferred_exact_keys = ["event_id", "trade_id", "id"]
    for key in preferred_exact_keys:
        if key in features_df.columns and key in results_df.columns:
            return [key]

    preferred_composite = ["ticker", "date", "announcement_date", "earnings_date"]
    keys = [column for column in preferred_composite if column in features_df.columns and column in results_df.columns]
    if keys:
        return keys

    return []


def _merge_features_and_results(features_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    merge_keys = _infer_merge_keys(features_df, results_df)
    if merge_keys:
        merged = features_df.merge(results_df, on=merge_keys, how="inner", suffixes=("", "_result"))
    else:
        merged = pd.concat(
            [features_df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1
        )

    _validate_columns(merged, _RESULT_REQUIRED_COLUMNS, "analyze_characteristics")
    merged["pnl"] = pd.to_numeric(merged["pnl"], errors="coerce")
    return merged.dropna(subset=["pnl"]).copy()


def analyze_characteristics(df: pd.DataFrame, results: Any) -> dict[str, pd.DataFrame]:
    """Merge features + results and compare top-vs-bottom event characteristics."""
    results_df = _coerce_results_frame(results)
    merged = _merge_features_and_results(df, results_df)

    top_events = get_top_events(merged, n=min(_DEFAULT_TOP_BOTTOM_N, len(merged)))
    bottom_events = get_bottom_events(merged, n=min(_DEFAULT_TOP_BOTTOM_N, len(merged)))

    numeric_rows = []
    for column in _NUMERIC_COMPARE_COLUMNS:
        if column not in merged.columns:
            continue

        top_numeric = pd.to_numeric(top_events[column], errors="coerce")
        bottom_numeric = pd.to_numeric(bottom_events[column], errors="coerce")
        numeric_rows.append(
            {
                "metric": column,
                "top_mean": top_numeric.mean(),
                "bottom_mean": bottom_numeric.mean(),
                "difference": top_numeric.mean() - bottom_numeric.mean(),
            }
        )

    numeric_summary = pd.DataFrame(numeric_rows)

    categorical_frames: dict[str, pd.DataFrame] = {}
    for column in _CATEGORICAL_COMPARE_COLUMNS:
        if column not in merged.columns:
            continue

        top_counts = top_events[column].astype("string").fillna("unknown").value_counts(normalize=True)
        bottom_counts = bottom_events[column].astype("string").fillna("unknown").value_counts(normalize=True)
        comparison = pd.concat([top_counts, bottom_counts], axis=1).fillna(0)
        comparison.columns = ["top_share", "bottom_share"]
        comparison["share_diff"] = comparison["top_share"] - comparison["bottom_share"]
        comparison = comparison.reset_index().rename(columns={"index": column})
        categorical_frames[column] = comparison.sort_values("share_diff", ascending=False).reset_index(drop=True)

    return {
        "merged": merged,
        "top_events": top_events,
        "bottom_events": bottom_events,
        "numeric_summary": numeric_summary,
        "ticker_comparison": categorical_frames.get("ticker", pd.DataFrame()),
        "announcement_timing_comparison": categorical_frames.get("announcement_timing", pd.DataFrame()),
    }


def run_event_filtering(path: str | Path) -> dict[str, pd.DataFrame]:
    """Load results and print top/worst events plus top-vs-bottom summary."""
    results_df = _load_frame(path)

    analysis = analyze_characteristics(results_df, results_df)

    print("\n=== Top Events by PnL ===")
    print(analysis["top_events"].head(_DEFAULT_TOP_BOTTOM_N).to_string(index=False))

    print("\n=== Worst Events by PnL ===")
    print(analysis["bottom_events"].head(_DEFAULT_TOP_BOTTOM_N).to_string(index=False))

    print("\n=== Top vs Bottom Summary ===")
    if analysis["numeric_summary"].empty:
        print("No numeric comparison columns found.")
    else:
        print(analysis["numeric_summary"].to_string(index=False))

    if not analysis["ticker_comparison"].empty:
        print("\nTicker distribution (top vs bottom):")
        print(analysis["ticker_comparison"].to_string(index=False))

    if not analysis["announcement_timing_comparison"].empty:
        print("\nAnnouncement timing distribution (top vs bottom):")
        print(analysis["announcement_timing_comparison"].to_string(index=False))

    return analysis
