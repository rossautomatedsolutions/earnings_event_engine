"""Run a fast, repeatable end-to-end batch test for the earnings pipeline.

Pipeline stages executed:
1) events
2) prices
3) options (synthetic ATM straddle snapshot)
4) features
5) validation
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure local packages are importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from earnings_event_engine.utils.debug_logger import log_event
from earnings_event_engine.validation import build_data_coverage_report, run_event_diagnostics
from scripts.run_event_pipeline_test import (
    generate_features,
    load_prices_and_compute_moves,
    retrieve_and_normalize_events,
)

TEST_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD"]
START_DATE = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=365 * 2)
END_DATE = pd.Timestamp.utcnow().normalize()
MAX_EVENTS = 40
OUTPUT_DIR = REPO_ROOT / "data" / "outputs" / "test_batch"


def build_synthetic_options(events_with_prices: pd.DataFrame) -> pd.DataFrame:
    """Build a lightweight synthetic options snapshot with call+put legs per event."""
    if events_with_prices.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "option_type",
                "strike",
                "underlying_price",
                "bid",
                "ask",
                "mid",
            ]
        )

    price_anchor = pd.to_numeric(events_with_prices["pre_close"], errors="coerce")
    fallback_anchor = pd.to_numeric(events_with_prices["post_close"], errors="coerce")
    underlying = price_anchor.fillna(fallback_anchor)

    valid = events_with_prices.loc[underlying.notna(), ["event_id"]].copy()
    valid["underlying_price"] = underlying[underlying.notna()].astype(float).to_numpy()

    if valid.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "option_type",
                "strike",
                "underlying_price",
                "bid",
                "ask",
                "mid",
            ]
        )

    # Keep spreads tight and mids positive so options diagnostics focus on core pipeline wiring.
    valid["strike"] = valid["underlying_price"]
    valid["mid"] = np.maximum(valid["underlying_price"] * 0.015, 0.5)
    valid["bid"] = valid["mid"] * 0.95
    valid["ask"] = valid["mid"] * 1.05

    calls = valid.copy()
    calls["option_type"] = "call"
    puts = valid.copy()
    puts["option_type"] = "put"

    return pd.concat([calls, puts], ignore_index=True)


def safe_to_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def main() -> None:
    log_event(
        "events",
        "Starting small batch pipeline test",
        {
            "tickers": TEST_TICKERS,
            "start_date": START_DATE.date().isoformat(),
            "end_date": END_DATE.date().isoformat(),
            "max_events": MAX_EVENTS,
        },
    )

    events = retrieve_and_normalize_events(TEST_TICKERS, START_DATE, END_DATE, MAX_EVENTS)
    log_event("events", "Retrieved earnings events", {"event_count": len(events)})

    events_with_prices = load_prices_and_compute_moves(events)
    log_event("prices", "Loaded price windows and computed move fields", {"event_count": len(events_with_prices)})

    options_df = build_synthetic_options(events_with_prices)
    log_event("options", "Built synthetic options snapshots", {"option_rows": len(options_df)})

    features_df = generate_features(events_with_prices)
    log_event("features", "Generated features", {"feature_rows": len(features_df)})

    if events_with_prices.empty:
        clean_events = events_with_prices.copy()
        diagnostics_summary = pd.DataFrame(
            [
                {"category": "overview", "metric": "total_events", "count": 0, "pct_of_events": 0.0},
                {"category": "overview", "metric": "valid_events", "count": 0, "pct_of_events": 0.0},
                {"category": "overview", "metric": "invalid_events", "count": 0, "pct_of_events": 0.0},
            ]
        )
        diagnostics_flags = pd.DataFrame(
            columns=["event_id", "invalid_event", "valid_event", "failure_reasons"]
        )
        coverage_summary = diagnostics_summary.copy()
        coverage_by_ticker = pd.DataFrame(columns=["ticker", "num_events"])
        coverage_by_year = pd.DataFrame(columns=["event_year", "events_per_year"])
    else:
        options_for_validation = options_df if not options_df.empty else None

        clean_events, diagnostics_summary, diagnostics_flags = run_event_diagnostics(
            events_with_prices,
            options_df=options_for_validation,
            event_key_col="event_id",
            pre_close_col="pre_close",
            post_open_col="post_open",
            post_close_col="post_close",
            implied_move_col="implied_move",
            timing_col="announcement_timing",
            mapped_session_col="mapped_session",
        )

        coverage_summary, coverage_by_ticker, coverage_by_year = build_data_coverage_report(
            diagnostics_flags.merge(
                events_with_prices[
                    ["event_id", "ticker", "event_date", "announcement_timing", "mapped_session"]
                ],
                on="event_id",
                how="left",
            ),
            ticker_col="ticker",
            event_date_col="event_date",
            timing_col="announcement_timing",
            mapped_session_col="mapped_session",
        )

    final_features = features_df.merge(
        diagnostics_flags[["event_id", "valid_event", "invalid_event", "failure_reasons"]],
        on="event_id",
        how="left",
    )

    print("\n=== Diagnostics summary ===")
    print(diagnostics_summary.sort_values(["category", "metric"]).to_string(index=False))

    print("\n=== Coverage summary ===")
    print(coverage_summary.sort_values(["category", "metric"]).to_string(index=False))

    print("\n=== Sample output rows (events) ===")
    event_sample_cols = [
        "event_id",
        "ticker",
        "event_date",
        "announcement_timing",
        "pre_close",
        "post_open",
        "implied_move",
    ]
    present_event_cols = [c for c in event_sample_cols if c in events_with_prices.columns]
    print(events_with_prices[present_event_cols].head(8).round(4).to_string(index=False))

    print("\n=== Sample output rows (features) ===")
    feature_sample_cols = [
        "event_id",
        "ticker",
        "event_date",
        "realized_gap_move",
        "surprise_vs_implied",
        "valid_event",
        "failure_reasons",
    ]
    present_feature_cols = [c for c in feature_sample_cols if c in final_features.columns]
    print(final_features[present_feature_cols].head(8).round(4).to_string(index=False))

    log_event("validation", "Validation complete", {"valid_events": int(diagnostics_flags["valid_event"].sum())})

    safe_to_parquet(events_with_prices, OUTPUT_DIR / "events.parquet")
    safe_to_parquet(options_df, OUTPUT_DIR / "options.parquet")
    safe_to_parquet(final_features, OUTPUT_DIR / "features.parquet")
    safe_to_parquet(clean_events, OUTPUT_DIR / "clean_events.parquet")
    safe_to_parquet(diagnostics_summary, OUTPUT_DIR / "diagnostics.parquet")
    safe_to_parquet(diagnostics_flags, OUTPUT_DIR / "diagnostics_flags.parquet")
    safe_to_parquet(coverage_summary, OUTPUT_DIR / "coverage_summary.parquet")
    safe_to_parquet(coverage_by_ticker, OUTPUT_DIR / "coverage_by_ticker.parquet")
    safe_to_parquet(coverage_by_year, OUTPUT_DIR / "coverage_by_year.parquet")

    print("\nArtifacts written:")
    for artifact in [
        "events.parquet",
        "options.parquet",
        "features.parquet",
        "clean_events.parquet",
        "diagnostics.parquet",
        "diagnostics_flags.parquet",
        "coverage_summary.parquet",
        "coverage_by_ticker.parquet",
        "coverage_by_year.parquet",
    ]:
        print(f"- {OUTPUT_DIR / artifact}")


if __name__ == "__main__":
    main()
