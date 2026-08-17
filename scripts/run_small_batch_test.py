"""Run a fast, repeatable end-to-end batch test for the earnings pipeline.

Pipeline stages executed:
1) events
2) prices
3) options (synthetic ATM straddle snapshot)
4) features
5) validation
"""

from __future__ import annotations

import argparse
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

from data.earnings_calendar import EarningsCalendarRetrievalError
from earnings_event_engine.utils.debug_logger import log_event
from earnings_event_engine.validation import build_data_coverage_report, run_event_diagnostics
from scripts.run_event_pipeline_test import (
    _append_summary_metric,
    _base_diagnostics_summary,
    _build_failure_counts,
    _build_coverage_input_df,
    _empty_events_frame,
    _empty_diagnostics_flags,
    _empty_enriched_events_frame,
    generate_features,
    load_prices_and_compute_moves,
    retrieve_and_normalize_events,
)

# Broad, liquid universe for better small-batch statistics:
# - Primarily SPY / S&P 500 constituents
# - Includes a liquid mid-cap sleeve for cross-section diversity
TEST_TICKERS = [
    # SPY components (large cap)
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "AVGO", "JPM",
    "V", "MA", "XOM", "JNJ", "WMT", "PG", "LLY", "UNH", "HD", "COST",
    "MRK", "ABBV", "CVX", "KO", "PEP", "BAC", "ORCL", "CRM", "NFLX", "ADBE",
    "CSCO", "TMO", "ACN", "MCD", "DHR", "LIN", "ABT", "WFC", "DIS", "INTC",
    "VZ", "CMCSA", "TXN", "PM", "QCOM", "NEE", "HON", "UPS", "RTX", "LOW",
    # Liquid mid-cap sleeve
    "RCL", "NCLH", "DAL", "UAL", "ETSY", "DKNG", "PINS", "ROKU", "HOOD", "SOFI",
    "CVNA", "RIVN", "SNAP", "CHWY", "U", "TWLO", "FSLY", "AFRM", "CROX", "ONON",
]
START_DATE = pd.Timestamp.now("UTC").normalize() - pd.Timedelta(days=365 * 2)
END_DATE = pd.Timestamp.now("UTC").normalize()
MAX_EVENTS = 240
OUTPUT_DIR = REPO_ROOT / "data" / "outputs" / "test_batch"


def build_synthetic_options(events_with_prices: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
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

    valid_mask = underlying.notna()
    skipped = events_with_prices.loc[~valid_mask, ["event_id"]].copy()
    if not skipped.empty:
        skipped_ids = skipped["event_id"].astype(str).tolist()
        message = (
            "Missing pre_close/post_close; synthetic options not generated for some events"
        )
        details = {"skipped_count": len(skipped_ids), "event_ids": skipped_ids}
        log_event("options", message, details)
        print(f"[options] WARNING: {message}. skipped_count={len(skipped_ids)}")
        if debug:
            print(f"[options][debug] skipped_event_ids={skipped_ids}")

    valid = events_with_prices.loc[valid_mask, ["event_id"]].copy()
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


def build_output_artifacts(
    *,
    events: pd.DataFrame,
    events_with_prices: pd.DataFrame,
    options_df: pd.DataFrame,
    final_features: pd.DataFrame,
    clean_events: pd.DataFrame,
    diagnostics_output: pd.DataFrame,
    diagnostics_flags: pd.DataFrame,
    coverage_output: pd.DataFrame,
    coverage_by_ticker: pd.DataFrame,
    coverage_by_year: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Keep the core artifact contract aligned with run_event_pipeline_test.py."""
    return {
        "events.parquet": events,
        "events_with_prices.parquet": events_with_prices,
        "options.parquet": options_df,
        "features.parquet": final_features,
        "clean_events.parquet": clean_events,
        "diagnostics.parquet": diagnostics_output,
        "diagnostics_flags.parquet": diagnostics_flags,
        "coverage_summary.parquet": coverage_output,
        "coverage_by_ticker.parquet": coverage_by_ticker,
        "coverage_by_year.parquet": coverage_by_year,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small-batch earnings pipeline test.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable per-event diagnostics and verbose failure logging.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow zero raw or zero validated rows without returning a non-zero exit code.",
    )
    return parser.parse_args()


def _compute_realized_move(events_with_prices: pd.DataFrame, features_df: pd.DataFrame) -> pd.Series:
    if "realized_gap_move" in features_df.columns:
        by_event = features_df.set_index("event_id")["realized_gap_move"]
        return events_with_prices["event_id"].map(by_event)
    return (
        pd.to_numeric(events_with_prices["post_open"], errors="coerce")
        / pd.to_numeric(events_with_prices["pre_close"], errors="coerce")
        - 1.0
    )


def _print_debug_diagnostics(
    events_with_prices: pd.DataFrame,
    features_df: pd.DataFrame,
    diagnostics_flags: pd.DataFrame,
) -> None:
    realized_move = _compute_realized_move(events_with_prices, features_df)
    debug_rows = events_with_prices[
        ["event_id", "announcement_timing", "mapped_session", "implied_move"]
    ].copy()
    debug_rows["realized_move"] = realized_move
    debug_rows = debug_rows.merge(
        diagnostics_flags[["event_id", "valid_event", "invalid_event", "failure_reasons"]],
        on="event_id",
        how="left",
    )

    print("\n=== Per-event debug diagnostics ===")
    for row in debug_rows.itertuples(index=False):
        print(
            "[event]"
            f" event_id={row.event_id}"
            f" implied_move={row.implied_move}"
            f" realized_move={row.realized_move}"
            f" alignment=timing:{row.announcement_timing}|session:{row.mapped_session}"
            f" valid={row.valid_event}"
            f" failure_reasons={row.failure_reasons}"
        )
        if bool(row.invalid_event):
            print(
                "[event][failure]"
                f" event_id={row.event_id}"
                f" reasons={row.failure_reasons}"
            )


def main() -> None:
    args = parse_args()
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

    events = _empty_events_frame()
    events_with_prices = _empty_enriched_events_frame()
    options_df = pd.DataFrame(
        columns=["event_id", "option_type", "strike", "underlying_price", "bid", "ask", "mid"]
    )
    features_df = generate_features(events_with_prices)
    clean_events = events_with_prices.copy()
    diagnostics_summary = _base_diagnostics_summary()
    diagnostics_flags = _empty_diagnostics_flags()
    coverage_summary = diagnostics_summary.copy()
    coverage_by_ticker = pd.DataFrame(columns=["ticker", "num_events"])
    coverage_by_year = pd.DataFrame(columns=["event_year", "events_per_year"])
    runtime_metrics: list[str] = []
    failure_messages: list[str] = []

    try:
        events = retrieve_and_normalize_events(TEST_TICKERS, START_DATE, END_DATE, MAX_EVENTS)
        log_event("events", "Retrieved earnings events", {"event_count": len(events)})

        events_with_prices = load_prices_and_compute_moves(events)
        log_event("prices", "Loaded price windows and computed move fields", {"event_count": len(events_with_prices)})

        options_df = build_synthetic_options(events_with_prices, debug=args.debug)
        log_event("options", "Built synthetic options snapshots", {"option_rows": len(options_df)})

        features_df = generate_features(events_with_prices)
        log_event("features", "Generated features", {"feature_rows": len(features_df)})

        if not events_with_prices.empty:
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

            coverage_report = build_data_coverage_report(
                _build_coverage_input_df(diagnostics_flags, events_with_prices),
                ticker_col="ticker",
                event_date_col="event_date",
                price_valid_col="price_alignment_valid",
                timing_col="announcement_timing",
                mapped_session_col="mapped_session",
            )
            coverage_summary = coverage_report.summary
            coverage_by_ticker = coverage_report.by_ticker
            coverage_by_year = coverage_report.by_year
            print("[DEBUG] Coverage Summary:")
            print(coverage_summary)
    except EarningsCalendarRetrievalError as exc:
        failure_messages.append(str(exc))
        runtime_metrics.append("retrieval_failure")
        log_event("events", "Earnings retrieval failed", {"error": str(exc)})

    final_features = features_df.merge(
        diagnostics_flags[["event_id", "valid_event", "invalid_event", "failure_reasons"]],
        on="event_id",
        how="left",
    )

    raw_event_count = len(events)
    valid_event_count = len(clean_events)
    feature_row_count = len(final_features)

    if raw_event_count == 0:
        runtime_metrics.append("zero_raw_events")
        if not args.allow_empty and not failure_messages:
            failure_messages.append("Zero raw events produced during the retrieval stage in live mode.")

    if raw_event_count > 0 and valid_event_count == 0:
        runtime_metrics.append("zero_valid_events")
        if not args.allow_empty:
            failure_messages.append("Zero valid/clean events produced during the validation stage in live mode.")

    diagnostics_output = diagnostics_summary.copy()
    coverage_output = coverage_summary.copy()
    for metric in runtime_metrics:
        if metric not in diagnostics_output["metric"].tolist():
            diagnostics_output = _append_summary_metric(diagnostics_output, metric)
        if metric not in coverage_output["metric"].tolist():
            coverage_output = _append_summary_metric(coverage_output, metric)

    failed_events = diagnostics_flags.loc[diagnostics_flags["invalid_event"].fillna(False)].copy()
    if not failed_events.empty:
        for row in failed_events.itertuples(index=False):
            print(f"[failure] event_id={row.event_id} reasons={row.failure_reasons}")

    if args.debug and not events_with_prices.empty:
        _print_debug_diagnostics(events_with_prices, features_df, diagnostics_flags)

    failure_counts = _build_failure_counts(diagnostics_flags)
    if not failure_counts.empty:
        failures = failure_counts.rename(columns={"failure_reason": "metric"}).copy()
        failures.insert(0, "category", "failure_counts")
        failures["pct_of_events"] = failures["count"] / max(len(diagnostics_flags), 1)
        diagnostics_output = pd.concat([diagnostics_output, failures], ignore_index=True, sort=False)

    print("\n=== Diagnostics summary ===")
    print(diagnostics_output.sort_values(["category", "metric"]).to_string(index=False))

    print("\n=== Coverage summary ===")
    if "category" in coverage_output.columns:
        coverage_output = coverage_output.sort_values(["category", "metric"])
    else:
        coverage_output = coverage_output.sort_values(["metric"])
    print(coverage_output.to_string(index=False))

    print(f"\n[INFO] raw_events={raw_event_count}")
    print(f"[INFO] clean_events={valid_event_count}")
    print(f"[INFO] feature_rows={feature_row_count}")

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
    event_display = events_with_prices[present_event_cols].head(8).copy()
    event_numeric_cols = event_display.select_dtypes(include=[np.number]).columns
    event_display.loc[:, event_numeric_cols] = event_display.loc[:, event_numeric_cols].round(4)
    print(event_display.to_string(index=False))

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
    feature_display = final_features[present_feature_cols].head(8).copy()
    feature_numeric_cols = feature_display.select_dtypes(include=[np.number]).columns
    feature_display.loc[:, feature_numeric_cols] = feature_display.loc[:, feature_numeric_cols].round(4)
    print(feature_display.to_string(index=False))

    log_event("validation", "Validation complete", {"valid_events": int(diagnostics_flags["valid_event"].sum())})

    artifacts = build_output_artifacts(
        events=events,
        events_with_prices=events_with_prices,
        options_df=options_df,
        final_features=final_features,
        clean_events=clean_events,
        diagnostics_output=diagnostics_output,
        diagnostics_flags=diagnostics_flags,
        coverage_output=coverage_output,
        coverage_by_ticker=coverage_by_ticker,
        coverage_by_year=coverage_by_year,
    )

    for artifact_name, artifact_df in artifacts.items():
        safe_to_parquet(artifact_df, OUTPUT_DIR / artifact_name)

    print("\nArtifacts written:")
    for artifact_name in artifacts:
        print(f"- {OUTPUT_DIR / artifact_name}")

    if failure_messages:
        raise SystemExit("\n".join(failure_messages))


if __name__ == "__main__":
    main()
