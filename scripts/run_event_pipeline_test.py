"""End-to-end earnings event pipeline smoke-test runner.

This script is intended to be the standard quick way to validate the system
from raw event retrieval through validation outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure local ``src`` package is importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.earnings_calendar import get_earnings_dates
from earnings_event_engine.validation import build_data_coverage_report, run_event_diagnostics
from utils.datetime_utils import (
    assert_same_timezone,
    normalize_datetime_index,
    to_naive,
    to_naive_datetime,
    validate_datetime,
)


def parse_args() -> argparse.Namespace:
    today = to_naive(pd.Timestamp.utcnow()).normalize()
    parser = argparse.ArgumentParser(
        description="Run the earnings event pipeline end-to-end for quick testing."
    )
    parser.add_argument(
        "--tickers",
        required=True,
        help="Comma-separated ticker list (example: AAPL,MSFT,NVDA).",
    )
    parser.add_argument(
        "--start-date",
        default=(today - pd.Timedelta(days=365 * 2)).date().isoformat(),
        help="Inclusive start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        default=today.date().isoformat(),
        help="Inclusive end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=50,
        help="Maximum events to process after filtering (for quick tests).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/pipeline_test",
        help="Directory where parquet artifacts will be written.",
    )
    return parser.parse_args()


def _normalize_ticker_list(raw: str) -> list[str]:
    tickers = [token.strip().upper() for token in raw.split(",")]
    return [ticker for ticker in tickers if ticker]


def _infer_timing(earnings_ts: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(earnings_ts, errors="coerce")
    naive = parsed.where(parsed.isna(), parsed.map(to_naive))
    hours = naive.dt.hour
    # yfinance often returns timestamps where earlier hours are BMO-like and later AMC-like.
    return np.where(hours.fillna(16) < 12, "bmo", "amc")


def retrieve_and_normalize_events(
    tickers: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    max_events: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        df = get_earnings_dates(ticker)
        if df.empty:
            continue
        frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=[
                "event_id",
                "ticker",
                "earnings_date",
                "event_date",
                "announcement_timing",
                "mapped_session",
            ]
        )

    events = pd.concat(frames, ignore_index=True)
    parsed_earnings = pd.to_datetime(events["earnings_date"], errors="coerce")
    events = events.loc[parsed_earnings.notna()].copy()
    events["earnings_date"] = parsed_earnings.loc[parsed_earnings.notna()].map(to_naive)
    events = events[
        (events["earnings_date"].dt.normalize() >= start_date)
        & (events["earnings_date"].dt.normalize() <= end_date)
    ].copy()

    events["ticker"] = events["ticker"].astype(str).str.upper()
    events["event_date"] = events["earnings_date"].dt.normalize()
    events["announcement_timing"] = _infer_timing(events["earnings_date"])
    events["mapped_session"] = np.where(
        events["announcement_timing"].eq("bmo"), "same_day", "next_day"
    )

    events = (
        events.sort_values(["event_date", "ticker"])
        .drop_duplicates(subset=["ticker", "event_date"], keep="first")
        .head(max_events)
        .reset_index(drop=True)
    )
    events.insert(0, "event_id", np.arange(1, len(events) + 1, dtype=int))
    return events


def _download_price_history(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    try:
        import yfinance as yf  # type: ignore

        history = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
    except Exception:
        return pd.DataFrame(columns=["open", "close"])

    if history is None or history.empty:
        return pd.DataFrame(columns=["open", "close"])

    cols = {}
    if "Open" in history.columns:
        cols["Open"] = "open"
    if "Close" in history.columns:
        cols["Close"] = "close"

    out = history[list(cols.keys())].rename(columns=cols)
    out.index = normalize_datetime_index(pd.to_datetime(out.index, errors="raise"))
    out = out.sort_index()
    return out


def load_prices_and_compute_moves(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        empty = events.copy()
        for col in ("pre_close", "post_open", "post_close", "implied_move"):
            if col not in empty.columns:
                empty[col] = pd.Series(dtype="float")
        return empty

    enriched: list[pd.DataFrame] = []
    for ticker, group in events.groupby("ticker", sort=False):
        start = group["event_date"].min() - pd.Timedelta(days=45)
        end = group["event_date"].max() + pd.Timedelta(days=10)
        history = _download_price_history(ticker, start, end)

        group = group.copy()
        group["pre_close"] = pd.NA
        group["post_open"] = pd.NA
        group["post_close"] = pd.NA
        group["implied_move"] = pd.NA

        if history.empty:
            enriched.append(group)
            continue

        # Historical-volatility proxy for implied move when option chain snapshots are unavailable.
        history["daily_ret"] = history["close"].pct_change()
        history["hv20"] = history["daily_ret"].rolling(20, min_periods=10).std(ddof=1)

        idx = normalize_datetime_index(history.index)
        for row_idx, row in group.iterrows():
            event_date = to_naive_datetime(row["event_date"])
            if event_date is not None:
                event_date = event_date.normalize()
            idx = idx.sort_values()

            if event_date is None:
                raise ValueError("event_date is None")

            if not isinstance(idx, pd.DatetimeIndex):
                raise TypeError("Index must be DatetimeIndex")

            assert_same_timezone(idx, event_date)
            print(f"[DEBUG] event_date={event_date}, idx_sample={idx[:3]}")

            pre_pos = idx.searchsorted(event_date, side="left") - 1
            post_pos = idx.searchsorted(event_date, side="left")

            if pre_pos >= 0:
                pre_date = idx[pre_pos]
                group.at[row_idx, "pre_close"] = history.iloc[pre_pos]["close"]
                hv = history.loc[pre_date, "hv20"]
                if pd.notna(hv):
                    # Convert annualized daily-vol estimate to one-session expected absolute move.
                    group.at[row_idx, "implied_move"] = float(hv)

            if post_pos < len(history):
                group.at[row_idx, "post_open"] = history.iloc[post_pos]["open"]
                group.at[row_idx, "post_close"] = history.iloc[post_pos]["close"]

        enriched.append(group)

    return pd.concat(enriched, ignore_index=True)


def generate_features(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["event_id", "ticker", "event_date"])

    features = events.copy()
    pre_close = pd.to_numeric(features["pre_close"], errors="coerce")
    post_open = pd.to_numeric(features["post_open"], errors="coerce")
    post_close = pd.to_numeric(features["post_close"], errors="coerce")
    implied_move = pd.to_numeric(features["implied_move"], errors="coerce")

    features["realized_gap_move"] = (post_open / pre_close - 1.0).abs()
    features["realized_close_move"] = (post_close / pre_close - 1.0).abs()
    features["iv_minus_realized_gap"] = implied_move - features["realized_gap_move"]
    features["surprise_vs_implied"] = features["realized_gap_move"] - implied_move

    features = features.sort_values(["ticker", "event_date"]).copy()
    features["days_since_prev_event"] = (
        features.groupby("ticker")["event_date"].diff().dt.days.astype("float")
    )

    return features


def _safe_to_parquet(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(output_path, index=False)
    except Exception as exc:  # pragma: no cover - defensive runtime path
        raise RuntimeError(
            f"Failed writing parquet to {output_path}. Install pyarrow or fastparquet."
        ) from exc


def main() -> None:
    args = parse_args()
    tickers = _normalize_ticker_list(args.tickers)

    if not tickers:
        raise ValueError("At least one ticker must be provided via --tickers.")
    if args.max_events <= 0:
        raise ValueError("--max-events must be > 0.")

    start_date = to_naive(validate_datetime(args.start_date)).normalize()
    end_date = to_naive(validate_datetime(args.end_date)).normalize()
    if start_date > end_date:
        raise ValueError("--start-date must be <= --end-date.")

    # Pipeline stages.
    events = retrieve_and_normalize_events(tickers, start_date, end_date, args.max_events)
    events_with_prices = load_prices_and_compute_moves(events)
    feature_df = generate_features(events_with_prices)

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
    else:
        clean_events, diagnostics_summary, diagnostics_flags = run_event_diagnostics(
            events_with_prices,
            options_df=None,
            event_key_col="event_id",
            pre_close_col="pre_close",
            post_open_col="post_open",
            post_close_col="post_close",
            implied_move_col="implied_move",
            timing_col="announcement_timing",
            mapped_session_col="mapped_session",
        )

        # Coverage report provides additional validation-layer visibility.
        _ = build_data_coverage_report(
            diagnostics_flags.merge(
                events_with_prices[["event_id", "ticker", "event_date", "announcement_timing", "mapped_session"]],
                on="event_id",
                how="left",
            ),
            ticker_col="ticker",
            event_date_col="event_date",
            timing_col="announcement_timing",
            mapped_session_col="mapped_session",
        )

    final_features = feature_df.merge(
        diagnostics_flags[["event_id", "valid_event", "invalid_event", "failure_reasons"]],
        on="event_id",
        how="left",
    )

    failure_counts = (
        diagnostics_flags.loc[diagnostics_flags["invalid_event"], "failure_reasons"]
        .str.split(",")
        .explode()
        .dropna()
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .rename_axis("failure_reason")
        .reset_index(name="count")
    )

    print("\n=== Diagnostics summary ===")
    if diagnostics_summary.empty:
        print("No diagnostics rows produced.")
    else:
        print(diagnostics_summary.sort_values(["category", "metric"]).to_string(index=False))

    print("\n=== Final dataset sample ===")
    sample_cols = [
        "event_id",
        "ticker",
        "event_date",
        "pre_close",
        "post_open",
        "implied_move",
        "realized_gap_move",
        "surprise_vs_implied",
        "valid_event",
    ]
    if final_features.empty:
        print("No rows produced.")
    else:
        present_cols = [col for col in sample_cols if col in final_features.columns]
        print(final_features[present_cols].head(10).round(4).to_string(index=False))

    print("\n=== Failure counts ===")
    if failure_counts.empty:
        print("No failures detected.")
    else:
        print(failure_counts.to_string(index=False))

    output_dir = Path(args.output_dir)
    _safe_to_parquet(clean_events, output_dir / "events.parquet")
    _safe_to_parquet(final_features, output_dir / "features.parquet")

    diagnostics_output = diagnostics_summary.copy()
    if not failure_counts.empty:
        failures = failure_counts.rename(columns={"failure_reason": "metric"}).copy()
        failures.insert(0, "category", "failure_counts")
        failures["pct_of_events"] = failures["count"] / max(len(diagnostics_flags), 1)
        diagnostics_output = pd.concat([diagnostics_output, failures], ignore_index=True, sort=False)

    _safe_to_parquet(diagnostics_output, output_dir / "diagnostics.parquet")

    print("\nArtifacts written:")
    print(f"- {output_dir / 'events.parquet'}")
    print(f"- {output_dir / 'features.parquet'}")
    print(f"- {output_dir / 'diagnostics.parquet'}")


if __name__ == "__main__":
    main()
