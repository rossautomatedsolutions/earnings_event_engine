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
    inferred = np.where(hours.fillna(16) < 12, "BMO", "AMC")
    return pd.Series(inferred, index=earnings_ts.index, dtype="object")


def _normalize_timing(events: pd.DataFrame) -> pd.Series:
    timing_aliases = {
        "amc": "AMC",
        "after market close": "AMC",
        "after_close": "AMC",
        "post_market": "AMC",
        "post-close": "AMC",
        "post_close": "AMC",
        "bmo": "BMO",
        "before market open": "BMO",
        "before_open": "BMO",
        "pre_market": "BMO",
        "pre-open": "BMO",
        "pre_open": "BMO",
    }
    timing_cols = [
        col
        for col in ("timing", "announcement_timing", "earnings_timing", "time_of_day", "session")
        if col in events.columns
    ]

    normalized = pd.Series("UNKNOWN", index=events.index, dtype="object")
    for col in timing_cols:
        raw = (
            events[col]
            .astype("string")
            .str.strip()
            .str.lower()
        )
        mapped = raw.map(timing_aliases).fillna("UNKNOWN")
        normalized = np.where(normalized.eq("UNKNOWN"), mapped, normalized)
        normalized = pd.Series(normalized, index=events.index, dtype="object")

    inferred = _infer_timing(events["earnings_date"])
    missing = normalized.eq("UNKNOWN")
    normalized.loc[missing] = inferred.loc[missing]
    normalized = normalized.fillna("AMC")
    return normalized.replace("UNKNOWN", "AMC")


def _build_coverage_input_df(
    diagnostics_flags: pd.DataFrame,
    events_with_prices: pd.DataFrame,
) -> pd.DataFrame:
    required_event_columns = [
        "event_id",
        "ticker",
        "event_date",
        "announcement_timing",
        "mapped_session",
        "price_alignment_valid",
    ]
    missing_columns = [col for col in required_event_columns if col not in events_with_prices.columns]
    if missing_columns:
        raise AssertionError(
            "Coverage input is missing required event columns: "
            + ", ".join(missing_columns)
        )

    coverage_input = diagnostics_flags.merge(
        events_with_prices[required_event_columns],
        on="event_id",
        how="left",
    )

    if "price_alignment_valid" not in coverage_input.columns:
        raise AssertionError("Coverage input must include price_alignment_valid.")

    return coverage_input


def retrieve_and_normalize_events(
    tickers: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    max_events: int,
) -> pd.DataFrame:
    import pandas as pd

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

    # Normalize earnings_date column
    events["earnings_date"] = pd.to_datetime(events["earnings_date"], errors="coerce")

    # Force tz-naive
    if hasattr(events["earnings_date"].dt, "tz") and events["earnings_date"].dt.tz is not None:
        events["earnings_date"] = events["earnings_date"].dt.tz_convert(None)

    parsed_earnings = pd.to_datetime(events["earnings_date"], errors="coerce")
    events = events.loc[parsed_earnings.notna()].copy()
    events["earnings_date"] = parsed_earnings.loc[parsed_earnings.notna()]

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if start_date.tzinfo is not None:
        start_date = start_date.tz_convert(None)

    if end_date.tzinfo is not None:
        end_date = end_date.tz_convert(None)

    print("[DEBUG] earnings_date dtype:", events["earnings_date"].dtype)
    print("[DEBUG] start_date:", start_date)
    print("[DEBUG] end_date:", end_date)

    mask = (
        (events["earnings_date"] >= start_date)
        & (events["earnings_date"] <= end_date)
    )
    events = events.loc[mask].copy()

    events["ticker"] = events["ticker"].astype(str).str.upper()
    events["event_date"] = events["earnings_date"].dt.normalize()
    events["timing"] = _normalize_timing(events)
    events["announcement_timing"] = events["timing"]
    events["mapped_session"] = np.where(events["timing"].eq("BMO"), "same_day", "next_day")

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

    def extract_scalar(x):
        if isinstance(x, pd.Series):
            if len(x) == 1:
                return float(x.iloc[0])
            return float(x.squeeze())
        return float(x)

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
        group["price_alignment_valid"] = False

        if history.empty:
            enriched.append(group)
            continue

        # Historical-volatility proxy for implied move when option chain snapshots are unavailable.
        history["daily_ret"] = history["close"].pct_change()
        history["hv20"] = history["daily_ret"].rolling(20, min_periods=10).std(ddof=1)

        idx = normalize_datetime_index(history.index).sort_values()
        debug_printed = 0
        for row_idx, row in group.iterrows():
            event_date = to_naive_datetime(row["event_date"])
            if event_date is not None:
                event_date = event_date.normalize()

            if event_date is None:
                continue

            if not isinstance(idx, pd.DatetimeIndex):
                continue

            assert_same_timezone(idx, event_date)

            timing = str(row.get("timing", "AMC")).upper()
            if timing not in {"AMC", "BMO"}:
                timing = "AMC"

            pos = idx.searchsorted(event_date, side="left")
            exact_match = pos < len(idx) and idx[pos] == event_date

            pre_idx = None
            post_idx = None
            if timing == "AMC":
                pre_idx = pos if exact_match else pos - 1
                post_idx = pos + 1
            elif timing == "BMO":
                pre_idx = pos - 1
                post_idx = pos

            has_pre = pre_idx is not None and 0 <= pre_idx < len(idx)
            has_post = post_idx is not None and 0 <= post_idx < len(idx)

            if has_pre:
                pre_date = idx[pre_idx]
                group.at[row_idx, "pre_close"] = history.iloc[pre_idx]["close"]
                hv = history.loc[pre_date, "hv20"]
                if isinstance(hv, pd.Series):
                    hv = hv.iloc[-1] if not hv.empty else None
                if hv is not None and pd.notna(hv):
                    group.at[row_idx, "implied_move"] = float(hv)

            if has_post:
                group.at[row_idx, "post_open"] = history.iloc[post_idx]["open"]
                group.at[row_idx, "post_close"] = history.iloc[post_idx]["close"]

            pre_close = group.at[row_idx, "pre_close"]
            post_open = group.at[row_idx, "post_open"]
            post_close = group.at[row_idx, "post_close"]

            try:
                pre_close = extract_scalar(pre_close)
            except Exception:
                pre_close = None
            try:
                post_open = extract_scalar(post_open)
            except Exception:
                post_open = None
            try:
                post_close = extract_scalar(post_close)
            except Exception:
                post_close = None

            assert not isinstance(pre_close, pd.Series)
            assert not isinstance(post_open, pd.Series)
            assert not isinstance(post_close, pd.Series)

            group.at[row_idx, "pre_close"] = pre_close
            group.at[row_idx, "post_open"] = post_open
            group.at[row_idx, "post_close"] = post_close

            valid = (
                pd.notna(pre_close) and pre_close > 0
                and pd.notna(post_open) and post_open > 0
                and pd.notna(post_close) and post_close > 0
            )
            group.at[row_idx, "price_alignment_valid"] = bool(valid)

            if debug_printed < 5:
                print(
                    "[DEBUG] price_alignment "
                    f"event_date={event_date.date()} timing={timing} "
                    f"pre_close={group.at[row_idx, 'pre_close']} "
                    f"post_open={group.at[row_idx, 'post_open']} "
                    f"post_close={group.at[row_idx, 'post_close']}"
                )
                debug_printed += 1

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

    macro_regime_path = REPO_ROOT / "data" / "inputs" / "macro_regime.csv"
    if macro_regime_path.exists():
        macro_regime_df = pd.read_csv(macro_regime_path)
        required_cols = {"event_date", "macro_regime"}
        if required_cols.issubset(macro_regime_df.columns):
            macro_regime_df = macro_regime_df.loc[:, ["event_date", "macro_regime"]].copy()
            macro_regime_df["event_date"] = pd.to_datetime(
                macro_regime_df["event_date"],
                errors="coerce",
            ).dt.normalize()
            macro_regime_df = macro_regime_df.dropna(subset=["event_date"]).drop_duplicates(
                subset=["event_date"],
                keep="last",
            )
            features["event_date"] = pd.to_datetime(features["event_date"], errors="coerce").dt.normalize()
            features = features.merge(macro_regime_df, on="event_date", how="left")

    if "macro_regime" not in features.columns:
        features["macro_regime"] = "unknown"
    else:
        features["macro_regime"] = (
            features["macro_regime"]
            .astype("string")
            .str.strip()
            .fillna("unknown")
            .replace("", "unknown")
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
            _build_coverage_input_df(diagnostics_flags, events_with_prices),
            ticker_col="ticker",
            event_date_col="event_date",
            price_valid_col="price_alignment_valid",
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
