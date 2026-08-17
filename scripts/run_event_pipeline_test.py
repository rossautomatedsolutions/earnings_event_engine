"""End-to-end earnings event pipeline smoke-test runner.

This script is the standard quick way to validate the system from raw event
retrieval through validation outputs. It now supports an explicit deterministic
fixture mode for offline contract testing.
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

from data.earnings_calendar import EarningsCalendarRetrievalError, get_earnings_dates
from earnings_event_engine.validation import build_data_coverage_report, run_event_diagnostics
from src.events import align_event_to_sessions
from src.events.timing import (
    TIMING_COLUMN_CANDIDATES,
    normalize_announcement_timing,
    resolve_announcement_timing,
    strip_series_timezone_preserving_wall_clock,
    strip_timezone_preserving_wall_clock,
)
from utils.datetime_utils import (
    assert_same_timezone,
    normalize_datetime_index,
    to_naive,
    validate_datetime,
)


EVENT_BASE_COLUMNS = [
    "event_id",
    "ticker",
    "earnings_date",
    "event_date",
    "timing",
    "announcement_timing",
    "mapped_session",
]
ENRICHED_EVENT_COLUMNS = [
    *EVENT_BASE_COLUMNS,
    "pre_close",
    "post_open",
    "post_close",
    "implied_move",
    "price_alignment_valid",
]
FEATURE_COLUMNS = [
    "event_id",
    "ticker",
    "earnings_date",
    "event_date",
    "timing",
    "announcement_timing",
    "mapped_session",
    "price_alignment_valid",
    "pre_close",
    "post_open",
    "post_close",
    "implied_move",
    "realized_gap_move",
    "realized_close_move",
    "iv_minus_realized_gap",
    "surprise_vs_implied",
    "days_since_prev_event",
    "macro_regime",
]
DIAGNOSTICS_FLAG_COLUMNS = ["event_id", "invalid_event", "valid_event", "failure_reasons"]
SUMMARY_COLUMNS = ["category", "metric", "count", "pct_of_events"]
SUPPORTED_TABLE_SUFFIXES = (".parquet", ".csv")


def parse_args() -> argparse.Namespace:
    today = to_naive(pd.Timestamp.now("UTC")).normalize()
    parser = argparse.ArgumentParser(
        description="Run the earnings event pipeline end-to-end for quick testing."
    )
    parser.add_argument(
        "--tickers",
        help="Comma-separated ticker list (example: AAPL,MSFT,NVDA). Required in live mode.",
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
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=None,
        help=(
            "Optional deterministic offline fixture directory containing "
            "earnings_events(.csv|.parquet), prices(.csv|.parquet), and optional "
            "options(.csv|.parquet)."
        ),
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow zero raw or zero validated rows without returning a non-zero exit code.",
    )
    return parser.parse_args()


def _normalize_ticker_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    tickers = [token.strip().upper() for token in raw.split(",")]
    return [ticker for ticker in tickers if ticker]


def _empty_events_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_BASE_COLUMNS)


def _empty_enriched_events_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=ENRICHED_EVENT_COLUMNS)


def _empty_features_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=FEATURE_COLUMNS)


def _empty_diagnostics_flags() -> pd.DataFrame:
    return pd.DataFrame(columns=DIAGNOSTICS_FLAG_COLUMNS)


def _base_diagnostics_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"category": "overview", "metric": "total_events", "count": 0, "pct_of_events": 0.0},
            {"category": "overview", "metric": "valid_events", "count": 0, "pct_of_events": 0.0},
            {"category": "overview", "metric": "invalid_events", "count": 0, "pct_of_events": 0.0},
        ],
        columns=SUMMARY_COLUMNS,
    )


def _append_summary_metric(summary_df: pd.DataFrame, metric: str, *, count: int = 1) -> pd.DataFrame:
    out = summary_df.copy()
    out.loc[len(out)] = {
        "category": "runtime",
        "metric": metric,
        "count": count,
        "pct_of_events": 0.0,
    }
    return out


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format for {path}. Expected .parquet or .csv.")


def _resolve_fixture_table(fixture_dir: Path, stem: str, *, required: bool = True) -> Path | None:
    for suffix in SUPPORTED_TABLE_SUFFIXES:
        candidate = fixture_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate

    if required:
        supported = ", ".join(f"{stem}{suffix}" for suffix in SUPPORTED_TABLE_SUFFIXES)
        raise FileNotFoundError(
            f"Could not find required fixture table in {fixture_dir}. Expected one of: {supported}"
        )
    return None


def load_fixture_tables(fixture_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    fixture_path = Path(fixture_dir)
    events_path = _resolve_fixture_table(fixture_path, "earnings_events", required=True)
    prices_path = _resolve_fixture_table(fixture_path, "prices", required=True)
    options_path = _resolve_fixture_table(fixture_path, "options", required=False)

    events_df = _read_table(events_path)
    prices_df = _read_table(prices_path)
    options_df = _read_table(options_path) if options_path is not None else None
    return events_df, prices_df, options_df


def _has_text(value: object) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    return bool(str(value).strip())


def _resolve_event_timing(row: pd.Series, timing_cols: list[str]) -> str | None:
    explicit_unsupported_raw: object | None = None
    for col in timing_cols:
        raw_value = row.get(col)
        normalized = normalize_announcement_timing(raw_value)
        if normalized is not None:
            return normalized
        if explicit_unsupported_raw is None and _has_text(raw_value):
            explicit_unsupported_raw = raw_value

    resolution = resolve_announcement_timing(explicit_unsupported_raw, row.get("earnings_date"))
    return resolution.timing


def _normalize_timing(events: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    timing_cols = [col for col in TIMING_COLUMN_CANDIDATES if col in events.columns]

    resolved_timing: list[object] = []
    mapped_sessions: list[object] = []
    for _, row in events.iterrows():
        timing = _resolve_event_timing(row, timing_cols)
        if timing == "BMO":
            resolved_timing.append("BMO")
            mapped_sessions.append("same_day")
        elif timing == "AMC":
            resolved_timing.append("AMC")
            mapped_sessions.append("next_day")
        else:
            resolved_timing.append(pd.NA)
            mapped_sessions.append(pd.NA)

    return (
        pd.Series(resolved_timing, index=events.index, dtype="object"),
        pd.Series(mapped_sessions, index=events.index, dtype="object"),
    )


def _normalize_events_frame(
    events: pd.DataFrame,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    max_events: int,
    requested_tickers: list[str] | None = None,
) -> pd.DataFrame:
    if events is None or events.empty:
        return _empty_events_frame()

    working = events.copy()
    if "ticker" not in working.columns:
        raise ValueError("Events input must include a 'ticker' column.")
    if "earnings_date" not in working.columns:
        if "event_date" in working.columns:
            working["earnings_date"] = working["event_date"]
        else:
            raise ValueError("Events input must include an 'earnings_date' column.")

    if requested_tickers:
        ticker_mask = working["ticker"].astype(str).str.upper().isin(requested_tickers)
        working = working.loc[ticker_mask].copy()
        if working.empty:
            return _empty_events_frame()

    working["earnings_date"] = strip_series_timezone_preserving_wall_clock(working["earnings_date"])

    parsed_earnings = pd.to_datetime(working["earnings_date"], errors="coerce")
    working = working.loc[parsed_earnings.notna()].copy()
    if working.empty:
        return _empty_events_frame()
    working["earnings_date"] = parsed_earnings.loc[parsed_earnings.notna()]

    print("[DEBUG] earnings_date dtype:", working["earnings_date"].dtype)
    print("[DEBUG] start_date:", start_date)
    print("[DEBUG] end_date:", end_date)

    mask = (working["earnings_date"] >= start_date) & (working["earnings_date"] <= end_date)
    working = working.loc[mask].copy()
    if working.empty:
        return _empty_events_frame()

    working["ticker"] = working["ticker"].astype(str).str.upper()
    working["event_date"] = working["earnings_date"].dt.normalize()
    working["timing"], working["mapped_session"] = _normalize_timing(working)
    working["announcement_timing"] = working["timing"]

    working = (
        working.sort_values(["event_date", "ticker"])
        .drop_duplicates(subset=["ticker", "event_date"], keep="first")
        .head(max_events)
        .reset_index(drop=True)
    )
    working.insert(0, "event_id", np.arange(1, len(working) + 1, dtype=int))

    ordered_cols = EVENT_BASE_COLUMNS + [col for col in working.columns if col not in EVENT_BASE_COLUMNS]
    return working.loc[:, ordered_cols]


def retrieve_and_normalize_events(
    tickers: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    max_events: int,
    *,
    events_source_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if events_source_df is not None:
        return _normalize_events_frame(
            events_source_df,
            start_date=start_date,
            end_date=end_date,
            max_events=max_events,
            requested_tickers=tickers,
        )

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        df = get_earnings_dates(ticker)
        if df.empty:
            continue
        frames.append(df)

    if not frames:
        return _empty_events_frame()

    return _normalize_events_frame(
        pd.concat(frames, ignore_index=True),
        start_date=start_date,
        end_date=end_date,
        max_events=max_events,
        requested_tickers=tickers,
    )


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


def _prepare_fixture_price_panel(prices_df: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "date", "open", "close"}
    missing = sorted(required.difference(prices_df.columns))
    if missing:
        raise ValueError(f"Fixture prices are missing required columns: {missing}")

    out = prices_df.copy()
    out["ticker"] = out["ticker"].astype("string").str.upper().str.strip()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["open"] = pd.to_numeric(out["open"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["ticker", "date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    return out.drop_duplicates(subset=["ticker", "date"], keep="last")


def _load_price_history_from_panel(
    prepared_panel: pd.DataFrame,
    *,
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    filtered = prepared_panel.loc[
        (prepared_panel["ticker"] == str(ticker).upper())
        & (prepared_panel["date"] >= start.normalize())
        & (prepared_panel["date"] <= end.normalize())
    ].copy()
    if filtered.empty:
        return pd.DataFrame(columns=["open", "close"])

    filtered = filtered.set_index("date")[["open", "close"]]
    filtered.index = normalize_datetime_index(filtered.index)
    return filtered.sort_index()


def load_prices_and_compute_moves(
    events: pd.DataFrame,
    *,
    price_panel_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if events.empty:
        return _empty_enriched_events_frame()

    prepared_panel = _prepare_fixture_price_panel(price_panel_df) if price_panel_df is not None else None

    def extract_scalar(x: object) -> float:
        if isinstance(x, pd.Series):
            if len(x) == 1:
                return float(x.iloc[0])
            return float(x.squeeze())
        return float(x)

    enriched: list[pd.DataFrame] = []
    for ticker, group in events.groupby("ticker", sort=False):
        start = group["event_date"].min() - pd.Timedelta(days=45)
        end = group["event_date"].max() + pd.Timedelta(days=10)
        if prepared_panel is None:
            history = _download_price_history(ticker, start, end)
        else:
            history = _load_price_history_from_panel(
                prepared_panel,
                ticker=ticker,
                start=start,
                end=end,
            )

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
            event_date = strip_timezone_preserving_wall_clock(row.get("earnings_date", row["event_date"]))

            if pd.isna(event_date) or not isinstance(idx, pd.DatetimeIndex):
                continue

            assert_same_timezone(idx, event_date)

            timing = normalize_announcement_timing(row.get("announcement_timing", row.get("timing")))
            if timing is None:
                continue

            alignment = align_event_to_sessions(
                event_date=event_date,
                announcement_timing=timing,
                price_index=idx,
            )
            if alignment["alignment_flag"] != "ok":
                continue

            pre_date = pd.Timestamp(alignment["pre_event_date"])
            post_date = pd.Timestamp(alignment["post_event_date"])

            group.at[row_idx, "pre_close"] = history.loc[pre_date, "close"]
            hv = history.loc[pre_date, "hv20"]
            if isinstance(hv, pd.Series):
                hv = hv.iloc[-1] if not hv.empty else None
            if hv is not None and pd.notna(hv):
                group.at[row_idx, "implied_move"] = float(hv)

            group.at[row_idx, "post_open"] = history.loc[post_date, "open"]
            group.at[row_idx, "post_close"] = history.loc[post_date, "close"]

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

            group.at[row_idx, "pre_close"] = pre_close
            group.at[row_idx, "post_open"] = post_open
            group.at[row_idx, "post_close"] = post_close

            valid = (
                pd.notna(pre_close)
                and pre_close > 0
                and pd.notna(post_open)
                and post_open > 0
                and pd.notna(post_close)
                and post_close > 0
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

    out = pd.concat(enriched, ignore_index=True)
    for column in ENRICHED_EVENT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    ordered_cols = ENRICHED_EVENT_COLUMNS + [col for col in out.columns if col not in ENRICHED_EVENT_COLUMNS]
    return out.loc[:, ordered_cols]


def generate_features(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return _empty_features_frame()

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
            features["macro_regime"].astype("string").str.strip().fillna("unknown").replace("", "unknown")
        )

    for column in FEATURE_COLUMNS:
        if column not in features.columns:
            features[column] = pd.NA
    return features.loc[:, FEATURE_COLUMNS]


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
            "Coverage input is missing required event columns: " + ", ".join(missing_columns)
        )

    coverage_input = diagnostics_flags.merge(
        events_with_prices[required_event_columns],
        on="event_id",
        how="left",
    )

    if "price_alignment_valid" not in coverage_input.columns:
        raise AssertionError("Coverage input must include price_alignment_valid.")

    return coverage_input


def load_fixture_options_for_events(
    events_df: pd.DataFrame,
    options_source_df: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if options_source_df is None:
        return None
    if events_df.empty:
        return pd.DataFrame(columns=["event_id", "option_type", "strike", "underlying_price", "bid", "ask", "mid"])

    required = {"ticker", "event_date", "option_type", "strike", "underlying_price", "bid", "ask", "mid"}
    missing = sorted(required.difference(options_source_df.columns))
    if missing:
        raise ValueError(f"Fixture options are missing required columns: {missing}")

    options = options_source_df.copy()
    options["ticker"] = options["ticker"].astype("string").str.upper().str.strip()
    options["event_date"] = pd.to_datetime(options["event_date"], errors="coerce").dt.normalize()

    mapping = events_df[["event_id", "ticker", "event_date"]].copy()
    merged = mapping.merge(options, on=["ticker", "event_date"], how="inner")

    return merged[["event_id", "option_type", "strike", "underlying_price", "bid", "ask", "mid"]]


def _build_failure_counts(diagnostics_flags: pd.DataFrame) -> pd.DataFrame:
    if diagnostics_flags.empty:
        return pd.DataFrame(columns=["failure_reason", "count"])
    if "invalid_event" not in diagnostics_flags.columns or "failure_reasons" not in diagnostics_flags.columns:
        return pd.DataFrame(columns=["failure_reason", "count"])

    return (
        diagnostics_flags.loc[diagnostics_flags["invalid_event"], "failure_reasons"]
        .astype("string")
        .str.split(",")
        .explode()
        .dropna()
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .rename_axis("failure_reason")
        .reset_index(name="count")
    )


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

    if args.max_events <= 0:
        raise ValueError("--max-events must be > 0.")
    if args.fixture_dir is None and not tickers:
        raise ValueError("At least one ticker must be provided via --tickers in live mode.")

    start_date = to_naive(validate_datetime(args.start_date)).normalize()
    end_date = to_naive(validate_datetime(args.end_date)).normalize()
    if start_date > end_date:
        raise ValueError("--start-date must be <= --end-date.")

    mode_label = "fixture" if args.fixture_dir is not None else "live"
    print(f"[INFO] run_mode={mode_label}")
    print(f"[INFO] allow_empty={args.allow_empty}")

    events = _empty_events_frame()
    events_with_prices = _empty_enriched_events_frame()
    clean_events = _empty_enriched_events_frame()
    feature_df = _empty_features_frame()
    diagnostics_summary = _base_diagnostics_summary()
    diagnostics_flags = _empty_diagnostics_flags()
    runtime_metrics: list[str] = []
    failure_messages: list[str] = []

    try:
        if args.fixture_dir is not None:
            fixture_events, fixture_prices, fixture_options_source = load_fixture_tables(args.fixture_dir)
            if not tickers:
                tickers = (
                    fixture_events["ticker"].astype("string").str.upper().dropna().drop_duplicates().tolist()
                    if "ticker" in fixture_events.columns
                    else []
                )
            print(f"[INFO] fixture_dir={Path(args.fixture_dir).resolve()}")
            events = retrieve_and_normalize_events(
                tickers,
                start_date,
                end_date,
                args.max_events,
                events_source_df=fixture_events,
            )
            events_with_prices = load_prices_and_compute_moves(events, price_panel_df=fixture_prices)
            feature_df = generate_features(events_with_prices)
            options_df = load_fixture_options_for_events(events, fixture_options_source)
        else:
            events = retrieve_and_normalize_events(tickers, start_date, end_date, args.max_events)
            events_with_prices = load_prices_and_compute_moves(events)
            feature_df = generate_features(events_with_prices)
            options_df = None

        if not events_with_prices.empty:
            clean_events, diagnostics_summary, diagnostics_flags = run_event_diagnostics(
                events_with_prices,
                options_df=options_df,
                event_key_col="event_id",
                pre_close_col="pre_close",
                post_open_col="post_open",
                post_close_col="post_close",
                implied_move_col="implied_move",
                timing_col="announcement_timing",
                mapped_session_col="mapped_session",
            )

            _ = build_data_coverage_report(
                _build_coverage_input_df(diagnostics_flags, events_with_prices),
                ticker_col="ticker",
                event_date_col="event_date",
                price_valid_col="price_alignment_valid",
                timing_col="announcement_timing",
                mapped_session_col="mapped_session",
            )
    except EarningsCalendarRetrievalError as exc:
        failure_messages.append(str(exc))
        runtime_metrics.append("retrieval_failure")
        options_df = None

    final_features = feature_df.merge(
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
            failure_messages.append(f"Zero raw events produced during the retrieval stage in {mode_label} mode.")

    if raw_event_count > 0 and valid_event_count == 0:
        runtime_metrics.append("zero_valid_events")
        if not args.allow_empty:
            failure_messages.append(
                f"Zero valid/clean events produced during the validation stage in {mode_label} mode."
            )

    if feature_row_count == 0 and raw_event_count > 0:
        runtime_metrics.append("zero_feature_rows")

    diagnostics_output = diagnostics_summary.copy()
    for metric in runtime_metrics:
        if metric not in diagnostics_output["metric"].tolist():
            diagnostics_output = _append_summary_metric(diagnostics_output, metric)

    failure_counts = _build_failure_counts(diagnostics_flags)
    if not failure_counts.empty:
        failures = failure_counts.rename(columns={"failure_reason": "metric"}).copy()
        failures.insert(0, "category", "failure_counts")
        failures["pct_of_events"] = failures["count"] / max(len(diagnostics_flags), 1)
        diagnostics_output = pd.concat([diagnostics_output, failures], ignore_index=True, sort=False)

    print(f"[INFO] raw_events={raw_event_count}")
    print(f"[INFO] clean_events={valid_event_count}")
    print(f"[INFO] feature_rows={feature_row_count}")

    print("\n=== Diagnostics summary ===")
    if diagnostics_output.empty:
        print("No diagnostics rows produced.")
    else:
        print(diagnostics_output.sort_values(["category", "metric"]).to_string(index=False))

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
        display_df = final_features[present_cols].head(10).copy()
        numeric_cols = display_df.select_dtypes(include=[np.number]).columns
        display_df.loc[:, numeric_cols] = display_df.loc[:, numeric_cols].round(4)
        print(display_df.to_string(index=False))

    print("\n=== Failure counts ===")
    if failure_counts.empty:
        print("No failures detected.")
    else:
        print(failure_counts.to_string(index=False))

    output_dir = Path(args.output_dir)
    _safe_to_parquet(events, output_dir / "events.parquet")
    _safe_to_parquet(clean_events, output_dir / "clean_events.parquet")
    _safe_to_parquet(final_features, output_dir / "features.parquet")
    _safe_to_parquet(diagnostics_output, output_dir / "diagnostics.parquet")

    print("\nArtifacts written:")
    print(f"- {output_dir / 'events.parquet'}")
    print(f"- {output_dir / 'clean_events.parquet'}")
    print(f"- {output_dir / 'features.parquet'}")
    print(f"- {output_dir / 'diagnostics.parquet'}")

    if failure_messages:
        raise SystemExit("\n".join(failure_messages))


if __name__ == "__main__":
    main()
