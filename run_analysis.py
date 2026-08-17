"""CLI workflow for historical earnings-event analysis.

This module computes realized post-earnings move statistics from historical
price data. Event-time IV-dependent metrics are only populated when explicit
per-event IV inputs are supplied; a current option-chain snapshot must never be
reused across historical events. Any return metric derived from event-time IV in
this module is an expected-move proxy, not observed historical option P&L.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd

from data.earnings_calendar import get_earnings_dates
from src.events import align_event_to_sessions


@dataclass
class AnalysisResult:
    """Container for key summary outputs."""

    ticker: str
    earnings_events: int
    usable_events: int
    events_with_event_time_iv: int
    avg_abs_move_pct: float
    avg_signed_move_pct: float
    avg_event_time_iv_minus_realized_pct: float
    avg_event_time_expected_move_proxy_return_pct: float


EVENT_ROW_COLUMNS = [
    "ticker",
    "earnings_date",
    "pre_day",
    "post_day",
    "pre_close",
    "post_close",
    "signed_move_pct",
    "abs_move_pct",
    "realized_vol_annual_pct",
    "event_time_iv_available",
    "event_time_iv_annual_pct",
    "expected_move_pct",
    "event_time_iv_minus_realized_pct",
    "event_time_expected_move_proxy_return_pct",
]

_TIMING_COLUMN_CANDIDATES = (
    "announcement_timing",
    "timing",
    "earnings_timing",
    "earnings_time",
    "time_of_day",
    "session",
)
_TIMING_ALIASES = {
    "amc": "AMC",
    "after_close": "AMC",
    "after_market_close": "AMC",
    "after_market": "AMC",
    "post_close": "AMC",
    "post_market": "AMC",
    "bmo": "BMO",
    "before_open": "BMO",
    "before_market_open": "BMO",
    "before_market": "BMO",
    "pre_market": "BMO",
    "pre_open": "BMO",
}
_PREMARKET_CUTOFF = time(hour=9, minute=30)
_POSTCLOSE_CUTOFF = time(hour=16, minute=0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run earnings-event analysis for a ticker.")
    parser.add_argument("--ticker", "-t", required=True, help="Ticker symbol to analyze, e.g. AAPL")
    parser.add_argument(
        "--window-days",
        type=int,
        default=20,
        help="Trading-day lookback used for pre-earnings realized volatility (default: 20)",
    )
    return parser.parse_args()


def _load_history(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Load adjusted daily close history for a ticker within a date range."""
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
        return pd.DataFrame(columns=["close"])

    if history is None or history.empty:
        return pd.DataFrame(columns=["close"])

    close = history[["Close"]].rename(columns={"Close": "close"}).copy()
    close.index = pd.to_datetime(close.index)
    close = close[~close.index.duplicated(keep="last")]
    return close.sort_index()


def _estimate_current_front_month_iv(ticker: str) -> float | None:
    """Estimate a current front-month ATM IV snapshot.

    This value reflects the option chain and spot available at call time, not
    the IV that prevailed at a historical earnings event. It must stay separate
    from historical event analysis unless a caller is explicitly performing
    present-day snapshot inspection.
    """
    try:
        import yfinance as yf  # type: ignore

        tk = yf.Ticker(ticker)
        expiries = tk.options
        if not expiries:
            return None

        chain = tk.option_chain(expiries[0])
        calls = chain.calls
        puts = chain.puts
        if calls.empty or puts.empty:
            return None

        # Approximate ATM as strike nearest to latest close.
        hist = tk.history(period="5d", auto_adjust=True)
        if hist.empty:
            return None
        spot = float(hist["Close"].dropna().iloc[-1])

        call_idx = (calls["strike"] - spot).abs().idxmin()
        put_idx = (puts["strike"] - spot).abs().idxmin()

        ivs = [calls.loc[call_idx, "impliedVolatility"], puts.loc[put_idx, "impliedVolatility"]]
        ivs = [float(iv) for iv in ivs if pd.notna(iv)]
        if not ivs:
            return None

        return float(np.mean(ivs))
    except Exception:
        return None


def _empty_event_time_iv_inputs() -> pd.DataFrame:
    return pd.DataFrame(columns=["ticker", "earnings_date", "event_time_iv_annual"])


def _first_present(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _strip_timezone_preserving_wall_clock(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if hasattr(parsed.dt, "tz") and parsed.dt.tz is not None:
        return parsed.dt.tz_localize(None)
    return parsed


def _normalize_timing_value(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None

    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return None
    return _TIMING_ALIASES.get(normalized)


def _infer_timing_from_timestamp(earnings_date: pd.Timestamp) -> str | None:
    event_ts = pd.Timestamp(earnings_date)
    if pd.isna(event_ts):
        return None

    if event_ts == event_ts.normalize():
        return None

    event_time = event_ts.time()
    if event_time < _PREMARKET_CUTOFF:
        return "BMO"
    if event_time >= _POSTCLOSE_CUTOFF:
        return "AMC"
    return None


def _raise_unresolvable_timing(
    *,
    ticker: str,
    earnings_date: pd.Timestamp,
    raw_timing: object,
    reason: str,
) -> None:
    raw_label = "<missing>" if raw_timing is None or pd.isna(raw_timing) else repr(raw_timing)
    raise ValueError(
        "Unable to resolve announcement timing for "
        f"ticker={ticker} earnings_date={pd.Timestamp(earnings_date).isoformat(sep=' ')}: "
        f"{raw_label} ({reason})"
    )


def _prepare_earnings_events(ticker: str, earnings_df: pd.DataFrame) -> pd.DataFrame:
    if earnings_df is None or earnings_df.empty:
        return pd.DataFrame(columns=["ticker", "earnings_date", "announcement_timing"])

    if "earnings_date" not in earnings_df.columns:
        raise ValueError("earnings_df must contain an 'earnings_date' column")

    prepared = earnings_df.copy()
    prepared["ticker"] = prepared.get("ticker", ticker)
    prepared["ticker"] = prepared["ticker"].astype("string").fillna(ticker).str.upper().str.strip()
    prepared["earnings_date"] = _strip_timezone_preserving_wall_clock(prepared["earnings_date"])
    prepared = prepared.dropna(subset=["earnings_date"]).copy()

    if prepared.empty:
        return pd.DataFrame(columns=["ticker", "earnings_date", "announcement_timing"])

    timing_col = _first_present(prepared.columns, _TIMING_COLUMN_CANDIDATES)
    resolved_timing: list[str] = []
    for row in prepared.itertuples(index=False):
        raw_timing = getattr(row, timing_col) if timing_col is not None else None
        ticker_value = str(getattr(row, "ticker", ticker)).strip().upper() or str(ticker).strip().upper()
        earnings_date = pd.Timestamp(getattr(row, "earnings_date"))

        normalized_timing = _normalize_timing_value(raw_timing)
        if normalized_timing is not None:
            resolved_timing.append(normalized_timing)
            continue

        if raw_timing is not None and not pd.isna(raw_timing) and str(raw_timing).strip():
            _raise_unresolvable_timing(
                ticker=ticker_value,
                earnings_date=earnings_date,
                raw_timing=raw_timing,
                reason="unsupported timing label; expected AMC/BMO-style session tags",
            )

        inferred_timing = _infer_timing_from_timestamp(earnings_date)
        if inferred_timing is None:
            _raise_unresolvable_timing(
                ticker=ticker_value,
                earnings_date=earnings_date,
                raw_timing=raw_timing,
                reason=(
                    "missing or ambiguous timing; provide announcement_timing or a clearly "
                    "pre-market/post-close timestamp"
                ),
            )

        resolved_timing.append(inferred_timing)

    prepared["announcement_timing"] = resolved_timing
    return (
        prepared.loc[:, ["ticker", "earnings_date", "announcement_timing"]]
        .sort_values(["ticker", "earnings_date"])
        .reset_index(drop=True)
    )


def _raise_invalid_event_time_iv_value(
    row: pd.Series,
    raw_value: object,
    *,
    reason: str,
) -> None:
    ticker = row.get("ticker", "UNKNOWN")
    earnings_date = row.get("earnings_date")
    if pd.isna(earnings_date):
        earnings_date_label = "missing"
    else:
        earnings_date_label = pd.Timestamp(earnings_date).date().isoformat()

    raise ValueError(
        "Invalid event_time_iv_annual for "
        f"ticker={ticker} earnings_date={earnings_date_label}: {raw_value!r} ({reason})"
    )


def _prepare_event_time_iv_inputs(
    ticker: str,
    event_time_iv_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Validate optional event-time IV inputs for one ticker.

    Expected columns:
    - ``earnings_date``: event date/timestamp for the earnings event
    - ``event_time_iv_annual``: annualized IV in decimal units (0.60 = 60%)
    - optional ``ticker`` column when the input contains multiple names
    """
    if event_time_iv_df is None or event_time_iv_df.empty:
        return _empty_event_time_iv_inputs()

    required = {"earnings_date", "event_time_iv_annual"}
    missing = sorted(required - set(event_time_iv_df.columns))
    if missing:
        raise ValueError(f"event_time_iv_df is missing required columns: {missing}")

    normalized_ticker = str(ticker).strip().upper()
    prepared = event_time_iv_df.copy()
    if "ticker" in prepared.columns:
        prepared["ticker"] = prepared["ticker"].astype("string").str.upper().str.strip()
        prepared = prepared.loc[prepared["ticker"] == normalized_ticker].copy()
    else:
        prepared["ticker"] = normalized_ticker

    if prepared.empty:
        return _empty_event_time_iv_inputs()

    prepared["earnings_date"] = pd.to_datetime(prepared["earnings_date"], errors="coerce")
    if hasattr(prepared["earnings_date"].dt, "tz") and prepared["earnings_date"].dt.tz is not None:
        prepared["earnings_date"] = prepared["earnings_date"].dt.tz_localize(None)
    prepared["earnings_date"] = prepared["earnings_date"].dt.normalize()
    prepared = prepared.dropna(subset=["earnings_date"]).copy()

    raw_iv_values = prepared["event_time_iv_annual"].copy()
    coerced_iv_values = pd.to_numeric(raw_iv_values, errors="coerce")

    invalid_non_numeric_mask = raw_iv_values.notna() & coerced_iv_values.isna()
    if invalid_non_numeric_mask.any():
        invalid_index = invalid_non_numeric_mask[invalid_non_numeric_mask].index[0]
        _raise_invalid_event_time_iv_value(
            prepared.loc[invalid_index],
            raw_iv_values.loc[invalid_index],
            reason="must be numeric or null",
        )

    invalid_non_finite_mask = coerced_iv_values.notna() & ~np.isfinite(coerced_iv_values)
    if invalid_non_finite_mask.any():
        invalid_index = invalid_non_finite_mask[invalid_non_finite_mask].index[0]
        _raise_invalid_event_time_iv_value(
            prepared.loc[invalid_index],
            raw_iv_values.loc[invalid_index],
            reason="must be finite",
        )

    invalid_negative_mask = coerced_iv_values.notna() & (coerced_iv_values < 0)
    if invalid_negative_mask.any():
        invalid_index = invalid_negative_mask[invalid_negative_mask].index[0]
        _raise_invalid_event_time_iv_value(
            prepared.loc[invalid_index],
            raw_iv_values.loc[invalid_index],
            reason="must be non-negative",
        )

    prepared["event_time_iv_annual"] = coerced_iv_values

    duplicate_mask = prepared.duplicated(subset=["ticker", "earnings_date"], keep=False)
    if duplicate_mask.any():
        duplicate_dates = (
            prepared.loc[duplicate_mask, "earnings_date"].dt.date.astype(str).drop_duplicates().tolist()
        )
        raise ValueError(
            "event_time_iv_df contains duplicate event-time IV rows for the same ticker/date: "
            + ", ".join(duplicate_dates)
        )

    return (
        prepared.loc[:, ["ticker", "earnings_date", "event_time_iv_annual"]]
        .sort_values(["ticker", "earnings_date"])
        .reset_index(drop=True)
    )


def _compute_event_rows(
    ticker: str,
    earnings_events: pd.DataFrame,
    history: pd.DataFrame,
    lookback_window: int,
    event_time_iv_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    if history.empty or earnings_events.empty:
        return pd.DataFrame(columns=EVENT_ROW_COLUMNS)

    daily_ret = history["close"].pct_change()
    event_time_iv_inputs = _prepare_event_time_iv_inputs(ticker, event_time_iv_df)
    iv_lookup: dict[pd.Timestamp, float] = {}
    if not event_time_iv_inputs.empty:
        iv_lookup = event_time_iv_inputs.set_index("earnings_date")["event_time_iv_annual"].to_dict()

    normalized_earnings_events = earnings_events.copy()
    normalized_earnings_events["earnings_date"] = _strip_timezone_preserving_wall_clock(
        normalized_earnings_events["earnings_date"]
    )
    normalized_earnings_events = normalized_earnings_events.sort_values("earnings_date").reset_index(drop=True)

    for event in normalized_earnings_events.itertuples(index=False):
        earnings_date = pd.Timestamp(event.earnings_date)
        alignment = align_event_to_sessions(
            event_date=earnings_date,
            announcement_timing=event.announcement_timing,
            price_index=history.index,
        )
        if alignment["alignment_flag"] != "ok":
            continue

        pre_day = pd.Timestamp(alignment["pre_event_date"])
        post_day = pd.Timestamp(alignment["post_event_date"])

        pre_price = float(history.loc[pre_day, "close"])
        post_price = float(history.loc[post_day, "close"])
        if pre_price <= 0:
            continue

        signed_move = (post_price / pre_price) - 1.0
        abs_move = abs(signed_move)

        window_rets = daily_ret.loc[:pre_day].dropna().tail(lookback_window)
        if window_rets.empty:
            continue

        realized_daily = float(window_rets.std(ddof=1)) if len(window_rets) > 1 else float(window_rets.std(ddof=0))
        realized_annual = realized_daily * math.sqrt(252)

        event_date_key = pd.Timestamp(earnings_date).normalize()
        event_time_iv = iv_lookup.get(event_date_key)
        event_time_iv_available = event_time_iv is not None and pd.notna(event_time_iv)
        iv_minus_realized = np.nan if not event_time_iv_available else float(event_time_iv) - realized_annual

        # Expected-move proxy: compare realized absolute move with the IV-implied daily move.
        # This is not observed historical option-premium P&L.
        expected_move = np.nan if not event_time_iv_available else float(event_time_iv) / math.sqrt(252)
        expected_move_proxy_return = np.nan if np.isnan(expected_move) else (abs_move / expected_move) - 1.0

        rows.append(
            {
                "ticker": ticker,
                "earnings_date": earnings_date.date().isoformat(),
                "pre_day": pre_day.date().isoformat(),
                "post_day": post_day.date().isoformat(),
                "pre_close": pre_price,
                "post_close": post_price,
                "signed_move_pct": signed_move * 100.0,
                "abs_move_pct": abs_move * 100.0,
                "realized_vol_annual_pct": realized_annual * 100.0,
                "event_time_iv_available": bool(event_time_iv_available),
                "event_time_iv_annual_pct": np.nan if not event_time_iv_available else float(event_time_iv) * 100.0,
                "expected_move_pct": expected_move * 100.0,
                "event_time_iv_minus_realized_pct": iv_minus_realized * 100.0,
                "event_time_expected_move_proxy_return_pct": expected_move_proxy_return * 100.0,
            }
        )

    return pd.DataFrame(rows, columns=EVENT_ROW_COLUMNS)


def _summarize(ticker: str, earnings_events: int, event_rows: pd.DataFrame) -> AnalysisResult:
    usable = len(event_rows)
    if usable == 0:
        return AnalysisResult(
            ticker=ticker,
            earnings_events=earnings_events,
            usable_events=0,
            events_with_event_time_iv=0,
            avg_abs_move_pct=float("nan"),
            avg_signed_move_pct=float("nan"),
            avg_event_time_iv_minus_realized_pct=float("nan"),
            avg_event_time_expected_move_proxy_return_pct=float("nan"),
        )

    event_time_iv_available = event_rows.get("event_time_iv_available", pd.Series(dtype="bool"))
    return AnalysisResult(
        ticker=ticker,
        earnings_events=earnings_events,
        usable_events=usable,
        events_with_event_time_iv=int(pd.Series(event_time_iv_available).fillna(False).sum()),
        avg_abs_move_pct=float(event_rows["abs_move_pct"].mean()),
        avg_signed_move_pct=float(event_rows["signed_move_pct"].mean()),
        avg_event_time_iv_minus_realized_pct=float(
            event_rows["event_time_iv_minus_realized_pct"].mean(skipna=True)
        ),
        avg_event_time_expected_move_proxy_return_pct=float(
            event_rows["event_time_expected_move_proxy_return_pct"].mean(skipna=True)
        ),
    )


def run_analysis(
    ticker: str,
    window_days: int = 20,
    event_time_iv_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, AnalysisResult]:
    """Run historical earnings move analysis.

    Parameters
    ----------
    ticker
        Single ticker symbol to analyze.
    window_days
        Trading-day lookback used for pre-event realized volatility.
    event_time_iv_df
        Optional explicit per-event IV inputs with columns ``earnings_date`` and
        ``event_time_iv_annual`` in decimal units. When omitted, historical
        price-based metrics still populate while IV-dependent metrics remain
        unavailable. Any return metric produced from these inputs is an
        expected-move proxy rather than observed option-premium P&L.
    """
    normalized_ticker = str(ticker).strip().upper()
    earnings_df = get_earnings_dates(normalized_ticker)

    if earnings_df.empty:
        empty_summary = AnalysisResult(
            ticker=normalized_ticker,
            earnings_events=0,
            usable_events=0,
            events_with_event_time_iv=0,
            avg_abs_move_pct=float("nan"),
            avg_signed_move_pct=float("nan"),
            avg_event_time_iv_minus_realized_pct=float("nan"),
            avg_event_time_expected_move_proxy_return_pct=float("nan"),
        )
        return pd.DataFrame(columns=EVENT_ROW_COLUMNS), empty_summary

    earnings_events = _prepare_earnings_events(normalized_ticker, earnings_df)
    if earnings_events.empty:
        return pd.DataFrame(columns=EVENT_ROW_COLUMNS), _summarize(
            normalized_ticker,
            0,
            pd.DataFrame(columns=EVENT_ROW_COLUMNS),
        )

    hist_start = earnings_events["earnings_date"].min() - pd.Timedelta(days=max(45, window_days * 3))
    hist_end = earnings_events["earnings_date"].max() + pd.Timedelta(days=5)

    history = _load_history(normalized_ticker, hist_start, hist_end)
    rows = _compute_event_rows(
        normalized_ticker,
        earnings_events,
        history,
        window_days,
        event_time_iv_df=event_time_iv_df,
    )
    summary = _summarize(normalized_ticker, int(len(earnings_events)), rows)
    return rows, summary


def _print_report(summary: AnalysisResult, rows: pd.DataFrame) -> None:
    print(f"Ticker: {summary.ticker}")
    print(f"Earnings dates loaded: {summary.earnings_events}")
    print(f"Usable events: {summary.usable_events}")
    print(f"Events with event-time IV input: {summary.events_with_event_time_iv}")

    print("\nSummary stats")
    print(f"- Avg abs move (%): {summary.avg_abs_move_pct:.2f}")
    print(f"- Avg signed move (%): {summary.avg_signed_move_pct:.2f}")
    print(f"- Avg event-time IV - realized (% points): {summary.avg_event_time_iv_minus_realized_pct:.2f}")
    print(
        "- Avg event-time expected-move proxy return (%): "
        f"{summary.avg_event_time_expected_move_proxy_return_pct:.2f}"
    )

    if summary.events_with_event_time_iv == 0:
        print(
            "\nIV-dependent metrics are unavailable because no explicit event-time option "
            "snapshot inputs were supplied."
        )
    else:
        print(
            "\nExpected-move proxy return uses event-time IV as a volatility proxy. "
            "It is not observed historical straddle P&L; actual option-return analysis "
            "requires event-time option premium snapshots."
        )

    if not rows.empty:
        display_cols = [
            "earnings_date",
            "pre_day",
            "post_day",
            "abs_move_pct",
            "signed_move_pct",
            "event_time_iv_available",
            "event_time_iv_annual_pct",
            "event_time_iv_minus_realized_pct",
            "event_time_expected_move_proxy_return_pct",
        ]
        print("\nPer-event table")
        print(rows[display_cols].round(2).to_string(index=False))


def main() -> None:
    args = _parse_args()
    rows, summary = run_analysis(args.ticker, args.window_days)
    _print_report(summary, rows)


if __name__ == "__main__":
    main()
