"""CLI workflow for earnings event analysis."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.earnings_calendar import get_earnings_dates


@dataclass
class AnalysisResult:
    """Container for key summary outputs."""

    ticker: str
    earnings_events: int
    usable_events: int
    avg_abs_move_pct: float
    avg_signed_move_pct: float
    avg_iv_minus_realized_pct: float
    avg_straddle_return_pct: float


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


def _estimate_iv(ticker: str) -> float | None:
    """Estimate front-month ATM implied volatility from option chain data."""
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


def _compute_event_rows(
    ticker: str,
    earnings_dates: pd.Series,
    history: pd.DataFrame,
    lookback_window: int,
    iv_estimate: float | None,
) -> pd.DataFrame:
    rows: list[dict] = []
    if history.empty:
        return pd.DataFrame()

    daily_ret = history["close"].pct_change()

    for earnings_date in earnings_dates.sort_values().dt.tz_localize(None):
        prior = history.index[history.index < earnings_date]
        after = history.index[history.index > earnings_date]
        if len(prior) == 0 or len(after) == 0:
            continue

        pre_day = prior[-1]
        post_day = after[0]

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

        iv_minus_realized = np.nan if iv_estimate is None else iv_estimate - realized_annual

        # Straddle approximation: pay estimated expected move from IV and receive realized absolute move.
        expected_move = np.nan if iv_estimate is None else iv_estimate / math.sqrt(252)
        straddle_ret = np.nan if np.isnan(expected_move) else (abs_move / expected_move) - 1.0

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
                "iv_estimate_annual_pct": np.nan if iv_estimate is None else iv_estimate * 100.0,
                "iv_minus_realized_pct": iv_minus_realized * 100.0,
                "straddle_return_pct": straddle_ret * 100.0,
            }
        )

    return pd.DataFrame(rows)


def _summarize(ticker: str, earnings_events: int, event_rows: pd.DataFrame) -> AnalysisResult:
    usable = len(event_rows)
    if usable == 0:
        return AnalysisResult(
            ticker=ticker,
            earnings_events=earnings_events,
            usable_events=0,
            avg_abs_move_pct=float("nan"),
            avg_signed_move_pct=float("nan"),
            avg_iv_minus_realized_pct=float("nan"),
            avg_straddle_return_pct=float("nan"),
        )

    return AnalysisResult(
        ticker=ticker,
        earnings_events=earnings_events,
        usable_events=usable,
        avg_abs_move_pct=float(event_rows["abs_move_pct"].mean()),
        avg_signed_move_pct=float(event_rows["signed_move_pct"].mean()),
        avg_iv_minus_realized_pct=float(event_rows["iv_minus_realized_pct"].mean(skipna=True)),
        avg_straddle_return_pct=float(event_rows["straddle_return_pct"].mean(skipna=True)),
    )


def run_analysis(ticker: str, window_days: int = 20) -> tuple[pd.DataFrame, AnalysisResult]:
    """Run full analysis workflow and return per-event table plus summary stats."""
    normalized_ticker = str(ticker).strip().upper()
    earnings_df = get_earnings_dates(normalized_ticker)

    if earnings_df.empty:
        empty_summary = AnalysisResult(
            ticker=normalized_ticker,
            earnings_events=0,
            usable_events=0,
            avg_abs_move_pct=float("nan"),
            avg_signed_move_pct=float("nan"),
            avg_iv_minus_realized_pct=float("nan"),
            avg_straddle_return_pct=float("nan"),
        )
        return pd.DataFrame(), empty_summary

    earnings_series = pd.to_datetime(earnings_df["earnings_date"], errors="coerce").dropna().sort_values()
    if earnings_series.empty:
        return pd.DataFrame(), _summarize(normalized_ticker, 0, pd.DataFrame())

    hist_start = earnings_series.min() - pd.Timedelta(days=max(45, window_days * 3))
    hist_end = earnings_series.max() + pd.Timedelta(days=5)

    history = _load_history(normalized_ticker, hist_start, hist_end)
    iv_estimate = _estimate_iv(normalized_ticker)
    rows = _compute_event_rows(normalized_ticker, earnings_series, history, window_days, iv_estimate)
    summary = _summarize(normalized_ticker, int(len(earnings_series)), rows)
    return rows, summary


def _print_report(summary: AnalysisResult, rows: pd.DataFrame) -> None:
    print(f"Ticker: {summary.ticker}")
    print(f"Earnings dates loaded: {summary.earnings_events}")
    print(f"Usable events: {summary.usable_events}")

    print("\nSummary stats")
    print(f"- Avg abs move (%): {summary.avg_abs_move_pct:.2f}")
    print(f"- Avg signed move (%): {summary.avg_signed_move_pct:.2f}")
    print(f"- Avg IV - realized (% points): {summary.avg_iv_minus_realized_pct:.2f}")
    print(f"- Avg straddle return (%): {summary.avg_straddle_return_pct:.2f}")

    if not rows.empty:
        display_cols = [
            "earnings_date",
            "pre_day",
            "post_day",
            "abs_move_pct",
            "signed_move_pct",
            "iv_minus_realized_pct",
            "straddle_return_pct",
        ]
        print("\nPer-event table")
        print(rows[display_cols].round(2).to_string(index=False))


def main() -> None:
    args = _parse_args()
    rows, summary = run_analysis(args.ticker, args.window_days)
    _print_report(summary, rows)


if __name__ == "__main__":
    main()
