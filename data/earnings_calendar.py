"""Utilities for fetching earnings-calendar dates."""

from __future__ import annotations

import pandas as pd


class EarningsCalendarRetrievalError(RuntimeError):
    """Raised when a calendar request fails before a valid response is returned."""

    def __init__(self, ticker: str, message: str) -> None:
        self.ticker = ticker
        super().__init__(f"Failed to retrieve earnings dates for {ticker}: {message}")


def _empty_earnings_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["ticker", "earnings_date"])


def _to_session_clock_timestamp(value: object) -> pd.Timestamp:
    """Drop timezone metadata while preserving the provider's wall-clock time.

    Earnings timestamps are used for session classification (BMO vs AMC), so we
    preserve the original clock reading instead of converting to UTC first.
    """
    parsed = pd.Timestamp(pd.to_datetime(value))
    if parsed.tzinfo is not None:
        return parsed.tz_localize(None)
    return parsed


def get_earnings_dates(ticker: str) -> pd.DataFrame:
    """Return earnings dates for a ticker as ``[ticker, earnings_date]``.

    Successful retrieval returns either a populated DataFrame or an empty
    DataFrame when the provider has no rows for the ticker. Retrieval failures
    raise ``EarningsCalendarRetrievalError`` so callers can distinguish them
    from legitimate empty responses.
    """
    normalized_ticker = str(ticker).strip().upper()

    if not normalized_ticker:
        return _empty_earnings_frame()

    try:
        import yfinance as yf  # type: ignore

        earnings_dates = yf.Ticker(normalized_ticker).get_earnings_dates()
    except Exception as exc:
        raise EarningsCalendarRetrievalError(normalized_ticker, str(exc)) from exc

    if earnings_dates is None or earnings_dates.empty:
        return _empty_earnings_frame()

    result = earnings_dates.reset_index().rename(columns={"Earnings Date": "earnings_date"})

    if "earnings_date" not in result.columns:
        date_col = result.columns[0]
        result = result.rename(columns={date_col: "earnings_date"})

    result = result[["earnings_date"]].copy()
    result.insert(0, "ticker", normalized_ticker)
    parsed = pd.to_datetime(result["earnings_date"], errors="coerce")
    valid_mask = parsed.notna()
    result = result.loc[valid_mask].copy()
    result["earnings_date"] = parsed.loc[valid_mask].map(_to_session_clock_timestamp)
    return result.reset_index(drop=True)
