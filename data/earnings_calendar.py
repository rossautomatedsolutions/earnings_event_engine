"""Utilities for fetching earnings-calendar dates."""

from __future__ import annotations

import pandas as pd
from src.utils.datetime_utils import to_naive


class EarningsCalendarRetrievalError(RuntimeError):
    """Raised when a calendar request fails before a valid response is returned."""

    def __init__(self, ticker: str, message: str) -> None:
        self.ticker = ticker
        super().__init__(f"Failed to retrieve earnings dates for {ticker}: {message}")


def _empty_earnings_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["ticker", "earnings_date"])


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
    result["earnings_date"] = parsed.loc[valid_mask].map(to_naive)
    return result.reset_index(drop=True)
