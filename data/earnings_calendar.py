"""Utilities for fetching upcoming earnings-calendar dates."""

from __future__ import annotations

import pandas as pd


def get_earnings_dates(ticker: str) -> pd.DataFrame:
    """Return earnings dates for a ticker as ``[ticker, earnings_date]``.

    The function uses ``yfinance`` when available and falls back to an empty
    DataFrame with the expected schema when data cannot be fetched.
    """
    columns = ["ticker", "earnings_date"]
    normalized_ticker = str(ticker).strip().upper()

    if not normalized_ticker:
        return pd.DataFrame(columns=columns)

    try:
        import yfinance as yf  # type: ignore

        earnings_dates = yf.Ticker(normalized_ticker).get_earnings_dates()
        if earnings_dates is None or earnings_dates.empty:
            return pd.DataFrame(columns=columns)

        result = earnings_dates.reset_index().rename(columns={"Earnings Date": "earnings_date"})

        if "earnings_date" not in result.columns:
            date_col = result.columns[0]
            result = result.rename(columns={date_col: "earnings_date"})

        result = result[["earnings_date"]].copy()
        result.insert(0, "ticker", normalized_ticker)
        result["earnings_date"] = pd.to_datetime(result["earnings_date"], errors="coerce")
        return result.dropna(subset=["earnings_date"]).reset_index(drop=True)

    except Exception:
        return pd.DataFrame(columns=columns)
