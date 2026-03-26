"""Utilities for measuring price moves around earnings events."""

from __future__ import annotations

from typing import Iterable

import pandas as pd



def compute_earnings_move(price_df: pd.DataFrame, earnings_dates: Iterable) -> pd.DataFrame:
    """Compute pre-/post-earnings move metrics for each earnings date.

    Parameters
    ----------
    price_df
        Price history that includes open and close columns. The trading date can be
        supplied either as a ``DatetimeIndex`` or via a ``date`` column.
    earnings_dates
        Iterable of earnings event dates.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by ``earnings_date`` with these columns:
        - ``pre_close``: close from the last trading session before the earnings date
        - ``post_open``: open from the first trading session on/after the earnings date
        - ``percent_move``: ``(post_open - pre_close) / pre_close * 100``
    """
    if price_df.empty:
        return pd.DataFrame(columns=["pre_close", "post_open", "percent_move"]).rename_axis(
            "earnings_date"
        )

    working_df = price_df.copy()

    if not isinstance(working_df.index, pd.DatetimeIndex):
        if "date" not in working_df.columns:
            raise ValueError("price_df must have a DatetimeIndex or a 'date' column.")
        working_df["date"] = pd.to_datetime(working_df["date"])
        working_df = working_df.set_index("date")

    working_df = working_df.sort_index()

    open_col = "open" if "open" in working_df.columns else "Open"
    close_col = "close" if "close" in working_df.columns else "Close"

    if open_col not in working_df.columns or close_col not in working_df.columns:
        raise ValueError("price_df must include open/close columns (open/close or Open/Close).")

    records = []
    index = working_df.index

    for earnings_date in pd.to_datetime(list(earnings_dates)):
        pre_pos = index.searchsorted(earnings_date, side="left") - 1
        post_pos = index.searchsorted(earnings_date, side="left")

        pre_close = pd.NA
        post_open = pd.NA
        percent_move = pd.NA

        if pre_pos >= 0:
            pre_close = working_df.iloc[pre_pos][close_col]

        if post_pos < len(working_df):
            post_open = working_df.iloc[post_pos][open_col]

        if pd.notna(pre_close) and pd.notna(post_open) and pre_close != 0:
            percent_move = (post_open - pre_close) / pre_close * 100

        records.append(
            {
                "earnings_date": earnings_date,
                "pre_close": pre_close,
                "post_open": post_open,
                "percent_move": percent_move,
            }
        )

    result = pd.DataFrame(records).set_index("earnings_date")
    return result
