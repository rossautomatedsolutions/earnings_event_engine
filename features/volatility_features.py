"""Volatility-oriented feature utilities.

This module provides helpers for comparing the move implied by option
prices (via an at-the-money straddle) with realized post-earnings moves.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


_DATE_CANDIDATES = ("date", "quote_date", "as_of_date", "timestamp")
_EXPIRY_CANDIDATES = ("expiration", "expiry", "expiration_date", "exp_date")
_STRIKE_CANDIDATES = ("strike", "strike_price")
_SPOT_CANDIDATES = ("underlying_price", "underlying_close", "spot", "close", "price")
_TYPE_CANDIDATES = ("option_type", "type", "right")
_PRICE_CANDIDATES = ("mid", "mark", "last", "price", "close")


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...], required: bool = True) -> str | None:
    """Return the first matching column name from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"Could not find any required columns from: {candidates}")
    return None


def _normalize_option_type(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"c", "call", "calls"}:
        return "call"
    if text in {"p", "put", "puts"}:
        return "put"
    return text


def _coerce_earnings_dates(earnings_dates: Iterable) -> pd.DatetimeIndex:
    idx = pd.to_datetime(list(earnings_dates), errors="coerce")
    idx = pd.DatetimeIndex(idx[~pd.isna(idx)]).normalize().unique().sort_values()
    return idx


def compute_iv_vs_realized(option_df: pd.DataFrame, earnings_dates: Iterable) -> pd.DataFrame:
    """Compute implied vs realized move around earnings events.

    Args:
        option_df: Option/underlying data. Supported shapes:
            1) Long options format with columns for date, expiration,
               strike, option_type and option price.
            2) Wide format with ``call_price`` and ``put_price`` columns,
               plus date/expiration/strike and an underlying spot column.
            Underlying price history should be present in one of the
            supported spot columns.
        earnings_dates: Iterable of earnings event dates.

    Returns:
        DataFrame with columns:
            - earnings_date
            - implied_move
            - realized_move
            - difference (implied_move - realized_move)
        Each move is expressed as a decimal return (e.g., 0.05 == 5%).
    """
    if option_df is None or option_df.empty:
        return pd.DataFrame(columns=["earnings_date", "implied_move", "realized_move", "difference"])

    df = option_df.copy()

    date_col = _pick_column(df, _DATE_CANDIDATES)
    expiry_col = _pick_column(df, _EXPIRY_CANDIDATES, required=False)
    strike_col = _pick_column(df, _STRIKE_CANDIDATES, required=False)
    spot_col = _pick_column(df, _SPOT_CANDIDATES)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    if expiry_col:
        df[expiry_col] = pd.to_datetime(df[expiry_col], errors="coerce").dt.normalize()

    # Price history for realized move uses unique daily underlying closes.
    daily_spot = (
        df[[date_col, spot_col]]
        .dropna(subset=[date_col, spot_col])
        .sort_values(date_col)
        .drop_duplicates(subset=[date_col], keep="last")
        .set_index(date_col)[spot_col]
    )

    earnings_idx = _coerce_earnings_dates(earnings_dates)
    if earnings_idx.empty:
        return pd.DataFrame(columns=["earnings_date", "implied_move", "realized_move", "difference"])

    # Prepare option pricing view for implied move calculation.
    if {"call_price", "put_price"}.issubset(df.columns):
        option_view = df.copy()
        option_view["call_price"] = pd.to_numeric(option_view["call_price"], errors="coerce")
        option_view["put_price"] = pd.to_numeric(option_view["put_price"], errors="coerce")
    else:
        type_col = _pick_column(df, _TYPE_CANDIDATES)
        price_col = _pick_column(df, _PRICE_CANDIDATES)

        option_rows = df[[c for c in [date_col, expiry_col, strike_col, spot_col, type_col, price_col] if c is not None]].copy()
        option_rows[type_col] = option_rows[type_col].map(_normalize_option_type)
        option_rows[price_col] = pd.to_numeric(option_rows[price_col], errors="coerce")

        if strike_col is None:
            raise ValueError("A strike column is required when call_price/put_price are not provided.")
        if expiry_col is None:
            raise ValueError("An expiration column is required when call_price/put_price are not provided.")

        pivoted = (
            option_rows.pivot_table(
                index=[date_col, expiry_col, strike_col, spot_col],
                columns=type_col,
                values=price_col,
                aggfunc="mean",
            )
            .reset_index()
            .rename_axis(columns=None)
        )
        if "call" not in pivoted.columns or "put" not in pivoted.columns:
            raise ValueError("Option rows must contain both call and put prices to compute straddle implied move.")

        option_view = pivoted.rename(columns={"call": "call_price", "put": "put_price"})

    records = []
    for event_date in earnings_idx:
        pre_dates = daily_spot.index[daily_spot.index <= event_date]
        post_dates = daily_spot.index[daily_spot.index > event_date]

        realized_move = pd.NA
        if len(pre_dates) and len(post_dates):
            pre_date = pre_dates[-1]
            post_date = post_dates[0]
            pre_price = daily_spot.loc[pre_date]
            post_price = daily_spot.loc[post_date]
            if pd.notna(pre_price) and pre_price != 0 and pd.notna(post_price):
                realized_move = abs((post_price - pre_price) / pre_price)

        implied_move = pd.NA
        if expiry_col and strike_col:
            candidates = option_view[option_view[date_col] <= event_date].copy()
            if not candidates.empty:
                candidates = candidates[candidates[expiry_col] >= event_date]
            if not candidates.empty:
                nearest_quote_date = candidates[date_col].max()
                candidates = candidates[candidates[date_col] == nearest_quote_date]
                nearest_expiry = candidates[expiry_col].min()
                candidates = candidates[candidates[expiry_col] == nearest_expiry]

                candidates["atm_gap"] = (pd.to_numeric(candidates[strike_col], errors="coerce") - pd.to_numeric(candidates[spot_col], errors="coerce")).abs()
                atm_row = candidates.sort_values("atm_gap").iloc[0]
                call_px = pd.to_numeric(atm_row.get("call_price"), errors="coerce")
                put_px = pd.to_numeric(atm_row.get("put_price"), errors="coerce")
                spot_px = pd.to_numeric(atm_row.get(spot_col), errors="coerce")
                if pd.notna(call_px) and pd.notna(put_px) and pd.notna(spot_px) and spot_px != 0:
                    implied_move = (call_px + put_px) / spot_px

        difference = pd.NA
        if pd.notna(implied_move) and pd.notna(realized_move):
            difference = implied_move - realized_move

        records.append(
            {
                "earnings_date": event_date,
                "implied_move": implied_move,
                "realized_move": realized_move,
                "difference": difference,
            }
        )

    return pd.DataFrame.from_records(records).sort_values("earnings_date").reset_index(drop=True)
