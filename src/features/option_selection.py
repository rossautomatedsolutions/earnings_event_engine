"""Option chain selection helpers for earnings-event implied move calculations."""

from __future__ import annotations

from typing import Any

import pandas as pd

_EXPIRY_CANDIDATES = ("expiration", "expiry", "expiration_date", "exp_date")
_STRIKE_CANDIDATES = ("strike", "strike_price")
_OPTION_TYPE_CANDIDATES = ("option_type", "type", "right")
_BID_CANDIDATES = ("bid", "best_bid")
_ASK_CANDIDATES = ("ask", "best_ask")
_MID_CANDIDATES = ("mid", "mark", "last", "price", "close")


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...], required: bool = True) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"Missing required column. Expected one of: {candidates}")
    return None


def _normalize_option_type(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"c", "call", "calls"}:
        return "call"
    if text in {"p", "put", "puts"}:
        return "put"
    return text


def _to_mid(bid: float, ask: float, fallback_mid: float) -> float:
    if pd.notna(bid) and pd.notna(ask) and bid >= 0 and ask >= 0 and ask >= bid:
        return (bid + ask) / 2.0
    return fallback_mid


def _spread_is_extreme(bid: float, ask: float, mid: float) -> bool:
    if pd.isna(bid) or pd.isna(ask) or pd.isna(mid) or mid <= 0:
        return True
    if ask < bid:
        return True
    spread = ask - bid
    # Practical sanity bound: >60% of mid usually indicates poor liquidity/data quality.
    return spread / mid > 0.60


def select_event_options(chain_df: pd.DataFrame, underlying_price: float, event_date: Any) -> dict[str, Any]:
    """Select ATM event-expiry call/put and compute implied move snapshot.

    Args:
        chain_df: Option chain for a single symbol and quote date.
        underlying_price: Spot price used to identify ATM strike.
        event_date: Earnings event date; selected expiry must be strictly after this date.

    Returns:
        Dictionary with:
            - snapshot: clean selected option snapshot and computed metrics.
            - quality_flags: dictionary of validation flags.
    """
    quality_flags = {
        "missing_required_columns": False,
        "invalid_underlying_price": False,
        "no_event_expiry_found": False,
        "missing_legs": False,
        "invalid_prices": False,
        "extreme_spread": False,
    }

    empty_snapshot = {
        "event_date": pd.NaT,
        "selected_expiry": pd.NaT,
        "atm_strike": pd.NA,
        "call_mid": pd.NA,
        "put_mid": pd.NA,
        "call_bid": pd.NA,
        "call_ask": pd.NA,
        "put_bid": pd.NA,
        "put_ask": pd.NA,
        "call_spread": pd.NA,
        "put_spread": pd.NA,
        "straddle_mid": pd.NA,
        "implied_move_dollars": pd.NA,
        "implied_move_pct": pd.NA,
    }

    if chain_df is None or chain_df.empty:
        quality_flags["missing_required_columns"] = True
        quality_flags["no_event_expiry_found"] = True
        return {"snapshot": empty_snapshot, "quality_flags": quality_flags}

    df = chain_df.copy()

    try:
        expiry_col = _pick_column(df, _EXPIRY_CANDIDATES)
        strike_col = _pick_column(df, _STRIKE_CANDIDATES)
        type_col = _pick_column(df, _OPTION_TYPE_CANDIDATES)
    except ValueError:
        quality_flags["missing_required_columns"] = True
        return {"snapshot": empty_snapshot, "quality_flags": quality_flags}

    bid_col = _pick_column(df, _BID_CANDIDATES, required=False)
    ask_col = _pick_column(df, _ASK_CANDIDATES, required=False)
    mid_col = _pick_column(df, _MID_CANDIDATES, required=False)

    if bid_col is None and ask_col is None and mid_col is None:
        quality_flags["missing_required_columns"] = True
        return {"snapshot": empty_snapshot, "quality_flags": quality_flags}

    event_ts = pd.Timestamp(event_date).normalize()
    spot = pd.to_numeric(pd.Series([underlying_price]), errors="coerce").iloc[0]

    if pd.isna(spot) or spot <= 0:
        quality_flags["invalid_underlying_price"] = True
        return {
            "snapshot": {**empty_snapshot, "event_date": event_ts},
            "quality_flags": quality_flags,
        }

    df[expiry_col] = pd.to_datetime(df[expiry_col], errors="coerce").dt.normalize()
    df[strike_col] = pd.to_numeric(df[strike_col], errors="coerce")
    df[type_col] = df[type_col].map(_normalize_option_type)

    for col in (bid_col, ask_col, mid_col):
        if col:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df[expiry_col].notna() & df[strike_col].notna()]
    df = df[df[expiry_col] > event_ts]
    if df.empty:
        quality_flags["no_event_expiry_found"] = True
        return {
            "snapshot": {**empty_snapshot, "event_date": event_ts},
            "quality_flags": quality_flags,
        }

    selected_expiry = df[expiry_col].min()
    exp_df = df[df[expiry_col] == selected_expiry].copy()

    exp_df["atm_gap"] = (exp_df[strike_col] - spot).abs()
    atm_strike = exp_df.sort_values(["atm_gap", strike_col]).iloc[0][strike_col]

    strike_df = exp_df[exp_df[strike_col] == atm_strike].copy()

    call_rows = strike_df[strike_df[type_col] == "call"]
    put_rows = strike_df[strike_df[type_col] == "put"]

    if call_rows.empty or put_rows.empty:
        quality_flags["missing_legs"] = True
        return {
            "snapshot": {
                **empty_snapshot,
                "event_date": event_ts,
                "selected_expiry": selected_expiry,
                "atm_strike": atm_strike,
            },
            "quality_flags": quality_flags,
        }

    call_row = call_rows.iloc[0]
    put_row = put_rows.iloc[0]

    call_bid = call_row.get(bid_col) if bid_col else pd.NA
    call_ask = call_row.get(ask_col) if ask_col else pd.NA
    put_bid = put_row.get(bid_col) if bid_col else pd.NA
    put_ask = put_row.get(ask_col) if ask_col else pd.NA
    call_mid_fallback = call_row.get(mid_col) if mid_col else pd.NA
    put_mid_fallback = put_row.get(mid_col) if mid_col else pd.NA

    call_mid = _to_mid(call_bid, call_ask, call_mid_fallback)
    put_mid = _to_mid(put_bid, put_ask, put_mid_fallback)

    if (
        pd.isna(call_mid)
        or pd.isna(put_mid)
        or call_mid <= 0
        or put_mid <= 0
    ):
        quality_flags["invalid_prices"] = True

    call_spread = (call_ask - call_bid) if pd.notna(call_bid) and pd.notna(call_ask) else pd.NA
    put_spread = (put_ask - put_bid) if pd.notna(put_bid) and pd.notna(put_ask) else pd.NA

    if _spread_is_extreme(call_bid, call_ask, call_mid) or _spread_is_extreme(put_bid, put_ask, put_mid):
        quality_flags["extreme_spread"] = True

    straddle_mid = call_mid + put_mid if pd.notna(call_mid) and pd.notna(put_mid) else pd.NA
    implied_move_dollars = straddle_mid if pd.notna(straddle_mid) else pd.NA
    implied_move_pct = (implied_move_dollars / spot) if pd.notna(implied_move_dollars) and spot > 0 else pd.NA

    snapshot = {
        "event_date": event_ts,
        "selected_expiry": selected_expiry,
        "atm_strike": atm_strike,
        "call_mid": call_mid,
        "put_mid": put_mid,
        "call_bid": call_bid,
        "call_ask": call_ask,
        "put_bid": put_bid,
        "put_ask": put_ask,
        "call_spread": call_spread,
        "put_spread": put_spread,
        "straddle_mid": straddle_mid,
        "implied_move_dollars": implied_move_dollars,
        "implied_move_pct": implied_move_pct,
    }

    return {"snapshot": snapshot, "quality_flags": quality_flags}
