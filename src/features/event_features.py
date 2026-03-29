"""Event-level feature engineering for earnings move research.

The goal of this module is to create a clean, event-indexed dataset with:
- realized post-event moves,
- option-implied expected moves,
- pre-event context features.

Assumptions
-----------
1) ``events_df`` contains one row per event with at least:
   - ``symbol``
   - ``event_date``
   - optional ``announcement_timing`` (e.g. after_close, before_open)
2) ``prices_df`` contains daily OHLCV data with at least:
   - ``symbol``, ``date``, ``open``, ``close``
   - optional ``volume``
3) ``options_df`` contains one (or more) pre-event implied move observations with:
   - ``symbol``
   - an event-key column: ``event_date`` (preferred) or ``date``
   - either ``implied_move_pct`` directly,
     or ``implied_move_dollars`` + a price anchor,
     or ``call_price`` + ``put_price`` + an underlying spot column.

No-lookahead policy
-------------------
- Pre-event context (5d return, 20d realized vol, average volume) uses only data up to
  and including the selected pre-event close date.
- Price move features use only the aligned pre/post sessions for each event.
- Implied move uses the event row from ``options_df`` (or nearest eligible observation
  keyed to that event), and never post-event prices.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.events.event_alignment import align_event_to_sessions


_REQUIRED_EVENT_COLUMNS = {"symbol", "event_date"}
_REQUIRED_PRICE_COLUMNS = {"symbol", "date", "open", "close"}

_OPTION_EVENT_KEY_CANDIDATES = ("event_date", "date")
_IMPLIED_PCT_CANDIDATES = ("implied_move_pct", "iv_move_pct", "expected_move_pct")
_IMPLIED_DOLLAR_CANDIDATES = ("implied_move_dollars", "implied_move", "expected_move_dollars")
_SPOT_CANDIDATES = ("underlying_price", "underlying_close", "spot", "close")
_CALL_CANDIDATES = ("call_price", "call_mid", "atm_call")
_PUT_CANDIDATES = ("put_price", "put_mid", "atm_put")

_OUTPUT_COLUMNS = [
    "symbol",
    "event_date",
    "announcement_timing",
    "alignment_type",
    "alignment_flag",
    "pre_event_date",
    "post_event_date",
    "pre_close",
    "post_open",
    "post_close",
    "gap_move_pct",
    "close_to_close_pct",
    "abs_gap_move_pct",
    "abs_close_to_close_pct",
    "implied_move_pct",
    "implied_move_dollars",
    "implied_minus_realized",
    "abs_diff",
    "pre_5d_return",
    "pre_20d_realized_vol",
    "pre_avg_volume_20d",
]


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _validate_inputs(events_df: pd.DataFrame, prices_df: pd.DataFrame) -> None:
    missing_events = _REQUIRED_EVENT_COLUMNS.difference(events_df.columns)
    if missing_events:
        raise ValueError(f"events_df missing required columns: {sorted(missing_events)}")

    missing_prices = _REQUIRED_PRICE_COLUMNS.difference(prices_df.columns)
    if missing_prices:
        raise ValueError(f"prices_df missing required columns: {sorted(missing_prices)}")


def _prepare_prices(prices_df: pd.DataFrame) -> pd.DataFrame:
    px = prices_df.copy()
    px["date"] = pd.to_datetime(px["date"], errors="coerce").dt.normalize()
    px["open"] = pd.to_numeric(px["open"], errors="coerce")
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    if "volume" in px.columns:
        px["volume"] = pd.to_numeric(px["volume"], errors="coerce")

    px = (
        px.dropna(subset=["symbol", "date", "open", "close"])
        .sort_values(["symbol", "date"])
        .drop_duplicates(subset=["symbol", "date"], keep="last")
        .reset_index(drop=True)
    )
    return px


def _prepare_events(events_df: pd.DataFrame) -> pd.DataFrame:
    ev = events_df.copy()
    ev["event_date"] = pd.to_datetime(ev["event_date"], errors="coerce").dt.normalize()
    if "announcement_timing" not in ev.columns:
        ev["announcement_timing"] = pd.NA

    ev = (
        ev.dropna(subset=["symbol", "event_date"])
        .sort_values(["symbol", "event_date"])
        .reset_index(drop=True)
    )
    return ev


def _prepare_options(options_df: pd.DataFrame) -> pd.DataFrame:
    if options_df is None or options_df.empty:
        return pd.DataFrame()

    opt = options_df.copy()
    event_key_col = _pick_column(opt, _OPTION_EVENT_KEY_CANDIDATES)
    if event_key_col is None or "symbol" not in opt.columns:
        return pd.DataFrame()

    opt["_option_event_key"] = pd.to_datetime(opt[event_key_col], errors="coerce").dt.normalize()
    opt = opt.dropna(subset=["symbol", "_option_event_key"]).copy()

    for col in _IMPLIED_PCT_CANDIDATES + _IMPLIED_DOLLAR_CANDIDATES + _SPOT_CANDIDATES + _CALL_CANDIDATES + _PUT_CANDIDATES:
        if col in opt.columns:
            opt[col] = pd.to_numeric(opt[col], errors="coerce")

    return opt.sort_values(["symbol", "_option_event_key"]).reset_index(drop=True)


def _compute_implied_fields(option_rows: pd.DataFrame, pre_close: float | None) -> tuple[Any, Any]:
    if option_rows.empty:
        return pd.NA, pd.NA

    row = option_rows.iloc[0]

    implied_pct_col = _pick_column(option_rows, _IMPLIED_PCT_CANDIDATES)
    implied_dollar_col = _pick_column(option_rows, _IMPLIED_DOLLAR_CANDIDATES)
    spot_col = _pick_column(option_rows, _SPOT_CANDIDATES)
    call_col = _pick_column(option_rows, _CALL_CANDIDATES)
    put_col = _pick_column(option_rows, _PUT_CANDIDATES)

    implied_move_pct = pd.to_numeric(row.get(implied_pct_col), errors="coerce") if implied_pct_col else pd.NA
    implied_move_dollars = pd.to_numeric(row.get(implied_dollar_col), errors="coerce") if implied_dollar_col else pd.NA

    if pd.isna(implied_move_pct) and pd.notna(implied_move_dollars):
        anchor = pre_close
        if (anchor is None or pd.isna(anchor) or anchor == 0) and spot_col:
            anchor = pd.to_numeric(row.get(spot_col), errors="coerce")
        if pd.notna(anchor) and anchor != 0:
            implied_move_pct = implied_move_dollars / anchor

    if pd.isna(implied_move_dollars) and pd.notna(implied_move_pct):
        anchor = pre_close
        if (anchor is None or pd.isna(anchor)) and spot_col:
            anchor = pd.to_numeric(row.get(spot_col), errors="coerce")
        if pd.notna(anchor):
            implied_move_dollars = implied_move_pct * anchor

    if pd.isna(implied_move_pct) and call_col and put_col:
        call_px = pd.to_numeric(row.get(call_col), errors="coerce")
        put_px = pd.to_numeric(row.get(put_col), errors="coerce")
        anchor = pre_close
        if (anchor is None or pd.isna(anchor) or anchor == 0) and spot_col:
            anchor = pd.to_numeric(row.get(spot_col), errors="coerce")
        if pd.notna(call_px) and pd.notna(put_px):
            implied_move_dollars = call_px + put_px
            if pd.notna(anchor) and anchor != 0:
                implied_move_pct = implied_move_dollars / anchor

    return implied_move_pct, implied_move_dollars


def build_event_features(events_df: pd.DataFrame, prices_df: pd.DataFrame, options_df: pd.DataFrame) -> pd.DataFrame:
    """Build the main event-level research dataset.

    Returns one row per input event with aligned price, realized move,
    implied move, and pre-event context features.
    """
    _validate_inputs(events_df=events_df, prices_df=prices_df)

    ev = _prepare_events(events_df)
    px = _prepare_prices(prices_df)
    opt = _prepare_options(options_df)

    records: list[dict[str, Any]] = []

    for event in ev.itertuples(index=False):
        symbol = event.symbol
        event_date = pd.Timestamp(event.event_date).normalize()
        timing = getattr(event, "announcement_timing", pd.NA)

        sym_px = px[px["symbol"] == symbol]
        if sym_px.empty:
            records.append({
                "symbol": symbol,
                "event_date": event_date,
                "announcement_timing": timing,
                "alignment_type": "unknown",
                "alignment_flag": "missing_symbol_prices",
            })
            continue

        aligned = align_event_to_sessions(
            event_date=event_date,
            announcement_timing=timing,
            price_index=sym_px["date"],
        )

        pre_event_date = aligned["pre_event_date"]
        post_event_date = aligned["post_event_date"]

        pre_row = sym_px[sym_px["date"] == pre_event_date].head(1)
        post_row = sym_px[sym_px["date"] == post_event_date].head(1)

        pre_close = pd.to_numeric(pre_row["close"].iloc[0], errors="coerce") if not pre_row.empty else pd.NA
        post_open = pd.to_numeric(post_row["open"].iloc[0], errors="coerce") if not post_row.empty else pd.NA
        post_close = pd.to_numeric(post_row["close"].iloc[0], errors="coerce") if not post_row.empty else pd.NA

        gap_move_pct = pd.NA
        close_to_close_pct = pd.NA
        if pd.notna(pre_close) and pre_close != 0 and pd.notna(post_open):
            gap_move_pct = (post_open - pre_close) / pre_close
        if pd.notna(pre_close) and pre_close != 0 and pd.notna(post_close):
            close_to_close_pct = (post_close - pre_close) / pre_close

        abs_gap_move_pct = abs(gap_move_pct) if pd.notna(gap_move_pct) else pd.NA
        abs_close_to_close_pct = abs(close_to_close_pct) if pd.notna(close_to_close_pct) else pd.NA

        implied_move_pct = pd.NA
        implied_move_dollars = pd.NA
        if not opt.empty:
            sym_opt = opt[(opt["symbol"] == symbol) & (opt["_option_event_key"] == event_date)]
            implied_move_pct, implied_move_dollars = _compute_implied_fields(sym_opt, pre_close=pre_close)

        implied_minus_realized = pd.NA
        abs_diff = pd.NA
        if pd.notna(implied_move_pct) and pd.notna(abs_close_to_close_pct):
            implied_minus_realized = implied_move_pct - abs_close_to_close_pct
            abs_diff = abs(implied_minus_realized)

        pre_5d_return = pd.NA
        pre_20d_realized_vol = pd.NA
        pre_avg_volume_20d = pd.NA

        if pd.notna(pre_event_date):
            hist = sym_px[sym_px["date"] <= pre_event_date].sort_values("date").copy()
            hist["daily_ret"] = hist["close"].pct_change()

            if len(hist) >= 6:
                close_t = hist["close"].iloc[-1]
                close_t5 = hist["close"].iloc[-6]
                if pd.notna(close_t) and pd.notna(close_t5) and close_t5 != 0:
                    pre_5d_return = (close_t / close_t5) - 1

            if len(hist) >= 21:
                trailing_ret = hist["daily_ret"].iloc[-20:]
                pre_20d_realized_vol = trailing_ret.std(ddof=1) * (252**0.5)

            if "volume" in hist.columns and len(hist) >= 20:
                pre_avg_volume_20d = hist["volume"].iloc[-20:].mean()

        records.append(
            {
                "symbol": symbol,
                "event_date": event_date,
                "announcement_timing": timing,
                "alignment_type": aligned["alignment_type"],
                "alignment_flag": aligned["alignment_flag"],
                "pre_event_date": pre_event_date,
                "post_event_date": post_event_date,
                "pre_close": pre_close,
                "post_open": post_open,
                "post_close": post_close,
                "gap_move_pct": gap_move_pct,
                "close_to_close_pct": close_to_close_pct,
                "abs_gap_move_pct": abs_gap_move_pct,
                "abs_close_to_close_pct": abs_close_to_close_pct,
                "implied_move_pct": implied_move_pct,
                "implied_move_dollars": implied_move_dollars,
                "implied_minus_realized": implied_minus_realized,
                "abs_diff": abs_diff,
                "pre_5d_return": pre_5d_return,
                "pre_20d_realized_vol": pre_20d_realized_vol,
                "pre_avg_volume_20d": pre_avg_volume_20d,
            }
        )

    event_features_df = pd.DataFrame.from_records(records)
    if event_features_df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    for col in _OUTPUT_COLUMNS:
        if col not in event_features_df.columns:
            event_features_df[col] = pd.NA

    event_features_df = (
        event_features_df[_OUTPUT_COLUMNS]
        .sort_values(["symbol", "event_date"])
        .reset_index(drop=True)
    )
    return event_features_df
