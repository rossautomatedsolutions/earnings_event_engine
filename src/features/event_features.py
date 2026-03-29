"""Build event-level research features from events, prices, and options snapshots.

The output is one row per earnings event with aligned pre/post prices, realized
moves, implied move inputs, and pre-event context features.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.events.event_alignment import align_event_to_sessions

_SYMBOL_CANDIDATES = ("symbol", "ticker")
_EVENT_DATE_CANDIDATES = ("event_date", "earnings_date", "date")
_TIMING_CANDIDATES = ("announcement_timing", "earnings_time", "timing")
_PRE_EVENT_DATE_CANDIDATES = ("pre_event_date",)
_POST_EVENT_DATE_CANDIDATES = ("post_event_date",)
_PRICE_DATE_CANDIDATES = ("date", "trading_date", "timestamp")
_OPEN_CANDIDATES = ("open", "open_price")
_CLOSE_CANDIDATES = ("close", "close_price", "adj_close")
_VOLUME_CANDIDATES = ("volume", "shares_traded")
_IMPLIED_PCT_CANDIDATES = ("implied_move_pct", "implied_move")
_IMPLIED_DOLLAR_CANDIDATES = ("implied_move_dollars", "implied_move_usd", "straddle_mid")


OUTPUT_SCHEMA = [
    "symbol",
    "event_date",
    "announcement_timing",
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
    "pre_20d_avg_volume",
]


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...], label: str, required: bool = True) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"Could not find required {label} column. Tried: {list(candidates)}")
    return None


def _normalize_symbol(series: pd.Series) -> pd.Series:
    return series.astype("string").str.upper().str.strip()


def _prepare_prices(prices_df: pd.DataFrame) -> tuple[pd.DataFrame, str, str, str]:
    if prices_df is None or prices_df.empty:
        raise ValueError("prices_df must be a non-empty DataFrame")

    symbol_col = _pick_column(prices_df, _SYMBOL_CANDIDATES, "symbol")
    date_col = _pick_column(prices_df, _PRICE_DATE_CANDIDATES, "price date")
    open_col = _pick_column(prices_df, _OPEN_CANDIDATES, "open price")
    close_col = _pick_column(prices_df, _CLOSE_CANDIDATES, "close price")
    volume_col = _pick_column(prices_df, _VOLUME_CANDIDATES, "volume", required=False)

    df = prices_df.copy()
    df[symbol_col] = _normalize_symbol(df[symbol_col])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    for numeric_col in (open_col, close_col, volume_col):
        if numeric_col is not None:
            df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")

    df = df.dropna(subset=[symbol_col, date_col]).sort_values([symbol_col, date_col]).reset_index(drop=True)
    return df, symbol_col, date_col, volume_col


def _prepare_options(options_df: pd.DataFrame | None, events: pd.DataFrame) -> pd.DataFrame:
    if options_df is None or options_df.empty:
        return events[["symbol", "event_date"]].assign(implied_move_pct=pd.NA, implied_move_dollars=pd.NA)

    symbol_col = _pick_column(options_df, _SYMBOL_CANDIDATES, "symbol")
    event_col = _pick_column(options_df, _EVENT_DATE_CANDIDATES, "event date")
    implied_pct_col = _pick_column(options_df, _IMPLIED_PCT_CANDIDATES, "implied move pct", required=False)
    implied_dollar_col = _pick_column(options_df, _IMPLIED_DOLLAR_CANDIDATES, "implied move dollars", required=False)

    if implied_pct_col is None and implied_dollar_col is None:
        raise ValueError("options_df must contain at least one implied move column")

    df = options_df.copy()
    df[symbol_col] = _normalize_symbol(df[symbol_col])
    df[event_col] = pd.to_datetime(df[event_col], errors="coerce").dt.normalize()

    if implied_pct_col is not None:
        df[implied_pct_col] = pd.to_numeric(df[implied_pct_col], errors="coerce")
    if implied_dollar_col is not None:
        df[implied_dollar_col] = pd.to_numeric(df[implied_dollar_col], errors="coerce")

    selected = df[[symbol_col, event_col]].copy()
    selected["implied_move_pct"] = df[implied_pct_col] if implied_pct_col else pd.NA
    selected["implied_move_dollars"] = df[implied_dollar_col] if implied_dollar_col else pd.NA

    return (
        selected.dropna(subset=[symbol_col, event_col])
        .sort_values([symbol_col, event_col])
        .drop_duplicates(subset=[symbol_col, event_col], keep="first")
        .rename(columns={symbol_col: "symbol", event_col: "event_date"})
        .reset_index(drop=True)
    )


def _with_aligned_dates(events: pd.DataFrame, prices: pd.DataFrame, price_symbol_col: str, price_date_col: str) -> pd.DataFrame:
    pre_date_col = _pick_column(events, _PRE_EVENT_DATE_CANDIDATES, "pre_event_date", required=False)
    post_date_col = _pick_column(events, _POST_EVENT_DATE_CANDIDATES, "post_event_date", required=False)

    if pre_date_col and post_date_col:
        events["pre_event_date"] = pd.to_datetime(events[pre_date_col], errors="coerce").dt.normalize()
        events["post_event_date"] = pd.to_datetime(events[post_date_col], errors="coerce").dt.normalize()
        return events

    timing_col = _pick_column(events, _TIMING_CANDIDATES, "announcement timing")

    price_index_by_symbol: dict[str, pd.Series] = {
        symbol: grp[price_date_col] for symbol, grp in prices.groupby(price_symbol_col, dropna=False)
    }

    alignments = []
    for row in events.itertuples(index=False):
        symbol = getattr(row, "symbol")
        trading_index = price_index_by_symbol.get(symbol)

        if trading_index is None or trading_index.empty:
            alignments.append((pd.NaT, pd.NaT))
            continue

        aligned = align_event_to_sessions(
            event_date=getattr(row, "event_date"),
            announcement_timing=getattr(row, timing_col),
            price_index=trading_index,
        )
        alignments.append((aligned["pre_event_date"], aligned["post_event_date"]))

    events[["pre_event_date", "post_event_date"]] = pd.DataFrame(alignments, index=events.index)
    return events


def build_event_features(events_df: pd.DataFrame, prices_df: pd.DataFrame, options_df: pd.DataFrame) -> pd.DataFrame:
    """Create one-row-per-event research features.

    Assumptions (explicit):
    - `events_df` has one row per earnings event with symbol and event date.
    - If pre/post event dates are absent, they are inferred from `announcement_timing`
      using market-session alignment rules from `align_event_to_sessions`.
    - `options_df` is expected to already represent event-level implied move snapshots
      (one row per symbol/event date or collapsible to one via first-row dedupe).
    - All return/move percentages are decimal units (0.05 = 5%).

    Bias controls:
    - Pre-event context features only use prices strictly up to `pre_event_date`.
    - Rolling volatility and averages are shifted so event-day/post-event data is never used.
    """

    if events_df is None or events_df.empty:
        raise ValueError("events_df must be a non-empty DataFrame")

    price_df, price_symbol_col, price_date_col, volume_col = _prepare_prices(prices_df)

    event_symbol_col = _pick_column(events_df, _SYMBOL_CANDIDATES, "symbol")
    event_date_col = _pick_column(events_df, _EVENT_DATE_CANDIDATES, "event date")
    timing_col = _pick_column(events_df, _TIMING_CANDIDATES, "announcement timing")

    events = events_df.copy()
    events["symbol"] = _normalize_symbol(events[event_symbol_col])
    events["event_date"] = pd.to_datetime(events[event_date_col], errors="coerce").dt.normalize()
    events["announcement_timing"] = events[timing_col]
    events = events.dropna(subset=["symbol", "event_date"]).reset_index(drop=True)

    if events.empty:
        return pd.DataFrame(columns=OUTPUT_SCHEMA)

    events = _with_aligned_dates(events, price_df, price_symbol_col, price_date_col)

    merge_prices = price_df.rename(
        columns={
            price_symbol_col: "symbol",
            price_date_col: "price_date",
            _pick_column(price_df, _OPEN_CANDIDATES, "open price"): "open",
            _pick_column(price_df, _CLOSE_CANDIDATES, "close price"): "close",
        }
    )
    if volume_col:
        merge_prices = merge_prices.rename(columns={volume_col: "volume"})
    else:
        merge_prices["volume"] = pd.NA

    grouped = merge_prices.sort_values(["symbol", "price_date"]).copy()
    grouped["daily_return"] = grouped.groupby("symbol", dropna=False)["close"].pct_change()
    grouped["pre_5d_return"] = grouped.groupby("symbol", dropna=False)["close"].pct_change(5)
    grouped["pre_20d_realized_vol"] = (
        grouped.groupby("symbol", dropna=False)["daily_return"]
        .rolling(20, min_periods=20)
        .std()
        .reset_index(level=0, drop=True)
    )
    grouped["pre_20d_avg_volume"] = (
        grouped.groupby("symbol", dropna=False)["volume"]
        .rolling(20, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Shift by one day to keep pre-event context strictly historical.
    grouped[["pre_5d_return", "pre_20d_realized_vol", "pre_20d_avg_volume"]] = grouped.groupby(
        "symbol", dropna=False
    )[["pre_5d_return", "pre_20d_realized_vol", "pre_20d_avg_volume"]].shift(1)

    pre_snap = grouped[["symbol", "price_date", "close", "pre_5d_return", "pre_20d_realized_vol", "pre_20d_avg_volume"]].rename(
        columns={"price_date": "pre_event_date", "close": "pre_close"}
    )
    post_snap = grouped[["symbol", "price_date", "open", "close"]].rename(
        columns={"price_date": "post_event_date", "open": "post_open", "close": "post_close"}
    )

    out = events.merge(pre_snap, on=["symbol", "pre_event_date"], how="left")
    out = out.merge(post_snap, on=["symbol", "post_event_date"], how="left")

    out["gap_move_pct"] = (out["post_open"] - out["pre_close"]) / out["pre_close"]
    out["close_to_close_pct"] = (out["post_close"] - out["pre_close"]) / out["pre_close"]
    out["abs_gap_move_pct"] = out["gap_move_pct"].abs()
    out["abs_close_to_close_pct"] = out["close_to_close_pct"].abs()

    implied = _prepare_options(options_df, events)
    out = out.merge(implied, on=["symbol", "event_date"], how="left")

    # Realized comparator: close-to-close absolute move.
    out["implied_minus_realized"] = out["implied_move_pct"] - out["abs_close_to_close_pct"]
    out["abs_diff"] = out["implied_minus_realized"].abs()

    event_features_df = out.reindex(columns=OUTPUT_SCHEMA).sort_values(["event_date", "symbol"]).reset_index(drop=True)
    return event_features_df
