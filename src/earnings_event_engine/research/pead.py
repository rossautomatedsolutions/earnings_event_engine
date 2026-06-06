"""Minimal PEAD research utilities for validated earnings-event datasets.

These functions are research-only helpers. They quantify announcement reaction
and short forward drift for event datasets that already contain the needed price
columns. They do not fetch data, generate signals, or make recommendations.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


DEFAULT_WINDOWS: tuple[int, ...] = (1, 3, 5, 10, 20)
DEFAULT_POST_CLOSE_CANDIDATES: tuple[str, ...] = ("post_close", "announcement_close")
CORE_REQUIRED_COLUMNS: tuple[str, ...] = ("event_id", "ticker", "event_date", "pre_close")
PRICE_REQUIRED_COLUMNS: tuple[str, ...] = ("ticker", "date", "close")


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    as_str = series.astype(str).str.strip().str.lower()
    truthy = as_str.isin({"1", "true", "t", "yes", "y"})
    falsy = as_str.isin({"0", "false", "f", "no", "n", "nan", "none", ""})
    out = pd.Series(False, index=series.index)
    out.loc[truthy] = True
    remainder = ~truthy & ~falsy
    non_missing_remainder = remainder & series.notna()
    out.loc[non_missing_remainder] = series.loc[non_missing_remainder].astype(bool)
    return out


def _raise_missing_columns(events_df: pd.DataFrame, required_columns: Sequence[str]) -> None:
    missing = [column for column in required_columns if column not in events_df.columns]
    if missing:
        raise ValueError(
            "events_df is missing required columns for PEAD research: "
            f"{missing}. Expected a validated event dataset with announcement and "
            "forward-close price columns already present."
        )


def _resolve_post_close_column(
    events_df: pd.DataFrame,
    *,
    post_close_col: str | None = None,
) -> str:
    if post_close_col is not None:
        _raise_missing_columns(events_df, [post_close_col])
        return post_close_col

    for candidate in DEFAULT_POST_CLOSE_CANDIDATES:
        if candidate in events_df.columns:
            return candidate

    raise ValueError(
        "events_df is missing a post-event close column. Expected one of "
        f"{list(DEFAULT_POST_CLOSE_CANDIDATES)}."
    )


def _coerce_price_columns(events_df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = events_df.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _normalize_tickers(series: pd.Series) -> pd.Series:
    return series.astype("string").str.upper().str.strip()


def _resolve_mapped_session(events_df: pd.DataFrame) -> pd.Series:
    if "mapped_session" in events_df.columns:
        session = events_df["mapped_session"].astype("string").str.strip().str.lower()
        return session.where(session.isin({"same_day", "next_day"}), "next_day")

    if "announcement_timing" in events_df.columns:
        timing = events_df["announcement_timing"].astype("string").str.strip().str.upper()
        session = pd.Series("next_day", index=events_df.index, dtype="string")
        session.loc[timing.eq("BMO")] = "same_day"
        return session

    return pd.Series("same_day", index=events_df.index, dtype="string")


def _resolve_forward_anchor_dates(
    events_df: pd.DataFrame,
    price_dates_by_ticker: dict[str, pd.DatetimeIndex],
    *,
    baseline_date_col: str | None = None,
) -> pd.Series:
    if baseline_date_col is not None:
        _raise_missing_columns(events_df, [baseline_date_col])
        return pd.to_datetime(events_df[baseline_date_col], errors="coerce").dt.normalize()

    if "post_event_date" in events_df.columns:
        return pd.to_datetime(events_df["post_event_date"], errors="coerce").dt.normalize()

    event_dates = pd.to_datetime(events_df["event_date"], errors="coerce").dt.normalize()
    mapped_session = _resolve_mapped_session(events_df)
    anchor_dates = pd.Series(pd.NaT, index=events_df.index, dtype="datetime64[ns]")

    for idx, ticker in events_df["ticker"].items():
        event_date = event_dates.loc[idx]
        if pd.isna(event_date):
            continue

        price_dates = price_dates_by_ticker.get(str(ticker))
        if price_dates is None or len(price_dates) == 0:
            continue

        session = mapped_session.loc[idx]
        insert_pos = price_dates.searchsorted(event_date, side="left")
        if session == "same_day":
            if insert_pos < len(price_dates):
                anchor_dates.loc[idx] = price_dates[insert_pos]
        else:
            next_pos = price_dates.searchsorted(event_date, side="right")
            if next_pos < len(price_dates):
                anchor_dates.loc[idx] = price_dates[next_pos]

    return anchor_dates


def _valid_research_mask(events_df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=events_df.index, dtype=bool)

    if "valid_event" in events_df.columns:
        mask &= _coerce_bool(events_df["valid_event"])

    if "price_alignment_valid" in events_df.columns:
        mask &= _coerce_bool(events_df["price_alignment_valid"])

    return mask


def add_forward_close_columns(
    events_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    baseline_date_col: str | None = None,
) -> pd.DataFrame:
    """Add forward close columns from an already-local ticker/date/close price panel.

    Expected contracts:
    - `events_df` must include `event_id`, `ticker`, and `event_date`
    - `prices_df` must include `ticker`, `date`, and `close`
    - If `post_event_date` is present, it is used as the forward baseline date
    - Otherwise, the helper falls back to `mapped_session` or `announcement_timing`
      to infer whether the forward baseline should start on the same trading day
      or the next trading day after `event_date`
    """
    if events_df is None or prices_df is None:
        raise ValueError("events_df and prices_df must be provided.")

    _raise_missing_columns(events_df, ["event_id", "ticker", "event_date"])
    _raise_missing_columns(prices_df, PRICE_REQUIRED_COLUMNS)

    out = events_df.copy()
    out["ticker"] = _normalize_tickers(out["ticker"])
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce").dt.normalize()

    prices = prices_df.copy()
    prices["ticker"] = _normalize_tickers(prices["ticker"])
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.normalize()
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices = prices.dropna(subset=["ticker", "date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    prices = prices.drop_duplicates(subset=["ticker", "date"], keep="last")

    price_dates_by_ticker = {
        ticker: pd.DatetimeIndex(group["date"])
        for ticker, group in prices.groupby("ticker", sort=False, dropna=False)
    }
    close_by_key = prices.set_index(["ticker", "date"])["close"]

    anchor_dates = _resolve_forward_anchor_dates(
        out,
        price_dates_by_ticker,
        baseline_date_col=baseline_date_col,
    )
    out["forward_anchor_date"] = anchor_dates

    for window in windows:
        forward_col = f"forward_close_{window}d"
        out[forward_col] = pd.NA

        for idx, ticker in out["ticker"].items():
            anchor_date = anchor_dates.loc[idx]
            if pd.isna(anchor_date):
                continue

            price_dates = price_dates_by_ticker.get(str(ticker))
            if price_dates is None or len(price_dates) == 0:
                continue

            anchor_pos = price_dates.searchsorted(anchor_date, side="left")
            if anchor_pos >= len(price_dates) or price_dates[anchor_pos] != anchor_date:
                continue

            forward_pos = anchor_pos + int(window)
            if forward_pos >= len(price_dates):
                continue

            forward_date = price_dates[forward_pos]
            close_value = close_by_key.get((str(ticker), forward_date), pd.NA)
            out.at[idx, forward_col] = close_value

    return out


def calculate_announcement_reaction(
    events_df: pd.DataFrame,
    *,
    post_close_col: str | None = None,
) -> pd.DataFrame:
    """Add a conservative close-to-close announcement reaction return."""
    if events_df is None:
        raise ValueError("events_df must be provided.")

    resolved_post_close_col = _resolve_post_close_column(events_df, post_close_col=post_close_col)
    _raise_missing_columns(events_df, [*CORE_REQUIRED_COLUMNS, resolved_post_close_col])

    out = _coerce_price_columns(events_df, ["pre_close", resolved_post_close_col])
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")

    valid_mask = _valid_research_mask(out)
    baseline_valid = out["pre_close"].notna() & (out["pre_close"] > 0)
    announcement_valid = out[resolved_post_close_col].notna() & (out[resolved_post_close_col] > 0)
    calculable = valid_mask & baseline_valid & announcement_valid

    out["reaction_return"] = pd.NA
    out.loc[calculable, "reaction_return"] = (
        out.loc[calculable, resolved_post_close_col] / out.loc[calculable, "pre_close"] - 1.0
    )
    out["announcement_close_col"] = resolved_post_close_col
    return out


def calculate_forward_drift(
    events_df: pd.DataFrame,
    *,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    post_close_col: str | None = None,
) -> pd.DataFrame:
    """Add forward drift returns using precomputed forward close columns."""
    if events_df is None:
        raise ValueError("events_df must be provided.")

    resolved_post_close_col = _resolve_post_close_column(events_df, post_close_col=post_close_col)
    required_forward_columns = [f"forward_close_{window}d" for window in windows]
    _raise_missing_columns(events_df, [*CORE_REQUIRED_COLUMNS, resolved_post_close_col, *required_forward_columns])

    price_columns = [resolved_post_close_col, *required_forward_columns]
    out = _coerce_price_columns(events_df, price_columns)
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")

    valid_mask = _valid_research_mask(out)
    baseline_valid = out[resolved_post_close_col].notna() & (out[resolved_post_close_col] > 0)

    for window in windows:
        forward_close_col = f"forward_close_{window}d"
        return_col = f"forward_return_{window}d"
        forward_valid = out[forward_close_col].notna() & (out[forward_close_col] > 0)
        calculable = valid_mask & baseline_valid & forward_valid
        out[return_col] = pd.NA
        out.loc[calculable, return_col] = (
            out.loc[calculable, forward_close_col] / out.loc[calculable, resolved_post_close_col] - 1.0
        )

    out["announcement_close_col"] = resolved_post_close_col
    return out


def assign_reaction_bucket(
    events_df: pd.DataFrame,
    *,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Bucket events by the sign of the announcement reaction."""
    if events_df is None:
        raise ValueError("events_df must be provided.")

    _raise_missing_columns(events_df, ["reaction_return"])
    out = events_df.copy()
    reaction_return = pd.to_numeric(out["reaction_return"], errors="coerce")

    out["reaction_bucket"] = "neutral_reaction"
    out.loc[reaction_return > threshold, "reaction_bucket"] = "positive_reaction"
    out.loc[reaction_return < -threshold, "reaction_bucket"] = "negative_reaction"
    out.loc[reaction_return.isna(), "reaction_bucket"] = pd.NA
    return out


def summarize_pead_by_bucket(
    events_df: pd.DataFrame,
    *,
    windows: Sequence[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Summarize forward drift by reaction bucket and horizon."""
    if events_df is None:
        raise ValueError("events_df must be provided.")

    required_columns = ["reaction_bucket", "reaction_return", *[f"forward_return_{window}d" for window in windows]]
    _raise_missing_columns(events_df, required_columns)

    frames: list[pd.DataFrame] = []
    working = events_df.copy()

    for window in windows:
        return_col = f"forward_return_{window}d"
        numeric_returns = pd.to_numeric(working[return_col], errors="coerce")
        per_window = pd.DataFrame(
            {
                "reaction_bucket": working["reaction_bucket"],
                "forward_return": numeric_returns,
            }
        ).dropna(subset=["reaction_bucket", "forward_return"])

        if per_window.empty:
            continue

        summary = (
            per_window.groupby("reaction_bucket", dropna=False)
            .agg(
                event_count=("forward_return", "size"),
                mean_forward_return=("forward_return", "mean"),
                median_forward_return=("forward_return", "median"),
                positive_rate=("forward_return", lambda values: (values > 0).mean()),
            )
            .reset_index()
        )
        summary["window_days"] = window
        frames.append(summary)

    if not frames:
        return pd.DataFrame(
            columns=[
                "reaction_bucket",
                "window_days",
                "event_count",
                "mean_forward_return",
                "median_forward_return",
                "positive_rate",
            ]
        )

    return pd.concat(frames, ignore_index=True)[
        [
            "reaction_bucket",
            "window_days",
            "event_count",
            "mean_forward_return",
            "median_forward_return",
            "positive_rate",
        ]
    ].sort_values(["window_days", "reaction_bucket"]).reset_index(drop=True)
