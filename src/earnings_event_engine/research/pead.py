"""Leakage-aware PEAD research helpers for validated earnings-event datasets.

These functions extend the repo's existing PEAD research path without rebuilding
the upstream event pipeline. They consume validated event artifacts plus local
daily price data, derive forward windows and pre-event context, and emit
research-only summaries. They do not create signals, recommendations, or live
jobs.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.events.event_alignment import align_event_to_sessions
from src.events.timing import normalize_announcement_timing


DEFAULT_WINDOWS: tuple[int, ...] = (1, 3, 5, 10, 20)
DEFAULT_POST_CLOSE_CANDIDATES: tuple[str, ...] = ("post_close", "announcement_close")
CORE_REQUIRED_COLUMNS: tuple[str, ...] = ("event_id", "ticker", "event_date", "pre_close")
PRICE_REQUIRED_COLUMNS: tuple[str, ...] = ("ticker", "date", "close")
PRICE_OPTIONAL_COLUMNS: tuple[str, ...] = ("open", "volume")
SIZE_COLUMN_CANDIDATES: tuple[str, ...] = (
    "market_cap",
    "market_capitalization",
    "company_market_cap",
    "float_market_cap",
)
EARNINGS_SURPRISE_CANDIDATES: tuple[str, ...] = (
    "earnings_surprise",
    "earnings_surprise_pct",
    "earnings_surprise_percent",
    "eps_surprise",
    "eps_surprise_pct",
    "eps_surprise_percent",
)
VOLATILITY_CONTEXT_CANDIDATES: tuple[str, ...] = (
    "pre_20d_realized_vol",
    "hv20_expected_move_proxy",
)
LIQUIDITY_CONTEXT_CANDIDATES: tuple[str, ...] = (
    "pre_20d_avg_dollar_volume",
    "pre_20d_avg_volume",
)
DEFAULT_CONDITION_COLUMNS: tuple[str, ...] = (
    "reaction_bucket",
    "gap_bucket",
    "timing_bucket",
    "implied_move_bucket",
    "realized_vs_implied_bucket",
    "volatility_context_bucket",
    "liquidity_bucket",
    "size_bucket",
    "earnings_surprise_bucket",
)


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


def _pick_first_present_column(
    events_df: pd.DataFrame,
    candidates: Sequence[str],
) -> str | None:
    for candidate in candidates:
        if candidate in events_df.columns:
            return candidate
    return None


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


def _resolve_timing_series(events_df: pd.DataFrame) -> pd.Series:
    for candidate in ("announcement_timing", "timing"):
        if candidate in events_df.columns:
            normalized = events_df[candidate].map(normalize_announcement_timing)
            timing = pd.Series(normalized, index=events_df.index, dtype="object")
            if timing.notna().any():
                return timing

    if "mapped_session" in events_df.columns:
        mapped = events_df["mapped_session"].astype("string").str.strip().str.lower()
        out = pd.Series(pd.NA, index=events_df.index, dtype="object")
        out.loc[mapped.eq("same_day")] = "BMO"
        out.loc[mapped.eq("next_day")] = "AMC"
        return out

    return pd.Series(pd.NA, index=events_df.index, dtype="object")


def _prepare_price_panel(prices_df: pd.DataFrame) -> pd.DataFrame:
    if prices_df is None:
        raise ValueError("prices_df must be provided.")

    _raise_missing_columns(prices_df, PRICE_REQUIRED_COLUMNS)
    prices = prices_df.copy()
    prices["ticker"] = _normalize_tickers(prices["ticker"])
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.normalize()
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")

    for optional_column in PRICE_OPTIONAL_COLUMNS:
        if optional_column in prices.columns:
            prices[optional_column] = pd.to_numeric(prices[optional_column], errors="coerce")
        else:
            prices[optional_column] = pd.Series(float("nan"), index=prices.index, dtype="float64")

    prices = prices.dropna(subset=["ticker", "date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    prices = prices.drop_duplicates(subset=["ticker", "date"], keep="last")

    if prices.empty:
        return prices

    prices["daily_return"] = prices.groupby("ticker", dropna=False)["close"].pct_change()
    prices["dollar_volume"] = prices["close"] * prices["volume"]
    prices["pre_20d_realized_vol"] = (
        prices.groupby("ticker", dropna=False)["daily_return"]
        .rolling(20, min_periods=20)
        .std(ddof=1)
        .reset_index(level=0, drop=True)
    )
    prices["pre_20d_avg_volume"] = (
        prices.groupby("ticker", dropna=False)["volume"]
        .rolling(20, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    prices["pre_20d_avg_dollar_volume"] = (
        prices.groupby("ticker", dropna=False)["dollar_volume"]
        .rolling(20, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    prices[["pre_20d_realized_vol", "pre_20d_avg_volume", "pre_20d_avg_dollar_volume"]] = prices.groupby(
        "ticker",
        dropna=False,
    )[["pre_20d_realized_vol", "pre_20d_avg_volume", "pre_20d_avg_dollar_volume"]].shift(1)
    return prices


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


def _resolve_session_context_dates(
    events_df: pd.DataFrame,
    price_dates_by_ticker: dict[str, pd.DatetimeIndex],
) -> pd.DataFrame:
    if {"pre_event_date", "post_event_date"}.issubset(events_df.columns):
        return pd.DataFrame(
            {
                "pre_event_date": pd.to_datetime(events_df["pre_event_date"], errors="coerce").dt.normalize(),
                "post_event_date": pd.to_datetime(events_df["post_event_date"], errors="coerce").dt.normalize(),
                "session_alignment_flag": "explicit_event_dates",
            },
            index=events_df.index,
        )

    event_dates = pd.to_datetime(events_df["event_date"], errors="coerce").dt.normalize()
    timings = _resolve_timing_series(events_df)
    out = pd.DataFrame(
        {
            "pre_event_date": pd.NaT,
            "post_event_date": pd.NaT,
            "session_alignment_flag": pd.NA,
        },
        index=events_df.index,
    )

    for idx, ticker in events_df["ticker"].items():
        event_date = event_dates.loc[idx]
        timing = timings.loc[idx]
        if pd.isna(event_date) or pd.isna(timing):
            out.loc[idx, "session_alignment_flag"] = "missing_timing_or_event_date"
            continue

        price_dates = price_dates_by_ticker.get(str(ticker))
        if price_dates is None or len(price_dates) == 0:
            out.loc[idx, "session_alignment_flag"] = "missing_price_context"
            continue

        alignment = align_event_to_sessions(
            event_date=event_date,
            announcement_timing=timing,
            price_index=price_dates,
        )
        out.loc[idx, "pre_event_date"] = alignment.get("pre_event_date", pd.NaT)
        out.loc[idx, "post_event_date"] = alignment.get("post_event_date", pd.NaT)
        out.loc[idx, "session_alignment_flag"] = alignment.get("alignment_flag", "unknown")

    return out


def _valid_research_mask(events_df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=events_df.index, dtype=bool)

    if "valid_event" in events_df.columns:
        mask &= _coerce_bool(events_df["valid_event"])

    if "price_alignment_valid" in events_df.columns:
        mask &= _coerce_bool(events_df["price_alignment_valid"])

    return mask


def _signed_bucket(
    series: pd.Series,
    *,
    threshold: float,
    positive_label: str,
    negative_label: str,
    neutral_label: str,
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=series.index, dtype="object")
    out.loc[numeric > threshold] = positive_label
    out.loc[numeric < -threshold] = negative_label
    out.loc[numeric.notna() & numeric.between(-threshold, threshold, inclusive="both")] = neutral_label
    return out


def _quantile_bucket(
    series: pd.Series,
    *,
    labels: Sequence[str],
    flat_label: str,
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=series.index, dtype="object")
    valid = numeric.dropna()
    if valid.empty:
        return out

    if valid.nunique(dropna=True) < 2:
        out.loc[valid.index] = flat_label
        return out

    def _bucket(values: pd.Series) -> pd.Series | None:
        try:
            bucketed = pd.qcut(values, q=len(labels), duplicates="drop")
        except ValueError:
            return None
        categories = list(bucketed.cat.categories)
        if len(categories) < 2:
            return None
        mapped_labels = list(labels[: len(categories)])
        label_map = {category: mapped_labels[idx] for idx, category in enumerate(categories)}
        return bucketed.map(label_map).astype("object")

    bucketed = _bucket(valid)
    if bucketed is None:
        ranked = valid.rank(method="first")
        bucketed = _bucket(ranked)

    if bucketed is None:
        out.loc[valid.index] = flat_label
        return out

    out.loc[bucketed.index] = bucketed
    return out


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
    prices = _prepare_price_panel(prices_df)

    out = events_df.copy()
    out["ticker"] = _normalize_tickers(out["ticker"])
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce").dt.normalize()

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


def add_pre_event_context_columns(events_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe pre-event volatility and liquidity context from local prices."""
    if events_df is None or prices_df is None:
        raise ValueError("events_df and prices_df must be provided.")

    _raise_missing_columns(events_df, ["event_id", "ticker", "event_date"])
    prices = _prepare_price_panel(prices_df)

    out = events_df.copy()
    out["ticker"] = _normalize_tickers(out["ticker"])
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce").dt.normalize()

    price_dates_by_ticker = {
        ticker: pd.DatetimeIndex(group["date"])
        for ticker, group in prices.groupby("ticker", sort=False, dropna=False)
    }
    session_context = _resolve_session_context_dates(out, price_dates_by_ticker)

    if "pre_event_date" not in out.columns:
        out["pre_event_date"] = session_context["pre_event_date"]
    else:
        out["pre_event_date"] = pd.to_datetime(out["pre_event_date"], errors="coerce").dt.normalize()
    if "post_event_date" not in out.columns:
        out["post_event_date"] = session_context["post_event_date"]
    else:
        out["post_event_date"] = pd.to_datetime(out["post_event_date"], errors="coerce").dt.normalize()
    out["session_alignment_flag"] = session_context["session_alignment_flag"]

    context_columns = [
        "ticker",
        "date",
        "pre_20d_realized_vol",
        "pre_20d_avg_volume",
        "pre_20d_avg_dollar_volume",
    ]
    context = prices[context_columns].rename(columns={"date": "pre_event_date"})
    out = out.merge(context, on=["ticker", "pre_event_date"], how="left")
    return out


def calculate_announcement_reaction(
    events_df: pd.DataFrame,
    *,
    post_close_col: str | None = None,
) -> pd.DataFrame:
    """Add conservative announcement move metrics anchored to validated event prices."""
    if events_df is None:
        raise ValueError("events_df must be provided.")

    resolved_post_close_col = _resolve_post_close_column(events_df, post_close_col=post_close_col)
    _raise_missing_columns(events_df, [*CORE_REQUIRED_COLUMNS, resolved_post_close_col])

    numeric_columns = ["pre_close", resolved_post_close_col]
    if "post_open" in events_df.columns:
        numeric_columns.append("post_open")

    out = _coerce_price_columns(events_df, numeric_columns)
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")

    valid_mask = _valid_research_mask(out)
    baseline_valid = out["pre_close"].notna() & (out["pre_close"] > 0)
    announcement_valid = out[resolved_post_close_col].notna() & (out[resolved_post_close_col] > 0)
    calculable = valid_mask & baseline_valid & announcement_valid

    out["reaction_return"] = pd.NA
    out.loc[calculable, "reaction_return"] = (
        out.loc[calculable, resolved_post_close_col] / out.loc[calculable, "pre_close"] - 1.0
    )

    out["gap_return"] = pd.NA
    out["post_session_return"] = pd.NA
    if "post_open" in out.columns:
        post_open_valid = out["post_open"].notna() & (out["post_open"] > 0)
        gap_calculable = valid_mask & baseline_valid & post_open_valid
        out.loc[gap_calculable, "gap_return"] = (
            out.loc[gap_calculable, "post_open"] / out.loc[gap_calculable, "pre_close"] - 1.0
        )
        session_follow_through = valid_mask & post_open_valid & announcement_valid
        out.loc[session_follow_through, "post_session_return"] = (
            out.loc[session_follow_through, resolved_post_close_col] / out.loc[session_follow_through, "post_open"]
            - 1.0
        )

    if "implied_move" in out.columns:
        implied_move = pd.to_numeric(out["implied_move"], errors="coerce")
        out["reaction_vs_implied"] = pd.NA
        comparable = calculable & implied_move.notna() & (implied_move > 0)
        out.loc[comparable, "reaction_vs_implied"] = out.loc[comparable, "reaction_return"] - implied_move.loc[comparable]

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
        total_return_col = f"total_return_{window}d"
        forward_valid = out[forward_close_col].notna() & (out[forward_close_col] > 0)
        calculable = valid_mask & baseline_valid & forward_valid
        out[return_col] = pd.NA
        out.loc[calculable, return_col] = (
            out.loc[calculable, forward_close_col] / out.loc[calculable, resolved_post_close_col] - 1.0
        )
        if "pre_close" in out.columns:
            pre_close_valid = out["pre_close"].notna() & (out["pre_close"] > 0)
            total_calculable = calculable & pre_close_valid
            out[total_return_col] = pd.NA
            out.loc[total_calculable, total_return_col] = (
                out.loc[total_calculable, forward_close_col] / out.loc[total_calculable, "pre_close"] - 1.0
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
    out["reaction_bucket"] = _signed_bucket(
        out["reaction_return"],
        threshold=threshold,
        positive_label="positive_reaction",
        negative_label="negative_reaction",
        neutral_label="neutral_reaction",
    )
    return out


def build_pead_condition_buckets(
    events_df: pd.DataFrame,
    *,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Add conservative conditioning buckets when contemporaneous data supports them."""
    if events_df is None:
        raise ValueError("events_df must be provided.")

    out = events_df.copy()
    if "reaction_bucket" not in out.columns and "reaction_return" in out.columns:
        out = assign_reaction_bucket(out, threshold=threshold)

    if "gap_return" in out.columns:
        out["gap_bucket"] = _signed_bucket(
            out["gap_return"],
            threshold=threshold,
            positive_label="positive_gap",
            negative_label="negative_gap",
            neutral_label="neutral_gap",
        )

    timing = _resolve_timing_series(out)
    out["timing_bucket"] = timing.fillna("unknown")

    if "implied_move" in out.columns:
        out["implied_move_bucket"] = _quantile_bucket(
            out["implied_move"],
            labels=("low_expected_move", "mid_expected_move", "high_expected_move"),
            flat_label="flat_expected_move",
        )

    if "surprise_vs_implied" in out.columns:
        out["realized_vs_implied_bucket"] = _signed_bucket(
            out["surprise_vs_implied"],
            threshold=threshold,
            positive_label="larger_than_implied",
            negative_label="smaller_than_implied",
            neutral_label="inline_with_implied",
        )

    volatility_source = _pick_first_present_column(out, VOLATILITY_CONTEXT_CANDIDATES)
    if volatility_source is not None:
        out["volatility_context_bucket"] = _quantile_bucket(
            out[volatility_source],
            labels=("low_volatility", "mid_volatility", "high_volatility"),
            flat_label="flat_volatility",
        )

    liquidity_source = _pick_first_present_column(out, LIQUIDITY_CONTEXT_CANDIDATES)
    if liquidity_source is not None:
        out["liquidity_bucket"] = _quantile_bucket(
            out[liquidity_source],
            labels=("low_liquidity", "mid_liquidity", "high_liquidity"),
            flat_label="flat_liquidity",
        )

    size_source = _pick_first_present_column(out, SIZE_COLUMN_CANDIDATES)
    if size_source is not None:
        out["size_bucket"] = _quantile_bucket(
            out[size_source],
            labels=("small_cap", "mid_cap", "large_cap"),
            flat_label="flat_cap",
        )

    earnings_surprise_source = _pick_first_present_column(out, EARNINGS_SURPRISE_CANDIDATES)
    if earnings_surprise_source is not None:
        out["earnings_surprise_bucket"] = _signed_bucket(
            out[earnings_surprise_source],
            threshold=threshold,
            positive_label="positive_surprise",
            negative_label="negative_surprise",
            neutral_label="inline_surprise",
        )

    return out


def resolve_pead_condition_sources(events_df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Return human-readable source metadata for PEAD conditioning outputs."""
    timing_source = _pick_first_present_column(events_df, ("announcement_timing", "timing", "mapped_session"))
    volatility_source = _pick_first_present_column(events_df, VOLATILITY_CONTEXT_CANDIDATES)
    liquidity_source = _pick_first_present_column(events_df, LIQUIDITY_CONTEXT_CANDIDATES)
    size_source = _pick_first_present_column(events_df, SIZE_COLUMN_CANDIDATES)
    earnings_surprise_source = _pick_first_present_column(events_df, EARNINGS_SURPRISE_CANDIDATES)

    return {
        "reaction_bucket": {
            "label": "announcement_reaction",
            "source_col": "reaction_return",
            "note": "Close-to-close announcement reaction.",
        },
        "gap_bucket": {
            "label": "gap_reaction",
            "source_col": "gap_return",
            "note": "Pre-close to post-open announcement gap.",
        },
        "timing_bucket": {
            "label": "announcement_timing",
            "source_col": timing_source or "unavailable",
            "note": "BMO vs AMC timing bucket.",
        },
        "implied_move_bucket": {
            "label": "implied_expectation",
            "source_col": "implied_move" if "implied_move" in events_df.columns else "unavailable",
            "note": "Quantile bucket of event-time implied move.",
        },
        "realized_vs_implied_bucket": {
            "label": "realized_vs_implied",
            "source_col": "surprise_vs_implied" if "surprise_vs_implied" in events_df.columns else "unavailable",
            "note": "Sign of realized gap move minus implied move.",
        },
        "volatility_context_bucket": {
            "label": "volatility_context",
            "source_col": volatility_source or "unavailable",
            "note": "Pre-event realized-vol or hv20 proxy bucket.",
        },
        "liquidity_bucket": {
            "label": "liquidity_context",
            "source_col": liquidity_source or "unavailable",
            "note": "Pre-event average volume or dollar-volume bucket.",
        },
        "size_bucket": {
            "label": "company_size",
            "source_col": size_source or "unavailable",
            "note": "Company size bucket when market-cap fields are present.",
        },
        "earnings_surprise_bucket": {
            "label": "fundamental_surprise",
            "source_col": earnings_surprise_source or "unavailable",
            "note": "Sign bucket of fundamental earnings surprise when present.",
        },
    }


def summarize_pead_by_window(
    events_df: pd.DataFrame,
    *,
    windows: Sequence[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Summarize baseline forward-drift behavior by horizon."""
    if events_df is None:
        raise ValueError("events_df must be provided.")

    required_columns = [f"forward_return_{window}d" for window in windows]
    _raise_missing_columns(events_df, required_columns)

    frames: list[pd.DataFrame] = []
    working = events_df.copy()

    for window in windows:
        return_col = f"forward_return_{window}d"
        numeric_returns = pd.to_numeric(working[return_col], errors="coerce")
        per_window = pd.DataFrame(
            {
                "forward_return": numeric_returns,
                "reaction_return": pd.to_numeric(working.get("reaction_return"), errors="coerce"),
                "gap_return": pd.to_numeric(working.get("gap_return"), errors="coerce"),
            }
        ).dropna(subset=["forward_return"])

        if per_window.empty:
            continue

        frames.append(
            pd.DataFrame(
                [
                    {
                        "window_days": window,
                        "event_count": len(per_window),
                        "mean_forward_return": per_window["forward_return"].mean(),
                        "median_forward_return": per_window["forward_return"].median(),
                        "positive_rate": (per_window["forward_return"] > 0).mean(),
                        "negative_rate": (per_window["forward_return"] < 0).mean(),
                        "mean_abs_forward_return": per_window["forward_return"].abs().mean(),
                        "mean_reaction_return": per_window["reaction_return"].mean(),
                        "mean_gap_return": per_window["gap_return"].mean(),
                    }
                ]
            )
        )

    if not frames:
        return pd.DataFrame(
            columns=[
                "window_days",
                "event_count",
                "mean_forward_return",
                "median_forward_return",
                "positive_rate",
                "negative_rate",
                "mean_abs_forward_return",
                "mean_reaction_return",
                "mean_gap_return",
            ]
        )

    return pd.concat(frames, ignore_index=True).sort_values("window_days").reset_index(drop=True)


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


def summarize_pead_by_condition(
    events_df: pd.DataFrame,
    *,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    condition_columns: Sequence[str] = DEFAULT_CONDITION_COLUMNS,
) -> pd.DataFrame:
    """Summarize forward drift across available conditioning buckets."""
    if events_df is None:
        raise ValueError("events_df must be provided.")

    required_columns = [f"forward_return_{window}d" for window in windows]
    _raise_missing_columns(events_df, required_columns)

    sources = resolve_pead_condition_sources(events_df)
    frames: list[pd.DataFrame] = []
    working = events_df.copy()

    for condition_column in condition_columns:
        if condition_column not in working.columns:
            continue

        info = sources.get(
            condition_column,
            {"label": condition_column, "source_col": condition_column, "note": ""},
        )
        for window in windows:
            return_col = f"forward_return_{window}d"
            per_window = pd.DataFrame(
                {
                    "condition_bucket": working[condition_column],
                    "forward_return": pd.to_numeric(working[return_col], errors="coerce"),
                    "reaction_return": pd.to_numeric(working.get("reaction_return"), errors="coerce"),
                    "gap_return": pd.to_numeric(working.get("gap_return"), errors="coerce"),
                }
            ).dropna(subset=["condition_bucket", "forward_return"])

            if per_window.empty:
                continue

            summary = (
                per_window.groupby("condition_bucket", dropna=False)
                .agg(
                    event_count=("forward_return", "size"),
                    mean_forward_return=("forward_return", "mean"),
                    median_forward_return=("forward_return", "median"),
                    positive_rate=("forward_return", lambda values: (values > 0).mean()),
                    mean_reaction_return=("reaction_return", "mean"),
                    mean_gap_return=("gap_return", "mean"),
                )
                .reset_index()
            )
            summary["condition_name"] = info["label"]
            summary["condition_column"] = condition_column
            summary["source_column"] = info["source_col"]
            summary["window_days"] = window
            frames.append(summary)

    if not frames:
        return pd.DataFrame(
            columns=[
                "condition_name",
                "condition_column",
                "source_column",
                "condition_bucket",
                "window_days",
                "event_count",
                "mean_forward_return",
                "median_forward_return",
                "positive_rate",
                "mean_reaction_return",
                "mean_gap_return",
            ]
        )

    return (
        pd.concat(frames, ignore_index=True)[
            [
                "condition_name",
                "condition_column",
                "source_column",
                "condition_bucket",
                "window_days",
                "event_count",
                "mean_forward_return",
                "median_forward_return",
                "positive_rate",
                "mean_reaction_return",
                "mean_gap_return",
            ]
        ]
        .sort_values(["condition_name", "window_days", "condition_bucket"])
        .reset_index(drop=True)
    )


def build_pead_data_coverage_report(
    events_df: pd.DataFrame,
    *,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    condition_columns: Sequence[str] = DEFAULT_CONDITION_COLUMNS,
) -> pd.DataFrame:
    """Document what PEAD coverage exists for windows, inputs, and conditions."""
    if events_df is None:
        raise ValueError("events_df must be provided.")

    total_events = len(events_df)
    valid_mask = _valid_research_mask(events_df)
    sources = resolve_pead_condition_sources(events_df)
    rows: list[dict[str, object]] = [
        {
            "coverage_group": "dataset",
            "metric": "input_events",
            "source_column": "event_id",
            "available_events": total_events,
            "total_events": total_events,
            "coverage_rate": 1.0 if total_events else 0.0,
            "note": "Rows received by the PEAD research stage.",
        },
        {
            "coverage_group": "dataset",
            "metric": "research_valid_events",
            "source_column": "valid_event|price_alignment_valid",
            "available_events": int(valid_mask.sum()),
            "total_events": total_events,
            "coverage_rate": float(valid_mask.mean()) if total_events else 0.0,
            "note": "Rows passing validated-event and price-alignment gates.",
        },
    ]

    event_level_metrics = {
        "reaction_return": "Close-to-close announcement reaction coverage.",
        "gap_return": "Pre-close to post-open gap coverage.",
        "post_session_return": "Post-open to post-close announcement-session follow-through coverage.",
    }
    for column, note in event_level_metrics.items():
        if column not in events_df.columns:
            continue
        available = int(pd.to_numeric(events_df[column], errors="coerce").notna().sum())
        rows.append(
            {
                "coverage_group": "event_metric",
                "metric": column,
                "source_column": column,
                "available_events": available,
                "total_events": total_events,
                "coverage_rate": available / total_events if total_events else 0.0,
                "note": note,
            }
        )

    for window in windows:
        forward_close_col = f"forward_close_{window}d"
        forward_return_col = f"forward_return_{window}d"
        if forward_close_col in events_df.columns:
            available = int(pd.to_numeric(events_df[forward_close_col], errors="coerce").notna().sum())
            rows.append(
                {
                    "coverage_group": "window",
                    "metric": forward_close_col,
                    "source_column": forward_close_col,
                    "available_events": available,
                    "total_events": total_events,
                    "coverage_rate": available / total_events if total_events else 0.0,
                    "note": "Events with usable forward close data for this horizon.",
                }
            )
        if forward_return_col in events_df.columns:
            available = int(pd.to_numeric(events_df[forward_return_col], errors="coerce").notna().sum())
            rows.append(
                {
                    "coverage_group": "window",
                    "metric": forward_return_col,
                    "source_column": forward_return_col,
                    "available_events": available,
                    "total_events": total_events,
                    "coverage_rate": available / total_events if total_events else 0.0,
                    "note": "Events with usable forward drift returns for this horizon.",
                }
            )

    for condition_column in condition_columns:
        info = sources.get(condition_column)
        source_column = info["source_col"] if info is not None else "unavailable"
        note = info["note"] if info is not None else ""

        if condition_column not in events_df.columns:
            rows.append(
                {
                    "coverage_group": "condition",
                    "metric": condition_column,
                    "source_column": source_column,
                    "available_events": 0,
                    "total_events": total_events,
                    "coverage_rate": 0.0,
                    "note": f"{note} Bucket not available on the current artifact.",
                }
            )
            continue

        available = int(events_df[condition_column].notna().sum())
        rows.append(
            {
                "coverage_group": "condition",
                "metric": condition_column,
                "source_column": source_column,
                "available_events": available,
                "total_events": total_events,
                "coverage_rate": available / total_events if total_events else 0.0,
                "note": note,
            }
        )

    return pd.DataFrame(rows)[
        [
            "coverage_group",
            "metric",
            "source_column",
            "available_events",
            "total_events",
            "coverage_rate",
            "note",
        ]
    ]


def assess_signal_readiness(
    events_df: pd.DataFrame,
    *,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    min_signal_events: int = 30,
) -> pd.DataFrame:
    """Provide a conservative watchlist/signal readiness assessment.

    The output is intentionally strict and never emits trade candidates. It only
    tells us whether the current research sample is large enough for a later
    manual review of a watchlist or signal design step.
    """
    if events_df is None:
        raise ValueError("events_df must be provided.")

    rows: list[dict[str, object]] = []
    bucket_summary = summarize_pead_by_bucket(events_df, windows=windows)
    min_bucket_events = max(5, min_signal_events // 4)

    for window in windows:
        return_col = f"forward_return_{window}d"
        usable_events = (
            int(pd.to_numeric(events_df[return_col], errors="coerce").notna().sum())
            if return_col in events_df.columns
            else 0
        )
        per_window = bucket_summary.loc[bucket_summary["window_days"] == window].copy()
        positive_events = int(
            per_window.loc[per_window["reaction_bucket"] == "positive_reaction", "event_count"].sum()
        )
        negative_events = int(
            per_window.loc[per_window["reaction_bucket"] == "negative_reaction", "event_count"].sum()
        )

        if usable_events < min_signal_events:
            status = "not_defensible"
            reason = (
                f"Only {usable_events} usable events at {window}d; require at least "
                f"{min_signal_events} before considering watchlist or signal design."
            )
        elif positive_events < min_bucket_events or negative_events < min_bucket_events:
            status = "not_defensible"
            reason = (
                f"Positive/negative reaction cohorts are too thin at {window}d "
                f"(positive={positive_events}, negative={negative_events})."
            )
        else:
            status = "needs_manual_review"
            reason = (
                "Coverage is large enough for a later manual signal review, but no "
                "automatic export is emitted from this research stage."
            )

        rows.append(
            {
                "window_days": window,
                "usable_events": usable_events,
                "positive_reaction_events": positive_events,
                "negative_reaction_events": negative_events,
                "status": status,
                "reason": reason,
            }
        )

    return pd.DataFrame(rows)[
        [
            "window_days",
            "usable_events",
            "positive_reaction_events",
            "negative_reaction_events",
            "status",
            "reason",
        ]
    ]


def build_pead_research_dataset(
    events_df: pd.DataFrame,
    *,
    prices_df: pd.DataFrame | None = None,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    threshold: float = 0.0,
    post_close_col: str | None = None,
) -> pd.DataFrame:
    """Build the full research dataset used by the PEAD CLI."""
    if events_df is None:
        raise ValueError("events_df must be provided.")

    working = events_df.copy()
    if prices_df is not None:
        working = add_forward_close_columns(working, prices_df, windows=windows)
        working = add_pre_event_context_columns(working, prices_df)

    working = calculate_announcement_reaction(working, post_close_col=post_close_col)
    working = calculate_forward_drift(working, windows=windows, post_close_col=post_close_col)
    working = assign_reaction_bucket(working, threshold=threshold)
    working = build_pead_condition_buckets(working, threshold=threshold)
    return working
