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


def _valid_research_mask(events_df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=events_df.index, dtype=bool)

    if "valid_event" in events_df.columns:
        mask &= _coerce_bool(events_df["valid_event"])

    if "price_alignment_valid" in events_df.columns:
        mask &= _coerce_bool(events_df["price_alignment_valid"])

    return mask


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
