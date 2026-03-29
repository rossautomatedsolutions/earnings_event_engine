"""Fast sanity checks for event-level features.

This module is intended for lightweight guardrails that quickly surface broken
logic or clearly bad source data before downstream analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_sanity_checks(event_features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Run high-signal sanity checks on event feature rows.

    Required columns:
    - ``realized_move``
    - ``implied_move``

    Optional price columns used for non-positive price checks (if present):
    - ``pre_close``
    - ``post_open``
    - ``post_close``
    - ``underlying_price``

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        A tuple of:
        1) DataFrame containing only flagged rows with check columns attached.
        2) Series with per-check flag counts plus totals.
    """

    required_cols = {"realized_move", "implied_move"}
    missing = sorted(required_cols.difference(event_features_df.columns))
    if missing:
        raise ValueError(
            "run_sanity_checks requires columns: "
            f"{sorted(required_cols)}; missing={missing}"
        )

    working = event_features_df.copy()

    realized_move = pd.to_numeric(working["realized_move"], errors="coerce")
    implied_move = pd.to_numeric(working["implied_move"], errors="coerce")

    # 1) Realized move sanity
    working["flag_realized_move_gt_50pct"] = (~realized_move.isna()) & (realized_move.abs() > 0.50)
    working["flag_realized_move_zero"] = (~realized_move.isna()) & (realized_move == 0)

    # 2) Implied move sanity
    working["flag_implied_move_lt_0_5pct"] = (~implied_move.isna()) & (implied_move < 0.005)
    working["flag_implied_move_gt_25pct"] = (~implied_move.isna()) & (implied_move > 0.25)

    # 3) Relationship checks
    denom = realized_move.abs().replace(0, np.nan)
    ratio = implied_move.abs() / denom
    working["implied_to_realized_ratio"] = ratio.replace([np.inf, -np.inf], np.nan)

    working["flag_ratio_lt_0_2x"] = (~working["implied_to_realized_ratio"].isna()) & (
        working["implied_to_realized_ratio"] < 0.2
    )
    working["flag_ratio_gt_3x"] = (~working["implied_to_realized_ratio"].isna()) & (
        working["implied_to_realized_ratio"] > 3.0
    )

    price_cols = [
        col
        for col in ["pre_close", "post_open", "post_close", "underlying_price"]
        if col in working.columns
    ]
    if price_cols:
        nonpositive_price = pd.Series(False, index=working.index)
        for col in price_cols:
            price = pd.to_numeric(working[col], errors="coerce")
            nonpositive_price = nonpositive_price | ((~price.isna()) & (price <= 0))
        working["flag_nonpositive_price"] = nonpositive_price
    else:
        working["flag_nonpositive_price"] = False

    check_cols = [
        "flag_realized_move_gt_50pct",
        "flag_realized_move_zero",
        "flag_implied_move_lt_0_5pct",
        "flag_implied_move_gt_25pct",
        "flag_ratio_lt_0_2x",
        "flag_ratio_gt_3x",
        "flag_nonpositive_price",
    ]

    working["is_flagged"] = working[check_cols].any(axis=1)
    flagged_rows = working.loc[working["is_flagged"]].copy()

    summary_counts = working[check_cols].sum().astype("int64")
    summary_counts.loc["total_flagged_rows"] = int(working["is_flagged"].sum())
    summary_counts.loc["total_rows"] = int(len(working))

    return flagged_rows, summary_counts
