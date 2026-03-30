"""Utilities to simulate long call/put vertical spreads on event moves.

The simulator uses percent-move abstractions so it can run directly on feature datasets
that contain implied and realized post-event moves.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("implied_move", "realized_gap_move")


def _validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns and return a defensive numeric copy."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    results = df.copy()
    for column in REQUIRED_COLUMNS:
        results[column] = pd.to_numeric(results[column], errors="coerce")

    return results


def _clip_non_negative(series: pd.Series) -> pd.Series:
    """Clip values below zero (vectorized helper)."""
    return series.clip(lower=0)


def simulate_vertical_spreads(
    df: pd.DataFrame,
    call_width: float = 0.08,
    put_width: float = 0.08,
    debit_fraction: float = 0.45,
    execution_slippage: float = 0.05,
) -> pd.DataFrame:
    """Simulate long call and long put vertical spread outcomes per event.

    Modeling assumptions (percent-of-underlying terms):
    - Spread width is capped by ``call_width``/``put_width``.
    - Entry debit is a fraction of spread width and implied move.
    - Exit value is intrinsic value clipped at spread width.
    - Execution slippage increases entry debit and reduces realized exit value.

    This structure gives:
    - lower upfront cost vs. long straddles (debit fraction),
    - capped payoff (spread width),
    - more realistic execution (slippage on entry + exit).
    """
    if call_width <= 0 or put_width <= 0:
        raise ValueError("call_width and put_width must be > 0")
    if not 0 < debit_fraction <= 1:
        raise ValueError("debit_fraction must be in (0, 1]")
    if not 0 <= execution_slippage < 1:
        raise ValueError("execution_slippage must be in [0, 1)")

    results = _validate_input(df)

    implied_abs = results["implied_move"].abs()
    realized = results["realized_gap_move"]

    call_base_debit = implied_abs.clip(upper=call_width) * debit_fraction
    put_base_debit = implied_abs.clip(upper=put_width) * debit_fraction

    results["call_spread_cost"] = call_base_debit * (1 + execution_slippage)
    results["put_spread_cost"] = put_base_debit * (1 + execution_slippage)

    call_intrinsic = _clip_non_negative(realized).clip(upper=call_width)
    put_intrinsic = _clip_non_negative(-realized).clip(upper=put_width)

    results["call_spread_payoff"] = call_intrinsic * (1 - execution_slippage)
    results["put_spread_payoff"] = put_intrinsic * (1 - execution_slippage)

    results["call_spread_pnl"] = results["call_spread_payoff"] - results["call_spread_cost"]
    results["put_spread_pnl"] = results["put_spread_payoff"] - results["put_spread_cost"]

    results["lower_cost_strategy"] = results[["call_spread_cost", "put_spread_cost"]].idxmin(axis=1)
    results["higher_pnl_strategy"] = results[["call_spread_pnl", "put_spread_pnl"]].idxmax(axis=1)

    return results


def compute_summary(results: pd.DataFrame) -> dict[str, float]:
    """Compute aggregate comparison metrics across call and put spreads."""
    required = {
        "call_spread_cost",
        "put_spread_cost",
        "call_spread_payoff",
        "put_spread_payoff",
        "call_spread_pnl",
        "put_spread_pnl",
    }
    missing = sorted(required.difference(results.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    call_pnl = pd.to_numeric(results["call_spread_pnl"], errors="coerce")
    put_pnl = pd.to_numeric(results["put_spread_pnl"], errors="coerce")

    if call_pnl.dropna().empty and put_pnl.dropna().empty:
        return {
            "avg_call_cost": float("nan"),
            "avg_put_cost": float("nan"),
            "avg_call_payoff": float("nan"),
            "avg_put_payoff": float("nan"),
            "avg_call_pnl": float("nan"),
            "avg_put_pnl": float("nan"),
            "call_win_rate": float("nan"),
            "put_win_rate": float("nan"),
            "cost_advantage_call_minus_put": float("nan"),
            "pnl_advantage_call_minus_put": float("nan"),
        }

    return {
        "avg_call_cost": float(pd.to_numeric(results["call_spread_cost"], errors="coerce").mean()),
        "avg_put_cost": float(pd.to_numeric(results["put_spread_cost"], errors="coerce").mean()),
        "avg_call_payoff": float(pd.to_numeric(results["call_spread_payoff"], errors="coerce").mean()),
        "avg_put_payoff": float(pd.to_numeric(results["put_spread_payoff"], errors="coerce").mean()),
        "avg_call_pnl": float(call_pnl.mean()),
        "avg_put_pnl": float(put_pnl.mean()),
        "call_win_rate": float((call_pnl > 0).mean()),
        "put_win_rate": float((put_pnl > 0).mean()),
        "cost_advantage_call_minus_put": float(
            pd.to_numeric(results["call_spread_cost"], errors="coerce").mean()
            - pd.to_numeric(results["put_spread_cost"], errors="coerce").mean()
        ),
        "pnl_advantage_call_minus_put": float(call_pnl.mean() - put_pnl.mean()),
    }


def _load_features(path: str | Path) -> pd.DataFrame:
    """Load a features DataFrame from CSV or Parquet."""
    source = Path(path)
    suffix = source.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)

    raise ValueError(f"Unsupported file type '{suffix}'. Use CSV or Parquet input.")


def run_vertical_spread_analysis(
    path: str | Path,
    call_width: float = 0.08,
    put_width: float = 0.08,
    debit_fraction: float = 0.45,
    execution_slippage: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Load features, run vertical spread simulation, and print summary metrics."""
    features = _load_features(path)
    results = simulate_vertical_spreads(
        features,
        call_width=call_width,
        put_width=put_width,
        debit_fraction=debit_fraction,
        execution_slippage=execution_slippage,
    )
    summary = compute_summary(results)

    print("Vertical Spread Summary")
    for key, value in summary.items():
        print(f"- {key}: {value:.6f}" if pd.notna(value) else f"- {key}: nan")

    return results, summary
