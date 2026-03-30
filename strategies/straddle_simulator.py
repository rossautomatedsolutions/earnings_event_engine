"""Utilities to evaluate long-straddle profitability from feature datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("implied_move", "realized_gap_move")


def _validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns and return a defensive copy."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    results = df.copy()
    for column in REQUIRED_COLUMNS:
        results[column] = pd.to_numeric(results[column], errors="coerce")

    return results


def simulate_straddle(df: pd.DataFrame, cost_multiplier: float = 1.5) -> pd.DataFrame:
    """Simulate long straddle outcomes for each event row.

    Assumption: ``implied_move`` approximates straddle cost (in percent terms).
    """
    results = _validate_input(df)

    results["cost"] = results["implied_move"] * cost_multiplier
    results["payoff"] = results["realized_gap_move"].abs()
    results["pnl"] = results["payoff"] - results["cost"]

    return results


def compute_summary(results: pd.DataFrame) -> dict[str, float]:
    """Compute strategy-level summary metrics from simulation results."""
    if "pnl" not in results.columns:
        raise KeyError("Missing required column: pnl")

    pnl = pd.to_numeric(results["pnl"], errors="coerce").dropna()
    if pnl.empty:
        return {
            "avg_pnl": float("nan"),
            "avg_cost": float("nan"),
            "avg_payoff": float("nan"),
            "win_rate": float("nan"),
            "median_pnl": float("nan"),
            "max_gain": float("nan"),
            "max_loss": float("nan"),
        }

    return {
        "avg_pnl": float(pnl.mean()),
        "avg_cost": float(pd.to_numeric(results["cost"], errors="coerce").mean()),
        "avg_payoff": float(pd.to_numeric(results["payoff"], errors="coerce").mean()),
        "win_rate": float((pnl > 0).mean()),
        "median_pnl": float(pnl.median()),
        "max_gain": float(pnl.max()),
        "max_loss": float(pnl.min()),
    }


def breakdown_by_ticker(results: pd.DataFrame) -> pd.DataFrame:
    """Return ticker-level long straddle metrics."""
    required = {"ticker", "pnl"}
    missing = sorted(required.difference(results.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    grouped = results.groupby("ticker", dropna=False)["pnl"]
    breakdown = grouped.agg(
        avg_pnl="mean",
        win_rate=lambda col: (col > 0).mean(),
        median_pnl="median",
        max_gain="max",
        max_loss="min",
        n_events="count",
    )
    return breakdown.sort_values("avg_pnl", ascending=False)


def _load_features(path: str | Path) -> pd.DataFrame:
    """Load a features DataFrame from CSV or Parquet."""
    source = Path(path)
    suffix = source.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)

    raise ValueError(f"Unsupported file type '{suffix}'. Use CSV or Parquet input.")


def run_straddle_analysis(
    path: str | Path,
    cost_multiplier: float = 1.5,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    """Load features, run simulation, and print summary and ticker breakdown."""
    features = _load_features(path)
    results = simulate_straddle(features, cost_multiplier=cost_multiplier)
    summary = compute_summary(results)

    print("Long Straddle Summary")
    for key, value in summary.items():
        print(f"- {key}: {value:.6f}" if pd.notna(value) else f"- {key}: nan")

    if "ticker" in results.columns:
        ticker_breakdown = breakdown_by_ticker(results)
        print("\nBreakdown by ticker")
        print(ticker_breakdown.round(6).to_string())
    else:
        ticker_breakdown = pd.DataFrame()
        print("\nBreakdown by ticker skipped (missing 'ticker' column).")

    return results, summary, ticker_breakdown
