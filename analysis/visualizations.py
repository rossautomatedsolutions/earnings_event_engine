"""Visualization helpers for earnings-event feature analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd



def plot_implied_vs_realized(df: pd.DataFrame) -> None:
    """Plot implied move against realized gap move as a scatter chart."""
    fig, ax = plt.subplots()
    ax.scatter(df["implied_move"], df["realized_gap_move"])

    combined = pd.concat([df["implied_move"], df["realized_gap_move"]], axis=0).dropna()
    if not combined.empty:
        lower = combined.min()
        upper = combined.max()
        ax.plot([lower, upper], [lower, upper])

    ax.set_xlabel("implied_move")
    ax.set_ylabel("realized_gap_move")
    ax.set_title("Implied vs Realized Gap Move")
    plt.show()



def plot_surprise_distribution(df: pd.DataFrame) -> None:
    """Plot distribution of surprise_vs_implied values."""
    fig, ax = plt.subplots()
    ax.hist(df["surprise_vs_implied"].dropna())
    ax.set_xlabel("surprise_vs_implied")
    ax.set_ylabel("count")
    ax.set_title("Surprise vs Implied Distribution")
    plt.show()



def plot_by_ticker(df: pd.DataFrame) -> None:
    """Plot average surprise_vs_implied per ticker as a bar chart."""
    ticker_avg = df.groupby("ticker", dropna=True)["surprise_vs_implied"].mean().sort_index()

    fig, ax = plt.subplots()
    ticker_avg.plot(kind="bar", ax=ax)
    ax.set_xlabel("ticker")
    ax.set_ylabel("avg surprise_vs_implied")
    ax.set_title("Average Surprise vs Implied by Ticker")
    plt.tight_layout()
    plt.show()



def run_all_plots(path: str | Path) -> None:
    """Load a feature file and run all visualization functions."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(file_path)
    else:
        raise ValueError(
            "Unsupported file format. Use one of: .csv, .parquet, .pkl, .pickle"
        )

    plot_implied_vs_realized(df)
    plot_surprise_distribution(df)
    plot_by_ticker(df)
