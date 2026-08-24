"""Run the bounded PEAD research pass on validated earnings-event artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.trading_foundation import load_trading_foundation_eod_panel
from earnings_event_engine.research import (
    DEFAULT_WINDOWS,
    assess_signal_readiness,
    build_pead_data_coverage_report,
    build_pead_research_dataset,
    resolve_pead_condition_sources,
    summarize_pead_by_bucket,
    summarize_pead_by_condition,
    summarize_pead_by_window,
)


DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "outputs" / "test_batch" / "clean_events.parquet"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "research"


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format for {path}. Expected .parquet or .csv.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded PEAD research on validated earnings-event artifacts.")
    parser.add_argument(
        "--events-path",
        type=Path,
        default=None,
        help=(
            "Path to a validated event artifact (.parquet or .csv). "
            "Prefer validated_features.parquet when option-implied surprise fields are needed."
        ),
    )
    parser.add_argument(
        "--clean-events-path",
        type=Path,
        default=None,
        help="Legacy alias for --events-path. Kept for backward compatibility.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PEAD research outputs will be written.",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="*",
        default=list(DEFAULT_WINDOWS),
        help="Forward windows in trading days.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Absolute threshold used for neutral reaction-style buckets.",
    )
    parser.add_argument(
        "--post-close-col",
        type=str,
        default=None,
        help="Optional explicit announcement close column. Defaults to post_close when available.",
    )
    parser.add_argument(
        "--prices-path",
        type=Path,
        default=None,
        help=(
            "Optional local price panel (.parquet or .csv) with ticker/date/close and optional "
            "open/volume columns for forward windows and context derivation."
        ),
    )
    parser.add_argument(
        "--historical-data-root",
        type=Path,
        default=None,
        help=(
            "Optional Trading_Foundation historical_data root used to load local EOD bars for the "
            "tickers present in the event artifact."
        ),
    )
    parser.add_argument(
        "--min-signal-events",
        type=int,
        default=30,
        help="Minimum usable events required before later watchlist/signal design is even reviewable.",
    )
    return parser.parse_args()


def _resolve_events_path(args: argparse.Namespace) -> Path:
    if args.events_path is not None and args.clean_events_path is not None:
        raise ValueError("Specify only one of --events-path or --clean-events-path.")
    if args.events_path is not None:
        return args.events_path
    if args.clean_events_path is not None:
        return args.clean_events_path
    return DEFAULT_INPUT_PATH


def _safe_to_parquet(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def _format_value(value: object, *, precision: int = 6) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def _markdown_table(df: pd.DataFrame, *, max_rows: int | None = None) -> list[str]:
    if df.empty:
        return ["No rows produced."]

    render_df = df.head(max_rows).copy() if max_rows is not None else df.copy()
    columns = render_df.columns.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in render_df.iterrows():
        lines.append("| " + " | ".join(_format_value(row[column]) for column in columns) + " |")
    return lines


def _load_prices_for_research(events_df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame | None, str]:
    if args.prices_path is not None and args.historical_data_root is not None:
        raise ValueError("Specify only one of --prices-path or --historical-data-root.")

    if args.prices_path is not None:
        return _read_table(args.prices_path), str(args.prices_path)

    if args.historical_data_root is not None:
        if "ticker" not in events_df.columns:
            raise ValueError("events_df must contain ticker when --historical-data-root is used.")
        tickers = (
            events_df["ticker"].astype("string").dropna().str.upper().str.strip().drop_duplicates().tolist()
        )
        price_panel = load_trading_foundation_eod_panel(args.historical_data_root, tickers)
        return price_panel, str(args.historical_data_root)

    return None, "embedded_forward_closes_only"


def _build_key_findings(
    horizon_summary: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    signal_assessment: pd.DataFrame,
) -> list[str]:
    findings: list[str] = []
    if horizon_summary.empty:
        return ["No forward-return summary rows were produced. Check price coverage and forward windows."]

    smallest_window = int(horizon_summary["window_days"].min())
    smallest_row = horizon_summary.loc[horizon_summary["window_days"] == smallest_window].iloc[0]
    findings.append(
        f"{smallest_window}d coverage: {int(smallest_row['event_count'])} usable events with mean forward return "
        f"{float(smallest_row['mean_forward_return']):.4f} and positive-rate {float(smallest_row['positive_rate']):.2%}."
    )

    if not bucket_summary.empty:
        one_day = bucket_summary.loc[bucket_summary["window_days"] == smallest_window].copy()
        positive = one_day.loc[one_day["reaction_bucket"] == "positive_reaction"]
        negative = one_day.loc[one_day["reaction_bucket"] == "negative_reaction"]
        if not positive.empty:
            row = positive.iloc[0]
            findings.append(
                "Positive reactions: "
                f"{int(row['event_count'])} events at {smallest_window}d with mean drift "
                f"{float(row['mean_forward_return']):.4f}."
            )
        if not negative.empty:
            row = negative.iloc[0]
            findings.append(
                "Negative reactions: "
                f"{int(row['event_count'])} events at {smallest_window}d with mean drift "
                f"{float(row['mean_forward_return']):.4f}."
            )

    if not signal_assessment.empty and (signal_assessment["status"] == "not_defensible").all():
        findings.append(
            "Signal readiness: sample coverage is still too thin for a defensible watchlist or event-driven export."
        )

    return findings


def _build_limitations(events_df: pd.DataFrame, coverage_df: pd.DataFrame) -> list[str]:
    limitations = [
        "Daily/session-level research only. No intraday path reconstruction or intraday options timing is inferred.",
        "Forward returns are left null when the local price history does not extend far enough for a requested horizon.",
        "Buckets use simple sign or quantile splits; no aggressive threshold optimization is performed.",
        "This stage does not emit a tradable signal or scheduled job. It is research-only by design.",
    ]

    zero_coverage_conditions = coverage_df.loc[
        (coverage_df["coverage_group"] == "condition") & (coverage_df["available_events"] == 0),
        "metric",
    ].astype(str)
    zero_coverage_set = set(zero_coverage_conditions.tolist())

    if "earnings_surprise_bucket" in zero_coverage_set:
        limitations.append(
            "Fundamental earnings surprise conditioning was unavailable because the input artifact did not expose a supported surprise field."
        )
    if "size_bucket" in zero_coverage_set:
        limitations.append(
            "Company-size conditioning was unavailable because the validated artifact did not expose a market-cap style field."
        )
    if "liquidity_bucket" in zero_coverage_set:
        limitations.append(
            "Liquidity conditioning requires pre-event volume data from the local price source; it was unavailable or insufficient in this run."
        )

    if "implied_move" not in events_df.columns:
        limitations.append(
            "Option-implied expectation conditioning was unavailable because `implied_move` was not present on the input artifact."
        )

    return limitations


def _write_summary_markdown(
    *,
    output_path: Path,
    input_path: Path,
    price_source_label: str,
    windows: list[int],
    threshold: float,
    min_signal_events: int,
    pead_events: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    condition_summary: pd.DataFrame,
    coverage_df: pd.DataFrame,
    signal_assessment: pd.DataFrame,
) -> None:
    sources = resolve_pead_condition_sources(pead_events)
    lines = [
        "# PEAD Research Summary",
        "",
        "Research-only artifact. This output extends the repo's validated PEAD path with forward-window outcomes, conditioning, coverage, and a conservative readiness assessment. It is not a strategy, recommendation, or live trading system.",
        "",
        "## Inputs",
        "",
        f"- Event artifact: `{input_path}`",
        f"- Price source: `{price_source_label}`",
        f"- Event rows received: `{len(pead_events)}`",
        f"- Drift windows: `{windows}`",
        f"- Neutral threshold: `{threshold}`",
        f"- Minimum signal-review event count: `{min_signal_events}`",
        "",
        "## Key Findings",
        "",
    ]
    for finding in _build_key_findings(horizon_summary, bucket_summary, signal_assessment):
        lines.append(f"- {finding}")

    lines.extend(["", "## Coverage", ""])
    lines.extend(_markdown_table(coverage_df, max_rows=20))

    if len(coverage_df) > 20:
        lines.extend(["", f"Coverage table truncated to 20 rows in markdown. Full detail is in `{output_path.parent / 'pead_data_coverage.csv'}`."])

    lines.extend(["", "## Baseline By Window", ""])
    lines.extend(_markdown_table(horizon_summary))

    lines.extend(["", "## Reaction Buckets", ""])
    lines.extend(_markdown_table(bucket_summary))

    lines.extend(["", "## Conditioning", ""])
    if condition_summary.empty:
        lines.append("No conditioning summary rows were produced.")
    else:
        condition_preview = (
            condition_summary.sort_values(["window_days", "condition_name", "event_count"], ascending=[True, True, False])
            .reset_index(drop=True)
        )
        lines.extend(_markdown_table(condition_preview, max_rows=30))

    lines.extend(["", "## Condition Sources", ""])
    source_rows = pd.DataFrame(
        [
            {
                "condition_column": condition_column,
                "condition_name": info["label"],
                "source_column": info["source_col"],
                "note": info["note"],
            }
            for condition_column, info in sources.items()
        ]
    )
    lines.extend(_markdown_table(source_rows))

    lines.extend(["", "## Signal Readiness", ""])
    lines.extend(_markdown_table(signal_assessment))
    lines.extend(
        [
            "",
            "No downstream signal/export is emitted automatically from this phase. A later watchlist or event-driven job should only be considered after reviewing a materially larger validated sample on the main machine.",
        ]
    )

    lines.extend(["", "## Limitations", ""])
    for limitation in _build_limitations(pead_events, coverage_df):
        lines.append(f"- {limitation}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    events_path = _resolve_events_path(args)
    events_df = _read_table(events_path)
    prices_df, price_source_label = _load_prices_for_research(events_df, args)

    pead_events = build_pead_research_dataset(
        events_df,
        prices_df=prices_df,
        windows=args.windows,
        threshold=args.threshold,
        post_close_col=args.post_close_col,
    )
    horizon_summary = summarize_pead_by_window(pead_events, windows=args.windows)
    bucket_summary = summarize_pead_by_bucket(pead_events, windows=args.windows)
    condition_summary = summarize_pead_by_condition(pead_events, windows=args.windows)
    coverage_df = build_pead_data_coverage_report(pead_events, windows=args.windows)
    signal_assessment = assess_signal_readiness(
        pead_events,
        windows=args.windows,
        min_signal_events=args.min_signal_events,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pead_events.to_csv(args.output_dir / "pead_event_returns.csv", index=False)
    _safe_to_parquet(pead_events, args.output_dir / "pead_event_returns.parquet")
    horizon_summary.to_csv(args.output_dir / "pead_horizon_summary.csv", index=False)
    bucket_summary.to_csv(args.output_dir / "pead_bucket_summary.csv", index=False)
    condition_summary.to_csv(args.output_dir / "pead_condition_summary.csv", index=False)
    coverage_df.to_csv(args.output_dir / "pead_data_coverage.csv", index=False)
    signal_assessment.to_csv(args.output_dir / "pead_signal_assessment.csv", index=False)
    _write_summary_markdown(
        output_path=args.output_dir / "pead_summary.md",
        input_path=events_path,
        price_source_label=price_source_label,
        windows=list(args.windows),
        threshold=args.threshold,
        min_signal_events=args.min_signal_events,
        pead_events=pead_events,
        horizon_summary=horizon_summary,
        bucket_summary=bucket_summary,
        condition_summary=condition_summary,
        coverage_df=coverage_df,
        signal_assessment=signal_assessment,
    )

    print("PEAD research artifacts written:")
    print(f"- {args.output_dir / 'pead_event_returns.csv'}")
    print(f"- {args.output_dir / 'pead_event_returns.parquet'}")
    print(f"- {args.output_dir / 'pead_horizon_summary.csv'}")
    print(f"- {args.output_dir / 'pead_bucket_summary.csv'}")
    print(f"- {args.output_dir / 'pead_condition_summary.csv'}")
    print(f"- {args.output_dir / 'pead_data_coverage.csv'}")
    print(f"- {args.output_dir / 'pead_signal_assessment.csv'}")
    print(f"- {args.output_dir / 'pead_summary.md'}")


if __name__ == "__main__":
    main()
