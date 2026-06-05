"""Run a minimal research-only PEAD pass on validated clean earnings events."""

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

from earnings_event_engine.research import (
    DEFAULT_WINDOWS,
    assign_reaction_bucket,
    calculate_announcement_reaction,
    calculate_forward_drift,
    summarize_pead_by_bucket,
)


DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "outputs" / "test_batch" / "clean_events.parquet"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "research"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal PEAD research on validated earnings events.")
    parser.add_argument(
        "--clean-events-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to a clean validated events parquet artifact.",
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
        help="Forward windows in trading days. Requires matching forward_close_{N}d columns.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Absolute threshold used for neutral reaction bucketing.",
    )
    parser.add_argument(
        "--post-close-col",
        type=str,
        default=None,
        help="Optional explicit announcement close column. Defaults to post_close when available.",
    )
    return parser.parse_args()


def _safe_to_parquet(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def _write_summary_markdown(
    summary_df: pd.DataFrame,
    *,
    input_path: Path,
    output_path: Path,
    windows: list[int],
    threshold: float,
    event_count: int,
) -> None:
    lines = [
        "# PEAD Research Summary",
        "",
        "Research-only artifact. This output measures announcement reaction and short-horizon drift for validated earnings events. It is not a strategy, recommendation, or live trading system.",
        "",
        f"- Input artifact: `{input_path}`",
        f"- Event count: `{event_count}`",
        f"- Drift windows: `{windows}`",
        f"- Neutral threshold: `{threshold}`",
        "",
    ]

    if summary_df.empty:
        lines.extend(
            [
                "No bucket summary rows were produced.",
                "",
                "Check that the input artifact contains `post_close` or `announcement_close` plus matching `forward_close_{N}d` columns.",
            ]
        )
    else:
        lines.append("| reaction_bucket | window_days | event_count | mean_forward_return | median_forward_return | positive_rate |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in summary_df.itertuples(index=False):
            lines.append(
                "| "
                f"{row.reaction_bucket} | "
                f"{row.window_days} | "
                f"{row.event_count} | "
                f"{row.mean_forward_return:.6f} | "
                f"{row.median_forward_return:.6f} | "
                f"{row.positive_rate:.6f} |"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    clean_events = pd.read_parquet(args.clean_events_path)

    pead_events = calculate_announcement_reaction(
        clean_events,
        post_close_col=args.post_close_col,
    )
    pead_events = calculate_forward_drift(
        pead_events,
        windows=args.windows,
        post_close_col=args.post_close_col,
    )
    pead_events = assign_reaction_bucket(pead_events, threshold=args.threshold)
    bucket_summary = summarize_pead_by_bucket(pead_events, windows=args.windows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pead_events.to_csv(args.output_dir / "pead_event_returns.csv", index=False)
    _safe_to_parquet(pead_events, args.output_dir / "pead_event_returns.parquet")
    bucket_summary.to_csv(args.output_dir / "pead_bucket_summary.csv", index=False)
    _write_summary_markdown(
        bucket_summary,
        input_path=args.clean_events_path,
        output_path=args.output_dir / "pead_summary.md",
        windows=list(args.windows),
        threshold=args.threshold,
        event_count=len(pead_events),
    )

    print("PEAD research artifacts written:")
    print(f"- {args.output_dir / 'pead_event_returns.csv'}")
    print(f"- {args.output_dir / 'pead_event_returns.parquet'}")
    print(f"- {args.output_dir / 'pead_bucket_summary.csv'}")
    print(f"- {args.output_dir / 'pead_summary.md'}")


if __name__ == "__main__":
    main()
