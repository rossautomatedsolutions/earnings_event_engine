"""Honest orchestration script for the end-to-end research pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

# Ensure local packages are importable when running as a script.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT_DIR / "src"
for path in (SRC_PATH, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.conditional_analysis import run_conditional_analysis
from analysis.earnings_analysis import run_full_analysis
from strategies.straddle_simulator import run_straddle_analysis


PIPELINE_SCRIPT = ROOT_DIR / "scripts" / "run_event_pipeline_test.py"
DEFAULT_FIXTURE_DIR = ROOT_DIR / "tests" / "fixtures" / "offline_pipeline"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "full_research"
VALIDATED_FEATURES_FILENAME = "validated_features.parquet"

EVENT_PIPELINE_STAGE = "event_pipeline"
ANALYSIS_STAGE = "earnings_analysis"
STRATEGY_STAGE = "strategy_simulation"
CONDITIONALS_STAGE = "conditional_analysis"
VISUALS_STAGE = "visualizations"

EVENT_KEY_COLUMN = "event_id"
VALID_EVENT_COLUMN = "valid_event"
INVALID_EVENT_COLUMN = "invalid_event"
REQUIRED_EVENT_COLUMNS = {"event_id", "ticker", "event_date", "announcement_timing"}
REQUIRED_CLEAN_EVENT_COLUMNS = {
    "event_id",
    "ticker",
    "event_date",
    "announcement_timing",
    "pre_close",
    "post_close",
}
REQUIRED_FEATURE_COLUMNS = {
    "ticker",
    "announcement_timing",
    "realized_gap_move",
    "implied_move",
    "surprise_vs_implied",
}
REQUIRED_DIAGNOSTICS_COLUMNS = {"category", "metric", "count", "pct_of_events"}
REQUIRED_PIPELINE_ARTIFACTS = {
    "events.parquet": REQUIRED_EVENT_COLUMNS,
    "clean_events.parquet": REQUIRED_CLEAN_EVENT_COLUMNS,
    "features.parquet": REQUIRED_FEATURE_COLUMNS,
    "diagnostics.parquet": REQUIRED_DIAGNOSTICS_COLUMNS,
}


class StageFailure(RuntimeError):
    """Raised when a required orchestration stage fails."""

    def __init__(self, stage_name: str, detail: str):
        super().__init__(detail)
        self.stage_name = stage_name
        self.detail = detail


@dataclass
class StageResult:
    """Summary of one orchestration stage."""

    name: str
    required: bool
    status: str
    detail: str
    elapsed_seconds: float


@dataclass
class PipelineArtifacts:
    """Validated artifact paths and row counts emitted by the pipeline stage."""

    output_dir: Path
    events_path: Path
    clean_events_path: Path
    features_path: Path
    diagnostics_path: Path
    raw_event_count: int
    clean_event_count: int
    feature_row_count: int
    validated_features_path: Path | None = None
    validated_feature_row_count: int = 0


@dataclass
class ResearchRunReport:
    """Full orchestration result returned to callers and tests."""

    status: str
    exit_code: int
    mode: str
    output_dir: Path
    stage_results: list[StageResult] = field(default_factory=list)
    failed_stage: str | None = None
    failure_detail: str | None = None
    pipeline_artifacts: PipelineArtifacts | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    parser = argparse.ArgumentParser(
        description=(
            "Run the canonical research pipeline with honest failure propagation. "
            "Use --fixture-dir for the deterministic offline path or --tickers for live mode."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where pipeline artifacts will be written.",
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=None,
        help=(
            "Deterministic offline fixture directory. When provided, the run stays fully offline "
            "and never falls back to live data."
        ),
    )
    parser.add_argument(
        "--tickers",
        default=None,
        help="Comma-separated ticker list required for live mode.",
    )
    parser.add_argument(
        "--start-date",
        default=(today - pd.Timedelta(days=365 * 2)).date().isoformat(),
        help="Inclusive start date in YYYY-MM-DD format for live mode.",
    )
    parser.add_argument(
        "--end-date",
        default=today.date().isoformat(),
        help="Inclusive end date in YYYY-MM-DD format for live mode.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=50,
        help="Maximum events to process in live mode.",
    )
    parser.add_argument(
        "--run-visuals",
        action="store_true",
        help=(
            "Run the optional matplotlib-based visualization stage after the required stages. "
            "This stage is excluded from the default canonical contract."
        ),
    )
    return parser.parse_args(argv)


def _print_header(report: ResearchRunReport, fixture_dir: Path | None) -> None:
    print("=== Full Research Pipeline ===")
    print(f"mode={report.mode}")
    print(f"output_dir={report.output_dir.resolve()}")
    if fixture_dir is not None:
        print(f"fixture_dir={fixture_dir.resolve()}")
    else:
        print("fixture_dir=<live mode>")


def _trim_output(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1]


def _echo_process_output(result: subprocess.CompletedProcess[str]) -> None:
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)


def _remove_stale_contract_artifacts(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for artifact_name in [*REQUIRED_PIPELINE_ARTIFACTS, VALIDATED_FEATURES_FILENAME]:
        (output_dir / artifact_name).unlink(missing_ok=True)


def _load_and_validate_artifact(
    path: Path,
    *,
    required_columns: set[str],
    require_non_empty: bool,
) -> pd.DataFrame:
    if not path.exists():
        raise StageFailure(EVENT_PIPELINE_STAGE, f"Missing required artifact: {path}")

    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - present artifact-level context
        raise StageFailure(
            EVENT_PIPELINE_STAGE,
            f"Unreadable required artifact {path}: {exc.__class__.__name__}: {exc}",
        ) from exc

    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise StageFailure(
            EVENT_PIPELINE_STAGE,
            f"Artifact {path} is missing required columns: {missing_columns}",
        )

    if require_non_empty and df.empty:
        raise StageFailure(EVENT_PIPELINE_STAGE, f"Artifact {path} contains zero rows.")

    return df


def validate_pipeline_artifacts(output_dir: Path) -> PipelineArtifacts:
    """Validate the required upstream contract before downstream stages run."""

    output_dir = Path(output_dir)
    events_path = output_dir / "events.parquet"
    clean_events_path = output_dir / "clean_events.parquet"
    features_path = output_dir / "features.parquet"
    diagnostics_path = output_dir / "diagnostics.parquet"

    events_df = _load_and_validate_artifact(
        events_path,
        required_columns=REQUIRED_EVENT_COLUMNS,
        require_non_empty=True,
    )
    clean_events_df = _load_and_validate_artifact(
        clean_events_path,
        required_columns=REQUIRED_CLEAN_EVENT_COLUMNS,
        require_non_empty=True,
    )
    features_df = _load_and_validate_artifact(
        features_path,
        required_columns=REQUIRED_FEATURE_COLUMNS,
        require_non_empty=True,
    )
    _load_and_validate_artifact(
        diagnostics_path,
        required_columns=REQUIRED_DIAGNOSTICS_COLUMNS,
        require_non_empty=False,
    )

    return PipelineArtifacts(
        output_dir=output_dir,
        events_path=events_path,
        clean_events_path=clean_events_path,
        features_path=features_path,
        diagnostics_path=diagnostics_path,
        raw_event_count=len(events_df),
        clean_event_count=len(clean_events_df),
        feature_row_count=len(features_df),
    )


def _coerce_strict_boolean_flags(series: pd.Series, *, artifact_name: str, column_name: str) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=bool)

    if pd.api.types.is_bool_dtype(series):
        normalized = series.astype("boolean")
    else:
        text_values = series.astype("string").str.strip().str.lower()
        mapping = {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        }
        invalid_mask = series.notna() & ~text_values.isin(mapping)
        if invalid_mask.any():
            invalid_values = sorted({str(value) for value in series.loc[invalid_mask].tolist()})
            raise StageFailure(
                EVENT_PIPELINE_STAGE,
                f"{artifact_name} has ambiguous values in {column_name}: {invalid_values}",
            )
        normalized = text_values.map(mapping).astype("boolean")

    if normalized.isna().any():
        raise StageFailure(
            EVENT_PIPELINE_STAGE,
            f"{artifact_name} is missing explicit boolean values in {column_name}.",
        )

    return normalized.astype(bool)


def _validate_unique_event_keys(df: pd.DataFrame, *, artifact_name: str) -> None:
    if EVENT_KEY_COLUMN not in df.columns:
        raise StageFailure(
            EVENT_PIPELINE_STAGE,
            f"{artifact_name} is missing required event key column '{EVENT_KEY_COLUMN}'.",
        )

    if df[EVENT_KEY_COLUMN].isna().any():
        raise StageFailure(
            EVENT_PIPELINE_STAGE,
            f"{artifact_name} contains null values in required event key column '{EVENT_KEY_COLUMN}'.",
        )

    duplicates = df.loc[df[EVENT_KEY_COLUMN].duplicated(), EVENT_KEY_COLUMN].tolist()
    if duplicates:
        duplicate_values = sorted({str(value) for value in duplicates})
        raise StageFailure(
            EVENT_PIPELINE_STAGE,
            f"{artifact_name} contains duplicate event keys: {duplicate_values}",
        )


def _write_validated_features_artifact(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def prepare_validated_research_artifact(pipeline_artifacts: PipelineArtifacts) -> tuple[PipelineArtifacts, pd.DataFrame]:
    """Build the validated-only feature artifact used by required downstream research."""

    clean_events_df = _load_and_validate_artifact(
        pipeline_artifacts.clean_events_path,
        required_columns=REQUIRED_CLEAN_EVENT_COLUMNS,
        require_non_empty=True,
    )
    required_feature_columns = REQUIRED_FEATURE_COLUMNS | {EVENT_KEY_COLUMN, VALID_EVENT_COLUMN}
    features_df = _load_and_validate_artifact(
        pipeline_artifacts.features_path,
        required_columns=required_feature_columns,
        require_non_empty=True,
    )

    _validate_unique_event_keys(clean_events_df, artifact_name="clean_events.parquet")
    _validate_unique_event_keys(features_df, artifact_name="features.parquet")

    valid_mask = _coerce_strict_boolean_flags(
        features_df[VALID_EVENT_COLUMN],
        artifact_name="features.parquet",
        column_name=VALID_EVENT_COLUMN,
    )

    if INVALID_EVENT_COLUMN in features_df.columns:
        invalid_mask = _coerce_strict_boolean_flags(
            features_df[INVALID_EVENT_COLUMN],
            artifact_name="features.parquet",
            column_name=INVALID_EVENT_COLUMN,
        )
        inconsistent_flags = valid_mask == invalid_mask
        if inconsistent_flags.any():
            event_ids = features_df.loc[inconsistent_flags, EVENT_KEY_COLUMN].astype(str).tolist()
            raise StageFailure(
                EVENT_PIPELINE_STAGE,
                f"features.parquet has inconsistent {VALID_EVENT_COLUMN}/{INVALID_EVENT_COLUMN} flags for event_ids={event_ids}",
            )

    validated_features_df = features_df.loc[valid_mask].copy()
    if validated_features_df.empty:
        raise StageFailure(
            EVENT_PIPELINE_STAGE,
            "Validated feature gate failed: features.parquet contains no rows with valid_event=True.",
        )

    clean_event_ids = set(clean_events_df[EVENT_KEY_COLUMN].tolist())
    validated_event_ids = set(validated_features_df[EVENT_KEY_COLUMN].tolist())
    if validated_event_ids != clean_event_ids:
        clean_only = sorted(str(value) for value in clean_event_ids - validated_event_ids)
        validated_only = sorted(str(value) for value in validated_event_ids - clean_event_ids)
        raise StageFailure(
            EVENT_PIPELINE_STAGE,
            "Validated feature rows are inconsistent with clean_events.parquet event_ids. "
            f"clean_only={clean_only} validated_only={validated_only}",
        )

    validated_features_path = pipeline_artifacts.output_dir / VALIDATED_FEATURES_FILENAME
    _write_validated_features_artifact(validated_features_df, validated_features_path)
    pipeline_artifacts.validated_features_path = validated_features_path
    pipeline_artifacts.validated_feature_row_count = len(validated_features_df)
    return pipeline_artifacts, validated_features_df


def run_canonical_event_pipeline_stage(
    *,
    output_dir: Path,
    fixture_dir: Path | None,
    tickers: str | None,
    start_date: str,
    end_date: str,
    max_events: int,
) -> tuple[PipelineArtifacts, str]:
    """Run the pipeline stage and enforce the validated-only downstream boundary."""

    pipeline_artifacts, detail = run_event_pipeline_stage(
        output_dir=output_dir,
        fixture_dir=fixture_dir,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        max_events=max_events,
    )
    pipeline_artifacts, _ = prepare_validated_research_artifact(pipeline_artifacts)
    detail = (
        f"{detail} "
        f"validated_feature_rows={pipeline_artifacts.validated_feature_row_count}"
    )
    return pipeline_artifacts, detail


def _require_validated_features_path(pipeline_artifacts: PipelineArtifacts) -> Path:
    if pipeline_artifacts.validated_features_path is None:
        raise StageFailure(
            EVENT_PIPELINE_STAGE,
            "Validated feature artifact was not prepared before downstream research stages.",
        )
    return pipeline_artifacts.validated_features_path


def run_event_pipeline_stage(
    *,
    output_dir: Path,
    fixture_dir: Path | None,
    tickers: str | None,
    start_date: str,
    end_date: str,
    max_events: int,
) -> tuple[PipelineArtifacts, str]:
    """Run the validated event pipeline entry point and enforce its artifact contract."""

    if fixture_dir is None and not tickers:
        raise StageFailure(
            EVENT_PIPELINE_STAGE,
            "Live mode requires --tickers. Use --fixture-dir for the deterministic offline path.",
        )

    _remove_stale_contract_artifacts(output_dir)

    command = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--output-dir",
        str(output_dir),
    ]
    if fixture_dir is not None:
        command.extend(["--fixture-dir", str(fixture_dir)])
    else:
        command.extend(
            [
                "--tickers",
                str(tickers),
                "--start-date",
                start_date,
                "--end-date",
                end_date,
                "--max-events",
                str(max_events),
            ]
        )

    print(f"pipeline_command={subprocess.list2cmdline(command)}")
    result = subprocess.run(
        command,
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    _echo_process_output(result)

    if result.returncode != 0:
        detail = _trim_output(result.stderr) or _trim_output(result.stdout)
        if not detail:
            detail = f"Pipeline command exited with code {result.returncode}."
        raise StageFailure(EVENT_PIPELINE_STAGE, detail)

    artifacts = validate_pipeline_artifacts(output_dir)
    detail = (
        f"raw_events={artifacts.raw_event_count} "
        f"clean_events={artifacts.clean_event_count} "
        f"feature_rows={artifacts.feature_row_count}"
    )
    return artifacts, detail


def run_analysis_stage(features_path: Path) -> tuple[dict[str, Any], str]:
    """Run the required downstream earnings-analysis stage."""

    analysis = run_full_analysis(str(features_path))
    by_ticker = analysis.get("by_ticker")
    by_timing = analysis.get("by_timing")

    if not isinstance(by_ticker, pd.DataFrame) or by_ticker.empty:
        raise StageFailure(ANALYSIS_STAGE, "Analysis stage produced an empty by_ticker output.")
    if not isinstance(by_timing, pd.DataFrame) or by_timing.empty:
        raise StageFailure(ANALYSIS_STAGE, "Analysis stage produced an empty by_timing output.")

    detail = f"by_ticker_rows={len(by_ticker)} by_timing_rows={len(by_timing)}"
    return analysis, detail


def run_strategy_stage(features_path: Path) -> tuple[pd.DataFrame, str]:
    """Run the required strategy simulation stage."""

    strategy_results, _, ticker_breakdown = run_straddle_analysis(str(features_path))
    if strategy_results.empty:
        raise StageFailure(
            STRATEGY_STAGE,
            "Strategy simulation produced zero rows from a non-empty feature artifact.",
        )

    detail = (
        f"simulation_rows={len(strategy_results)} "
        f"ticker_breakdown_rows={len(ticker_breakdown)}"
    )
    return strategy_results, detail


def run_conditionals_stage(strategy_results: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], str]:
    """Run the required conditional-analysis stage."""

    outputs = run_conditional_analysis(strategy_results)
    group_comparison = outputs.get("group_comparison")
    timing_comparison = outputs.get("timing_comparison")

    if not isinstance(group_comparison, pd.DataFrame) or group_comparison.empty:
        raise StageFailure(CONDITIONALS_STAGE, "Conditional analysis produced an empty group_comparison output.")
    if not isinstance(timing_comparison, pd.DataFrame) or timing_comparison.empty:
        raise StageFailure(CONDITIONALS_STAGE, "Conditional analysis produced an empty timing_comparison output.")

    for name, frame in outputs.items():
        print(f"[conditional_output] {name} rows={len(frame)}")

    detail = (
        f"group_rows={len(group_comparison)} "
        f"timing_rows={len(timing_comparison)}"
    )
    return outputs, detail


def run_visuals_stage(features_path: Path) -> tuple[None, str]:
    """Run the optional plotting stage only when explicitly requested."""

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from analysis.visualizations import run_all_plots
    except ModuleNotFoundError as exc:
        missing_dependency = exc.name or "matplotlib"
        raise StageFailure(
            VISUALS_STAGE,
            f"Visualization stage requires dependency '{missing_dependency}' when --run-visuals is enabled.",
        ) from exc

    run_all_plots(str(features_path))
    return None, "matplotlib plotting completed"


def _run_stage(
    report: ResearchRunReport,
    *,
    stage_name: str,
    required: bool,
    stage_fn,
) -> Any:
    start = time.perf_counter()
    print(f"\n=== STAGE: {stage_name} ===")
    try:
        value, detail = stage_fn()
    except StageFailure as exc:
        elapsed = time.perf_counter() - start
        report.failed_stage = exc.stage_name
        report.failure_detail = exc.detail
        report.stage_results.append(
            StageResult(
                name=stage_name,
                required=required,
                status="FAILED",
                detail=exc.detail,
                elapsed_seconds=elapsed,
            )
        )
        print(f"[FAILED] {stage_name}: {exc.detail}")
        raise
    except Exception as exc:  # noqa: BLE001 - preserve stage name in failure output
        elapsed = time.perf_counter() - start
        detail = f"{exc.__class__.__name__}: {exc}"
        report.failed_stage = stage_name
        report.failure_detail = detail
        report.stage_results.append(
            StageResult(
                name=stage_name,
                required=required,
                status="FAILED",
                detail=detail,
                elapsed_seconds=elapsed,
            )
        )
        print(f"[FAILED] {stage_name}: {detail}")
        raise StageFailure(stage_name, detail) from exc

    elapsed = time.perf_counter() - start
    report.stage_results.append(
        StageResult(
            name=stage_name,
            required=required,
            status="SUCCESS",
            detail=detail,
            elapsed_seconds=elapsed,
        )
    )
    print(f"[SUCCESS] {stage_name}: {detail}")
    return value


def _record_skipped_optional_stage(report: ResearchRunReport, stage_name: str, detail: str) -> None:
    report.stage_results.append(
        StageResult(
            name=stage_name,
            required=False,
            status="SKIPPED",
            detail=detail,
            elapsed_seconds=0.0,
        )
    )


def run_full_research_pipeline(
    *,
    output_dir: Path,
    fixture_dir: Path | None,
    tickers: str | None,
    start_date: str,
    end_date: str,
    max_events: int,
    run_visuals: bool,
) -> ResearchRunReport:
    """Run the required research stages and return an honest run report."""

    mode = "fixture" if fixture_dir is not None else "live"
    report = ResearchRunReport(
        status="FAILED",
        exit_code=1,
        mode=mode,
        output_dir=Path(output_dir),
    )

    _print_header(report, fixture_dir)

    try:
        pipeline_artifacts = _run_stage(
            report,
            stage_name=EVENT_PIPELINE_STAGE,
            required=True,
            stage_fn=lambda: run_canonical_event_pipeline_stage(
                output_dir=Path(output_dir),
                fixture_dir=fixture_dir,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                max_events=max_events,
            ),
        )
        report.pipeline_artifacts = pipeline_artifacts

        _run_stage(
            report,
            stage_name=ANALYSIS_STAGE,
            required=True,
            stage_fn=lambda: run_analysis_stage(_require_validated_features_path(pipeline_artifacts)),
        )

        strategy_results = _run_stage(
            report,
            stage_name=STRATEGY_STAGE,
            required=True,
            stage_fn=lambda: run_strategy_stage(_require_validated_features_path(pipeline_artifacts)),
        )

        _run_stage(
            report,
            stage_name=CONDITIONALS_STAGE,
            required=True,
            stage_fn=lambda: run_conditionals_stage(strategy_results),
        )

        if run_visuals:
            _run_stage(
                report,
                stage_name=VISUALS_STAGE,
                required=True,
                stage_fn=lambda: run_visuals_stage(_require_validated_features_path(pipeline_artifacts)),
            )
        else:
            _record_skipped_optional_stage(
                report,
                VISUALS_STAGE,
                "Optional stage disabled; canonical run excludes plotting and does not require matplotlib.",
            )
    except StageFailure:
        return report

    report.status = "SUCCESS"
    report.exit_code = 0
    return report


def print_run_summary(report: ResearchRunReport) -> None:
    """Emit a truthful final summary for humans and CLI tests."""

    print("\n=== RUN SUMMARY ===")
    for stage in report.stage_results:
        print(
            f"- {stage.name}: {stage.status} "
            f"(required={stage.required}, elapsed={stage.elapsed_seconds:.2f}s) "
            f"{stage.detail}"
        )

    if report.status == "SUCCESS":
        print("\nFINAL STATUS: SUCCESS")
    else:
        print("\nFINAL STATUS: FAILED")
        if report.failed_stage is not None:
            print(f"Failed stage: {report.failed_stage}")
        if report.failure_detail:
            print(f"Reason: {report.failure_detail}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_full_research_pipeline(
        output_dir=args.output_dir,
        fixture_dir=args.fixture_dir,
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        max_events=args.max_events,
        run_visuals=args.run_visuals,
    )
    print_run_summary(report)
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
