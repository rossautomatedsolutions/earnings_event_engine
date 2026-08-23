from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"


def _prioritize_repo_import_paths() -> None:
    for path in (str(REPO_ROOT), str(SRC_PATH)):
        while path in sys.path:
            sys.path.remove(path)

    sys.path.insert(0, str(SRC_PATH))
    sys.path.insert(0, str(REPO_ROOT))


_prioritize_repo_import_paths()

from scripts import run_full_research_pipeline as full_pipeline


FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "offline_pipeline"
TF_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "trading_foundation_historical_data"
FULL_PIPELINE_SCRIPT = REPO_ROOT / "scripts" / "run_full_research_pipeline.py"


class FullResearchPipelineTest(unittest.TestCase):
    def _run_cli(
        self,
        *extra_args: str,
        fixture_dir: Path | None = FIXTURE_DIR,
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        tmp_dir = Path(tempfile.mkdtemp())
        output_dir = tmp_dir / "outputs"
        command = [
            sys.executable,
            str(FULL_PIPELINE_SCRIPT),
            "--output-dir",
            str(output_dir),
        ]
        if fixture_dir is not None:
            command.extend(["--fixture-dir", str(fixture_dir)])
        command.extend(extra_args)

        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return result, output_dir

    def _make_fixture_dir(
        self,
        *,
        events_csv: str,
        prices_csv: str | None = None,
        options_csv: str | None = None,
    ) -> Path:
        tmp_dir = Path(tempfile.mkdtemp())
        fixture_dir = tmp_dir / "fixture"
        fixture_dir.mkdir(parents=True, exist_ok=True)
        (fixture_dir / "earnings_events.csv").write_text(events_csv, encoding="utf-8")

        if prices_csv is None:
            prices_csv = (FIXTURE_DIR / "prices.csv").read_text(encoding="utf-8")
        (fixture_dir / "prices.csv").write_text(prices_csv, encoding="utf-8")

        if options_csv is not None:
            (fixture_dir / "options.csv").write_text(options_csv, encoding="utf-8")

        return fixture_dir

    def _write_pipeline_artifacts(
        self,
        output_dir: Path,
        *,
        events_df: pd.DataFrame,
        clean_events_df: pd.DataFrame,
        features_df: pd.DataFrame,
        diagnostics_df: pd.DataFrame | None = None,
        validate_with_helper: bool = True,
    ) -> full_pipeline.PipelineArtifacts:
        output_dir.mkdir(parents=True, exist_ok=True)

        if diagnostics_df is None:
            diagnostics_df = pd.DataFrame(
                [
                    {
                        "category": "overview",
                        "metric": "total_events",
                        "count": len(events_df),
                        "pct_of_events": 1.0 if len(events_df) else 0.0,
                    },
                    {
                        "category": "overview",
                        "metric": "valid_events",
                        "count": len(clean_events_df),
                        "pct_of_events": len(clean_events_df) / max(len(events_df), 1),
                    },
                    {
                        "category": "overview",
                        "metric": "invalid_events",
                        "count": max(len(events_df) - len(clean_events_df), 0),
                        "pct_of_events": max(len(events_df) - len(clean_events_df), 0) / max(len(events_df), 1),
                    },
                ]
            )

        events_df.to_parquet(output_dir / "events.parquet", index=False)
        clean_events_df.to_parquet(output_dir / "clean_events.parquet", index=False)
        features_df.to_parquet(output_dir / "features.parquet", index=False)
        diagnostics_df.to_parquet(output_dir / "diagnostics.parquet", index=False)

        if validate_with_helper:
            return full_pipeline.validate_pipeline_artifacts(output_dir)

        return full_pipeline.PipelineArtifacts(
            output_dir=output_dir,
            events_path=output_dir / "events.parquet",
            clean_events_path=output_dir / "clean_events.parquet",
            features_path=output_dir / "features.parquet",
            diagnostics_path=output_dir / "diagnostics.parquet",
            raw_event_count=len(events_df),
            clean_event_count=len(clean_events_df),
            feature_row_count=len(features_df),
        )

    def _stage(self, report: full_pipeline.ResearchRunReport, name: str) -> full_pipeline.StageResult:
        for stage in report.stage_results:
            if stage.name == name:
                return stage
        self.fail(f"Stage {name!r} not found in report.")

    def test_callable_fixture_run_succeeds_and_marks_optional_visuals_skipped(self) -> None:
        output_dir = Path(tempfile.mkdtemp()) / "outputs"

        report = full_pipeline.run_full_research_pipeline(
            output_dir=output_dir,
            fixture_dir=FIXTURE_DIR,
            tickers=None,
            start_date="2025-01-01",
            end_date="2025-12-31",
            max_events=50,
            run_visuals=False,
        )

        self.assertEqual(report.status, "SUCCESS")
        self.assertEqual(report.exit_code, 0)
        self.assertIsNotNone(report.pipeline_artifacts)
        assert report.pipeline_artifacts is not None
        self.assertGreater(report.pipeline_artifacts.raw_event_count, 0)
        self.assertGreater(report.pipeline_artifacts.clean_event_count, 0)
        self.assertGreater(report.pipeline_artifacts.feature_row_count, 0)
        self.assertGreater(report.pipeline_artifacts.validated_feature_row_count, 0)
        self.assertIsNotNone(report.pipeline_artifacts.validated_features_path)
        assert report.pipeline_artifacts.validated_features_path is not None
        self.assertTrue(report.pipeline_artifacts.validated_features_path.exists())

        self.assertEqual(self._stage(report, full_pipeline.EVENT_PIPELINE_STAGE).status, "SUCCESS")
        self.assertEqual(self._stage(report, full_pipeline.ANALYSIS_STAGE).status, "SUCCESS")
        self.assertEqual(self._stage(report, full_pipeline.STRATEGY_STAGE).status, "SUCCESS")
        self.assertEqual(self._stage(report, full_pipeline.CONDITIONALS_STAGE).status, "SUCCESS")
        self.assertEqual(self._stage(report, full_pipeline.VISUALS_STAGE).status, "SKIPPED")

    def test_cli_fixture_run_exits_zero_and_reaches_required_downstream_analysis(self) -> None:
        result, output_dir = self._run_cli()
        combined_output = result.stdout + result.stderr

        self.assertEqual(result.returncode, 0, msg=combined_output)
        self.assertIn("FINAL STATUS: SUCCESS", combined_output)
        self.assertIn(f"- {full_pipeline.ANALYSIS_STAGE}: SUCCESS", combined_output)
        self.assertIn(f"- {full_pipeline.STRATEGY_STAGE}: SUCCESS", combined_output)
        self.assertIn(f"- {full_pipeline.CONDITIONALS_STAGE}: SUCCESS", combined_output)
        self.assertIn(f"- {full_pipeline.VISUALS_STAGE}: SKIPPED", combined_output)

        features = pd.read_parquet(output_dir / "features.parquet")
        validated_features = pd.read_parquet(output_dir / full_pipeline.VALIDATED_FEATURES_FILENAME)
        self.assertFalse(features.empty)
        self.assertFalse(validated_features.empty)
        validated_by_ticker = validated_features.set_index("ticker")
        self.assertAlmostEqual(float(validated_by_ticker.loc["AAPL", "implied_move"]), (4.8 + 4.7) / 104.5, places=6)
        self.assertAlmostEqual(float(validated_by_ticker.loc["MSFT", "implied_move"]), (5.2 + 5.1) / 210.0, places=6)
        self.assertTrue(validated_features["implied_move_available"].fillna(False).all())

    def test_cli_fixture_run_can_use_trading_foundation_historical_root(self) -> None:
        result, output_dir = self._run_cli("--historical-data-root", str(TF_FIXTURE_ROOT))
        combined_output = result.stdout + result.stderr

        self.assertEqual(result.returncode, 0, msg=combined_output)
        self.assertIn("historical_data_root=", combined_output)
        self.assertIn("option_data_source=trading_foundation_option_snapshot", combined_output)

        validated_features = pd.read_parquet(output_dir / full_pipeline.VALIDATED_FEATURES_FILENAME).set_index("ticker")
        self.assertAlmostEqual(float(validated_features.loc["AAPL", "implied_move"]), (6.0 + 5.8) / 106.0, places=6)
        self.assertAlmostEqual(float(validated_features.loc["MSFT", "implied_move"]), (6.2 + 6.0) / 205.0, places=6)
        self.assertEqual(
            pd.Timestamp(validated_features.loc["AAPL", "option_snapshot_quote_date"]).date().isoformat(),
            "2025-01-30",
        )
        self.assertEqual(
            pd.Timestamp(validated_features.loc["MSFT", "option_snapshot_quote_date"]).date().isoformat(),
            "2025-02-04",
        )

    def test_mixed_validity_fixture_routes_only_validated_rows_to_required_downstream_stages(self) -> None:
        fixture_dir = self._make_fixture_dir(
            events_csv=(FIXTURE_DIR / "earnings_events.csv").read_text(encoding="utf-8"),
            options_csv=(
                "ticker,event_date,option_type,strike,underlying_price,bid,ask,mid\n"
                "AAPL,2025-01-30,call,104.5,104.5,4.7,4.9,4.8\n"
                "AAPL,2025-01-30,put,104.5,104.5,4.6,4.8,4.7\n"
            ),
        )
        output_dir = Path(tempfile.mkdtemp()) / "outputs"
        captured: dict[str, list[str]] = {}

        original_run_full_analysis = full_pipeline.run_full_analysis
        original_run_straddle_analysis = full_pipeline.run_straddle_analysis
        original_run_conditional_analysis = full_pipeline.run_conditional_analysis

        def _capture_analysis(path: str) -> dict[str, object]:
            frame = pd.read_parquet(path)
            captured["analysis_tickers"] = frame["ticker"].astype(str).tolist()
            captured["analysis_event_ids"] = frame["event_id"].astype(str).tolist()
            return original_run_full_analysis(path)

        def _capture_strategy(path: str) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
            frame = pd.read_parquet(path)
            captured["strategy_tickers"] = frame["ticker"].astype(str).tolist()
            captured["strategy_event_ids"] = frame["event_id"].astype(str).tolist()
            return original_run_straddle_analysis(path)

        def _capture_conditionals(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
            captured["conditional_tickers"] = data["ticker"].astype(str).tolist()
            captured["conditional_event_ids"] = data["event_id"].astype(str).tolist()
            return original_run_conditional_analysis(data)

        with (
            patch.object(full_pipeline, "run_full_analysis", side_effect=_capture_analysis),
            patch.object(full_pipeline, "run_straddle_analysis", side_effect=_capture_strategy),
            patch.object(full_pipeline, "run_conditional_analysis", side_effect=_capture_conditionals),
        ):
            report = full_pipeline.run_full_research_pipeline(
                output_dir=output_dir,
                fixture_dir=fixture_dir,
                tickers=None,
                start_date="2025-01-01",
                end_date="2025-12-31",
                max_events=50,
                run_visuals=False,
            )

        raw_features = pd.read_parquet(output_dir / "features.parquet")
        validated_features = pd.read_parquet(output_dir / full_pipeline.VALIDATED_FEATURES_FILENAME)

        self.assertEqual(report.status, "SUCCESS")
        self.assertEqual(report.exit_code, 0)
        assert report.pipeline_artifacts is not None
        self.assertEqual(report.pipeline_artifacts.raw_event_count, 2)
        self.assertEqual(report.pipeline_artifacts.clean_event_count, 1)
        self.assertEqual(report.pipeline_artifacts.feature_row_count, 2)
        self.assertEqual(report.pipeline_artifacts.validated_feature_row_count, 1)
        self.assertCountEqual(raw_features["ticker"].astype(str).tolist(), ["AAPL", "MSFT"])
        self.assertEqual(validated_features["ticker"].astype(str).tolist(), ["AAPL"])
        raw_by_ticker = raw_features.set_index("ticker")
        self.assertAlmostEqual(float(raw_by_ticker.loc["AAPL", "implied_move"]), (4.8 + 4.7) / 104.5, places=6)
        self.assertTrue(pd.isna(raw_by_ticker.loc["MSFT", "implied_move"]))
        self.assertTrue(bool(raw_by_ticker.loc["AAPL", "implied_move_available"]))
        self.assertFalse(bool(raw_by_ticker.loc["MSFT", "implied_move_available"]))
        self.assertTrue(pd.notna(raw_by_ticker.loc["MSFT", "hv20_expected_move_proxy"]))
        self.assertTrue(pd.isna(raw_by_ticker.loc["MSFT", "surprise_vs_implied"]))
        self.assertEqual(captured["analysis_tickers"], ["AAPL"])
        self.assertEqual(captured["strategy_tickers"], ["AAPL"])
        self.assertEqual(captured["conditional_tickers"], ["AAPL"])
        self.assertEqual(captured["analysis_event_ids"], ["1"])
        self.assertEqual(captured["strategy_event_ids"], ["1"])
        self.assertEqual(captured["conditional_event_ids"], ["1"])

    def test_zero_validated_feature_rows_fail_before_downstream_research(self) -> None:
        output_dir = Path(tempfile.mkdtemp()) / "outputs"
        events_df = pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "ticker": "AAPL",
                    "event_date": "2025-01-30",
                    "announcement_timing": "AMC",
                }
            ]
        )
        clean_events_df = pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "ticker": "AAPL",
                    "event_date": "2025-01-30",
                    "announcement_timing": "AMC",
                    "pre_close": 100.0,
                    "post_close": 101.0,
                }
            ]
        )
        features_df = pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "ticker": "AAPL",
                    "announcement_timing": "AMC",
                    "realized_gap_move": 0.01,
                    "implied_move": 0.02,
                    "surprise_vs_implied": -0.01,
                    "valid_event": False,
                    "invalid_event": True,
                    "failure_reasons": "missing_options_data",
                }
            ]
        )

        def _fake_pipeline_stage(**kwargs: object) -> tuple[full_pipeline.PipelineArtifacts, str]:
            artifacts = self._write_pipeline_artifacts(
                Path(kwargs["output_dir"]),
                events_df=events_df,
                clean_events_df=clean_events_df,
                features_df=features_df,
            )
            return artifacts, "synthetic zero validated feature case"

        with (
            patch.object(full_pipeline, "run_event_pipeline_stage", side_effect=_fake_pipeline_stage),
            patch.object(full_pipeline, "run_full_analysis") as mock_analysis,
        ):
            report = full_pipeline.run_full_research_pipeline(
                output_dir=output_dir,
                fixture_dir=FIXTURE_DIR,
                tickers=None,
                start_date="2025-01-01",
                end_date="2025-12-31",
                max_events=50,
                run_visuals=False,
            )

        self.assertEqual(report.status, "FAILED")
        self.assertEqual(report.exit_code, 1)
        self.assertEqual(report.failed_stage, full_pipeline.EVENT_PIPELINE_STAGE)
        self.assertIn("valid_event=True", report.failure_detail or "")
        self.assertEqual(self._stage(report, full_pipeline.EVENT_PIPELINE_STAGE).status, "FAILED")
        self.assertFalse(mock_analysis.called)

    def test_missing_validation_flag_fails_before_downstream_research(self) -> None:
        output_dir = Path(tempfile.mkdtemp()) / "outputs"
        events_df = pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "ticker": "AAPL",
                    "event_date": "2025-01-30",
                    "announcement_timing": "AMC",
                }
            ]
        )
        clean_events_df = pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "ticker": "AAPL",
                    "event_date": "2025-01-30",
                    "announcement_timing": "AMC",
                    "pre_close": 100.0,
                    "post_close": 101.0,
                }
            ]
        )
        features_df = pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "ticker": "AAPL",
                    "announcement_timing": "AMC",
                    "realized_gap_move": 0.01,
                    "implied_move": 0.02,
                    "surprise_vs_implied": -0.01,
                }
            ]
        )

        def _fake_pipeline_stage(**kwargs: object) -> tuple[full_pipeline.PipelineArtifacts, str]:
            artifacts = self._write_pipeline_artifacts(
                Path(kwargs["output_dir"]),
                events_df=events_df,
                clean_events_df=clean_events_df,
                features_df=features_df,
            )
            return artifacts, "synthetic missing validation flag case"

        with (
            patch.object(full_pipeline, "run_event_pipeline_stage", side_effect=_fake_pipeline_stage),
            patch.object(full_pipeline, "run_full_analysis") as mock_analysis,
        ):
            report = full_pipeline.run_full_research_pipeline(
                output_dir=output_dir,
                fixture_dir=FIXTURE_DIR,
                tickers=None,
                start_date="2025-01-01",
                end_date="2025-12-31",
                max_events=50,
                run_visuals=False,
            )

        self.assertEqual(report.status, "FAILED")
        self.assertEqual(report.exit_code, 1)
        self.assertEqual(report.failed_stage, full_pipeline.EVENT_PIPELINE_STAGE)
        self.assertIn("valid_event", report.failure_detail or "")
        self.assertEqual(self._stage(report, full_pipeline.EVENT_PIPELINE_STAGE).status, "FAILED")
        self.assertFalse(mock_analysis.called)

    def test_zero_raw_events_failure_propagates_nonzero_and_identifies_stage(self) -> None:
        fixture_dir = self._make_fixture_dir(
            events_csv="ticker,earnings_date,announcement_timing\n",
            options_csv=(FIXTURE_DIR / "options.csv").read_text(encoding="utf-8"),
        )

        result, _ = self._run_cli(fixture_dir=fixture_dir)
        combined_output = result.stdout + result.stderr

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FINAL STATUS: FAILED", combined_output)
        self.assertIn(f"Failed stage: {full_pipeline.EVENT_PIPELINE_STAGE}", combined_output)
        self.assertIn("Zero raw events produced during the retrieval stage", combined_output)
        self.assertNotIn("FINAL STATUS: SUCCESS", combined_output)

    def test_missing_required_artifact_propagates_from_callable(self) -> None:
        output_dir = Path(tempfile.mkdtemp()) / "outputs"

        def _fake_pipeline_stage(**kwargs: object) -> tuple[full_pipeline.PipelineArtifacts, str]:
            stage_output_dir = Path(kwargs["output_dir"])
            stage_output_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {
                        "event_id": "AAPL_2024-01-25",
                        "ticker": "AAPL",
                        "event_date": "2024-01-25",
                        "announcement_timing": "AMC",
                    }
                ]
            ).to_parquet(stage_output_dir / "events.parquet", index=False)
            pd.DataFrame(
                [
                    {
                        "event_id": "AAPL_2024-01-25",
                        "ticker": "AAPL",
                        "event_date": "2024-01-25",
                        "announcement_timing": "AMC",
                        "pre_close": 100.0,
                        "post_close": 101.0,
                    }
                ]
            ).to_parquet(stage_output_dir / "clean_events.parquet", index=False)
            pd.DataFrame(
                [
                    {
                        "category": "overview",
                        "metric": "total_events",
                        "count": 1,
                        "pct_of_events": 1.0,
                    }
                ]
            ).to_parquet(stage_output_dir / "diagnostics.parquet", index=False)

            artifacts = full_pipeline.validate_pipeline_artifacts(stage_output_dir)
            return artifacts, "unexpected success"

        with patch.object(full_pipeline, "run_event_pipeline_stage", side_effect=_fake_pipeline_stage):
            report = full_pipeline.run_full_research_pipeline(
                output_dir=output_dir,
                fixture_dir=FIXTURE_DIR,
                tickers=None,
                start_date="2025-01-01",
                end_date="2025-12-31",
                max_events=50,
                run_visuals=False,
            )

        self.assertEqual(report.status, "FAILED")
        self.assertEqual(report.exit_code, 1)
        self.assertEqual(report.failed_stage, full_pipeline.EVENT_PIPELINE_STAGE)
        self.assertIn("Missing required artifact", report.failure_detail or "")
        self.assertEqual(self._stage(report, full_pipeline.EVENT_PIPELINE_STAGE).status, "FAILED")

    def test_downstream_analysis_exception_propagates_from_callable(self) -> None:
        output_dir = Path(tempfile.mkdtemp()) / "outputs"

        with patch.object(full_pipeline, "run_full_analysis", side_effect=RuntimeError("analysis exploded")):
            report = full_pipeline.run_full_research_pipeline(
                output_dir=output_dir,
                fixture_dir=FIXTURE_DIR,
                tickers=None,
                start_date="2025-01-01",
                end_date="2025-12-31",
                max_events=50,
                run_visuals=False,
            )

        self.assertEqual(report.status, "FAILED")
        self.assertEqual(report.exit_code, 1)
        self.assertEqual(report.failed_stage, full_pipeline.ANALYSIS_STAGE)
        self.assertIn("analysis exploded", report.failure_detail or "")
        self.assertEqual(self._stage(report, full_pipeline.ANALYSIS_STAGE).status, "FAILED")

    def test_strategy_exception_propagates_from_callable(self) -> None:
        output_dir = Path(tempfile.mkdtemp()) / "outputs"

        with patch.object(full_pipeline, "run_straddle_analysis", side_effect=RuntimeError("strategy exploded")):
            report = full_pipeline.run_full_research_pipeline(
                output_dir=output_dir,
                fixture_dir=FIXTURE_DIR,
                tickers=None,
                start_date="2025-01-01",
                end_date="2025-12-31",
                max_events=50,
                run_visuals=False,
            )

        self.assertEqual(report.status, "FAILED")
        self.assertEqual(report.exit_code, 1)
        self.assertEqual(report.failed_stage, full_pipeline.STRATEGY_STAGE)
        self.assertIn("strategy exploded", report.failure_detail or "")
        self.assertEqual(self._stage(report, full_pipeline.STRATEGY_STAGE).status, "FAILED")

    def test_conditional_exception_propagates_from_callable(self) -> None:
        output_dir = Path(tempfile.mkdtemp()) / "outputs"

        with patch.object(full_pipeline, "run_conditional_analysis", side_effect=RuntimeError("conditional exploded")):
            report = full_pipeline.run_full_research_pipeline(
                output_dir=output_dir,
                fixture_dir=FIXTURE_DIR,
                tickers=None,
                start_date="2025-01-01",
                end_date="2025-12-31",
                max_events=50,
                run_visuals=False,
            )

        self.assertEqual(report.status, "FAILED")
        self.assertEqual(report.exit_code, 1)
        self.assertEqual(report.failed_stage, full_pipeline.CONDITIONALS_STAGE)
        self.assertIn("conditional exploded", report.failure_detail or "")
        self.assertEqual(self._stage(report, full_pipeline.CONDITIONALS_STAGE).status, "FAILED")

    def test_run_visuals_requires_matplotlib_when_enabled(self) -> None:
        result, _ = self._run_cli("--run-visuals")
        combined_output = result.stdout + result.stderr

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FINAL STATUS: FAILED", combined_output)
        self.assertIn(f"Failed stage: {full_pipeline.VISUALS_STAGE}", combined_output)
        self.assertIn("matplotlib", combined_output)
        self.assertNotIn("FINAL STATUS: SUCCESS", combined_output)


if __name__ == "__main__":
    unittest.main()
