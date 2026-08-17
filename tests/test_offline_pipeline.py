from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import types
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

from data.earnings_calendar import EarningsCalendarRetrievalError, get_earnings_dates
from analysis.earnings_analysis import run_full_analysis


FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "offline_pipeline"
PIPELINE_SCRIPT = REPO_ROOT / "scripts" / "run_event_pipeline_test.py"
ANALYSIS_REQUIRED_COLUMNS = {
    "ticker",
    "announcement_timing",
    "realized_gap_move",
    "implied_move",
    "surprise_vs_implied",
}


class OfflinePipelineTest(unittest.TestCase):
    def _run_pipeline(self, *extra_args: str) -> tuple[subprocess.CompletedProcess[str], Path]:
        tmp_dir = Path(tempfile.mkdtemp())
        output_dir = tmp_dir / "outputs"
        cmd = [
            sys.executable,
            str(PIPELINE_SCRIPT),
            "--fixture-dir",
            str(FIXTURE_DIR),
            "--output-dir",
            str(output_dir),
        ]
        cmd.extend(extra_args)
        result = subprocess.run(
            cmd,
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

    def test_fixture_pipeline_cli_produces_non_zero_raw_clean_and_feature_rows(self) -> None:
        result, output_dir = self._run_pipeline()

        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

        raw_events = pd.read_parquet(output_dir / "events.parquet")
        clean_events = pd.read_parquet(output_dir / "clean_events.parquet")
        features = pd.read_parquet(output_dir / "features.parquet")

        self.assertEqual(len(raw_events), 2)
        self.assertEqual(len(clean_events), 2)
        self.assertEqual(len(features), 2)
        self.assertNotIn("pre_close", raw_events.columns)
        self.assertIn("pre_close", clean_events.columns)
        self.assertTrue(ANALYSIS_REQUIRED_COLUMNS.issubset(set(features.columns)))

    def test_run_full_analysis_succeeds_on_fixture_generated_feature_artifact(self) -> None:
        result, output_dir = self._run_pipeline()
        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

        analysis = run_full_analysis(str(output_dir / "features.parquet"))

        self.assertIn("basic_statistics", analysis)
        self.assertFalse(analysis["by_ticker"].empty)
        self.assertFalse(analysis["by_timing"].empty)

    def test_zero_raw_events_fail_by_default(self) -> None:
        fixture_dir = self._make_fixture_dir(
            events_csv="ticker,earnings_date,announcement_timing\n",
            options_csv=(FIXTURE_DIR / "options.csv").read_text(encoding="utf-8"),
        )
        output_dir = fixture_dir.parent / "outputs"

        result = subprocess.run(
            [
                sys.executable,
                str(PIPELINE_SCRIPT),
                "--fixture-dir",
                str(fixture_dir),
                "--output-dir",
                str(output_dir),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Zero raw events produced during the retrieval stage", result.stderr)

    def test_zero_valid_events_fail_by_default(self) -> None:
        fixture_dir = self._make_fixture_dir(
            events_csv=(FIXTURE_DIR / "earnings_events.csv").read_text(encoding="utf-8"),
            options_csv=None,
        )
        output_dir = fixture_dir.parent / "outputs"

        result = subprocess.run(
            [
                sys.executable,
                str(PIPELINE_SCRIPT),
                "--fixture-dir",
                str(fixture_dir),
                "--output-dir",
                str(output_dir),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Zero valid/clean events produced during the validation stage", result.stderr)

    def test_allow_empty_preserves_artifact_schema_and_returns_zero_exit(self) -> None:
        fixture_dir = self._make_fixture_dir(
            events_csv="ticker,earnings_date,announcement_timing\n",
            options_csv=(FIXTURE_DIR / "options.csv").read_text(encoding="utf-8"),
        )
        output_dir = fixture_dir.parent / "outputs"

        result = subprocess.run(
            [
                sys.executable,
                str(PIPELINE_SCRIPT),
                "--fixture-dir",
                str(fixture_dir),
                "--output-dir",
                str(output_dir),
                "--allow-empty",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

        diagnostics = pd.read_parquet(output_dir / "diagnostics.parquet")
        features = pd.read_parquet(output_dir / "features.parquet")

        self.assertIn("zero_raw_events", diagnostics["metric"].tolist())
        self.assertTrue(ANALYSIS_REQUIRED_COLUMNS.issubset(set(features.columns)))
        self.assertEqual(len(features), 0)

    def test_retrieval_failure_is_distinguishable_from_legitimate_empty(self) -> None:
        empty_module = types.SimpleNamespace(
            Ticker=lambda ticker: types.SimpleNamespace(get_earnings_dates=lambda: pd.DataFrame())
        )
        with patch.dict(sys.modules, {"yfinance": empty_module}):
            empty_result = get_earnings_dates("AAPL")

        self.assertTrue(empty_result.empty)

        def _raise_failure() -> pd.DataFrame:
            raise RuntimeError("network unavailable")

        failing_module = types.SimpleNamespace(
            Ticker=lambda ticker: types.SimpleNamespace(get_earnings_dates=_raise_failure)
        )
        with patch.dict(sys.modules, {"yfinance": failing_module}):
            with self.assertRaises(EarningsCalendarRetrievalError):
                get_earnings_dates("AAPL")


if __name__ == "__main__":
    unittest.main()
