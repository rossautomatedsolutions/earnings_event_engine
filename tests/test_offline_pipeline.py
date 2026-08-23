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
TF_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "trading_foundation_historical_data"
PIPELINE_SCRIPT = REPO_ROOT / "scripts" / "run_event_pipeline_test.py"
ANALYSIS_REQUIRED_COLUMNS = {
    "ticker",
    "announcement_timing",
    "hv20_expected_move_proxy",
    "realized_gap_move",
    "implied_move",
    "implied_move_available",
    "implied_move_source",
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

    def test_fixture_pipeline_uses_event_time_option_implied_moves(self) -> None:
        result, output_dir = self._run_pipeline()
        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

        features = pd.read_parquet(output_dir / "features.parquet").set_index("ticker")
        aapl = features.loc["AAPL"]
        msft = features.loc["MSFT"]

        self.assertAlmostEqual(float(aapl["implied_move"]), (4.8 + 4.7) / 104.5, places=6)
        self.assertAlmostEqual(float(msft["implied_move"]), (5.2 + 5.1) / 210.0, places=6)
        self.assertTrue(bool(aapl["implied_move_available"]))
        self.assertTrue(bool(msft["implied_move_available"]))
        self.assertEqual(str(aapl["implied_move_source"]), "event_option_straddle_mid_over_spot")
        self.assertEqual(str(msft["implied_move_source"]), "event_option_straddle_mid_over_spot")
        self.assertTrue(pd.notna(aapl["hv20_expected_move_proxy"]))
        self.assertTrue(pd.notna(msft["hv20_expected_move_proxy"]))
        self.assertGreater(abs(float(aapl["implied_move"]) - float(aapl["hv20_expected_move_proxy"])), 0.01)
        self.assertGreater(abs(float(msft["implied_move"]) - float(msft["hv20_expected_move_proxy"])), 0.01)

    def test_fixture_pipeline_can_consume_trading_foundation_historical_root(self) -> None:
        result, output_dir = self._run_pipeline("--historical-data-root", str(TF_FIXTURE_ROOT))
        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

        features = pd.read_parquet(output_dir / "features.parquet").set_index("ticker")
        aapl = features.loc["AAPL"]
        msft = features.loc["MSFT"]

        self.assertAlmostEqual(float(aapl["implied_move"]), (6.0 + 5.8) / 106.0, places=6)
        self.assertAlmostEqual(float(msft["implied_move"]), (6.2 + 6.0) / 205.0, places=6)
        self.assertEqual(pd.Timestamp(aapl["option_snapshot_quote_date"]).date().isoformat(), "2025-01-30")
        self.assertEqual(pd.Timestamp(msft["option_snapshot_quote_date"]).date().isoformat(), "2025-02-04")
        self.assertTrue(str(aapl["option_snapshot_source_path"]).endswith("AAPL_options_snapshot_2025-01-30.csv"))
        self.assertTrue(str(msft["option_snapshot_source_path"]).endswith("MSFT_options_snapshot_2025-02-04.csv"))

    def test_missing_option_data_leaves_implied_move_unavailable_without_hv_fallback(self) -> None:
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

        features = pd.read_parquet(output_dir / "features.parquet")
        self.assertTrue(features["implied_move"].isna().all())
        self.assertFalse(features["implied_move_available"].fillna(False).any())
        self.assertTrue(features["surprise_vs_implied"].isna().all())
        self.assertTrue(features["hv20_expected_move_proxy"].notna().any())

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
