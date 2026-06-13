from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from earnings_event_engine.research import (
    add_forward_close_columns,
    assign_reaction_bucket,
    calculate_announcement_reaction,
    calculate_forward_drift,
    summarize_pead_by_bucket,
)
from scripts.run_pead_research import _read_table


class PeadResearchTest(unittest.TestCase):
    def _base_events_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "event_id": [1, 2, 3],
                "ticker": ["AAPL", "MSFT", "NVDA"],
                "event_date": pd.to_datetime(["2025-01-30", "2025-02-05", "2025-02-26"]),
                "announcement_timing": ["AMC", "BMO", "BMO"],
                "mapped_session": ["next_day", "same_day", "same_day"],
                "valid_event": [True, True, True],
                "price_alignment_valid": [True, True, True],
                "pre_close": [100.0, 100.0, 100.0],
                "post_close": [110.0, 95.0, 100.0],
                "forward_close_1d": [112.0, 93.0, 101.0],
                "forward_close_3d": [115.0, 90.0, 99.0],
            }
        )

    def _prices_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ticker": [
                    "AAPL",
                    "AAPL",
                    "AAPL",
                    "AAPL",
                    "MSFT",
                    "MSFT",
                    "MSFT",
                    "MSFT",
                    "NVDA",
                    "NVDA",
                ],
                "date": pd.to_datetime(
                    [
                        "2025-01-30",
                        "2025-01-31",
                        "2025-02-03",
                        "2025-02-04",
                        "2025-02-05",
                        "2025-02-06",
                        "2025-02-07",
                        "2025-02-10",
                        "2025-02-26",
                        "2025-02-27",
                    ]
                ),
                "close": [108.0, 110.0, 112.0, 115.0, 95.0, 93.0, 90.0, 89.0, 100.0, 101.0],
            }
        )

    def test_add_forward_close_columns_uses_local_price_panel(self) -> None:
        events_df = self._base_events_df().drop(columns=["forward_close_1d", "forward_close_3d"])

        enriched = add_forward_close_columns(events_df, self._prices_df(), windows=(1, 3))

        self.assertEqual(enriched["forward_close_1d"].tolist(), [112.0, 93.0, 101.0])
        self.assertTrue(pd.isna(enriched.loc[0, "forward_close_3d"]))
        self.assertEqual(enriched.loc[1, "forward_close_3d"], 89.0)
        self.assertTrue(pd.isna(enriched.loc[2, "forward_close_3d"]))

    def test_add_forward_close_columns_keeps_tickers_isolated(self) -> None:
        events_df = self._base_events_df().drop(columns=["forward_close_1d", "forward_close_3d"])
        prices_df = self._prices_df().copy()
        prices_df.loc[prices_df["ticker"] == "AAPL", "close"] = [108.0, 210.0, 212.0, 215.0]

        enriched = add_forward_close_columns(events_df, prices_df, windows=(1,))

        self.assertEqual(enriched["forward_close_1d"].tolist(), [212.0, 93.0, 101.0])

    def test_add_forward_close_columns_leaves_missing_forward_rows_null(self) -> None:
        events_df = self._base_events_df().drop(columns=["forward_close_1d", "forward_close_3d"])

        enriched = add_forward_close_columns(events_df, self._prices_df(), windows=(3,))

        self.assertTrue(pd.isna(enriched.loc[2, "forward_close_3d"]))

    def test_derived_forward_close_columns_flow_through_pead_summary(self) -> None:
        events_df = self._base_events_df().drop(columns=["forward_close_1d", "forward_close_3d"])
        with_forward_closes = add_forward_close_columns(events_df, self._prices_df(), windows=(1, 3))
        with_reaction = calculate_announcement_reaction(with_forward_closes)
        with_drift = calculate_forward_drift(with_reaction, windows=(1, 3))
        bucketed = assign_reaction_bucket(with_drift)

        summary = summarize_pead_by_bucket(bucketed, windows=(1, 3))

        self.assertFalse(summary.empty)
        self.assertIn("positive_reaction", summary["reaction_bucket"].tolist())

    def test_calculate_announcement_reaction_uses_post_close_baseline(self) -> None:
        result = calculate_announcement_reaction(self._base_events_df())

        observed = result["reaction_return"].round(4).tolist()
        self.assertEqual(observed, [0.10, -0.05, 0.00])
        self.assertTrue((result["announcement_close_col"] == "post_close").all())

    def test_calculate_forward_drift_uses_post_close_as_denominator(self) -> None:
        with_reaction = calculate_announcement_reaction(self._base_events_df())
        result = calculate_forward_drift(with_reaction, windows=(1, 3))

        observed_1d = result["forward_return_1d"].round(4).tolist()
        observed_3d = result["forward_return_3d"].round(4).tolist()

        self.assertEqual(observed_1d, [0.0182, -0.0211, 0.01])
        self.assertEqual(observed_3d, [0.0455, -0.0526, -0.01])

    def test_assign_reaction_bucket_respects_threshold_signs(self) -> None:
        with_reaction = calculate_announcement_reaction(self._base_events_df())

        result = assign_reaction_bucket(with_reaction, threshold=0.0)

        self.assertEqual(
            result["reaction_bucket"].tolist(),
            ["positive_reaction", "negative_reaction", "neutral_reaction"],
        )

    def test_summarize_pead_by_bucket_returns_bucket_level_metrics(self) -> None:
        events_df = pd.DataFrame(
            {
                "reaction_bucket": [
                    "positive_reaction",
                    "positive_reaction",
                    "negative_reaction",
                    "negative_reaction",
                ],
                "reaction_return": [0.10, 0.05, -0.08, -0.03],
                "forward_return_1d": [0.02, -0.01, -0.03, 0.01],
                "forward_return_3d": [0.03, 0.00, -0.02, -0.01],
            }
        )

        summary = summarize_pead_by_bucket(events_df, windows=(1, 3))

        positive_1d = summary[
            (summary["reaction_bucket"] == "positive_reaction") & (summary["window_days"] == 1)
        ].iloc[0]
        negative_3d = summary[
            (summary["reaction_bucket"] == "negative_reaction") & (summary["window_days"] == 3)
        ].iloc[0]

        self.assertEqual(int(positive_1d["event_count"]), 2)
        self.assertAlmostEqual(float(positive_1d["mean_forward_return"]), 0.005)
        self.assertAlmostEqual(float(positive_1d["median_forward_return"]), 0.005)
        self.assertAlmostEqual(float(positive_1d["positive_rate"]), 0.5)

        self.assertEqual(int(negative_3d["event_count"]), 2)
        self.assertAlmostEqual(float(negative_3d["mean_forward_return"]), -0.015)
        self.assertAlmostEqual(float(negative_3d["median_forward_return"]), -0.015)
        self.assertAlmostEqual(float(negative_3d["positive_rate"]), 0.0)

    def test_missing_required_forward_columns_fail_clearly(self) -> None:
        incomplete = self._base_events_df().drop(columns=["forward_close_3d"])
        with_reaction = calculate_announcement_reaction(incomplete)

        with self.assertRaisesRegex(ValueError, "forward_close_3d"):
            calculate_forward_drift(with_reaction, windows=(1, 3))

    def test_read_table_loads_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "events.csv"
            expected = pd.DataFrame(
                {
                    "event_id": [1],
                    "ticker": ["AAPL"],
                    "pre_close": [100.0],
                }
            )
            expected.to_csv(csv_path, index=False)

            observed = _read_table(csv_path)

            pd.testing.assert_frame_equal(observed, expected)

    def test_read_table_loads_parquet_when_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            parquet_path = Path(tmp_dir) / "events.parquet"
            expected = pd.DataFrame(
                {
                    "event_id": [1],
                    "ticker": ["AAPL"],
                    "pre_close": [100.0],
                }
            )

            try:
                expected.to_parquet(parquet_path, index=False)
            except (ImportError, ModuleNotFoundError, ValueError, OSError) as exc:
                self.skipTest(f"Parquet support unavailable in current environment: {exc}")

            observed = _read_table(parquet_path)

            pd.testing.assert_frame_equal(observed, expected)

    def test_read_table_rejects_unsupported_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            unsupported_path = Path(tmp_dir) / "events.txt"
            unsupported_path.write_text("event_id,ticker\n1,AAPL\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unsupported input format|\\.parquet or \\.csv"):
                _read_table(unsupported_path)

    def test_read_table_treats_supported_suffixes_case_insensitively(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            expected = pd.DataFrame(
                {
                    "event_id": [1],
                    "ticker": ["AAPL"],
                    "pre_close": [100.0],
                }
            )

            csv_path = tmp_path / "events.CSV"
            expected.to_csv(csv_path, index=False)
            pd.testing.assert_frame_equal(_read_table(csv_path), expected)

            parquet_path = tmp_path / "events.PARQUET"
            try:
                expected.to_parquet(parquet_path, index=False)
            except (ImportError, ModuleNotFoundError, ValueError, OSError):
                return

            pd.testing.assert_frame_equal(_read_table(parquet_path), expected)

    def test_read_table_rejects_path_without_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            no_suffix_path = Path(tmp_dir) / "events"
            no_suffix_path.write_text("event_id,ticker\n1,AAPL\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unsupported input format|\\.parquet or \\.csv"):
                _read_table(no_suffix_path)

    def test_read_table_rejects_dotted_unsupported_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dotted_unsupported_path = Path(tmp_dir) / "events.data"
            dotted_unsupported_path.write_text("event_id,ticker\n1,AAPL\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unsupported input format|\\.parquet or \\.csv"):
                _read_table(dotted_unsupported_path)

    def test_read_table_rejects_trailing_dot_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            trailing_dot_path = Path(tmp_dir) / "events."
            trailing_dot_path.write_text("event_id,ticker\n1,AAPL\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unsupported input format|\\.parquet or \\.csv"):
                _read_table(trailing_dot_path)

    def test_read_table_rejects_unsupported_final_suffix_after_supported_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            double_extension_path = Path(tmp_dir) / "events.csv.bak"
            double_extension_path.write_text("event_id,ticker\n1,AAPL\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unsupported input format|\\.parquet or \\.csv"):
                _read_table(double_extension_path)

    def test_read_table_loads_multi_dot_csv_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            multi_dot_csv_path = Path(tmp_dir) / "events.backup.csv"
            expected = pd.DataFrame(
                {
                    "event_id": [1],
                    "ticker": ["AAPL"],
                    "pre_close": [100.0],
                }
            )
            expected.to_csv(multi_dot_csv_path, index=False)

            observed = _read_table(multi_dot_csv_path)

            pd.testing.assert_frame_equal(observed, expected)

    def test_read_table_loads_multi_dot_uppercase_csv_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            multi_dot_uppercase_csv_path = Path(tmp_dir) / "events.backup.CSV"
            expected = pd.DataFrame(
                {
                    "event_id": [1],
                    "ticker": ["AAPL"],
                    "pre_close": [100.0],
                }
            )
            expected.to_csv(multi_dot_uppercase_csv_path, index=False)

            observed = _read_table(multi_dot_uppercase_csv_path)

            pd.testing.assert_frame_equal(observed, expected)

    def test_read_table_loads_hidden_dot_csv_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            hidden_dot_csv_path = Path(tmp_dir) / ".events.csv"
            expected = pd.DataFrame(
                {
                    "event_id": [1],
                    "ticker": ["AAPL"],
                    "pre_close": [100.0],
                }
            )
            expected.to_csv(hidden_dot_csv_path, index=False)

            observed = _read_table(hidden_dot_csv_path)

            pd.testing.assert_frame_equal(observed, expected)

    def test_read_table_loads_hidden_dot_uppercase_csv_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            hidden_dot_uppercase_csv_path = Path(tmp_dir) / ".events.CSV"
            expected = pd.DataFrame(
                {
                    "event_id": [1],
                    "ticker": ["AAPL"],
                    "pre_close": [100.0],
                }
            )
            expected.to_csv(hidden_dot_uppercase_csv_path, index=False)

            observed = _read_table(hidden_dot_uppercase_csv_path)

            pd.testing.assert_frame_equal(observed, expected)

    def test_run_pead_research_cli_derives_forward_close_from_local_prices_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            events_path = tmp_path / "events.csv"
            prices_path = tmp_path / "prices.csv"
            output_dir = tmp_path / "outputs"

            pd.DataFrame(
                {
                    "event_id": [1],
                    "ticker": ["AAPL"],
                    "event_date": ["2025-01-30"],
                    "pre_close": [100.0],
                    "post_close": [110.0],
                    "mapped_session": ["next_day"],
                    "valid_event": [True],
                    "price_alignment_valid": [True],
                }
            ).to_csv(events_path, index=False)

            pd.DataFrame(
                {
                    "ticker": ["AAPL", "AAPL", "AAPL"],
                    "date": ["2025-01-30", "2025-01-31", "2025-02-03"],
                    "close": [108.0, 110.0, 112.0],
                }
            ).to_csv(prices_path, index=False)

            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "run_pead_research.py"),
                    "--clean-events-path",
                    str(events_path),
                    "--prices-path",
                    str(prices_path),
                    "--output-dir",
                    str(output_dir),
                    "--windows",
                    "1",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

            event_returns_path = output_dir / "pead_event_returns.csv"
            bucket_summary_path = output_dir / "pead_bucket_summary.csv"
            summary_md_path = output_dir / "pead_summary.md"

            self.assertTrue(event_returns_path.exists())
            self.assertTrue(bucket_summary_path.exists())
            self.assertTrue(summary_md_path.exists())

            event_returns = pd.read_csv(event_returns_path)
            self.assertIn("forward_close_1d", event_returns.columns)
            self.assertIn("forward_return_1d", event_returns.columns)
            self.assertFalse(pd.isna(event_returns.loc[0, "forward_close_1d"]))
            self.assertAlmostEqual(float(event_returns.loc[0, "forward_close_1d"]), 112.0)
            self.assertAlmostEqual(float(event_returns.loc[0, "forward_return_1d"]), 112.0 / 110.0 - 1.0)

    def test_run_pead_research_cli_fails_clearly_when_prices_csv_omits_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            events_path = tmp_path / "events.csv"
            prices_path = tmp_path / "prices.csv"
            output_dir = tmp_path / "outputs"

            pd.DataFrame(
                {
                    "event_id": [1],
                    "ticker": ["AAPL"],
                    "event_date": ["2025-01-30"],
                    "pre_close": [100.0],
                    "post_close": [110.0],
                    "mapped_session": ["next_day"],
                    "valid_event": [True],
                    "price_alignment_valid": [True],
                }
            ).to_csv(events_path, index=False)

            pd.DataFrame(
                {
                    "ticker": ["AAPL", "AAPL"],
                    "date": ["2025-01-30", "2025-01-31"],
                }
            ).to_csv(prices_path, index=False)

            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "run_pead_research.py"),
                    "--clean-events-path",
                    str(events_path),
                    "--prices-path",
                    str(prices_path),
                    "--output-dir",
                    str(output_dir),
                    "--windows",
                    "1",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)

            error_text = f"{result.stdout}\n{result.stderr}"
            self.assertIn("missing required columns", error_text)
            self.assertIn("close", error_text)

    def test_run_pead_research_cli_fails_clearly_when_events_csv_omits_baseline_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            events_path = tmp_path / "events.csv"
            output_dir = tmp_path / "outputs"

            pd.DataFrame(
                {
                    "event_id": [1],
                    "ticker": ["AAPL"],
                    "event_date": ["2025-01-30"],
                    "pre_close": [100.0],
                    "mapped_session": ["next_day"],
                    "valid_event": [True],
                    "price_alignment_valid": [True],
                    "forward_close_1d": [112.0],
                }
            ).to_csv(events_path, index=False)

            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "run_pead_research.py"),
                    "--clean-events-path",
                    str(events_path),
                    "--output-dir",
                    str(output_dir),
                    "--windows",
                    "1",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)

            error_text = f"{result.stdout}\n{result.stderr}"
            self.assertIn("post-event close column", error_text)
            self.assertIn("post_close", error_text)
            self.assertIn("announcement_close", error_text)

    def test_run_pead_research_cli_fails_clearly_when_post_close_override_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            events_path = tmp_path / "events.csv"
            output_dir = tmp_path / "outputs"

            pd.DataFrame(
                {
                    "event_id": [1],
                    "ticker": ["AAPL"],
                    "event_date": ["2025-01-30"],
                    "pre_close": [100.0],
                    "post_close": [110.0],
                    "mapped_session": ["next_day"],
                    "valid_event": [True],
                    "price_alignment_valid": [True],
                    "forward_close_1d": [112.0],
                }
            ).to_csv(events_path, index=False)

            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "run_pead_research.py"),
                    "--clean-events-path",
                    str(events_path),
                    "--output-dir",
                    str(output_dir),
                    "--windows",
                    "1",
                    "--post-close-col",
                    "nonexistent_post_close",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)

            error_text = f"{result.stdout}\n{result.stderr}"
            self.assertIn("missing required columns", error_text)
            self.assertIn("nonexistent_post_close", error_text)

    def test_run_pead_research_cli_fails_clearly_when_clean_events_suffix_is_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            events_path = tmp_path / "events.txt"
            output_dir = tmp_path / "outputs"

            events_path.write_text(
                "event_id,ticker,event_date,pre_close,post_close,forward_close_1d\n"
                "1,AAPL,2025-01-30,100.0,110.0,112.0\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "run_pead_research.py"),
                    "--clean-events-path",
                    str(events_path),
                    "--output-dir",
                    str(output_dir),
                    "--windows",
                    "1",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)

            error_text = f"{result.stdout}\n{result.stderr}"
            self.assertIn("Unsupported input format", error_text)
            self.assertIn(".parquet or .csv", error_text)

    def test_run_pead_research_cli_fails_clearly_when_prices_suffix_is_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            events_path = tmp_path / "events.csv"
            prices_path = tmp_path / "prices.txt"
            output_dir = tmp_path / "outputs"

            pd.DataFrame(
                {
                    "event_id": [1],
                    "ticker": ["AAPL"],
                    "event_date": ["2025-01-30"],
                    "pre_close": [100.0],
                    "post_close": [110.0],
                    "mapped_session": ["next_day"],
                    "valid_event": [True],
                    "price_alignment_valid": [True],
                }
            ).to_csv(events_path, index=False)

            prices_path.write_text(
                "ticker,date,close\n"
                "AAPL,2025-01-30,108.0\n"
                "AAPL,2025-01-31,110.0\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "run_pead_research.py"),
                    "--clean-events-path",
                    str(events_path),
                    "--prices-path",
                    str(prices_path),
                    "--output-dir",
                    str(output_dir),
                    "--windows",
                    "1",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)

            error_text = f"{result.stdout}\n{result.stderr}"
            self.assertIn("Unsupported input format", error_text)
            self.assertIn(".parquet or .csv", error_text)


if __name__ == "__main__":
    unittest.main()
