from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_analysis as run_analysis_module


class RunAnalysisIVContainmentTest(unittest.TestCase):
    def _earnings_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL"],
                "earnings_date": pd.to_datetime(["2025-01-30 16:05:00", "2025-02-13 16:05:00"]),
            }
        )

    def _history_df(self) -> pd.DataFrame:
        history = pd.DataFrame(
            {
                "close": [100.0, 101.0, 102.0, 104.0, 110.0, 108.0, 111.0, 113.0, 115.0, 120.0],
            },
            index=pd.to_datetime(
                [
                    "2025-01-27",
                    "2025-01-28",
                    "2025-01-29",
                    "2025-01-30",
                    "2025-01-31",
                    "2025-02-10",
                    "2025-02-11",
                    "2025-02-12",
                    "2025-02-13",
                    "2025-02-14",
                ]
            ),
        )
        history.index.name = "date"
        return history

    def _run_analysis(self, *, event_time_iv_df: pd.DataFrame | None = None):
        with (
            patch.object(run_analysis_module, "get_earnings_dates", return_value=self._earnings_df()),
            patch.object(run_analysis_module, "_load_history", return_value=self._history_df()),
        ):
            return run_analysis_module.run_analysis("AAPL", window_days=2, event_time_iv_df=event_time_iv_df)

    def _run_prepare_event_time_iv_inputs(self, event_time_iv_df: pd.DataFrame) -> pd.DataFrame:
        return run_analysis_module._prepare_event_time_iv_inputs("AAPL", event_time_iv_df)

    def test_historical_analysis_does_not_call_current_iv_snapshot_helper(self) -> None:
        with (
            patch.object(run_analysis_module, "get_earnings_dates", return_value=self._earnings_df()),
            patch.object(run_analysis_module, "_load_history", return_value=self._history_df()),
            patch.object(
                run_analysis_module,
                "_estimate_current_front_month_iv",
                side_effect=AssertionError("current IV helper should not be used for historical rows"),
            ),
        ):
            rows, summary = run_analysis_module.run_analysis("AAPL", window_days=2)

        self.assertEqual(len(rows), 2)
        self.assertEqual(summary.events_with_event_time_iv, 0)

    def test_historical_rows_without_event_time_iv_keep_iv_metrics_unavailable(self) -> None:
        rows, summary = self._run_analysis()

        self.assertEqual(len(rows), 2)
        self.assertEqual(summary.events_with_event_time_iv, 0)
        self.assertEqual(rows["event_time_iv_available"].tolist(), [False, False])
        self.assertTrue(rows["event_time_iv_annual_pct"].isna().all())
        self.assertTrue(rows["expected_move_pct"].isna().all())
        self.assertTrue(rows["event_time_iv_minus_realized_pct"].isna().all())
        self.assertTrue(rows["event_time_expected_move_proxy_return_pct"].isna().all())
        self.assertNotIn("event_time_straddle_return_pct", rows.columns)

    def test_non_iv_historical_fields_still_calculate_correctly(self) -> None:
        rows, summary = self._run_analysis()

        self.assertEqual(rows["pre_close"].tolist(), [104.0, 115.0])
        self.assertEqual(rows["post_close"].tolist(), [110.0, 120.0])
        self.assertAlmostEqual(rows.loc[0, "signed_move_pct"], (110.0 / 104.0 - 1.0) * 100.0, places=6)
        self.assertAlmostEqual(rows.loc[1, "signed_move_pct"], (120.0 / 115.0 - 1.0) * 100.0, places=6)
        self.assertAlmostEqual(rows.loc[0, "abs_move_pct"], abs(110.0 / 104.0 - 1.0) * 100.0, places=6)
        self.assertAlmostEqual(rows.loc[1, "abs_move_pct"], abs(120.0 / 115.0 - 1.0) * 100.0, places=6)
        self.assertTrue(pd.notna(summary.avg_abs_move_pct))
        self.assertTrue(pd.notna(summary.avg_signed_move_pct))

    def test_explicit_event_time_iv_input_only_applies_to_matching_event(self) -> None:
        event_time_iv_df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "earnings_date": ["2025-01-30", "2025-01-30"],
                "event_time_iv_annual": [0.60, 0.95],
            }
        )

        rows, summary = self._run_analysis(event_time_iv_df=event_time_iv_df)

        self.assertEqual(summary.events_with_event_time_iv, 1)
        self.assertEqual(rows["event_time_iv_available"].tolist(), [True, False])
        self.assertAlmostEqual(rows.loc[0, "event_time_iv_annual_pct"], 60.0, places=6)
        self.assertTrue(pd.isna(rows.loc[1, "event_time_iv_annual_pct"]))
        self.assertAlmostEqual(rows.loc[0, "expected_move_pct"], (0.60 / math.sqrt(252)) * 100.0, places=6)
        self.assertTrue(pd.isna(rows.loc[1, "expected_move_pct"]))

    def test_multiple_historical_events_keep_distinct_event_time_iv_values(self) -> None:
        event_time_iv_df = pd.DataFrame(
            {
                "earnings_date": ["2025-01-30", "2025-02-13"],
                "event_time_iv_annual": [0.60, 0.80],
            }
        )

        rows, summary = self._run_analysis(event_time_iv_df=event_time_iv_df)

        self.assertEqual(summary.events_with_event_time_iv, 2)
        self.assertEqual(rows["event_time_iv_available"].tolist(), [True, True])
        self.assertEqual(rows["event_time_iv_annual_pct"].round(6).tolist(), [60.0, 80.0])
        self.assertIn("event_time_expected_move_proxy_return_pct", rows.columns)
        self.assertNotIn("event_time_straddle_return_pct", rows.columns)
        self.assertNotAlmostEqual(
            rows.loc[0, "event_time_expected_move_proxy_return_pct"],
            rows.loc[1, "event_time_expected_move_proxy_return_pct"],
            places=6,
        )

    def test_duplicate_event_time_iv_inputs_fail_clearly(self) -> None:
        event_time_iv_df = pd.DataFrame(
            {
                "earnings_date": ["2025-01-30", "2025-01-30"],
                "event_time_iv_annual": [0.60, 0.62],
            }
        )

        with self.assertRaisesRegex(ValueError, "duplicate event-time IV rows"):
            self._run_analysis(event_time_iv_df=event_time_iv_df)

    def test_null_event_time_iv_is_allowed_and_remains_unavailable(self) -> None:
        event_time_iv_df = pd.DataFrame(
            {
                "earnings_date": ["2025-01-30", "2025-02-13"],
                "event_time_iv_annual": [None, 0.80],
            }
        )

        rows, summary = self._run_analysis(event_time_iv_df=event_time_iv_df)

        self.assertEqual(summary.events_with_event_time_iv, 1)
        self.assertEqual(rows["event_time_iv_available"].tolist(), [False, True])
        self.assertTrue(pd.isna(rows.loc[0, "event_time_iv_annual_pct"]))
        self.assertAlmostEqual(rows.loc[1, "event_time_iv_annual_pct"], 80.0, places=6)

    def test_valid_numeric_event_time_iv_input_is_preserved(self) -> None:
        event_time_iv_df = pd.DataFrame(
            {
                "earnings_date": ["2025-01-30"],
                "event_time_iv_annual": [0.60],
            }
        )

        prepared = self._run_prepare_event_time_iv_inputs(event_time_iv_df)

        self.assertEqual(prepared["event_time_iv_annual"].tolist(), [0.60])

    def test_malformed_string_event_time_iv_raises_clearly(self) -> None:
        event_time_iv_df = pd.DataFrame(
            {
                "earnings_date": ["2025-01-30"],
                "event_time_iv_annual": ["not-a-number"],
            }
        )

        with self.assertRaisesRegex(ValueError, "Invalid event_time_iv_annual"):
            self._run_prepare_event_time_iv_inputs(event_time_iv_df)

    def test_infinite_event_time_iv_raises_clearly(self) -> None:
        event_time_iv_df = pd.DataFrame(
            {
                "earnings_date": ["2025-01-30"],
                "event_time_iv_annual": [float("inf")],
            }
        )

        with self.assertRaisesRegex(ValueError, "must be finite"):
            self._run_prepare_event_time_iv_inputs(event_time_iv_df)

    def test_negative_event_time_iv_raises_clearly(self) -> None:
        event_time_iv_df = pd.DataFrame(
            {
                "earnings_date": ["2025-01-30"],
                "event_time_iv_annual": [-0.25],
            }
        )

        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            self._run_prepare_event_time_iv_inputs(event_time_iv_df)


if __name__ == "__main__":
    unittest.main()
