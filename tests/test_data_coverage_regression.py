from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from earnings_event_engine.validation import build_data_coverage_report
from scripts.run_event_pipeline_test import _build_coverage_input_df


class DataCoverageRegressionTest(unittest.TestCase):
    def test_explicit_price_alignment_valid_drives_full_coverage_without_price_columns(self) -> None:
        events_df = pd.DataFrame(
            {
                "event_id": [1, 2, 3],
                "ticker": ["AAPL", "MSFT", "NVDA"],
                "event_date": pd.to_datetime(["2025-01-30", "2025-02-05", "2025-02-26"]),
                "valid_event": [True, True, True],
                "price_alignment_valid": [True, True, True],
                "missing_options_data": [False, False, False],
                "missing_or_unknown_timing": [False, False, False],
                "announcement_timing": ["AMC", "BMO", "AMC"],
                "mapped_session": ["next_day", "same_day", "next_day"],
            }
        )

        report = build_data_coverage_report(
            events_df,
            ticker_col="ticker",
            event_date_col="event_date",
            price_valid_col="price_alignment_valid",
            timing_col="announcement_timing",
            mapped_session_col="mapped_session",
        )

        summary_counts = report.summary.set_index("metric")["count"]
        total_events = len(events_df)

        self.assertEqual(summary_counts["total_events"], total_events)
        self.assertEqual(summary_counts["events_with_valid_price_data"], total_events)
        self.assertEqual(summary_counts["fully_valid_events"], total_events)

    def test_script_coverage_input_keeps_price_alignment_valid(self) -> None:
        diagnostics_flags = pd.DataFrame(
            {
                "event_id": [1, 2],
                "valid_event": [True, True],
                "missing_options_data": [False, False],
                "missing_or_unknown_timing": [False, False],
            }
        )
        events_with_prices = pd.DataFrame(
            {
                "event_id": [1, 2],
                "ticker": ["AAPL", "MSFT"],
                "event_date": pd.to_datetime(["2025-01-30", "2025-02-05"]),
                "announcement_timing": ["AMC", "BMO"],
                "mapped_session": ["next_day", "same_day"],
                "price_alignment_valid": [True, True],
            }
        )

        coverage_input = _build_coverage_input_df(diagnostics_flags, events_with_prices)

        self.assertIn("price_alignment_valid", coverage_input.columns)

    def test_script_coverage_input_includes_expected_contract_columns(self) -> None:
        diagnostics_flags = pd.DataFrame(
            {
                "event_id": [1],
                "valid_event": [True],
                "missing_options_data": [False],
                "missing_or_unknown_timing": [False],
            }
        )
        events_with_prices = pd.DataFrame(
            {
                "event_id": [1],
                "ticker": ["AAPL"],
                "event_date": pd.to_datetime(["2025-01-30"]),
                "announcement_timing": ["AMC"],
                "mapped_session": ["next_day"],
                "price_alignment_valid": [True],
            }
        )

        coverage_input = _build_coverage_input_df(diagnostics_flags, events_with_prices)

        expected_columns = {
            "event_id",
            "ticker",
            "event_date",
            "announcement_timing",
            "mapped_session",
            "valid_event",
            "price_alignment_valid",
            "missing_options_data",
            "missing_or_unknown_timing",
        }

        self.assertTrue(expected_columns.issubset(set(coverage_input.columns)))

    def test_script_coverage_input_fails_when_price_alignment_valid_is_omitted(self) -> None:
        diagnostics_flags = pd.DataFrame({"event_id": [1], "valid_event": [True]})
        events_with_prices = pd.DataFrame(
            {
                "event_id": [1],
                "ticker": ["AAPL"],
                "event_date": pd.to_datetime(["2025-01-30"]),
                "announcement_timing": ["AMC"],
                "mapped_session": ["next_day"],
            }
        )

        with self.assertRaisesRegex(AssertionError, "price_alignment_valid"):
            _build_coverage_input_df(diagnostics_flags, events_with_prices)


if __name__ == "__main__":
    unittest.main()
