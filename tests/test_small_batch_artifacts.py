from __future__ import annotations

import importlib
import sys
import unittest
import warnings
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class SmallBatchArtifactContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.small_batch = importlib.import_module("scripts.run_small_batch_test")

    def test_build_output_artifacts_keeps_raw_and_price_enriched_events_separate(self) -> None:
        raw_events = pd.DataFrame(
            {
                "event_id": ["AAPL_2025-01-30"],
                "ticker": ["AAPL"],
                "earnings_date": pd.to_datetime(["2025-01-30 16:05:00"]),
                "event_date": pd.to_datetime(["2025-01-30"]),
                "timing": ["AMC"],
                "announcement_timing": ["AMC"],
                "mapped_session": ["next_day"],
            }
        )
        events_with_prices = raw_events.assign(
            pre_close=[106.0],
            post_open=[108.4],
            post_close=[107.8],
            implied_move=[0.01],
            price_alignment_valid=[True],
        )
        options_df = pd.DataFrame(
            {
                "event_id": ["AAPL_2025-01-30", "AAPL_2025-01-30"],
                "option_type": ["call", "put"],
                "strike": [104.5, 104.5],
                "underlying_price": [104.5, 104.5],
                "bid": [4.5, 4.6],
                "ask": [4.9, 4.8],
                "mid": [4.7, 4.7],
            }
        )
        final_features = pd.DataFrame(
            {
                "event_id": ["AAPL_2025-01-30"],
                "ticker": ["AAPL"],
                "event_date": pd.to_datetime(["2025-01-30"]),
                "realized_gap_move": [0.0226],
                "surprise_vs_implied": [0.0126],
                "valid_event": [True],
                "invalid_event": [False],
                "failure_reasons": [None],
            }
        )
        clean_events = events_with_prices.copy()
        diagnostics_output = pd.DataFrame(
            {
                "category": ["overview"],
                "metric": ["total_events"],
                "count": [1],
                "pct_of_events": [1.0],
            }
        )
        diagnostics_flags = pd.DataFrame(
            {
                "event_id": ["AAPL_2025-01-30"],
                "invalid_event": [False],
                "valid_event": [True],
                "failure_reasons": [None],
            }
        )
        coverage_output = diagnostics_output.copy()
        coverage_by_ticker = pd.DataFrame({"ticker": ["AAPL"], "num_events": [1]})
        coverage_by_year = pd.DataFrame({"event_year": [2025], "events_per_year": [1]})

        artifacts = self.small_batch.build_output_artifacts(
            events=raw_events,
            events_with_prices=events_with_prices,
            options_df=options_df,
            final_features=final_features,
            clean_events=clean_events,
            diagnostics_output=diagnostics_output,
            diagnostics_flags=diagnostics_flags,
            coverage_output=coverage_output,
            coverage_by_ticker=coverage_by_ticker,
            coverage_by_year=coverage_by_year,
        )

        self.assertEqual(
            list(artifacts),
            [
                "events.parquet",
                "events_with_prices.parquet",
                "options.parquet",
                "features.parquet",
                "clean_events.parquet",
                "diagnostics.parquet",
                "diagnostics_flags.parquet",
                "coverage_summary.parquet",
                "coverage_by_ticker.parquet",
                "coverage_by_year.parquet",
            ],
        )
        pd.testing.assert_frame_equal(artifacts["events.parquet"], raw_events)
        pd.testing.assert_frame_equal(artifacts["events_with_prices.parquet"], events_with_prices)
        self.assertNotIn("pre_close", artifacts["events.parquet"].columns)
        self.assertIn("pre_close", artifacts["events_with_prices.parquet"].columns)


if __name__ == "__main__":
    unittest.main()
