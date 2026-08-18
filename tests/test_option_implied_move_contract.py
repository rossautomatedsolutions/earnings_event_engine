from __future__ import annotations

import sys
import unittest
from pathlib import Path

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

from analysis import earnings_analysis as earnings_analysis_module
from scripts import run_event_pipeline_test as pipeline_module
from strategies import straddle_simulator as straddle_module
from src.features.option_selection import EVENT_OPTION_IMPLIED_MOVE_SOURCE


FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "offline_pipeline"


class OptionImpliedMoveContractTest(unittest.TestCase):
    def test_fixture_option_snapshot_maps_to_expected_events(self) -> None:
        fixture_events, fixture_prices, fixture_options = pipeline_module.load_fixture_tables(FIXTURE_DIR)
        events = pipeline_module.retrieve_and_normalize_events(
            ["AAPL", "MSFT"],
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2025-12-31"),
            50,
            events_source_df=fixture_events,
        )
        events_with_prices = pipeline_module.load_prices_and_compute_moves(events, price_panel_df=fixture_prices)
        options = pipeline_module.load_fixture_options_for_events(events, fixture_options)
        enriched = pipeline_module.attach_event_time_option_implied_moves(events_with_prices, options)

        by_ticker = enriched.set_index("ticker")
        self.assertAlmostEqual(float(by_ticker.loc["AAPL", "implied_move"]), (4.8 + 4.7) / 104.5, places=6)
        self.assertAlmostEqual(float(by_ticker.loc["MSFT", "implied_move"]), (5.2 + 5.1) / 210.0, places=6)
        self.assertTrue(bool(by_ticker.loc["AAPL", "option_snapshot_available"]))
        self.assertTrue(bool(by_ticker.loc["MSFT", "option_snapshot_available"]))
        self.assertTrue(bool(by_ticker.loc["AAPL", "implied_move_available"]))
        self.assertTrue(bool(by_ticker.loc["MSFT", "implied_move_available"]))
        self.assertEqual(str(by_ticker.loc["AAPL", "implied_move_source"]), EVENT_OPTION_IMPLIED_MOVE_SOURCE)
        self.assertEqual(str(by_ticker.loc["MSFT", "implied_move_source"]), EVENT_OPTION_IMPLIED_MOVE_SOURCE)

    def test_no_cross_event_option_leakage_for_same_ticker(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "ticker": "AAPL",
                    "event_date": pd.Timestamp("2025-01-30"),
                    "earnings_date": pd.Timestamp("2025-01-30 16:05:00"),
                    "timing": "AMC",
                    "announcement_timing": "AMC",
                    "mapped_session": "next_day",
                    "pre_close": 106.0,
                    "post_open": 108.4,
                    "post_close": 107.8,
                    "hv20_expected_move_proxy": 0.003187,
                    "price_alignment_valid": True,
                },
                {
                    "event_id": 2,
                    "ticker": "AAPL",
                    "event_date": pd.Timestamp("2025-02-13"),
                    "earnings_date": pd.Timestamp("2025-02-13 16:05:00"),
                    "timing": "AMC",
                    "announcement_timing": "AMC",
                    "mapped_session": "next_day",
                    "pre_close": 110.0,
                    "post_open": 112.0,
                    "post_close": 111.5,
                    "hv20_expected_move_proxy": 0.012000,
                    "price_alignment_valid": True,
                },
            ]
        )
        options_source = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "event_date": "2025-01-30",
                    "option_type": "call",
                    "strike": 104.5,
                    "underlying_price": 104.5,
                    "bid": 4.7,
                    "ask": 4.9,
                    "mid": 4.8,
                },
                {
                    "ticker": "AAPL",
                    "event_date": "2025-01-30",
                    "option_type": "put",
                    "strike": 104.5,
                    "underlying_price": 104.5,
                    "bid": 4.6,
                    "ask": 4.8,
                    "mid": 4.7,
                },
            ]
        )

        mapped_options = pipeline_module.load_fixture_options_for_events(events, options_source)
        enriched = pipeline_module.attach_event_time_option_implied_moves(events, mapped_options)

        first_row = enriched.loc[enriched["event_id"] == 1].iloc[0]
        second_row = enriched.loc[enriched["event_id"] == 2].iloc[0]
        self.assertAlmostEqual(float(first_row["implied_move"]), (4.8 + 4.7) / 104.5, places=6)
        self.assertTrue(bool(first_row["implied_move_available"]))
        self.assertTrue(pd.isna(second_row["implied_move"]))
        self.assertFalse(bool(second_row["implied_move_available"]))

    def test_incomplete_option_snapshot_does_not_manufacture_implied_move(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "ticker": "AAPL",
                    "event_date": pd.Timestamp("2025-01-30"),
                    "earnings_date": pd.Timestamp("2025-01-30 16:05:00"),
                    "timing": "AMC",
                    "announcement_timing": "AMC",
                    "mapped_session": "next_day",
                    "pre_close": 106.0,
                    "post_open": 108.4,
                    "post_close": 107.8,
                    "hv20_expected_move_proxy": 0.003187,
                    "price_alignment_valid": True,
                }
            ]
        )
        options = pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "option_type": "call",
                    "strike": 104.5,
                    "underlying_price": 104.5,
                    "bid": 4.7,
                    "ask": 4.9,
                    "mid": 4.8,
                }
            ]
        )

        enriched = pipeline_module.attach_event_time_option_implied_moves(events, options)
        row = enriched.iloc[0]
        self.assertTrue(pd.isna(row["implied_move"]))
        self.assertFalse(bool(row["option_snapshot_available"]))
        self.assertFalse(bool(row["implied_move_available"]))
        self.assertTrue(pd.isna(row["implied_move_source"]))

    def test_unusable_two_leg_snapshot_does_not_claim_option_snapshot_availability(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "ticker": "AAPL",
                    "event_date": pd.Timestamp("2025-01-30"),
                    "earnings_date": pd.Timestamp("2025-01-30 16:05:00"),
                    "timing": "AMC",
                    "announcement_timing": "AMC",
                    "mapped_session": "next_day",
                    "pre_close": 106.0,
                    "post_open": 108.4,
                    "post_close": 107.8,
                    "hv20_expected_move_proxy": 0.003187,
                    "price_alignment_valid": True,
                }
            ]
        )
        options = pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "option_type": "call",
                    "strike": 104.5,
                    "underlying_price": 104.5,
                    "bid": pd.NA,
                    "ask": pd.NA,
                    "mid": pd.NA,
                },
                {
                    "event_id": 1,
                    "option_type": "put",
                    "strike": 104.5,
                    "underlying_price": 104.5,
                    "bid": pd.NA,
                    "ask": pd.NA,
                    "mid": pd.NA,
                },
            ]
        )

        enriched = pipeline_module.attach_event_time_option_implied_moves(events, options)
        row = enriched.iloc[0]
        self.assertFalse(bool(row["option_snapshot_available"]))
        self.assertFalse(bool(row["implied_move_available"]))
        self.assertTrue(pd.isna(row["implied_move"]))
        self.assertTrue(pd.isna(row["option_call_mid"]))
        self.assertTrue(pd.isna(row["option_put_mid"]))

    def test_missing_call_snapshot_is_unavailable(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "ticker": "AAPL",
                    "event_date": pd.Timestamp("2025-01-30"),
                    "earnings_date": pd.Timestamp("2025-01-30 16:05:00"),
                    "timing": "AMC",
                    "announcement_timing": "AMC",
                    "mapped_session": "next_day",
                    "pre_close": 106.0,
                    "post_open": 108.4,
                    "post_close": 107.8,
                    "hv20_expected_move_proxy": 0.003187,
                    "price_alignment_valid": True,
                }
            ]
        )
        options = pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "option_type": "put",
                    "strike": 104.5,
                    "underlying_price": 104.5,
                    "bid": 4.6,
                    "ask": 4.8,
                    "mid": 4.7,
                }
            ]
        )

        enriched = pipeline_module.attach_event_time_option_implied_moves(events, options)
        row = enriched.iloc[0]
        self.assertFalse(bool(row["option_snapshot_available"]))
        self.assertFalse(bool(row["implied_move_available"]))
        self.assertTrue(pd.isna(row["implied_move"]))

    def test_invalid_underlying_price_keeps_snapshot_unavailable(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "ticker": "AAPL",
                    "event_date": pd.Timestamp("2025-01-30"),
                    "earnings_date": pd.Timestamp("2025-01-30 16:05:00"),
                    "timing": "AMC",
                    "announcement_timing": "AMC",
                    "mapped_session": "next_day",
                    "pre_close": 106.0,
                    "post_open": 108.4,
                    "post_close": 107.8,
                    "hv20_expected_move_proxy": 0.003187,
                    "price_alignment_valid": True,
                }
            ]
        )
        options = pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "option_type": "call",
                    "strike": 104.5,
                    "underlying_price": 0.0,
                    "bid": 4.7,
                    "ask": 4.9,
                    "mid": 4.8,
                },
                {
                    "event_id": 1,
                    "option_type": "put",
                    "strike": 104.5,
                    "underlying_price": 0.0,
                    "bid": 4.6,
                    "ask": 4.8,
                    "mid": 4.7,
                },
            ]
        )

        enriched = pipeline_module.attach_event_time_option_implied_moves(events, options)
        row = enriched.iloc[0]
        self.assertFalse(bool(row["option_snapshot_available"]))
        self.assertFalse(bool(row["implied_move_available"]))
        self.assertTrue(pd.isna(row["implied_move"]))
        self.assertAlmostEqual(float(row["option_straddle_mid"]), 9.5, places=6)

    def test_generate_features_keeps_hv_proxy_separate_and_surprise_null_without_real_implied_move(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "ticker": "AAPL",
                    "earnings_date": pd.Timestamp("2025-01-30 16:05:00"),
                    "event_date": pd.Timestamp("2025-01-30"),
                    "timing": "AMC",
                    "announcement_timing": "AMC",
                    "mapped_session": "next_day",
                    "pre_close": 106.0,
                    "post_open": 108.4,
                    "post_close": 107.8,
                    "hv20_expected_move_proxy": 0.003187,
                    "implied_move": pd.NA,
                    "implied_move_available": False,
                    "implied_move_source": pd.NA,
                    "price_alignment_valid": True,
                }
            ]
        )

        features = pipeline_module.generate_features(events)
        row = features.iloc[0]
        self.assertAlmostEqual(float(row["hv20_expected_move_proxy"]), 0.003187, places=6)
        self.assertTrue(pd.isna(row["implied_move"]))
        self.assertTrue(pd.isna(row["surprise_vs_implied"]))

    def test_earnings_analysis_excludes_rows_without_real_implied_move(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "announcement_timing": "AMC",
                    "realized_gap_move": 0.05,
                    "implied_move": 0.10,
                    "implied_move_available": True,
                    "implied_move_source": EVENT_OPTION_IMPLIED_MOVE_SOURCE,
                    "surprise_vs_implied": -0.05,
                },
                {
                    "ticker": "AAPL",
                    "announcement_timing": "AMC",
                    "realized_gap_move": 0.20,
                    "implied_move": pd.NA,
                    "implied_move_available": False,
                    "implied_move_source": pd.NA,
                    "surprise_vs_implied": pd.NA,
                },
            ]
        )

        stats = earnings_analysis_module.compute_basic_statistics(df)
        by_ticker = earnings_analysis_module.compute_by_ticker(df)

        self.assertEqual(stats["option_implied_event_count"], 1.0)
        self.assertAlmostEqual(stats["mean_realized_move"], 0.125, places=6)
        self.assertAlmostEqual(stats["mean_implied_move"], 0.10, places=6)
        self.assertAlmostEqual(stats["pct_overpriced"], 100.0, places=6)
        self.assertEqual(int(by_ticker.loc["AAPL", "event_count"]), 2)
        self.assertEqual(int(by_ticker.loc["AAPL", "option_implied_event_count"]), 1)

    def test_straddle_simulator_uses_only_option_eligible_rows(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "ticker": "AAPL",
                    "implied_move": 0.09,
                    "implied_move_available": True,
                    "implied_move_source": EVENT_OPTION_IMPLIED_MOVE_SOURCE,
                    "hv20_expected_move_proxy": 0.01,
                    "realized_gap_move": 0.02,
                },
                {
                    "event_id": 2,
                    "ticker": "MSFT",
                    "implied_move": pd.NA,
                    "implied_move_available": False,
                    "implied_move_source": pd.NA,
                    "hv20_expected_move_proxy": 0.02,
                    "realized_gap_move": 0.03,
                },
            ]
        )

        results = straddle_module.simulate_straddle(df)
        self.assertEqual(results["event_id"].astype(int).tolist(), [1])
        self.assertAlmostEqual(float(results.iloc[0]["cost"]), 0.09, places=6)
        self.assertAlmostEqual(float(results.iloc[0]["pnl"]), 0.02 - 0.09, places=6)


if __name__ == "__main__":
    unittest.main()
