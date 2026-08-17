from __future__ import annotations

import sys
import unittest
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.run_event_pipeline_test as pipeline_module


class RunEventPipelineTimingTest(unittest.TestCase):
    def _normalize_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        return pipeline_module._normalize_events_frame(
            events_df,
            start_date=pd.Timestamp("2025-01-01"),
            end_date=pd.Timestamp("2025-12-31") + pd.Timedelta(days=1),
            max_events=50,
        )

    def _price_panel(self, rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ticker": [ticker for ticker, _, _, _ in rows],
                "date": [date for _, date, _, _ in rows],
                "open": [open_price for _, _, open_price, _ in rows],
                "close": [close_price for _, _, _, close_price in rows],
            }
        )

    def test_explicit_bmo_weekday_uses_prior_pre_and_event_post_sessions(self) -> None:
        events_df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "earnings_date": ["2025-02-13 08:00:00"],
                "announcement_timing": ["BMO"],
            }
        )
        prices_df = self._price_panel(
            [
                ("AAPL", "2025-02-12", 103.0, 104.0),
                ("AAPL", "2025-02-13", 109.0, 110.0),
                ("AAPL", "2025-02-14", 107.0, 108.0),
            ]
        )

        events = self._normalize_events(events_df)
        enriched = pipeline_module.load_prices_and_compute_moves(events, price_panel_df=prices_df)

        self.assertEqual(events["announcement_timing"].tolist(), ["BMO"])
        self.assertEqual(events["mapped_session"].tolist(), ["same_day"])
        self.assertEqual(enriched["pre_close"].tolist(), [104.0])
        self.assertEqual(enriched["post_open"].tolist(), [109.0])
        self.assertEqual(enriched["post_close"].tolist(), [110.0])
        self.assertEqual(enriched["price_alignment_valid"].tolist(), [True])

    def test_explicit_amc_weekday_uses_event_pre_and_next_post_sessions(self) -> None:
        events_df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "earnings_date": ["2025-02-13 16:05:00"],
                "announcement_timing": ["AMC"],
            }
        )
        prices_df = self._price_panel(
            [
                ("AAPL", "2025-02-12", 103.0, 104.0),
                ("AAPL", "2025-02-13", 109.0, 110.0),
                ("AAPL", "2025-02-14", 107.0, 108.0),
            ]
        )

        events = self._normalize_events(events_df)
        enriched = pipeline_module.load_prices_and_compute_moves(events, price_panel_df=prices_df)

        self.assertEqual(events["announcement_timing"].tolist(), ["AMC"])
        self.assertEqual(events["mapped_session"].tolist(), ["next_day"])
        self.assertEqual(enriched["pre_close"].tolist(), [110.0])
        self.assertEqual(enriched["post_open"].tolist(), [107.0])
        self.assertEqual(enriched["post_close"].tolist(), [108.0])
        self.assertEqual(enriched["price_alignment_valid"].tolist(), [True])

    def test_missing_timing_infers_bmo_from_pre_market_timestamp(self) -> None:
        events_df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "earnings_date": ["2025-02-13 08:00:00"],
            }
        )

        events = self._normalize_events(events_df)

        self.assertEqual(events["announcement_timing"].tolist(), ["BMO"])
        self.assertEqual(events["mapped_session"].tolist(), ["same_day"])

    def test_missing_timing_infers_amc_from_after_close_timestamp(self) -> None:
        events_df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "earnings_date": ["2025-02-13 16:05:00"],
            }
        )

        events = self._normalize_events(events_df)

        self.assertEqual(events["announcement_timing"].tolist(), ["AMC"])
        self.assertEqual(events["mapped_session"].tolist(), ["next_day"])

    def test_ambiguous_regular_session_timestamp_does_not_become_amc(self) -> None:
        events_df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "earnings_date": ["2025-02-13 13:00:00"],
            }
        )
        prices_df = self._price_panel(
            [
                ("AAPL", "2025-02-12", 103.0, 104.0),
                ("AAPL", "2025-02-13", 109.0, 110.0),
                ("AAPL", "2025-02-14", 107.0, 108.0),
            ]
        )

        events = self._normalize_events(events_df)
        enriched = pipeline_module.load_prices_and_compute_moves(events, price_panel_df=prices_df)

        self.assertTrue(pd.isna(events.loc[0, "announcement_timing"]))
        self.assertTrue(pd.isna(events.loc[0, "mapped_session"]))
        self.assertTrue(pd.isna(enriched.loc[0, "pre_close"]))
        self.assertTrue(pd.isna(enriched.loc[0, "post_open"]))
        self.assertFalse(bool(enriched.loc[0, "price_alignment_valid"]))

    def test_unsupported_timing_label_does_not_become_amc(self) -> None:
        events_df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "earnings_date": ["2025-02-13 16:05:00"],
                "announcement_timing": ["DURING_MARKET"],
            }
        )

        events = self._normalize_events(events_df)

        self.assertTrue(pd.isna(events.loc[0, "announcement_timing"]))
        self.assertTrue(pd.isna(events.loc[0, "mapped_session"]))

    def test_tz_aware_bmo_timestamp_is_inferred_safely(self) -> None:
        events_df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "earnings_date": [pd.Timestamp("2025-02-13 08:00:00", tz=ZoneInfo("America/New_York"))],
            }
        )

        events = self._normalize_events(events_df)

        self.assertEqual(events["announcement_timing"].tolist(), ["BMO"])
        self.assertEqual(events["earnings_date"].tolist(), [pd.Timestamp("2025-02-13 08:00:00")])

    def test_tz_aware_amc_timestamp_is_inferred_safely(self) -> None:
        events_df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "earnings_date": [pd.Timestamp("2025-02-13 16:05:00", tz=ZoneInfo("America/New_York"))],
            }
        )

        events = self._normalize_events(events_df)

        self.assertEqual(events["announcement_timing"].tolist(), ["AMC"])
        self.assertEqual(events["earnings_date"].tolist(), [pd.Timestamp("2025-02-13 16:05:00")])

    def test_bmo_holiday_gap_uses_prior_and_next_available_sessions(self) -> None:
        events_df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "earnings_date": ["2025-02-17 08:00:00"],
                "announcement_timing": ["BMO"],
            }
        )
        prices_df = self._price_panel(
            [
                ("AAPL", "2025-02-14", 103.0, 104.0),
                ("AAPL", "2025-02-18", 110.0, 111.0),
                ("AAPL", "2025-02-19", 111.0, 112.0),
            ]
        )

        events = self._normalize_events(events_df)
        enriched = pipeline_module.load_prices_and_compute_moves(events, price_panel_df=prices_df)

        self.assertEqual(enriched["pre_close"].tolist(), [104.0])
        self.assertEqual(enriched["post_open"].tolist(), [110.0])
        self.assertEqual(enriched["post_close"].tolist(), [111.0])
        self.assertEqual(enriched["price_alignment_valid"].tolist(), [True])

    def test_missing_timestamp_drops_event_during_normalization(self) -> None:
        events_df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "earnings_date": [pd.NaT],
            }
        )

        events = self._normalize_events(events_df)

        self.assertTrue(events.empty)


if __name__ == "__main__":
    unittest.main()
