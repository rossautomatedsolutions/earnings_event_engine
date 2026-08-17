from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_analysis as run_analysis_module
from data.earnings_calendar import get_earnings_dates


class RunAnalysisSessionAlignmentTest(unittest.TestCase):
    def _history_df(self, closes_by_date: dict[str, float]) -> pd.DataFrame:
        history = pd.DataFrame(
            {"close": list(closes_by_date.values())},
            index=pd.to_datetime(list(closes_by_date.keys())),
        )
        history.index.name = "date"
        return history

    def _run_analysis(self, earnings_df: pd.DataFrame, history_df: pd.DataFrame):
        with (
            patch.object(run_analysis_module, "get_earnings_dates", return_value=earnings_df),
            patch.object(run_analysis_module, "_load_history", return_value=history_df),
        ):
            return run_analysis_module.run_analysis("AAPL", window_days=2)

    def test_bmo_weekday_uses_prior_session_as_pre_and_event_session_as_post(self) -> None:
        earnings_df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "earnings_date": pd.to_datetime(["2025-02-13 08:00:00"]),
                "announcement_timing": ["BMO"],
            }
        )
        history_df = self._history_df(
            {
                "2025-02-10": 98.0,
                "2025-02-11": 100.0,
                "2025-02-12": 104.0,
                "2025-02-13": 110.0,
                "2025-02-14": 108.0,
            }
        )

        rows, summary = self._run_analysis(earnings_df, history_df)

        self.assertEqual(summary.usable_events, 1)
        self.assertEqual(rows["pre_day"].tolist(), ["2025-02-12"])
        self.assertEqual(rows["post_day"].tolist(), ["2025-02-13"])
        self.assertEqual(rows["pre_close"].tolist(), [104.0])
        self.assertEqual(rows["post_close"].tolist(), [110.0])
        self.assertAlmostEqual(rows.loc[0, "signed_move_pct"], (110.0 / 104.0 - 1.0) * 100.0, places=6)
        self.assertAlmostEqual(rows.loc[0, "abs_move_pct"], abs(110.0 / 104.0 - 1.0) * 100.0, places=6)

    def test_amc_weekday_uses_event_session_as_pre_and_next_session_as_post(self) -> None:
        earnings_df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "earnings_date": pd.to_datetime(["2025-02-13 16:05:00"]),
                "announcement_timing": ["AMC"],
            }
        )
        history_df = self._history_df(
            {
                "2025-02-10": 98.0,
                "2025-02-11": 100.0,
                "2025-02-12": 104.0,
                "2025-02-13": 110.0,
                "2025-02-14": 108.0,
            }
        )

        rows, summary = self._run_analysis(earnings_df, history_df)

        self.assertEqual(summary.usable_events, 1)
        self.assertEqual(rows["pre_day"].tolist(), ["2025-02-13"])
        self.assertEqual(rows["post_day"].tolist(), ["2025-02-14"])
        self.assertEqual(rows["pre_close"].tolist(), [110.0])
        self.assertEqual(rows["post_close"].tolist(), [108.0])
        self.assertAlmostEqual(rows.loc[0, "signed_move_pct"], (108.0 / 110.0 - 1.0) * 100.0, places=6)
        self.assertAlmostEqual(rows.loc[0, "abs_move_pct"], abs(108.0 / 110.0 - 1.0) * 100.0, places=6)

    def test_bmo_monday_uses_friday_as_pre_and_monday_as_post(self) -> None:
        earnings_df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "earnings_date": pd.to_datetime(["2025-02-24 08:00:00"]),
                "announcement_timing": ["BMO"],
            }
        )
        history_df = self._history_df(
            {
                "2025-02-19": 97.0,
                "2025-02-20": 100.0,
                "2025-02-21": 104.0,
                "2025-02-24": 111.0,
                "2025-02-25": 112.0,
            }
        )

        rows, summary = self._run_analysis(earnings_df, history_df)

        self.assertEqual(summary.usable_events, 1)
        self.assertEqual(rows["pre_day"].tolist(), ["2025-02-21"])
        self.assertEqual(rows["post_day"].tolist(), ["2025-02-24"])
        self.assertEqual(rows["pre_close"].tolist(), [104.0])
        self.assertEqual(rows["post_close"].tolist(), [111.0])

    def test_amc_friday_uses_friday_as_pre_and_monday_as_post(self) -> None:
        earnings_df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "earnings_date": pd.to_datetime(["2025-02-21 16:05:00"]),
                "announcement_timing": ["AMC"],
            }
        )
        history_df = self._history_df(
            {
                "2025-02-18": 97.0,
                "2025-02-19": 99.0,
                "2025-02-20": 100.0,
                "2025-02-21": 104.0,
                "2025-02-24": 111.0,
            }
        )

        rows, summary = self._run_analysis(earnings_df, history_df)

        self.assertEqual(summary.usable_events, 1)
        self.assertEqual(rows["pre_day"].tolist(), ["2025-02-21"])
        self.assertEqual(rows["post_day"].tolist(), ["2025-02-24"])
        self.assertEqual(rows["pre_close"].tolist(), [104.0])
        self.assertEqual(rows["post_close"].tolist(), [111.0])

    def test_bmo_holiday_gap_uses_prior_and_next_available_sessions(self) -> None:
        earnings_df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "earnings_date": pd.to_datetime(["2025-02-17 08:00:00"]),
                "announcement_timing": ["BMO"],
            }
        )
        history_df = self._history_df(
            {
                "2025-02-12": 96.0,
                "2025-02-13": 100.0,
                "2025-02-14": 104.0,
                "2025-02-18": 111.0,
                "2025-02-19": 112.0,
            }
        )

        rows, summary = self._run_analysis(earnings_df, history_df)

        self.assertEqual(summary.usable_events, 1)
        self.assertEqual(rows["pre_day"].tolist(), ["2025-02-14"])
        self.assertEqual(rows["post_day"].tolist(), ["2025-02-18"])
        self.assertEqual(rows["pre_close"].tolist(), [104.0])
        self.assertEqual(rows["post_close"].tolist(), [111.0])
        self.assertAlmostEqual(rows.loc[0, "signed_move_pct"], (111.0 / 104.0 - 1.0) * 100.0, places=6)

    def test_infers_bmo_from_pre_market_timestamp_when_label_is_missing(self) -> None:
        earnings_df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "earnings_date": pd.to_datetime(["2025-02-13 08:00:00"]),
            }
        )
        history_df = self._history_df(
            {
                "2025-02-10": 98.0,
                "2025-02-11": 100.0,
                "2025-02-12": 104.0,
                "2025-02-13": 110.0,
                "2025-02-14": 108.0,
            }
        )

        rows, summary = self._run_analysis(earnings_df, history_df)

        self.assertEqual(summary.usable_events, 1)
        self.assertEqual(rows["pre_day"].tolist(), ["2025-02-12"])
        self.assertEqual(rows["post_day"].tolist(), ["2025-02-13"])

    def test_ambiguous_or_unsupported_timing_raises_instead_of_guessing(self) -> None:
        ambiguous_timing_df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "earnings_date": pd.to_datetime(["2025-02-13 13:00:00"]),
            }
        )
        unsupported_timing_df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "earnings_date": pd.to_datetime(["2025-02-13 13:00:00"]),
                "announcement_timing": ["DURING_MARKET"],
            }
        )
        history_df = self._history_df(
            {
                "2025-02-10": 98.0,
                "2025-02-11": 100.0,
                "2025-02-12": 104.0,
                "2025-02-13": 110.0,
                "2025-02-14": 108.0,
            }
        )

        with self.assertRaisesRegex(ValueError, "missing or ambiguous timing"):
            self._run_analysis(ambiguous_timing_df, history_df)

        with self.assertRaisesRegex(ValueError, "unsupported timing label"):
            self._run_analysis(unsupported_timing_df, history_df)

    def test_tz_aware_bmo_timestamp_keeps_local_wall_clock_for_alignment(self) -> None:
        earnings_df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "earnings_date": [
                    pd.Timestamp("2025-02-13 08:00:00", tz=ZoneInfo("America/New_York")),
                ],
            }
        )
        history_df = self._history_df(
            {
                "2025-02-10": 98.0,
                "2025-02-11": 100.0,
                "2025-02-12": 104.0,
                "2025-02-13": 110.0,
                "2025-02-14": 108.0,
            }
        )

        rows, summary = self._run_analysis(earnings_df, history_df)

        self.assertEqual(summary.usable_events, 1)
        self.assertEqual(rows["pre_day"].tolist(), ["2025-02-12"])
        self.assertEqual(rows["post_day"].tolist(), ["2025-02-13"])


class EarningsCalendarSessionClockTest(unittest.TestCase):
    def test_get_earnings_dates_preserves_wall_clock_for_tz_aware_timestamp(self) -> None:
        provider_frame = pd.DataFrame(
            {"EPS Estimate": [1.0]},
            index=pd.DatetimeIndex(
                [pd.Timestamp("2025-02-13 08:00:00", tz=ZoneInfo("America/New_York"))],
                name="Earnings Date",
            ),
        )
        fake_module = types.SimpleNamespace(
            Ticker=lambda ticker: types.SimpleNamespace(get_earnings_dates=lambda: provider_frame)
        )

        with patch.dict(sys.modules, {"yfinance": fake_module}):
            result = get_earnings_dates("AAPL")

        self.assertEqual(result.loc[0, "ticker"], "AAPL")
        self.assertEqual(result.loc[0, "earnings_date"], pd.Timestamp("2025-02-13 08:00:00"))


if __name__ == "__main__":
    unittest.main()
