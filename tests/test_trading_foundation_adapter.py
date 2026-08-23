from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

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

from data.trading_foundation import (
    load_trading_foundation_eod_panel,
    load_trading_foundation_options_for_events,
)
from earnings_event_engine.validation import run_event_diagnostics
from scripts import run_event_pipeline_test as pipeline_module


OFFLINE_FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "offline_pipeline"
TF_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "trading_foundation_historical_data"


class TradingFoundationAdapterTest(unittest.TestCase):
    def _copy_tree(self, source: Path, destination: Path) -> None:
        for path in source.rglob("*"):
            relative = path.relative_to(source)
            target = destination / relative
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    def _build_events_with_local_prices(self, historical_root: Path) -> pd.DataFrame:
        fixture_events, _, _ = pipeline_module.load_fixture_tables(OFFLINE_FIXTURE_DIR)
        events = pipeline_module.retrieve_and_normalize_events(
            ["AAPL", "MSFT"],
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2025-12-31"),
            50,
            events_source_df=fixture_events,
        )
        price_panel = load_trading_foundation_eod_panel(historical_root, ["AAPL", "MSFT"])
        return pipeline_module.load_prices_and_compute_moves(events, price_panel_df=price_panel)

    def test_load_trading_foundation_eod_panel_parses_canonical_timestamps(self) -> None:
        panel = load_trading_foundation_eod_panel(TF_FIXTURE_ROOT, ["AAPL", "MSFT"])

        self.assertEqual(sorted(panel["ticker"].dropna().unique().tolist()), ["AAPL", "MSFT"])
        self.assertIn(pd.Timestamp("2025-01-30"), panel.loc[panel["ticker"] == "AAPL", "date"].tolist())
        self.assertIn(pd.Timestamp("2025-02-04"), panel.loc[panel["ticker"] == "MSFT", "date"].tolist())

    def test_load_trading_foundation_options_uses_pre_event_snapshot_dates_and_dedupes_rows(self) -> None:
        events_with_prices = self._build_events_with_local_prices(TF_FIXTURE_ROOT)

        load_result = load_trading_foundation_options_for_events(events_with_prices, TF_FIXTURE_ROOT)
        options_df = load_result.options_df
        report = load_result.report
        event_flags = load_result.event_diagnostics_df.set_index("event_id")
        enriched = pipeline_module.attach_event_time_option_implied_moves(events_with_prices, options_df)
        by_ticker = enriched.set_index("ticker")

        self.assertEqual(report["requested_events"], 2)
        self.assertEqual(report["events_with_selected_options"], 2)
        self.assertEqual(report["emitted_option_rows"], 4)
        self.assertEqual(report["missing_option_snapshot_directory_events"], 0)
        self.assertEqual(report["missing_snapshot_file_events"], 0)
        self.assertNotIn("BADSYMBOL", options_df["option_symbol"].astype(str).tolist())
        self.assertNotIn("INVALIDROW", options_df["option_symbol"].astype(str).tolist())
        self.assertEqual(str(event_flags.loc[1, "option_coverage_status"]), "successful_selection")
        self.assertEqual(str(event_flags.loc[2, "option_coverage_status"]), "successful_selection")

        msft_rows = options_df.loc[options_df["event_id"] == 2].copy()
        self.assertEqual(
            msft_rows["option_snapshot_quote_date"].dt.strftime("%Y-%m-%d").dropna().unique().tolist(),
            ["2025-02-04"],
        )

        self.assertAlmostEqual(float(by_ticker.loc["AAPL", "implied_move"]), (6.0 + 5.8) / 106.0, places=6)
        self.assertAlmostEqual(float(by_ticker.loc["MSFT", "implied_move"]), (6.2 + 6.0) / 205.0, places=6)
        self.assertEqual(
            pd.Timestamp(by_ticker.loc["AAPL", "option_snapshot_quote_date"]).date().isoformat(),
            "2025-01-30",
        )
        self.assertEqual(
            pd.Timestamp(by_ticker.loc["MSFT", "option_snapshot_quote_date"]).date().isoformat(),
            "2025-02-04",
        )

    def test_missing_snapshot_file_is_reported_without_crashing(self) -> None:
        tmp_root = Path(tempfile.mkdtemp())
        eod_dir = tmp_root / "AAPL" / "eod"
        eod_dir.mkdir(parents=True, exist_ok=True)
        source_eod = TF_FIXTURE_ROOT / "AAPL" / "eod" / "AAPL_1D.csv"
        (eod_dir / "AAPL_1D.csv").write_text(source_eod.read_text(encoding="utf-8"), encoding="utf-8")

        fixture_events = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "earnings_date": "2025-01-30 16:05:00",
                    "announcement_timing": "AMC",
                }
            ]
        )
        events = pipeline_module.retrieve_and_normalize_events(
            ["AAPL"],
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2025-12-31"),
            50,
            events_source_df=fixture_events,
        )
        price_panel = load_trading_foundation_eod_panel(tmp_root, ["AAPL"])
        events_with_prices = pipeline_module.load_prices_and_compute_moves(events, price_panel_df=price_panel)

        load_result = load_trading_foundation_options_for_events(events_with_prices, tmp_root)
        options_df = load_result.options_df
        report = load_result.report
        event_flags = load_result.event_diagnostics_df.set_index("event_id")

        self.assertTrue(options_df.empty)
        self.assertEqual(report["missing_option_snapshot_directory_events"], 1)
        self.assertEqual(report["events_with_selected_options"], 0)
        self.assertEqual(str(event_flags.loc[1, "option_coverage_status"]), "missing_option_snapshot_directory")
        self.assertTrue(bool(event_flags.loc[1, "missing_option_snapshot_directory"]))
        self.assertTrue(bool(event_flags.loc[1, "missing_options_data"]))

    def test_missing_required_snapshot_column_raises_clear_error(self) -> None:
        tmp_root = Path(tempfile.mkdtemp())
        eod_dir = tmp_root / "AAPL" / "eod"
        snapshot_dir = tmp_root / "AAPL" / "option_snapshot"
        eod_dir.mkdir(parents=True, exist_ok=True)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        source_eod = TF_FIXTURE_ROOT / "AAPL" / "eod" / "AAPL_1D.csv"
        (eod_dir / "AAPL_1D.csv").write_text(source_eod.read_text(encoding="utf-8"), encoding="utf-8")
        (snapshot_dir / "AAPL_options_snapshot_2025-01-30.csv").write_text(
            "\n".join(
                [
                    "option_symbol,ask_price,last_trade_price",
                    "AAPL250131C00106000,6.10,6.00",
                    "AAPL250131P00106000,5.90,5.80",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        fixture_events = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "earnings_date": "2025-01-30 16:05:00",
                    "announcement_timing": "AMC",
                }
            ]
        )
        events = pipeline_module.retrieve_and_normalize_events(
            ["AAPL"],
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2025-12-31"),
            50,
            events_source_df=fixture_events,
        )
        price_panel = load_trading_foundation_eod_panel(tmp_root, ["AAPL"])
        events_with_prices = pipeline_module.load_prices_and_compute_moves(events, price_panel_df=price_panel)

        load_result = load_trading_foundation_options_for_events(events_with_prices, tmp_root)
        event_flags = load_result.event_diagnostics_df.set_index("event_id")

        self.assertTrue(load_result.options_df.empty)
        self.assertEqual(load_result.report["snapshot_unparseable_events"], 1)
        self.assertEqual(str(event_flags.loc[1, "option_coverage_status"]), "snapshot_unparseable")
        self.assertTrue(bool(event_flags.loc[1, "snapshot_unparseable"]))

    def test_zero_option_snapshots_for_all_events_emits_stable_event_diagnostics(self) -> None:
        tmp_root = Path(tempfile.mkdtemp())
        self._copy_tree(TF_FIXTURE_ROOT / "AAPL" / "eod", tmp_root / "AAPL" / "eod")
        self._copy_tree(TF_FIXTURE_ROOT / "MSFT" / "eod", tmp_root / "MSFT" / "eod")

        events_with_prices = self._build_events_with_local_prices(tmp_root)
        load_result = load_trading_foundation_options_for_events(events_with_prices, tmp_root)
        flags = load_result.event_diagnostics_df

        self.assertTrue(load_result.options_df.empty)
        self.assertEqual(load_result.report["missing_option_snapshot_directory_events"], 2)
        self.assertEqual(
            sorted(flags["option_coverage_status"].astype(str).unique().tolist()),
            ["missing_option_snapshot_directory"],
        )
        self.assertTrue({"has_call_leg", "has_put_leg"}.issubset(flags.columns))
        self.assertFalse(flags["has_call_leg"].fillna(False).any())
        self.assertFalse(flags["has_put_leg"].fillna(False).any())

    def test_mixed_events_some_have_options_and_some_do_not(self) -> None:
        tmp_root = Path(tempfile.mkdtemp())
        self._copy_tree(TF_FIXTURE_ROOT / "AAPL", tmp_root / "AAPL")
        self._copy_tree(TF_FIXTURE_ROOT / "MSFT" / "eod", tmp_root / "MSFT" / "eod")

        events_with_prices = self._build_events_with_local_prices(tmp_root)
        load_result = load_trading_foundation_options_for_events(events_with_prices, tmp_root)
        flags = load_result.event_diagnostics_df.set_index("event_id")

        self.assertEqual(load_result.report["events_with_selected_options"], 1)
        self.assertEqual(load_result.report["missing_option_snapshot_directory_events"], 1)
        self.assertEqual(len(load_result.options_df), 2)
        self.assertEqual(str(flags.loc[1, "option_coverage_status"]), "successful_selection")
        self.assertEqual(str(flags.loc[2, "option_coverage_status"]), "missing_option_snapshot_directory")

    def test_run_event_diagnostics_accepts_zero_selected_option_rows_with_explicit_event_flags(self) -> None:
        tmp_root = Path(tempfile.mkdtemp())
        self._copy_tree(TF_FIXTURE_ROOT / "AAPL" / "eod", tmp_root / "AAPL" / "eod")
        self._copy_tree(TF_FIXTURE_ROOT / "MSFT" / "eod", tmp_root / "MSFT" / "eod")

        events_with_prices = self._build_events_with_local_prices(tmp_root)
        load_result = load_trading_foundation_options_for_events(events_with_prices, tmp_root)

        clean_events, diagnostics_summary, diagnostics_flags = run_event_diagnostics(
            events_with_prices,
            options_df=load_result.options_df,
            options_event_flags_df=load_result.event_diagnostics_df,
            event_key_col="event_id",
            pre_close_col="pre_close",
            post_open_col="post_open",
            post_close_col="post_close",
            implied_move_col="implied_move",
            timing_col="announcement_timing",
            mapped_session_col="mapped_session",
        )

        self.assertEqual(len(clean_events), 0)
        self.assertEqual(sorted(diagnostics_flags["event_id"].astype(int).tolist()), [1, 2])
        self.assertTrue(diagnostics_flags["missing_option_snapshot_directory"].fillna(False).all())
        self.assertIn("missing_option_snapshot_directory", diagnostics_summary["metric"].astype(str).tolist())


if __name__ == "__main__":
    unittest.main()
