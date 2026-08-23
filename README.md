# earnings_event_engine

Lightweight tooling for running and validating the earnings event pipeline, including a targeted offline regression guard for coverage-input consistency.

## Entry Points

- `scripts/run_small_batch_test.py`
- `scripts/run_event_pipeline_test.py`
- `scripts/run_full_research_pipeline.py`

## Full Research Orchestrator

`scripts/run_full_research_pipeline.py` is the honest orchestration layer for the canonical multi-stage research run.

Canonical offline demo notebook:

- `notebooks/01_earnings_event_engine_demo.ipynb`: screen-recording-friendly walkthrough that runs the deterministic fixture through the real canonical pipeline, inspects the emitted artifacts, and shows how validated-only downstream research consumes `validated_features.parquet`

To run the notebook interactively, open it in Jupyter from the repo root and execute top to bottom. To execute it headlessly:

```powershell
jupyter nbconvert --to notebook --execute notebooks\01_earnings_event_engine_demo.ipynb --inplace --ExecutePreprocessor.timeout=600
```

Deterministic offline command:

```powershell
python scripts\run_full_research_pipeline.py --fixture-dir tests\fixtures\offline_pipeline --output-dir outputs\full_research_fixture
```

Required stages in the canonical run:

- `event_pipeline`: runs `scripts/run_event_pipeline_test.py` in fixture or live mode, validates `events.parquet`, `clean_events.parquet`, `features.parquet`, and `diagnostics.parquet`, and derives `validated_features.parquet` for downstream research
- `earnings_analysis`: runs `analysis.earnings_analysis.run_full_analysis()` on the validated-only research subset, keeping price-only summaries separate from option-implied pricing conclusions
- `strategy_simulation`: runs `strategies.straddle_simulator.run_straddle_analysis()` on the validated-only research subset using event-time option-derived `implied_move` values only
- `conditional_analysis`: runs `analysis.conditional_analysis.run_conditional_analysis()` on the strategy results

Optional stage:

- `visualizations`: enabled only with `--run-visuals`; this is intentionally excluded from the default contract because it depends on `matplotlib` and is reporting-only rather than core research/data plumbing

Exit semantics:

- `SUCCESS`: every selected required stage completed and its required upstream artifacts were present, readable, schema-valid, and non-empty
- `FAILED`: any required stage failed, any required artifact was missing or malformed, or a user-selected optional stage could not run

Mode behavior:

- `--fixture-dir ...`: deterministic offline mode using only local fixtures; it never falls back to live data
- no `--fixture-dir`: live mode; this requires `--tickers`, and the current pipeline does not silently substitute HV20 for missing event-time option snapshots

What deterministic full-run success proves:

- orchestration wiring is honest end to end
- the validated pipeline artifact contract is produced and the validated-only research subset is what downstream stages actually consume
- event-time option-derived `implied_move` semantics are preserved when fixture option snapshots are supplied
- the required research stages can complete without network access when fixtures are supplied

What deterministic full-run success does **not** prove:

- live market-data availability
- plotting support unless `--run-visuals` is explicitly enabled and succeeds
- research efficacy, profitability, or production readiness

`scripts/run_pead_research.py` remains a separate explicit research entry point. It is not part of the canonical full orchestrator because it depends on clean-event and price-panel research inputs that are adjacent to, but not guaranteed by, the core feature-artifact contract.

## Live vs Offline Validation

`scripts/run_event_pipeline_test.py` now supports two explicit modes:

- Live mode: fetches earnings dates and price history from the configured market-data sources.
- Fixture mode: reads deterministic local fixtures from `tests/fixtures/offline_pipeline` and exercises the real normalization, price-alignment, validation, and feature-generation logic without network access.

Exact deterministic validation command:

```powershell
python scripts\run_event_pipeline_test.py --fixture-dir tests\fixtures\offline_pipeline --output-dir outputs\pipeline_fixture
```

The fixture run writes:

- `events.parquet`: normalized raw events before price enrichment and validation
- `clean_events.parquet`: validated clean events
- `features.parquet`: feature-level audit artifact that may still include invalid rows, with validation flags attached
- `diagnostics.parquet`: overview plus runtime/failure diagnostics

When `scripts/run_full_research_pipeline.py` runs successfully, it also writes:

- `validated_features.parquet`: validated-only feature rows derived from `features.parquet` using the canonical `valid_event` flag and cross-checked against `clean_events.parquet`

Core artifact contract shared by both entry points:

- `events.parquet`: normalized raw events after retrieval or fixture loading
- `clean_events.parquet`: validated clean events
- `features.parquet`: enriched feature-level rows retained for diagnostics and auditability, including invalid rows when present
- `implied_move`: event-time option-derived implied move from explicit call/put snapshot data, computed as `(call_mid + put_mid) / underlying_price`
- `hv20_expected_move_proxy`: separate historical-volatility expected-move proxy retained for context only
- `surprise_vs_implied`: only populated when real event-time `implied_move` is available
- `validated_features.parquet`: orchestrator-only artifact used by required downstream research stages

Additional live small-batch artifacts from `scripts/run_small_batch_test.py`:

- `events_with_prices.parquet`: price-enriched intermediate retained for debugging and inspection
- `options.parquet`: synthetic options snapshot used by the live small-batch validation path
- coverage and diagnostics parquet outputs for batch-level validation summaries

What the fixture proves:

- pipeline wiring works end to end without network access
- the event, clean-event, and feature artifact contracts are non-empty and stable
- the full orchestrator can restrict downstream research to validated feature rows only
- fixture option snapshots map to event-time `implied_move` values without HV proxy substitution

What the fixture does **not** prove:

- live data availability or market-data quality
- correctness of historical option snapshots
- successful option-based strategy output is still a simplified research proxy unless actual historical option entry/exit premiums are modeled
- trading profitability or research efficacy

## Trading_Foundation Historical Data

`scripts/run_event_pipeline_test.py` and `scripts/run_full_research_pipeline.py` now support:

```powershell
--historical-data-root <path-to-Trading_Foundation\historical_data>
```

When this flag is provided, the pipeline:

- loads daily underlying bars from `historical_data/<TICKER>/eod/<TICKER>_1D.csv`
- loads daily historical option chains from `historical_data/<TICKER>/option_snapshot/*_options_snapshot_YYYY-MM-DD.csv`
- derives `expiration`, `strike`, and `option_type` from OCC-style `option_symbol`
- selects the first expiration strictly after the earnings event and the ATM call/put pair nearest the pre-event underlying close
- preserves `hv20_expected_move_proxy` as a separate context-only field rather than a fallback implied move

Current repo requirements mapped to Trading Foundation:

- Required now: underlying daily EOD bars and daily historical `option_snapshot`
- Not currently required by this repo: underlying intraday bars, `option_ohlc`, IV surface files, IV rank/percentile files

Timing behavior for daily snapshots:

- `AMC` events use the same-day pre-event trading session snapshot date
- `BMO` events use the prior trading-session snapshot date

Representative command using canonical local historical data:

```powershell
python scripts\run_full_research_pipeline.py --tickers AAPL,MSFT --start-date 2025-01-01 --end-date 2025-12-31 --max-events 20 --historical-data-root C:\Users\RAS\OneDrive\RAS\Trading\Live\Trading_Foundation\historical_data --output-dir outputs\main_machine_tf_validation
```

The feature artifact now includes option provenance columns such as:

- `pre_event_date`
- `option_snapshot_quote_date`
- `option_snapshot_source_path`
- `option_selected_expiration`
- `option_selected_strike`

These make it possible to audit whether daily historical snapshots were aligned to the intended pre-event session.

## Historical IV Caveat

`run_analysis.py` no longer reuses a current option-chain IV snapshot across historical earnings rows.

- Historical price-based outputs remain valid without options data: realized move, signed move, pre/post price alignment, and realized-volatility context.
- Historical IV-dependent outputs now require explicit event-time IV inputs keyed to the earnings event date.
- When event-time IV inputs are missing, `run_analysis.py` leaves IV-dependent fields unavailable instead of silently substituting current data.
- Any event-time return metric in `run_analysis.py` is an expected-move/volatility proxy, not observed historical straddle P&L.
- Actual historical option-return analysis requires event-time option premium snapshots.
- Any old historical outputs generated before this containment change that treated a current option chain as event-time IV should be considered stale and invalid for research conclusions.

## Daily-Bar Session Alignment Contract

`run_analysis.py` now aligns earnings events to daily bars using available trading sessions, not calendar-day arithmetic.

- `BMO` (before market open): pre-event session is the prior available trading session, and post-event session is the first available trading session on or after the event date.
- `AMC` (after market close): pre-event session is the last available trading session on or before the event date, and post-event session is the next available trading session after the event date.
- Weekends, holidays, and sparse fixtures are handled from the sorted price-history dates that actually exist in the input history.
- Missing or ambiguous timing is not silently guessed. `run_analysis.py` accepts explicit `announcement_timing`/`timing` labels or clearly pre-market/post-close timestamps; otherwise it fails fast.
- Daily-bar outputs remain limited to session-level price moves. They do not reconstruct intraday paths or intraday option behavior.

## Zero-Row Failure Semantics

By default, both pipeline entry points now fail non-zero when:

- the retrieval stage produces zero raw events
- the validation stage produces zero clean/valid events

`data/earnings_calendar.py` also now raises a retrieval error when the upstream request fails, so callers can distinguish fetch failure from a legitimate empty response.

If you need an explicit empty-data diagnostic run for `run_event_pipeline_test.py`, use:

```powershell
python scripts\run_event_pipeline_test.py --fixture-dir tests\fixtures\offline_pipeline --allow-empty
```

## Regression Check

```powershell
python -m unittest tests.test_data_coverage_regression tests.test_trading_foundation_adapter tests.test_offline_pipeline tests.test_full_research_pipeline -v
```

## Compile Check

```powershell
python -m compileall data\trading_foundation.py tests\test_trading_foundation_adapter.py scripts\run_event_pipeline_test.py scripts\run_full_research_pipeline.py
```

Live data runs may require local market-data environment setup, while the regression guard above is offline and deterministic.

## Minimal PEAD Research

The repo now includes a small research-only PEAD layer that measures announcement reaction and short-horizon forward drift for validated earnings events. It operates only on artifacts that already contain the needed price columns and does not fetch data, generate recommendations, or add live execution.

Expected PEAD input contract:
- Core columns: `event_id`, `ticker`, `event_date`, `pre_close`
- Announcement baseline: `post_close` or `announcement_close`
- Forward drift columns for each requested window: `forward_close_{N}d`
- Optional validity filters when present: `valid_event`, `price_alignment_valid`

If your validated event artifact does not already include `forward_close_{N}d`, the PEAD module now includes an offline helper that derives those columns from a local price panel with `ticker`, `date`, and `close`. The script can use that helper directly via `--prices-path`.

Offline test command:

```powershell
python -m unittest tests.test_data_coverage_regression tests.test_pead_research -v
```

Research script command:

```powershell
python scripts\run_pead_research.py --clean-events-path data\outputs\test_batch\clean_events.parquet --output-dir outputs\research
```

With a local price panel:

```powershell
python scripts\run_pead_research.py --clean-events-path data\outputs\test_batch\clean_events.parquet --prices-path data\outputs\local_prices.parquet --output-dir outputs\research
```
