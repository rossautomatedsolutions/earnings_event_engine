# earnings_event_engine

Lightweight tooling for running and validating the earnings event pipeline, including a targeted offline regression guard for coverage-input consistency.

## Entry Points

- `scripts/run_small_batch_test.py`
- `scripts/run_event_pipeline_test.py`

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
- `features.parquet`: analysis-ready feature artifact
- `diagnostics.parquet`: overview plus runtime/failure diagnostics

Core artifact contract shared by both entry points:

- `events.parquet`: normalized raw events after retrieval or fixture loading
- `clean_events.parquet`: validated clean events
- `features.parquet`: enriched feature-level rows used downstream

Additional live small-batch artifacts from `scripts/run_small_batch_test.py`:

- `events_with_prices.parquet`: price-enriched intermediate retained for debugging and inspection
- `options.parquet`: synthetic options snapshot used by the live small-batch validation path
- coverage and diagnostics parquet outputs for batch-level validation summaries

What the fixture proves:

- pipeline wiring works end to end without network access
- the event, clean-event, and feature artifact contracts are non-empty and stable
- downstream `analysis.earnings_analysis.run_full_analysis()` can consume the produced feature artifact

What the fixture does **not** prove:

- live data availability or market-data quality
- correctness of historical option snapshots
- trading profitability or research efficacy

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
python -m unittest tests.test_data_coverage_regression -v
```

## Compile Check

```powershell
python -m compileall tests\test_data_coverage_regression.py scripts\run_event_pipeline_test.py scripts\run_small_batch_test.py
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
