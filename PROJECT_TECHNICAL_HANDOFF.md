# Project Technical Handoff

Prepared on 2026-08-18 after inspecting the repository code, README, notebook, tests, fixture data, generated validation artifacts, and recent git history.

## 2026-08-23 Addendum: Trading_Foundation Historical Options Integration

This handoff predates the canonical `Trading_Foundation` historical-data adapter now present in this repo.

Superseding status as of 2026-08-23:

- The repo can now consume canonical-style `historical_data/<TICKER>/eod/*_1D.csv` plus `historical_data/<TICKER>/option_snapshot/*_options_snapshot_YYYY-MM-DD.csv`.
- The current earnings-event research path requires those two datasets only. It does not currently require intraday bars, `option_ohlc`, or IV surface/rank files.
- `AMC` events use same-session daily option snapshots; `BMO` events use the prior trading-session snapshot date.
- The adapter emits provenance fields including `pre_event_date`, `option_snapshot_quote_date`, `option_snapshot_source_path`, `option_selected_expiration`, and `option_selected_strike`.
- Older statements below that describe the repo as having no historical options snapshot/provider support should be treated as superseded by this addendum.
- Remaining uncertainty is now main-machine data coverage and once-per-day snapshot timing sufficiency for specific research questions, not the absence of an integration path.

See `docs/HISTORICAL_OPTIONS_DATA_STATUS.md` for the current field mapping, temporal assumptions, main-machine command, and readiness classification.

Validation performed for this handoff:

- `python -m unittest tests.test_offline_pipeline tests.test_full_research_pipeline tests.test_option_implied_move_contract tests.test_run_event_pipeline_timing tests.test_run_analysis_session_alignment tests.test_run_analysis_iv_containment tests.test_pead_research tests.test_data_coverage_regression tests.test_small_batch_artifacts -v`
- Result: `95` tests passed.
- `python scripts\run_full_research_pipeline.py --fixture-dir tests\fixtures\offline_pipeline --output-dir outputs\technical_handoff_full_research`
- Result: canonical fixture pipeline completed successfully and emitted contract artifacts.
- `python scripts\run_pead_research.py --events-path outputs\technical_handoff_full_research\validated_features.parquet --historical-data-root tests\fixtures\trading_foundation_historical_data --output-dir outputs\technical_handoff_pead_phase2 --windows 1 3 5 10 20`
- Result: PEAD research artifacts emitted successfully.

## 1. Executive Summary

This repository is a validated earnings-event research pipeline, not a full trading platform. Its real strength today is deterministic, test-backed event normalization, timing/session alignment, price enrichment, event-time option-implied move attachment, validation gating, and a small set of downstream research utilities that consume the validated artifact contract.

The current maturity level is best described as:

- Strong for offline, fixture-backed research plumbing and contract validation.
- Moderate for live ad hoc data retrieval through `yfinance`.
- Weak or nonexistent for broker integration, production execution, portfolio management, and real backtesting.

What is actually working today:

- Deterministic offline event pipeline from fixture inputs to `events.parquet`, `clean_events.parquet`, `features.parquet`, `diagnostics.parquet`, and orchestrator-only `validated_features.parquet`.
- Explicit BMO/AMC timing normalization and trading-session alignment with fail-fast behavior for ambiguous timing in historical analysis.
- Event-time option-implied move support from supplied call/put snapshots keyed to event date.
- Strict separation between real event-time option-implied data and historical-volatility proxies.
- Validated-only downstream research routing in the full orchestrator.
- Basic earnings analysis, simplified long-straddle proxy analysis, conditional high-IV/low-IV analysis, and minimal PEAD utilities.
- A canonical demo notebook that walks the deterministic pipeline.

What is experimental, partial, deprecated, or placeholder:

- `run_small_batch_test.py` is a live smoke-test path that uses live `yfinance` data plus synthetic options, not a canonical research or production path.
- `strategies/vertical_spread_simulator.py` is a research proxy, not a broker-accurate options backtest.
- `run_analysis.py` is a standalone historical move analysis utility with optional event-time IV inputs; it is not part of the canonical multi-stage pipeline.
- `src/features/event_features.py`, `src/features/option_selection.py`, `src/analysis/implied_vs_realized.py`, `src/earnings_event_engine/analysis/cohort_analysis.py`, and `analysis/event_filtering.py` are useful research helpers but are not part of the canonical orchestrator.
- `data/loader.py`, `features/engineering.py`, `analysis/event_analysis.py`, `strategies/base_strategy.py`, and `backtest/engine.py` are placeholders and should not be treated as real system capabilities.

Intended role in a broader trading/alpha-generation stack:

- This repo should serve as the event-centric preprocessing, validation, and research-input layer for earnings-event and adjacent volatility/event studies.
- Future modules should reuse its canonical artifacts and alignment/validation logic instead of rebuilding raw event plumbing.
- It is currently best positioned as an upstream research substrate, not as the final execution/risk platform.

## 2. Repository / System Architecture

### High-level architecture

| Area | Primary files | Responsibility | Current status |
| --- | --- | --- | --- |
| Data ingestion | `data/earnings_calendar.py`, `scripts/run_event_pipeline_test.py` | Fetch or load earnings events and local fixture tables | Active |
| Preprocessing / ETL | `scripts/run_event_pipeline_test.py`, `src/events/timing.py`, `src/events/event_alignment.py` | Normalize tickers/dates/timing, map events to trading sessions, load price history | Active |
| Universe construction | `scripts/run_event_pipeline_test.py`, `scripts/run_small_batch_test.py` | Either explicit `--tickers`, fixture tickers, or hardcoded smoke-test universe | Minimal |
| Feature generation | `scripts/run_event_pipeline_test.py`, `src/features/event_features.py` | Build realized move, implied move, timing, context, and derived research fields | Active |
| Screeners / filters | `analysis/conditional_analysis.py`, `analysis/event_filtering.py` | Median IV splits, top/bottom PnL event filtering | Partial |
| Signal / strategy logic | `strategies/straddle_simulator.py`, `strategies/vertical_spread_simulator.py`, `src/earnings_event_engine/research/pead.py` | Research proxies and PEAD bucketing logic | Partial |
| Ranking / scoring | `analysis/conditional_analysis.py`, `analysis/event_filtering.py`, `src/earnings_event_engine/validation/data_quality_score.py` | Group comparisons, event ranking by PnL, event data quality scoring | Partial |
| Validation | `src/earnings_event_engine/validation/*`, `src/validation/pipeline_health.py` | Event diagnostics, coverage reporting, quality scoring, pipeline health | Active |
| Backtesting / research | `run_analysis.py`, `analysis/earnings_analysis.py`, `strategies/*.py`, `scripts/run_pead_research.py` | Historical analysis and simplified strategy studies | Partial |
| Portfolio / risk management | None beyond validation gates | No real portfolio/risk engine | Not implemented |
| Execution / OMS | None | No broker/execution integration | Not implemented |
| Reporting / artifacts | `scripts/run_event_pipeline_test.py`, `scripts/run_full_research_pipeline.py`, `scripts/run_pead_research.py`, `src/earnings_event_engine/debug/event_snapshot.py` | Emit parquet/csv/md/json artifacts and debug snapshots | Active |
| Notebook | `notebooks/01_earnings_event_engine_demo.ipynb` | Canonical walkthrough of deterministic fixture path | Active |
| APIs / external services | `yfinance` only | Earnings dates and price history in live paths | Minimal |

### Canonical end-to-end data flow

```text
fixture/live earnings events
  -> event normalization
  -> timing resolution (explicit or timestamp-inferred BMO/AMC)
  -> trading-session alignment
  -> price enrichment (pre_close, post_open, post_close, HV20 proxy)
  -> event-time option snapshot attachment
  -> feature generation
  -> event diagnostics / coverage checks
  -> clean_events.parquet + features.parquet + diagnostics.parquet
  -> validated_features.parquet gate
  -> earnings analysis
  -> simplified straddle proxy
  -> conditional analysis
```

PEAD is adjacent, not in the canonical full orchestrator:

```text
clean_events.parquet
  (+ local price panel when forward_close_Nd columns are missing)
  -> reaction_return
  -> forward_return_Nd
  -> reaction buckets
  -> PEAD summary artifacts
```

### Canonical entry points

| Entry point | Purpose | Recommended use |
| --- | --- | --- |
| `scripts/run_event_pipeline_test.py` | Canonical event pipeline contract validation | Use for event artifact generation and validation |
| `scripts/run_full_research_pipeline.py` | Canonical multi-stage research orchestrator | Use when downstream research should consume validated-only features |
| `scripts/run_pead_research.py` | Bounded PEAD research run | Use for post-earnings drift analysis on validated events plus local price data |
| `notebooks/01_earnings_event_engine_demo.ipynb` | Human-readable walkthrough | Use to onboard a new agent or teammate |
| `run_analysis.py` | Standalone historical earnings move analysis | Use only for focused historical analysis, not as the main orchestrator |

## 3. Data Contracts

### Canonical input contracts

| Input | Producer / source | Row grain | Required fields | Optional fields | Frequency / granularity | Point-in-time assumptions | Downstream consumers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fixture events: `earnings_events.csv` or `.parquet` | Local fixture or user-supplied table | One row per earnings event | `ticker`, `earnings_date` | `announcement_timing` or any supported timing alias column | Event-level | If timing is omitted, only clearly pre-market or post-close timestamps are inferred | `scripts/run_event_pipeline_test.py` |
| Fixture prices: `prices.csv` or `.parquet` | Local fixture or local price panel | One row per ticker/date | `ticker`, `date`, `open`, `close` | None | Daily bars | Daily sessions only; no intraday reconstruction | `scripts/run_event_pipeline_test.py`, `src/earnings_event_engine/research/pead.py` |
| Fixture options: `options.csv` or `.parquet` | Local fixture or user-supplied event-time options snapshot | One row per option leg | `ticker`, `event_date`, `option_type`, `strike`, `underlying_price`, `bid`, `ask`, `mid` | None | Event-level option-leg rows | Must represent event-time option pricing, not a current chain reused for history | `scripts/run_event_pipeline_test.py` |
| Live earnings dates | `yfinance.Ticker(...).get_earnings_dates()` | One row per earnings event | provider columns mapped to `earnings_date` | provider-specific extras ignored | Event-level | Retrieval failures raise `EarningsCalendarRetrievalError`; empty response is allowed | `data/earnings_calendar.py` |
| Live prices | `yfinance.download(...)` | One row per trading day | `Open`, `Close` | None | Daily bars | Used only for live runs and ad hoc research | `scripts/run_event_pipeline_test.py`, `run_analysis.py` |
| Macro regime file: `data/inputs/macro_regime.csv` | Optional local file | One row per event date | `event_date`, `macro_regime` | None | Daily / event-date keyed | Joined by normalized event date only | `scripts/run_event_pipeline_test.py` |
| Historical event-time IV inputs | User-supplied DataFrame to `run_analysis.py` | One row per ticker/event date | `earnings_date`, `event_time_iv_annual` | `ticker` | Event-level | Must be true event-time IV; duplicates and invalid numeric values fail | `run_analysis.py` |
| PEAD local price panel | Local file path passed to `scripts/run_pead_research.py` | One row per ticker/date | `ticker`, `date`, `close` | `open`, `volume` | Daily bars | Used to derive `forward_close_Nd` columns and pre-event context when missing | `src/earnings_event_engine/research/pead.py` |
| PEAD Trading_Foundation local EOD root | `--historical-data-root` passed to `scripts/run_pead_research.py` | Per-ticker local EOD trees | Canonical Trading_Foundation `ticker/eod/*_1D.csv` | None | Daily bars | Used to derive forward windows and price context without a flat price panel | `data/trading_foundation.py`, `src/earnings_event_engine/research/pead.py` |

### Canonical output contracts

#### `events.parquet`

| Property | Value |
| --- | --- |
| Producer | `scripts/run_event_pipeline_test.py` |
| Row grain | One normalized earnings event |
| Required columns in orchestrator validation | `event_id`, `ticker`, `event_date`, `announcement_timing` |
| Observed schema in validated fixture run | `event_id`, `ticker`, `earnings_date`, `event_date`, `timing`, `announcement_timing`, `mapped_session` |
| Purpose | Raw normalized events before price enrichment and validation |
| Canonical persistence | Arbitrary run output directory; example: `outputs/technical_handoff_full_research/events.parquet` |
| Downstream consumers | `scripts/run_full_research_pipeline.py` for artifact validation only |

#### `clean_events.parquet`

| Property | Value |
| --- | --- |
| Producer | `scripts/run_event_pipeline_test.py` via `run_event_diagnostics()` |
| Row grain | One validated event |
| Required columns in orchestrator validation | `event_id`, `ticker`, `event_date`, `announcement_timing`, `pre_close`, `post_close` |
| Observed schema in validated fixture run | `event_id`, `ticker`, `earnings_date`, `event_date`, `timing`, `announcement_timing`, `mapped_session`, `pre_close`, `post_open`, `post_close`, `hv20_expected_move_proxy`, `price_alignment_valid`, `option_snapshot_available`, `implied_move_available`, `implied_move_source`, `option_call_mid`, `option_put_mid`, `option_straddle_mid`, `implied_move_dollars`, `implied_move` |
| Purpose | Validated event rows for downstream research and PEAD |
| Canonical persistence | Arbitrary run output directory; example: `outputs/technical_handoff_full_research/clean_events.parquet` |
| Downstream consumers | `scripts/run_full_research_pipeline.py`, `scripts/run_pead_research.py` |

#### `features.parquet`

| Property | Value |
| --- | --- |
| Producer | `scripts/run_event_pipeline_test.py` |
| Row grain | One event feature row, including invalid rows |
| Required columns in orchestrator validation | `ticker`, `announcement_timing`, `realized_gap_move`, `implied_move`, `surprise_vs_implied` |
| Observed schema in validated fixture run | `event_id`, `ticker`, `earnings_date`, `event_date`, `timing`, `announcement_timing`, `mapped_session`, `price_alignment_valid`, `option_snapshot_available`, `implied_move_available`, `implied_move_source`, `pre_close`, `post_open`, `post_close`, `hv20_expected_move_proxy`, `option_call_mid`, `option_put_mid`, `option_straddle_mid`, `implied_move_dollars`, `implied_move`, `realized_gap_move`, `realized_close_move`, `iv_minus_realized_gap`, `surprise_vs_implied`, `days_since_prev_event`, `macro_regime`, `valid_event`, `invalid_event`, `failure_reasons` |
| Purpose | Full audit artifact with validation flags attached |
| Canonical persistence | Arbitrary run output directory; example: `outputs/technical_handoff_full_research/features.parquet` |
| Downstream consumers | Diagnostics, ad hoc research, but not the canonical validated-only downstream path |

#### `validated_features.parquet`

| Property | Value |
| --- | --- |
| Producer | `scripts/run_full_research_pipeline.py` via `prepare_validated_research_artifact()` |
| Row grain | One validated event feature row |
| Schema | Same as `features.parquet`, filtered to `valid_event == True` and cross-checked against `clean_events.parquet` by `event_id` |
| Purpose | Canonical downstream research input artifact |
| Canonical persistence | Arbitrary full-run output directory; example: `outputs/technical_handoff_full_research/validated_features.parquet` |
| Downstream consumers | `analysis/earnings_analysis.py`, `strategies/straddle_simulator.py`, conditional analysis, future research modules |

#### `diagnostics.parquet`

| Property | Value |
| --- | --- |
| Producer | `scripts/run_event_pipeline_test.py` |
| Row grain | One metric row |
| Schema | `category`, `metric`, `count`, `pct_of_events` |
| Purpose | Summary diagnostics, failure breakdown, runtime metrics |
| Canonical persistence | Arbitrary run output directory; example: `outputs/technical_handoff_full_research/diagnostics.parquet` |
| Downstream consumers | Humans, debugging, pipeline audits |

### Additional non-canonical or adjacent artifacts

| Artifact | Producer | Purpose | Status |
| --- | --- | --- | --- |
| `events_with_prices.parquet` | `scripts/run_small_batch_test.py` | Price-enriched intermediate for smoke-test debugging | Live smoke-test only |
| `options.parquet` | `scripts/run_small_batch_test.py` | Synthetic options snapshot | Live smoke-test only |
| `diagnostics_flags.parquet`, `coverage_summary.parquet`, `coverage_by_ticker.parquet`, `coverage_by_year.parquet` | `scripts/run_small_batch_test.py` | Extra validation/coverage outputs | Useful, not canonical full-run outputs |
| `pead_event_returns.csv` / `.parquet` | `scripts/run_pead_research.py` | PEAD event-level reaction and drift outputs | Active research output |
| `pead_horizon_summary.csv` | `scripts/run_pead_research.py` | Baseline PEAD summary by forward horizon | Active research output |
| `pead_bucket_summary.csv` | `scripts/run_pead_research.py` | Bucketed PEAD summary | Active research output |
| `pead_condition_summary.csv` | `scripts/run_pead_research.py` | Conditioned PEAD summary by bucket and horizon | Active research output |
| `pead_data_coverage.csv` | `scripts/run_pead_research.py` | Coverage report for windows, event metrics, and condition buckets | Active research output |
| `pead_signal_assessment.csv` | `scripts/run_pead_research.py` | Conservative watchlist / signal readiness assessment | Active research output |
| `pead_summary.md` | `scripts/run_pead_research.py` | Human-readable PEAD summary | Active research output |
| `data/outputs/debug_snapshots/<event_id>/...` | `src/earnings_event_engine/debug/event_snapshot.py` | Per-event debug bundles | Utility, not integrated into canonical scripts |

### Canonical artifacts future projects should consume

Use these in this order:

1. `validated_features.parquet` for any earnings-event research that needs realized move plus event-time implied-move semantics and validated-only rows.
2. `clean_events.parquet` for PEAD or any extension that needs validated events but will derive additional future price columns itself.
3. `diagnostics.parquet` to audit coverage and failure modes.

Do not treat `features.parquet` as the canonical downstream research input unless the research explicitly needs invalid rows for diagnostics or error analysis.

## 4. Signals, Screens, and Trading Logic

### Implemented event and feature logic

| Logic | Location | Purpose | Formula / behavior | Inputs | Outputs | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Timing normalization | `scripts/run_event_pipeline_test.py`, `src/events/timing.py` | Resolve event timing to BMO/AMC when possible | Explicit timing aliases are normalized; otherwise timestamps before `09:30` infer `BMO`, timestamps at or after `16:00` infer `AMC`; ambiguous timestamps remain unresolved | Event timing labels or timestamps | `announcement_timing`, `mapped_session` | Validated |
| Trading-session alignment | `src/events/event_alignment.py` | Map event to pre/post trading sessions | `AMC`: pre = last session on/before event date, post = next session after. `BMO`: pre = prior session, post = first session on/after event date | Event date, timing, price index | `pre_event_date`, `post_event_date` or aligned price fields | Validated |
| HV20 expected move proxy | `scripts/run_event_pipeline_test.py` | Context-only volatility proxy | 20-day rolling standard deviation of daily returns at pre-event date, min periods `10` | Daily close history | `hv20_expected_move_proxy` | Validated, context only |
| Event-time implied move | `scripts/run_event_pipeline_test.py` | Attach option-implied move from supplied event options | ATM call/put mids are added and divided by `underlying_price` | Event-mapped call/put legs | `option_call_mid`, `option_put_mid`, `option_straddle_mid`, `implied_move_dollars`, `implied_move` | Validated |
| Realized gap move | `scripts/run_event_pipeline_test.py` | Primary realized move metric | `abs(post_open / pre_close - 1.0)` | `pre_close`, `post_open` | `realized_gap_move` | Validated |
| Realized close move | `scripts/run_event_pipeline_test.py` | Secondary realized move metric | `abs(post_close / pre_close - 1.0)` | `pre_close`, `post_close` | `realized_close_move` | Implemented |
| IV minus realized gap | `scripts/run_event_pipeline_test.py` | Compare implied vs realized | `implied_move - realized_gap_move`, but only when real implied move exists | `implied_move`, `realized_gap_move` | `iv_minus_realized_gap` | Validated |
| Surprise vs implied | `scripts/run_event_pipeline_test.py` | Realized minus implied surprise measure | `realized_gap_move - implied_move`, but only when real implied move exists | `implied_move`, `realized_gap_move` | `surprise_vs_implied` | Validated |
| Days since previous event | `scripts/run_event_pipeline_test.py` | Event spacing context | Ticker-grouped diff of sorted `event_date` in days | `ticker`, `event_date` | `days_since_prev_event` | Implemented |
| Macro regime join | `scripts/run_event_pipeline_test.py` | Optional regime tagging | Left join on normalized `event_date`; defaults to `"unknown"` | `data/inputs/macro_regime.csv` when present | `macro_regime` | Implemented, optional |

### Implemented validation and gating logic

| Logic | Location | Purpose | Behavior | Status |
| --- | --- | --- | --- | --- |
| Event diagnostics | `src/earnings_event_engine/validation/event_diagnostics.py` | Reject bad event rows | Flags missing prices, nonpositive prices, extreme price jumps over `50%`, missing implied move, implied move outside `[0, 30%]`, missing timing, inconsistent session mapping, missing options data, bad spreads, ATM distance over `5%`, missing call/put legs, zero mids | Validated |
| Zero-row failure semantics | `scripts/run_event_pipeline_test.py` | Fail loudly on empty pipelines | Nonzero exit when raw events or valid events are zero, unless `--allow-empty` is set | Validated |
| Validated-only downstream gate | `scripts/run_full_research_pipeline.py` | Prevent invalid rows from contaminating downstream research | Filters `features.parquet` to `valid_event == True`, requires non-empty validated rows, cross-checks `event_id` set against `clean_events.parquet` | Validated |
| IV containment | `run_analysis.py` | Prevent historical IV leakage | No current option-chain IV reuse for historical rows; IV-dependent fields remain unavailable without explicit event-time IV inputs | Validated |

### Implemented research / strategy logic

| Logic | Location | Purpose | Formula / behavior | Status |
| --- | --- | --- | --- | --- |
| Earnings summary stats | `analysis/earnings_analysis.py` | Summarize realized vs implied earnings behavior | Computes event counts, option-eligible counts, means, overpricing/underpricing rates, by-ticker and by-timing aggregates | Validated |
| Long straddle proxy | `strategies/straddle_simulator.py` | Simplified options-event strategy proxy | `cost = implied_move * cost_multiplier`; `payoff = abs(realized_gap_move)`; `pnl = payoff - cost` | Validated, proxy only |
| High-IV / low-IV split | `analysis/conditional_analysis.py` | Segment strategy outcomes by IV regime | Splits on sample median `implied_move` | Validated |
| Timing comparison | `analysis/conditional_analysis.py` | Compare AMC vs BMO outcomes | Aggregates `event_count`, `avg_realized_gap_move`, `avg_pnl` by timing | Validated |
| Macro-regime comparison | `analysis/conditional_analysis.py` | Compare outcomes by regime | Aggregates `event_count`, `avg_pnl`, `win_rate` by `macro_regime` | Implemented |
| Vertical spread proxy | `strategies/vertical_spread_simulator.py` | Alternative options-event proxy | Uses capped spread width, debit fraction, and execution slippage to compute call/put vertical spread costs, payoffs, and PnLs | Implemented, not canonical |
| Historical move analysis | `run_analysis.py` | Standalone historical earnings move analysis | Computes signed move, absolute move, realized vol, optional event-time IV comparison, and expected-move proxy return | Validated, standalone |

### Implemented PEAD logic

| Logic | Location | Purpose | Formula / behavior | Status |
| --- | --- | --- | --- | --- |
| Forward close derivation | `src/earnings_event_engine/research/pead.py` | Add local future closes to validated events | Uses `post_event_date` or `mapped_session` / `announcement_timing` to anchor forward windows per ticker | Validated |
| Pre-event context derivation | `src/earnings_event_engine/research/pead.py` | Add leakage-safe volatility / liquidity context | Uses trading-session alignment plus shifted 20-day rolling features from local prices | Implemented |
| Announcement reaction | `src/earnings_event_engine/research/pead.py` | Measure immediate event reaction | `reaction_return = post_close / pre_close - 1.0` | Validated |
| Gap reaction | `src/earnings_event_engine/research/pead.py` | Measure immediate open reaction | `gap_return = post_open / pre_close - 1.0` when `post_open` exists | Implemented |
| Forward drift | `src/earnings_event_engine/research/pead.py` | Measure post-announcement continuation or reversal | `forward_return_Nd = forward_close_Nd / post_close - 1.0` | Validated |
| Reaction bucket | `src/earnings_event_engine/research/pead.py` | Bucket by sign / neutrality | Positive if `reaction_return > threshold`, negative if `< -threshold`, else neutral | Validated |
| Bucket summary | `src/earnings_event_engine/research/pead.py` | Summarize drift by bucket and horizon | Aggregates event count, mean/median forward return, positive rate | Validated |
| Conditional PEAD buckets | `src/earnings_event_engine/research/pead.py` | Segment by timing, implied expectations, realized-vs-implied surprise, volatility, liquidity, size, and fundamental surprise where supported | Uses sign buckets or simple quantile buckets with no aggressive optimization | Implemented |
| Coverage and readiness reporting | `src/earnings_event_engine/research/pead.py`, `scripts/run_pead_research.py` | Document available horizons and whether a later watchlist/signal design step is even reviewable | Emits CSV summaries plus markdown packet; never emits live candidates automatically | Implemented |

### Implemented but non-canonical research helpers

| Logic | Location | Notes | Status |
| --- | --- | --- | --- |
| Implied vs realized cohort analysis | `src/analysis/implied_vs_realized.py` | General cohort utility, not wired into main orchestrator | Implemented |
| Multi-factor cohort bucketing | `src/earnings_event_engine/analysis/cohort_analysis.py` | Buckets on implied move, volume, pre-earnings return, timing | Implemented |
| Event top/bottom PnL filtering | `analysis/event_filtering.py` | Useful after a strategy result table exists | Implemented |
| Raw option-chain selector | `src/features/option_selection.py` | Reusable when future work has raw chains instead of pre-keyed event option rows | Implemented |

### Not implemented

- General screener engine.
- Persistent ranked-candidate artifact contract.
- Signal object abstraction.
- Entry/exit rules tied to actual marketable historical options premiums.
- Portfolio construction rules beyond one-event-at-a-time research proxies.

## 5. Risk Controls

### Controls actually enforced in code

| Control | Location | Enforcement |
| --- | --- | --- |
| Live mode requires explicit tickers | `scripts/run_event_pipeline_test.py`, `scripts/run_full_research_pipeline.py` | Raises on live mode without `--tickers` |
| Zero raw events fail | `scripts/run_event_pipeline_test.py` | Nonzero exit unless `--allow-empty` |
| Zero valid events fail | `scripts/run_event_pipeline_test.py` | Nonzero exit unless `--allow-empty` |
| Required artifact schema validation | `scripts/run_full_research_pipeline.py` | Missing artifacts, bad schemas, unreadable parquet, empty required artifacts fail the run |
| Validated-only downstream boundary | `scripts/run_full_research_pipeline.py` | Downstream stages only receive `validated_features.parquet` |
| No silent timing guessing in `run_analysis.py` | `run_analysis.py` | Unsupported or ambiguous timing raises |
| No historical IV leakage | `run_analysis.py`, `scripts/run_event_pipeline_test.py` | Missing event-time options do not trigger HV substitution for implied move fields |
| Diagnostics-based invalidation | `src/earnings_event_engine/validation/event_diagnostics.py` | Missing option data, bad spreads, missing legs, price issues, timing issues can invalidate events |

### Controls present as observability, not enforcement

| Utility | Location | Notes |
| --- | --- | --- |
| Coverage report | `src/earnings_event_engine/validation/data_coverage.py` | Reports coverage, does not itself stop the run |
| Event quality score | `src/earnings_event_engine/validation/data_quality_score.py` | Scores data quality, not used as a gate |
| Pipeline health score | `src/validation/pipeline_health.py` | Summary reliability metric, not wired into canonical CLI |
| Debug event snapshots | `src/earnings_event_engine/debug/event_snapshot.py` | Useful for incident/debug workflows, not auto-invoked |

### Controls not implemented

- Kill switch.
- Account- or strategy-level loss limits.
- Exposure limits.
- Position sizing engine.
- Drawdown controls.
- Broker-level fail-safe behavior.
- Portfolio-level allocation rules.
- Order throttling, order cancel/retry policy, or compliance checks.

Conclusion: the repo currently has strong data/validation controls and almost no live-trading risk controls.

## 6. Backtesting and Validation

### Available backtest / research workflows

| Workflow | Location | What it really is |
| --- | --- | --- |
| Canonical full research pipeline | `scripts/run_full_research_pipeline.py` | Offline/fixture or live orchestrated research run with downstream analysis and simplified strategy proxies |
| Event pipeline smoke test | `scripts/run_event_pipeline_test.py` | Artifact-generation and validation path, not a backtest engine |
| Small-batch live smoke test | `scripts/run_small_batch_test.py` | Live `yfinance` retrieval plus synthetic options for wiring/debugging |
| Historical earnings analysis | `run_analysis.py` | Historical move study with optional event-time IV inputs |
| Long straddle simulation | `strategies/straddle_simulator.py` | Simplified event-level PnL proxy |
| Vertical spread simulation | `strategies/vertical_spread_simulator.py` | Simplified event-level PnL proxy with slippage/debit assumptions |
| PEAD research | `scripts/run_pead_research.py` | Reaction and forward-drift summary, not a tradable strategy engine |

### Assumptions

- Prices are daily bars only.
- Event timing is session-level, not intraday path-level.
- Event-time option semantics require explicit event-time options snapshots.
- Historical-volatility proxies are retained for context only and are intentionally separate from true option-implied fields.
- The long-straddle and vertical-spread modules are payoff abstractions, not historical option-premium execution backtests.

### Transaction costs and slippage

- `strategies/straddle_simulator.py`: no explicit slippage or fees; uses `implied_move` as normalized cost.
- `strategies/vertical_spread_simulator.py`: has explicit `execution_slippage` and `debit_fraction` assumptions, but still remains a stylized proxy.
- No commissions, borrow, financing, queueing, assignment, or market impact modeling anywhere in the repo.

### Lookahead and data leakage protections

- `run_analysis.py` explicitly blocks reuse of current IV snapshots for historical events.
- `src/features/event_features.py` shifts pre-event rolling features by one trading day to avoid using event-day/post-event data.
- Session alignment is based on actual available trading sessions, not naive calendar arithmetic.
- Full orchestrator only promotes `valid_event == True` rows downstream.

### Train/test, walk-forward, and benchmarks

- No train/test split framework.
- No walk-forward evaluation framework.
- No benchmark series or strategy benchmark comparison framework.
- No portfolio-level performance series.

### Existing validated results

These results are implementation-validation results, not investment conclusions:

- The deterministic offline fixture path passes `95` tests as of 2026-08-18.
- The canonical full research fixture run emitted all required artifacts successfully.
- In the fixture run, `validated_features.parquet` contained `2` rows and preserved event-time option-implied move semantics for `AAPL` and `MSFT`.
- The fixture run's earnings analysis produced a mean realized gap move near `1.30%` and mean implied move near `7.00%`, but this is only a two-event fixture and is not research evidence.
- The simplified straddle proxy was negative on both fixture events, again useful only as a wiring check.

### Known methodological weaknesses

- No real historical option premium database is present.
- No realistic options execution model exists.
- Live retrieval is dependent on `yfinance`.
- The canonical deterministic fixture contains only two events.
- There is no out-of-sample or portfolio-level evaluation framework.
- The repo does not prove alpha, profitability, or production readiness.

## 7. Live / Paper Trading Integration

There is no real live or paper trading integration in this repository.

What does exist:

- `data/earnings_calendar.py` uses `yfinance` for earnings dates.
- `scripts/run_event_pipeline_test.py` and `run_analysis.py` use `yfinance` for daily price history in live mode.
- `scripts/run_small_batch_test.py` uses live `yfinance` data and synthetic options to validate plumbing on a broad ticker list.

What does not exist:

- No Alpaca integration.
- No broker abstraction.
- No order lifecycle handling.
- No paper-vs-live mode switch.
- No scheduling/orchestration daemon.
- No alerting.
- No account sync.
- No order or execution logging beyond console/debug output.
- No failure recovery around live orders because there are no live orders.

Current readiness level: research-only.

## 8. Existing Artifacts and Proven Outputs

### Most important repository artifacts

| Artifact | Path | Why it matters | Canonical? |
| --- | --- | --- | --- |
| Deterministic fixture events | `tests/fixtures/offline_pipeline/earnings_events.csv` | Minimal event contract for reproducible offline validation | Yes for testing |
| Deterministic fixture prices | `tests/fixtures/offline_pipeline/prices.csv` | Minimal price-panel contract | Yes for testing |
| Deterministic fixture options | `tests/fixtures/offline_pipeline/options.csv` | Minimal event-time options snapshot contract | Yes for testing |
| Demo notebook | `notebooks/01_earnings_event_engine_demo.ipynb` | Best onboarding walkthrough of the canonical path | Yes for onboarding |
| Full fixture validation outputs | `outputs/technical_handoff_full_research/*` | Example generated artifact set from a successful canonical run | Example output |
| PEAD validation outputs | `outputs/technical_handoff_pead/*` | Example generated PEAD outputs from validated events plus local prices | Example output |
| README | `README.md` | Current high-level repo contract and execution guidance | Active |
| Old insights note | `analysis/insights.md` | Historical note about a zero-row state | Stale |

### Canonical artifacts to treat as source-of-truth outputs

- `validated_features.parquet` for validated downstream event research.
- `clean_events.parquet` when research needs validated events plus additional local price enrichment.
- `diagnostics.parquet` for validation and coverage audit.

### Artifacts generated during this handoff

`outputs/technical_handoff_full_research`:

- `events.parquet`
- `clean_events.parquet`
- `features.parquet`
- `validated_features.parquet`
- `diagnostics.parquet`

`outputs/technical_handoff_pead_phase2`:

- `pead_event_returns.csv`
- `pead_event_returns.parquet`
- `pead_horizon_summary.csv`
- `pead_bucket_summary.csv`
- `pead_condition_summary.csv`
- `pead_data_coverage.csv`
- `pead_signal_assessment.csv`
- `pead_summary.md`

These are validation examples, not a versioned data store. The artifact names and schemas matter more than the exact directory names.

## 9. Reusable Components

### DO NOT REBUILD

| Reuse this | Exact location | Why it should be reused |
| --- | --- | --- |
| Canonical event pipeline | `scripts/run_event_pipeline_test.py` | It already normalizes events, aligns sessions, attaches option-implied moves, generates features, and writes contract artifacts |
| Validated-only downstream gate | `scripts/run_full_research_pipeline.py::prepare_validated_research_artifact` | This is the core promotion rule from raw/audit artifacts to research-grade inputs |
| Artifact schema validation | `scripts/run_full_research_pipeline.py::validate_pipeline_artifacts` | Prevents downstream work from silently running on malformed outputs |
| Timing normalization helpers | `src/events/timing.py` | Prevents duplicate, inconsistent BMO/AMC inference logic |
| Trading-session alignment | `src/events/event_alignment.py::align_event_to_sessions` | This is the canonical session-mapping logic and is heavily tested |
| Earnings date retrieval wrapper | `data/earnings_calendar.py::get_earnings_dates` | Preserves the repo's retrieval-error semantics |
| Event-time option implied move semantics | `scripts/run_event_pipeline_test.py::attach_event_time_option_implied_moves` | This is the implemented event-level mapping used by the canonical pipeline |
| Raw option-chain selection helper | `src/features/option_selection.py::select_event_options` | Use this when future work starts from raw chains instead of pre-keyed event option rows |
| Event diagnostics | `src/earnings_event_engine/validation/event_diagnostics.py::run_event_diagnostics` | This is the actual event validity contract |
| Coverage report | `src/earnings_event_engine/validation/data_coverage.py::build_data_coverage_report` | Reuse for coverage and completeness audits |
| Event quality score | `src/earnings_event_engine/validation/data_quality_score.py::score_event_data_quality` | Reuse for non-binary quality ranking |
| Pipeline health summary | `src/validation/pipeline_health.py::compute_pipeline_health` | Reuse for run-level health scoring |
| Historical IV containment and timing-safe analysis | `run_analysis.py` | Reuse its event-time IV handling instead of reintroducing historical IV leakage |
| Earnings analysis summaries | `analysis/earnings_analysis.py` | Reuse for consistent realized-vs-implied summary metrics |
| Straddle research proxy | `strategies/straddle_simulator.py` | Reuse for simple option-backed event studies rather than re-deriving the same proxy |
| Vertical spread research proxy | `strategies/vertical_spread_simulator.py` | Reuse if testing capped-payoff option abstractions |
| Conditional high-IV/low-IV analysis | `analysis/conditional_analysis.py` | Reuse for simple regime segmentation on strategy outputs |
| PEAD forward-close derivation and summary | `src/earnings_event_engine/research/pead.py` | This is the current PEAD research contract |
| Event debug snapshot writer | `src/earnings_event_engine/debug/event_snapshot.py::dump_event_snapshot` | Reuse for per-event forensic output instead of inventing new debug formats |
| Datetime normalization | `src/utils/datetime_utils.py` and compatibility re-export `utils/datetime_utils.py` | Reuse to keep timezone behavior consistent |
| Demo notebook | `notebooks/01_earnings_event_engine_demo.ipynb` | Reuse for agent or teammate orientation |

### What future work should not treat as reusable system foundations

- `data/loader.py`
- `features/engineering.py`
- `analysis/event_analysis.py`
- `strategies/base_strategy.py`
- `backtest/engine.py`

These are placeholders and should either be ignored or replaced deliberately.

## 10. Known Gaps / Technical Debt

### Genuine blockers

- There is no real historical event-time options dataset or options data provider implementation in the repo.
- There is no real backtest engine for event strategies with realistic fills, positions, portfolio state, or benchmarks.
- There is no broker or execution layer.
- There is no canonical signal/ranked-candidate output contract yet.
- Live research depends directly on `yfinance`, which is not a robust research data backend.

### Worthwhile improvements

- Mark placeholder modules more aggressively or move them out of the apparent main path.
- Consolidate the repo's mixed import/layout style between top-level modules and `src/earnings_event_engine`.
- Promote schema definitions into a dedicated, versioned contract module.
- Integrate pipeline health and event quality score into the canonical CLI outputs.
- Add richer deterministic fixtures with more events and more failure modes.
- Add tests for non-canonical helpers such as `src/features/event_features.py`, `src/features/option_selection.py`, `analysis/event_filtering.py`, and `strategies/vertical_spread_simulator.py`.
- Consider auto-emitting debug snapshots on validation failures.

### Optional enhancements

- Expand the notebook set beyond the single canonical demo.
- Add richer reporting tables and charts as optional outputs.
- Add more explicit artifact versioning and metadata manifests.
- Add local caching/persistence for live retrieval runs.

### Deliberately out of scope for this repo's current shape

- Merging SEC/fundamental/filing ML directly into this codebase.
- Pretending the placeholder backtest and strategy modules are production components.
- Building a generalized multi-strategy portfolio platform before extracting more value from the existing event pipeline.

### Stale or contradictory documentation / assumptions found

- `analysis/insights.md` is stale. It describes a zero-record state and says no conclusions can be drawn, but the repository now contains a deterministic fixture path, passing tests, and a working canonical offline research run.
- Several placeholder modules can create a false impression that the repo has a complete strategy/backtest stack when it does not.

## 11. Integration Contract for Future Alpha Research

This section contains recommended integration rules. They are not yet enforced by code, but they are the cleanest path that fits what already exists.

### Preferred flow

```text
universe
  -> earnings events
  -> market data
  -> validated event features
  -> signal logic
  -> ranked candidates
  -> research evidence
  -> optional strategy proxy
  -> future risk/execution layer
```

### What future modules should consume

Default starting point:

- `validated_features.parquet`

Use this when the research question is about:

- implied vs realized earnings moves
- option-overpricing/event-volatility studies
- candidate ranking on event features
- returns-based ML on validated earnings events

Alternative starting point:

- `clean_events.parquet` plus a local price panel

Use this when the research question needs:

- forward closes and PEAD-style drift windows
- custom future return horizons
- additional post-event price path features

### Minimum recommended input schema for future research modules

For a module consuming `validated_features.parquet`, assume these fields are canonical:

- `event_id`
- `ticker`
- `event_date`
- `announcement_timing`
- `mapped_session`
- `pre_close`
- `post_open`
- `post_close`
- `price_alignment_valid`
- `implied_move`
- `implied_move_available`
- `implied_move_source`
- `hv20_expected_move_proxy`
- `realized_gap_move`
- `realized_close_move`
- `iv_minus_realized_gap`
- `surprise_vs_implied`
- `days_since_prev_event`
- `macro_regime`
- `valid_event`
- `invalid_event`
- `failure_reasons`

### Minimum recommended output schema for a new alpha research module

This is recommended, not yet enforced:

| Field | Meaning |
| --- | --- |
| `event_id` | Join key back to canonical validated events |
| `ticker` | Symbol |
| `event_date` | Event date |
| `signal_name` | Stable research signal identifier |
| `signal_version` | Version string or git-derived version |
| `signal_value` | Raw signal number |
| `signal_score` | Normalized score used for ranking |
| `signal_direction` | `long_vol`, `short_vol`, `long`, `short`, `neutral`, etc. |
| `entry_basis` | Session or timing basis for assumed entry |
| `exit_horizon_days` | Holding horizon |
| `research_subset` | Description of any filtered subset |
| `source_artifact` | Upstream artifact path/name, typically `validated_features.parquet` |
| `notes` | Short explanation of assumptions or exclusions |

If a module produces ranked candidates, add:

- `rank`
- `rank_group`
- `selected_candidate`

If a module produces evaluation outputs, add:

- `target_return`
- `target_definition`
- `evidence_metric`
- `evidence_window`

### How signals should be represented

Recommended representation:

- one row per `event_id`
- raw signal column(s)
- one normalized `signal_score`
- explicit `signal_direction`
- explicit horizon and entry basis

Do not hide signal semantics in notebook-only variables or image outputs. Emit a parquet or csv artifact.

### How research should be promoted into the existing system

1. Start as a notebook or one-purpose script consuming `validated_features.parquet` or `clean_events.parquet`.
2. Emit a stable event-level artifact with the recommended schema above.
3. Add a deterministic fixture or unit test proving the module respects timing and point-in-time assumptions.
4. If the module is stable and generally useful, wrap it as a script under `scripts/`.
5. Only after the artifact contract is stable should it be added as an optional or required stage in `scripts/run_full_research_pipeline.py`.

### Guidance by future research area

Mean reversion:

- Reuse `validated_features.parquet`, session alignment, and price-validity gating.
- Add pre-event return/liquidity/volatility features rather than rebuilding event normalization.

PEAD / event-driven alpha:

- Reuse `clean_events.parquet` plus `add_forward_close_columns()`, `calculate_announcement_reaction()`, and `calculate_forward_drift()`.

Volatility / VIX / term-structure / options EV:

- Reuse event-time implied-move attachment and the validated-only downstream boundary.
- Extend from real historical option snapshots; do not reuse current chains for historical events.

Returns-based ML:

- Start from `validated_features.parquet`.
- Add lagged price/volume/volatility features in a new module or by adapting `src/features/event_features.py`.

Screener development:

- Build lightweight event-level ranking scripts on top of `validated_features.parquet`.
- Do not first build a generalized screener platform.

## 12. Relationship to Other Projects

This repo does not currently depend on an SEC/fundamental/filing ML stack.

That separation should be preserved:

- Keep SEC/fundamental/filing ML as a separate system.
- Integrate via clean artifacts, not by copying that project into this repo.
- The clean interface should be event-level features, signals, or results keyed by `ticker`, `event_date`, and ideally `event_id` when the event mapping is shared.

Price/returns-based ML is in scope here because it can naturally consume `validated_features.parquet` and adjacent local price panels. That should not be conflated with SEC/fundamental ML.

## 13. Recommended Next Uses

These are the highest-leverage things that can be built quickly from what already exists.

### Best immediate next uses

1. Build a small event-ranking notebook or script on top of `validated_features.parquet`.
   Use `surprise_vs_implied`, `realized_gap_move`, `announcement_timing`, and optional macro regime tags.
2. Extend PEAD research using `clean_events.parquet` plus a local price panel.
   The forward-close derivation and bucketing logic already exists.
3. Build a historical option-snapshot ingestion adapter that outputs the same event-level options contract as `tests/fixtures/offline_pipeline/options.csv`.
   This would unlock much more realistic volatility/event research without changing downstream modules.
4. Build returns-based ML datasets starting from `validated_features.parquet`.
   Add lagged price/volume/volatility features instead of rebuilding the event pipeline.
5. Compare alternate option-event strategy abstractions by reusing `strategies/straddle_simulator.py` and `strategies/vertical_spread_simulator.py`.

### Likely duplication to avoid

| Research area | What would be duplicated if rebuilt from scratch | Reuse instead |
| --- | --- | --- |
| Mean reversion | Event timing, session alignment, price window extraction | `src/events/*`, `scripts/run_event_pipeline_test.py`, `validated_features.parquet` |
| PEAD | Forward-close derivation and reaction bucketing | `src/earnings_event_engine/research/pead.py`, `clean_events.parquet` |
| Volatility/options research | Event-time implied-move semantics and missing-data behavior | `attach_event_time_option_implied_moves()`, `analysis/earnings_analysis.py`, `strategies/straddle_simulator.py` |
| Returns-based ML | Validated event set construction and key columns | `validated_features.parquet`, `src/features/event_features.py` as needed |
| Screener development | Candidate ranking on clean event features | `validated_features.parquet`, `analysis/event_filtering.py` as a lightweight helper |

### Recommended immediate next step

The single best next investment is a historical event-time options ingestion layer that emits the existing fixture-style option contract at scale. That would preserve all current downstream logic while upgrading the repo from plumbing validation into real volatility/event research.

## 14. Machine-Readable Quick Reference

```yaml
as_of: 2026-08-18
canonical_entry_points:
  event_pipeline: scripts/run_event_pipeline_test.py
  full_research: scripts/run_full_research_pipeline.py
  pead: scripts/run_pead_research.py
  demo_notebook: notebooks/01_earnings_event_engine_demo.ipynb
canonical_input_artifacts:
  fixture_events: tests/fixtures/offline_pipeline/earnings_events.csv
  fixture_prices: tests/fixtures/offline_pipeline/prices.csv
  fixture_options: tests/fixtures/offline_pipeline/options.csv
canonical_output_artifacts:
  raw_events: events.parquet
  clean_events: clean_events.parquet
  feature_audit: features.parquet
  validated_research_input: validated_features.parquet
  diagnostics: diagnostics.parquet
key_modules:
  timing: src/events/timing.py
  alignment: src/events/event_alignment.py
  diagnostics: src/earnings_event_engine/validation/event_diagnostics.py
  coverage: src/earnings_event_engine/validation/data_coverage.py
  quality_score: src/earnings_event_engine/validation/data_quality_score.py
  pipeline_health: src/validation/pipeline_health.py
  earnings_analysis: analysis/earnings_analysis.py
  straddle_proxy: strategies/straddle_simulator.py
  pead: src/earnings_event_engine/research/pead.py
  historical_analysis: run_analysis.py
reusable_utilities:
  - scripts/run_event_pipeline_test.py::attach_event_time_option_implied_moves
  - scripts/run_full_research_pipeline.py::prepare_validated_research_artifact
  - src/features/option_selection.py::select_event_options
  - src/earnings_event_engine/debug/event_snapshot.py::dump_event_snapshot
risk_layer:
  enforced:
    - zero-raw-event failure
    - zero-valid-event failure
    - artifact schema validation
    - validated-only downstream gate
    - no historical IV leakage
  missing:
    - broker risk controls
    - position sizing
    - exposure limits
    - drawdown controls
integration_contract:
  consume_default: validated_features.parquet
  consume_for_pead: clean_events.parquet plus local price panel
  join_key: event_id
  required_research_output_fields:
    - event_id
    - ticker
    - event_date
    - signal_name
    - signal_version
    - signal_value
    - signal_score
    - signal_direction
current_blockers:
  - no historical options snapshot store
  - no real backtest engine
  - no execution layer
  - no canonical signal artifact yet
recommended_immediate_next_step:
  build a historical event-time options ingestion layer that emits the existing event-level option contract
validation_summary:
  tests_passed: 95
  canonical_fixture_run: success
  pead_validation_run: success
stale_docs:
  - analysis/insights.md
do_not_rebuild:
  - timing normalization
  - session alignment
  - event diagnostics
  - validated-only artifact gate
  - event-time implied-move attachment
```
