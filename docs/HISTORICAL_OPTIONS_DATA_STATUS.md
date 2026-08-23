# Historical Options Data Status

Updated on 2026-08-23.

## Repository Purpose

This repository is an earnings-event research pipeline. Its current options use case is narrow and specific:

- align earnings events to pre/post daily trading sessions
- attach a pre-event option-implied move to each event from historical option snapshots
- compare realized gap moves vs event-time implied moves
- route only validated events into downstream research summaries and the simplified straddle proxy

It is not currently a full historical options backtester.

## Actual Historical-Options Requirement

This repo does **not** need a generalized historical options platform.

It currently needs only enough historical options data to answer this question per earnings event:

> What was the pre-event ATM straddle cost for the first expiration after the event?

That requirement maps to:

- `Trading_Foundation/historical_data/<TICKER>/eod`
- `Trading_Foundation/historical_data/<TICKER>/option_snapshot`

It does **not** currently require:

- underlying intraday bars
- `option_ohlc`
- IV surface files
- IV rank/percentile files
- contract lifecycle OHLC exits
- multiple intraday option-chain snapshots

## Exact Field Requirements

Required now by this repo:

| Repo need | Trading Foundation source | Notes |
| --- | --- | --- |
| `ticker` | path segment `<TICKER>` | derived from folder |
| `event_date` | earnings event input | comes from repo event retrieval/fixture |
| `announcement_timing` | earnings event input | used to choose the pre-event session |
| `pre_event_date` | derived from local EOD history + timing alignment | `AMC` uses same-day session, `BMO` uses prior trading session |
| `pre_close` | `eod/*_1D.csv -> close` | used as the underlying anchor for ATM selection and implied move normalization |
| `option_symbol` | `option_snapshot/*.csv -> option_symbol` | OCC symbol parsed into expiration/strike/type |
| `expiration` | derived from `option_symbol` | first expiration strictly after `event_date` |
| `strike` | derived from `option_symbol` | strike nearest `pre_close` |
| `option_type` | derived from `option_symbol` | call / put |
| `bid` | `bid_price` | used for midpoint calculation |
| `ask` | `ask_price` | used for midpoint calculation |
| `mid` | derived from `(bid + ask) / 2`, fallback `last_trade_price` | reused by existing implied-move logic |
| `option_snapshot_quote_date` | snapshot filename date | audit/provenance column |
| `option_snapshot_source_path` | snapshot file path | audit/provenance column |

Observed upstream snapshot columns from the sandbox `schema_summary.json`:

- `option_symbol`
- `delta`
- `gamma`
- `rho`
- `theta`
- `vega`
- `implied_volatility`
- `ask_price`
- `ask_size`
- `bid_price`
- `bid_size`
- `last_trade_price`
- `last_trade_size`
- `last_trade_time`

Only a subset is required by this repo today.

## Temporal Granularity

Current repo requirement:

- daily underlying bars
- once-per-day historical option chain snapshot aligned to the **pre-event trading session**

Current adapter behavior:

- `AMC` event on `YYYY-MM-DD`: use snapshot file dated `YYYY-MM-DD`
- `BMO` event on `YYYY-MM-DD`: use snapshot file dated the prior trading session

What this is sufficient for:

- event-time implied-move research at daily/session granularity
- historical candidate generation for ATM pre-event straddles
- validated event-level research summaries

What this is **not** sufficient for:

- intraday quote-path reconstruction
- timestamp-accurate quote selection within the session
- historical exit-price modeling
- contract lifecycle P&L from option bars

If a future research question requires those, the specific missing requirement is:

- finer quote timing or `option_ohlc` lifecycle coverage, not “historical options data” generically

## Historical Depth

Minimum functional integration:

- one valid pre-event `option_snapshot` file plus local EOD coverage for each event being analyzed

Research sample size requirement:

- depends on how many earnings events the user wants to study on the main machine
- must be audited from the actual `historical_data` root

The adapter is built and tested, but sandbox coverage is not a reliable proxy for main-machine symbol/date coverage.

## Existing Repo Functionality Reused

Reused as-is:

- `scripts/run_event_pipeline_test.py` event normalization and artifact contract
- `src/events/timing.py` timing normalization
- `src/events/event_alignment.py` session alignment
- `scripts/run_event_pipeline_test.py::attach_event_time_option_implied_moves`
- `analysis/earnings_analysis.py`
- `strategies/straddle_simulator.py`
- `scripts/run_full_research_pipeline.py` validated-only downstream gate

The adapter changes the **input source**, not the downstream research semantics.

## Empirical vs Proxy Behavior

Empirical now:

- `implied_move`
- `option_call_mid`
- `option_put_mid`
- `option_straddle_mid`
- `option_snapshot_quote_date`
- `option_snapshot_source_path`

Still proxy-only:

- `hv20_expected_move_proxy`
- straddle simulator P&L

The straddle simulator still compares realized move vs normalized event-time straddle cost. It is **not** a historical entry/exit premium backtest.

## Known Limitations

1. Daily `option_snapshot` files do not by themselves prove sub-session quote timing. They support session-level pre-event alignment, not intraday replay.
2. `option_ohlc` is not currently consumed, so the repo still does not model historical option exits.
3. The sandbox `Trading_Foundation` copy exposed a schema summary but not the full main-machine OneDrive sample files, so final empirical coverage still requires main-machine validation.
4. A file can exist yet still be unusable for this repo if the required pre-event snapshot date is missing or if no valid ATM call/put pair exists after OCC parsing.

## Canonical Main-Machine Root

Use:

```text
C:\Users\RAS\OneDrive\RAS\Trading\Live\Trading_Foundation\historical_data
```

## Exact Main-Machine Validation Command

Best end-to-end validation command:

```powershell
python scripts\run_full_research_pipeline.py --tickers AAPL,MSFT --start-date 2025-01-01 --end-date 2025-12-31 --max-events 20 --historical-data-root C:\Users\RAS\OneDrive\RAS\Trading\Live\Trading_Foundation\historical_data --output-dir outputs\main_machine_tf_validation
```

Pipeline-only validation command:

```powershell
python scripts\run_event_pipeline_test.py --tickers AAPL,MSFT --start-date 2025-01-01 --end-date 2025-12-31 --max-events 20 --historical-data-root C:\Users\RAS\OneDrive\RAS\Trading\Live\Trading_Foundation\historical_data --output-dir outputs\main_machine_tf_pipeline_validation
```

## Expected Output Files

From `run_full_research_pipeline.py`:

- `events.parquet`
- `clean_events.parquet`
- `features.parquet`
- `diagnostics.parquet`
- `validated_features.parquet`

## Main-Machine Sanity Checks

Expect:

1. non-zero `raw_events`, `clean_events`, and `validated_feature_rows`
2. `option_data_source=trading_foundation_option_snapshot` in CLI output
3. at least some rows with `implied_move_available=True`
4. populated `option_snapshot_quote_date` and `option_snapshot_source_path` on rows with `implied_move_available=True`
5. `BMO` rows where `option_snapshot_quote_date` equals `pre_event_date`, not the event date
6. `AMC` rows where `option_snapshot_quote_date` matches the event-date session

Helpful follow-up inspection command:

```powershell
python -c "import pandas as pd; df = pd.read_parquet(r'outputs\main_machine_tf_validation\validated_features.parquet'); cols = ['ticker','event_date','announcement_timing','pre_event_date','option_snapshot_quote_date','option_selected_expiration','implied_move','option_snapshot_source_path']; print(df[cols].head(20).to_string(index=False))"
```

## Failure Signals To Watch For

- `Zero valid/clean events produced during the validation stage`
- `missing_snapshot_file_events` reported in the event-pipeline output
- all `implied_move_available` values are false
- `option_snapshot_quote_date` is null on option-eligible rows
- malformed snapshot schema errors such as missing `option_symbol`, `bid_price`, or `ask_price`
- no valid post-event expiration found after OCC parsing

## What To Paste Back For Review

Paste back:

1. the full CLI summary from the main-machine run
2. the output of the parquet inspection command above
3. any non-zero `missing_snapshot_file_events` or validation failures

## Status

Current classification:

`ADAPTER_BUILT_BUT_COVERAGE_REQUIRES_MAIN_MACHINE_AUDIT`

Why:

- the repo now correctly consumes canonical-style `Trading_Foundation` daily EOD plus daily `option_snapshot` data
- the exact research need is no longer vague
- the remaining unknown is real main-machine symbol/date coverage and session-level snapshot availability, not adapter code shape
