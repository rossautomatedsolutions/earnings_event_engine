# PEAD Research Summary

Research-only artifact. This output extends the repo's validated PEAD path with forward-window outcomes, conditioning, coverage, and a conservative readiness assessment. It is not a strategy, recommendation, or live trading system.

## Inputs

- Event artifact: `outputs\technical_handoff_full_research\validated_features.parquet`
- Price source: `tests\fixtures\trading_foundation_historical_data`
- Event rows received: `2`
- Drift windows: `[1, 3, 5, 10, 20]`
- Neutral threshold: `0.0`
- Minimum signal-review event count: `30`

## Key Findings

- 1d coverage: 1 usable events with mean forward return 0.0039 and positive-rate 100.00%.
- Positive reactions: 1 events at 1d with mean drift 0.0039.
- Signal readiness: sample coverage is still too thin for a defensible watchlist or event-driven export.

## Coverage

| coverage_group | metric | source_column | available_events | total_events | coverage_rate | note |
| --- | --- | --- | --- | --- | --- | --- |
| dataset | input_events | event_id | 2 | 2 | 1.000000 | Rows received by the PEAD research stage. |
| dataset | research_valid_events | valid_event|price_alignment_valid | 2 | 2 | 1.000000 | Rows passing validated-event and price-alignment gates. |
| event_metric | reaction_return | reaction_return | 2 | 2 | 1.000000 | Close-to-close announcement reaction coverage. |
| event_metric | gap_return | gap_return | 2 | 2 | 1.000000 | Pre-close to post-open gap coverage. |
| event_metric | post_session_return | post_session_return | 2 | 2 | 1.000000 | Post-open to post-close announcement-session follow-through coverage. |
| window | forward_close_1d | forward_close_1d | 1 | 2 | 0.500000 | Events with usable forward close data for this horizon. |
| window | forward_return_1d | forward_return_1d | 1 | 2 | 0.500000 | Events with usable forward drift returns for this horizon. |
| window | forward_close_3d | forward_close_3d | 0 | 2 | 0.000000 | Events with usable forward close data for this horizon. |
| window | forward_return_3d | forward_return_3d | 0 | 2 | 0.000000 | Events with usable forward drift returns for this horizon. |
| window | forward_close_5d | forward_close_5d | 0 | 2 | 0.000000 | Events with usable forward close data for this horizon. |
| window | forward_return_5d | forward_return_5d | 0 | 2 | 0.000000 | Events with usable forward drift returns for this horizon. |
| window | forward_close_10d | forward_close_10d | 0 | 2 | 0.000000 | Events with usable forward close data for this horizon. |
| window | forward_return_10d | forward_return_10d | 0 | 2 | 0.000000 | Events with usable forward drift returns for this horizon. |
| window | forward_close_20d | forward_close_20d | 0 | 2 | 0.000000 | Events with usable forward close data for this horizon. |
| window | forward_return_20d | forward_return_20d | 0 | 2 | 0.000000 | Events with usable forward drift returns for this horizon. |
| condition | reaction_bucket | reaction_return | 2 | 2 | 1.000000 | Close-to-close announcement reaction. |
| condition | gap_bucket | gap_return | 2 | 2 | 1.000000 | Pre-close to post-open announcement gap. |
| condition | timing_bucket | announcement_timing | 2 | 2 | 1.000000 | BMO vs AMC timing bucket. |
| condition | implied_move_bucket | implied_move | 2 | 2 | 1.000000 | Quantile bucket of event-time implied move. |
| condition | realized_vs_implied_bucket | surprise_vs_implied | 2 | 2 | 1.000000 | Sign of realized gap move minus implied move. |

Coverage table truncated to 20 rows in markdown. Full detail is in `outputs\technical_handoff_pead_phase2\pead_data_coverage.csv`.

## Baseline By Window

| window_days | event_count | mean_forward_return | median_forward_return | positive_rate | negative_rate | mean_abs_forward_return | mean_reaction_return | mean_gap_return |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000000 | 1.000000 | 0.003880 | 0.003880 | 1.000000 | 0.000000 | 0.003880 | 0.005854 | 0.003415 |

## Reaction Buckets

| reaction_bucket | window_days | event_count | mean_forward_return | median_forward_return | positive_rate |
| --- | --- | --- | --- | --- | --- |
| positive_reaction | 1 | 1 | 0.003880 | 0.003880 | 1.000000 |

## Conditioning

| condition_name | condition_column | source_column | condition_bucket | window_days | event_count | mean_forward_return | median_forward_return | positive_rate | mean_reaction_return | mean_gap_return |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| announcement_reaction | reaction_bucket | reaction_return | positive_reaction | 1 | 1 | 0.003880 | 0.003880 | 1.000000 | 0.005854 | 0.003415 |
| announcement_timing | timing_bucket | announcement_timing | BMO | 1 | 1 | 0.003880 | 0.003880 | 1.000000 | 0.005854 | 0.003415 |
| gap_reaction | gap_bucket | gap_return | positive_gap | 1 | 1 | 0.003880 | 0.003880 | 1.000000 | 0.005854 | 0.003415 |
| implied_expectation | implied_move_bucket | implied_move | low_expected_move | 1 | 1 | 0.003880 | 0.003880 | 1.000000 | 0.005854 | 0.003415 |
| realized_vs_implied | realized_vs_implied_bucket | surprise_vs_implied | smaller_than_implied | 1 | 1 | 0.003880 | 0.003880 | 1.000000 | 0.005854 | 0.003415 |

## Condition Sources

| condition_column | condition_name | source_column | note |
| --- | --- | --- | --- |
| reaction_bucket | announcement_reaction | reaction_return | Close-to-close announcement reaction. |
| gap_bucket | gap_reaction | gap_return | Pre-close to post-open announcement gap. |
| timing_bucket | announcement_timing | announcement_timing | BMO vs AMC timing bucket. |
| implied_move_bucket | implied_expectation | implied_move | Quantile bucket of event-time implied move. |
| realized_vs_implied_bucket | realized_vs_implied | surprise_vs_implied | Sign of realized gap move minus implied move. |
| volatility_context_bucket | volatility_context | pre_20d_realized_vol | Pre-event realized-vol or hv20 proxy bucket. |
| liquidity_bucket | liquidity_context | pre_20d_avg_dollar_volume | Pre-event average volume or dollar-volume bucket. |
| size_bucket | company_size | unavailable | Company size bucket when market-cap fields are present. |
| earnings_surprise_bucket | fundamental_surprise | unavailable | Sign bucket of fundamental earnings surprise when present. |

## Signal Readiness

| window_days | usable_events | positive_reaction_events | negative_reaction_events | status | reason |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 0 | not_defensible | Only 1 usable events at 1d; require at least 30 before considering watchlist or signal design. |
| 3 | 0 | 0 | 0 | not_defensible | Only 0 usable events at 3d; require at least 30 before considering watchlist or signal design. |
| 5 | 0 | 0 | 0 | not_defensible | Only 0 usable events at 5d; require at least 30 before considering watchlist or signal design. |
| 10 | 0 | 0 | 0 | not_defensible | Only 0 usable events at 10d; require at least 30 before considering watchlist or signal design. |
| 20 | 0 | 0 | 0 | not_defensible | Only 0 usable events at 20d; require at least 30 before considering watchlist or signal design. |

No downstream signal/export is emitted automatically from this phase. A later watchlist or event-driven job should only be considered after reviewing a materially larger validated sample on the main machine.

## Limitations

- Daily/session-level research only. No intraday path reconstruction or intraday options timing is inferred.
- Forward returns are left null when the local price history does not extend far enough for a requested horizon.
- Buckets use simple sign or quantile splits; no aggressive threshold optimization is performed.
- This stage does not emit a tradable signal or scheduled job. It is research-only by design.
- Fundamental earnings surprise conditioning was unavailable because the input artifact did not expose a supported surprise field.
- Company-size conditioning was unavailable because the validated artifact did not expose a market-cap style field.
- Liquidity conditioning requires pre-event volume data from the local price source; it was unavailable or insufficient in this run.
