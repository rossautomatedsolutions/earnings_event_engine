# earnings_event_engine

Lightweight tooling for running and validating the earnings event pipeline, including a targeted offline regression guard for coverage-input consistency.

## Entry Points

- `scripts/run_small_batch_test.py`
- `scripts/run_event_pipeline_test.py`

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
