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
