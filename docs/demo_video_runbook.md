# Earnings Event Engine: Demo Video Runbook

## Purpose

Record the canonical two-to-three-minute, offline walkthrough in [`notebooks/01_earnings_event_engine_demo.ipynb`](../notebooks/01_earnings_event_engine_demo.ipynb). It demonstrates the real public CLI workflow using deterministic local fixtures; it does not make a trading or performance claim.

## Setup and launch

From the repository root:

```powershell
python -m pip install -r requirements.txt
jupyter notebook notebooks\01_earnings_event_engine_demo.ipynb
```

For a clean, non-interactive check of the same notebook:

```powershell
jupyter nbconvert --to notebook --execute notebooks\01_earnings_event_engine_demo.ipynb --inplace --ExecutePreprocessor.timeout=600
```

The notebook finds the repository root itself and creates a fresh temporary output directory per run. It does not call the network or overwrite repository artifacts.

## Data prerequisites

The recorded demo needs only tracked local fixtures:

- `tests/fixtures/offline_pipeline/earnings_events.csv`
- `tests/fixtures/offline_pipeline/prices.csv`
- `tests/fixtures/offline_pipeline/options.csv`
- `tests/fixtures/trading_foundation_historical_data/` for the PEAD-style local-price step

The representative fixture contains two schema-valid events: AAPL (AMC, 2025-01-30) and MSFT (BMO, 2025-02-05). Its explicit call/put mids demonstrate the event-time implied-move contract; they are fixture data, not a claim of broad historical-options coverage.

## Narration outline

| Time | Notebook cell | Suggested narration |
| --- | --- | --- |
| 0:00–0:20 | Title | “The Earnings Event Engine turns earnings inputs into audited research-ready event records. It validates timing, prices, and option context before analysis.” |
| 0:20–0:45 | Representative input contract | “This offline fixture keeps the demo deterministic: two earnings events, daily prices, and explicit pre-event call and put mids.” |
| 0:45–1:15 | Audited workflow | “The notebook runs the documented canonical CLI. It normalizes and aligns events, validates them, emits auditable artifacts, and uses only `validated_features.parquet` for downstream work.” |
| 1:15–2:05 | Event result | “The chart compares the aligned actual gap with the option-implied move reconstructed from the call and put mids. The two concepts remain separate from the HV20 proxy.” |
| 2:05–2:30 | PEAD-style follow-through | “The separate PEAD research step computes local forward-return context and reports coverage. One usable observation is explicitly not enough for a signal.” |
| 2:30–2:50 | Closing takeaway | “The asset is a reproducible validation boundary. With larger audited data, the same artifact contract supports further research without relaxing the guardrails.” |

## Pre-recording checklist

- Run **Restart Kernel and Run All** from the repository root.
- Confirm the canonical-run table shows four required `SUCCESS` stages.
- Confirm the PEAD readiness table says `not_defensible`; that is the intended honest fixture result.
- Keep the notebook at normal browser zoom on a 1080p screen; the seven cells are designed to fit without wide tables or log output.
- Close or hide personal terminals, file panes, and temporary-directory paths before recording.
- Record the fixture disclaimer and the research boundary; do not describe the output as a trading recommendation or backtest.

## Honest limitations

- The demo is fixture-backed and contains two events, so it validates workflow behavior rather than research efficacy.
- The event-time implied move is a call-plus-put midpoint normalized by underlying price. The separate HV20 field is context only, not an implied-move substitute.
- The PEAD-style output is daily/session-level and uses local EOD data. It is not an intraday reconstruction, a live signal, or a tradable export.
- The repository can consume Trading_Foundation-style historical option snapshots for supported research, but this demo does not claim option entry/exit P&L: it has no contract-lifecycle exit pricing model.
