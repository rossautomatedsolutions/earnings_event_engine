# Earnings Event Analysis Insights

- No statistical conclusions can be drawn yet because the current pipeline processed **0 records**.
- IV overpricing cannot be validated from current results; there are no realized-vs-implied comparisons in the dataset.
- Strategy timing edge cannot be identified because there are no simulated trades with event-window segmentation.
- No repeatable cross-sectional patterns can be extracted (sector, surprise magnitude, liquidity, or term structure) due to empty inputs.

## Next data needed for robust insights

- Event-level implied move before earnings and realized move after earnings.
  This must come from event-time option snapshots, not a current option chain reused across old events.
- Strategy PnL split by entry timing (e.g., T-5, T-1, close-to-open, open-to-close).
- Regime tags (VIX level, market trend, macro-event overlap) to detect conditional performance patterns.

## Historical IV integrity note

- Historical IV-dependent conclusions are unavailable unless event-time option inputs are supplied for each earnings event.
- Any return metric derived from event-time IV in `run_analysis.py` is an expected-move proxy, not observed historical straddle P&L.
- Actual option-return analysis requires event-time option premium snapshots aligned to each earnings event.
- Older outputs that reused a current option snapshot as if it existed at the historical event date should be treated as stale/invalid.
