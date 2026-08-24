"""Research modules for earnings-event datasets."""

from .pead import (
    DEFAULT_WINDOWS,
    assess_signal_readiness,
    add_forward_close_columns,
    add_pre_event_context_columns,
    assign_reaction_bucket,
    build_pead_condition_buckets,
    build_pead_data_coverage_report,
    build_pead_research_dataset,
    calculate_announcement_reaction,
    calculate_forward_drift,
    resolve_pead_condition_sources,
    summarize_pead_by_bucket,
    summarize_pead_by_condition,
    summarize_pead_by_window,
)

__all__ = [
    "DEFAULT_WINDOWS",
    "assess_signal_readiness",
    "add_forward_close_columns",
    "add_pre_event_context_columns",
    "assign_reaction_bucket",
    "build_pead_condition_buckets",
    "build_pead_data_coverage_report",
    "build_pead_research_dataset",
    "calculate_announcement_reaction",
    "calculate_forward_drift",
    "resolve_pead_condition_sources",
    "summarize_pead_by_bucket",
    "summarize_pead_by_condition",
    "summarize_pead_by_window",
]
