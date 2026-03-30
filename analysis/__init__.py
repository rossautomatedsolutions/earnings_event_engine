"""Analysis package for earnings-event research workflows."""

from analysis.conditional_analysis import (
    compare_by_macro_regime,
    compare_by_timing,
    compare_groups,
    filter_high_iv,
    filter_low_iv,
    run_conditional_analysis,
)

from analysis.event_filtering import (
    analyze_characteristics,
    get_bottom_events,
    get_top_events,
    run_event_filtering,
)

__all__ = [
    "filter_high_iv",
    "filter_low_iv",
    "compare_groups",
    "compare_by_timing",
    "compare_by_macro_regime",
    "run_conditional_analysis",
    "get_top_events",
    "get_bottom_events",
    "analyze_characteristics",
    "run_event_filtering",
]
