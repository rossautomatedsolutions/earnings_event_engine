"""Research modules for earnings-event datasets."""

from .pead import (
    DEFAULT_WINDOWS,
    assign_reaction_bucket,
    calculate_announcement_reaction,
    calculate_forward_drift,
    summarize_pead_by_bucket,
)

__all__ = [
    "DEFAULT_WINDOWS",
    "assign_reaction_bucket",
    "calculate_announcement_reaction",
    "calculate_forward_drift",
    "summarize_pead_by_bucket",
]
