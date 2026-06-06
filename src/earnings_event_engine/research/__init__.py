"""Research modules for earnings-event datasets."""

from .pead import (
    DEFAULT_WINDOWS,
    add_forward_close_columns,
    assign_reaction_bucket,
    calculate_announcement_reaction,
    calculate_forward_drift,
    summarize_pead_by_bucket,
)

__all__ = [
    "DEFAULT_WINDOWS",
    "add_forward_close_columns",
    "assign_reaction_bucket",
    "calculate_announcement_reaction",
    "calculate_forward_drift",
    "summarize_pead_by_bucket",
]
