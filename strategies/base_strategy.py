"""Placeholder strategy definitions."""


def generate_signals(features: list) -> list:
    """Generate placeholder trading signals from feature rows."""
    return ["HOLD" for _ in features]
