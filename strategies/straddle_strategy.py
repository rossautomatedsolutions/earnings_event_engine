"""Straddle strategy simulation utilities."""

from collections.abc import Mapping


REQUIRED_FIELDS = (
    "call_premium",
    "put_premium",
    "underlying_entry_price",
    "underlying_exit_price",
)


def _to_float(value: object, field_name: str) -> float:
    """Convert input value to float with a field-aware error message."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for '{field_name}': {value!r}") from exc


def _get_value(trade_row: Mapping[str, object], field_name: str) -> float:
    """Read and validate a numeric field from a trade row mapping."""
    if field_name not in trade_row:
        raise KeyError(f"Missing required field '{field_name}' in trade_row")
    return _to_float(trade_row[field_name], field_name)


def simulate_straddle(trade_row: Mapping[str, object]) -> tuple[float, float]:
    """Simulate a long straddle trade and return PnL and percent return.

    Expected ``trade_row`` fields:
    - ``call_premium``: premium paid for the call leg
    - ``put_premium``: premium paid for the put leg
    - ``underlying_entry_price``: underlying price at trade entry
    - ``underlying_exit_price``: underlying price at trade exit

    The model uses a simple payoff approximation:
    - total premium paid = call_premium + put_premium
    - straddle intrinsic value at exit = abs(underlying_exit_price - underlying_entry_price)
    - pnl = intrinsic_value - total_premium
    - return_pct = pnl / total_premium
    """
    if not isinstance(trade_row, Mapping):
        raise TypeError("trade_row must be a mapping (e.g., dict or pandas Series)")

    for field in REQUIRED_FIELDS:
        if field not in trade_row:
            raise KeyError(f"Missing required field '{field}' in trade_row")

    call_premium = _get_value(trade_row, "call_premium")
    put_premium = _get_value(trade_row, "put_premium")
    entry_price = _get_value(trade_row, "underlying_entry_price")
    exit_price = _get_value(trade_row, "underlying_exit_price")

    total_premium = call_premium + put_premium
    if total_premium <= 0:
        raise ValueError("Total premium must be greater than 0 for return calculation")

    intrinsic_value = abs(exit_price - entry_price)
    pnl = intrinsic_value - total_premium
    return_pct = pnl / total_premium
    return pnl, return_pct
