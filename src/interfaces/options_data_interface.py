"""Interface for options-derived data access used by earnings event analysis."""

from abc import ABC, abstractmethod


class OptionsDataProvider(ABC):
    """Abstract provider contract for options data around earnings events."""

    @abstractmethod
    def get_atm_straddle_price(self, ticker, event_date):
        """Return ATM straddle price for the ticker and earnings event date."""
        pass

    @abstractmethod
    def get_implied_move(self, ticker, event_date):
        """Return implied move estimate for the ticker and earnings event date."""
        pass
