"""Global configuration for the earnings_event_engine project."""

from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Standard project folders
DATA_DIR = BASE_DIR / "data"
FEATURES_DIR = BASE_DIR / "features"
ANALYSIS_DIR = BASE_DIR / "analysis"
STRATEGIES_DIR = BASE_DIR / "strategies"
BACKTEST_DIR = BASE_DIR / "backtest"
UTILS_DIR = BASE_DIR / "utils"

# Runtime defaults
DEFAULT_SYMBOL = "AAPL"
DEFAULT_LOOKBACK_DAYS = 252
