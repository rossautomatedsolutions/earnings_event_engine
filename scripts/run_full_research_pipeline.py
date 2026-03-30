"""Master orchestration script for the full research pipeline."""

import logging
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from analysis.conditional_analysis import run_conditional_analysis
from analysis.earnings_analysis import run_full_analysis
from analysis.visualizations import run_all_plots
from strategies.straddle_simulator import run_straddle_analysis, simulate_straddle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/outputs/test_batch")
FEATURES_PATH = OUTPUT_DIR / "features.parquet"


def run_data_pipeline() -> None:
    """Run the existing small-batch data pipeline script."""
    subprocess.run(["python", "scripts/run_small_batch_test.py"], check=True)


def run_analysis() -> None:
    """Run the full analysis layer on engineered features."""
    run_full_analysis(FEATURES_PATH)


def run_visuals() -> None:
    """Generate all research visualizations."""
    run_all_plots(FEATURES_PATH)


def run_strategy() -> None:
    """Run straddle strategy simulation."""
    run_straddle_analysis(FEATURES_PATH)


def run_conditionals() -> None:
    """Run conditional analysis module."""
    features = pd.read_parquet(FEATURES_PATH)
    strategy_results = simulate_straddle(features)
    run_conditional_analysis(strategy_results)


def _run_step(step_name: str, step_fn) -> None:
    """Execute a pipeline step with resilient error handling and timing."""
    start_time = time.time()
    try:
        logger.info("Starting %s", step_name)
        step_fn()
        elapsed = time.time() - start_time
        logger.info("Completed %s in %.2f seconds", step_name, elapsed)
    except Exception as exc:  # noqa: BLE001 - keep orchestration resilient
        elapsed = time.time() - start_time
        logger.exception("Failed %s after %.2f seconds: %s", step_name, elapsed, exc)


def main() -> None:
    print("\n=== STEP 1: DATA PIPELINE ===")
    _run_step("Data pipeline", run_data_pipeline)

    print("\n=== STEP 2: ANALYSIS ===")
    _run_step("Analysis", run_analysis)

    print("\n=== STEP 3: STRATEGY SIMULATION ===")
    _run_step("Strategy simulation", run_strategy)

    print("\n=== STEP 4: CONDITIONAL ANALYSIS ===")
    _run_step("Conditional analysis", run_conditionals)

    print("\n=== STEP 5: VISUALIZATION ===")
    _run_step("Visualization", run_visuals)

    print("\n=== COMPLETE ===")


if __name__ == "__main__":
    main()
