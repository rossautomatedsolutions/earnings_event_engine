"""Entry point for running analysis workflows in earnings_event_engine."""

from analysis.event_analysis import run_event_analysis


def main() -> None:
    """Run the default earnings event analysis workflow."""
    run_event_analysis()


if __name__ == "__main__":
    main()
