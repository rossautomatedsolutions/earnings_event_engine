"""Data coverage reporting for earnings-event datasets.

This module computes coverage metrics that make weak or biased data pockets obvious.
All APIs are DataFrame-in / DataFrame-out and optionally write report artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


_KNOWN_TIMING_VALUES = {
    "bmo",
    "pre_market",
    "before_open",
    "amc",
    "post_market",
    "after_close",
    "intraday",
    "during_market",
}


@dataclass(frozen=True)
class CoverageReport:
    """Container for coverage report tables."""

    summary: pd.DataFrame
    per_ticker: pd.DataFrame
    per_year: pd.DataFrame
    event_level: pd.DataFrame

    @property
    def by_ticker(self) -> pd.DataFrame:
        return self.per_ticker

    @property
    def by_year(self) -> pd.DataFrame:
        return self.per_year

    def __repr__(self) -> str:
        summary_descriptor = self.summary.shape if hasattr(self.summary, "shape") else type(self.summary)
        return f"CoverageReport(summary={summary_descriptor})"


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _safe_ratio(numerator: pd.Series | int, denominator: pd.Series | int) -> pd.Series:
    num = pd.Series(numerator)
    den = pd.Series(denominator)
    ratio = num.div(den.where(den != 0))
    return ratio.fillna(0.0)


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    as_str = series.astype(str).str.strip().str.lower()
    truthy = as_str.isin({"1", "true", "t", "yes", "y"})
    falsy = as_str.isin({"0", "false", "f", "no", "n", "nan", "none", ""})
    out = pd.Series(False, index=series.index)
    out.loc[truthy] = True
    remainder = ~truthy & ~falsy
    non_missing_remainder = remainder & series.notna()
    out.loc[non_missing_remainder] = series.loc[non_missing_remainder].astype(bool)
    return out


def _infer_price_validity(
    events_df: pd.DataFrame,
    *,
    price_valid_col: str,
    pre_close_col: str,
    post_open_col: str,
    post_close_col: str,
) -> pd.Series:
    if price_valid_col in events_df.columns:
        return _coerce_bool(events_df[price_valid_col])

    pre_close = pd.to_numeric(
        events_df.get(pre_close_col, pd.Series(index=events_df.index, dtype=float)),
        errors="coerce",
    )
    post_open = pd.to_numeric(
        events_df.get(post_open_col, pd.Series(index=events_df.index, dtype=float)),
        errors="coerce",
    )
    post_close = pd.to_numeric(
        events_df.get(post_close_col, pd.Series(index=events_df.index, dtype=float)),
        errors="coerce",
    )

    return pre_close.notna() & post_open.notna() & post_close.notna()


def _infer_options_validity(events_df: pd.DataFrame, *, options_valid_col: str) -> pd.Series:
    if options_valid_col in events_df.columns:
        return _coerce_bool(events_df[options_valid_col])

    if "missing_options_data" in events_df.columns:
        return ~_coerce_bool(events_df["missing_options_data"])

    if "has_options_data" in events_df.columns:
        return _coerce_bool(events_df["has_options_data"])

    return pd.Series(False, index=events_df.index)


def _infer_timing_validity(
    events_df: pd.DataFrame,
    *,
    timing_valid_col: str,
    timing_col: str,
    mapped_session_col: str,
) -> pd.Series:
    if timing_valid_col in events_df.columns:
        return _coerce_bool(events_df[timing_valid_col])

    if "missing_or_unknown_timing" in events_df.columns:
        return ~_coerce_bool(events_df["missing_or_unknown_timing"])

    timing = events_df.get(timing_col, pd.Series(index=events_df.index, dtype=object)).map(_normalize_text)
    mapped = events_df.get(mapped_session_col, pd.Series(index=events_df.index, dtype=object)).map(_normalize_text)

    timing_known = timing.isin(_KNOWN_TIMING_VALUES)
    has_mapping = mapped != ""
    return timing_known & has_mapping


def _build_event_level_table(
    events_df: pd.DataFrame,
    *,
    ticker_col: str,
    event_date_col: str,
    price_valid: pd.Series,
    options_valid: pd.Series,
    timing_valid: pd.Series,
) -> pd.DataFrame:
    event_level = pd.DataFrame(index=events_df.index)
    event_level["ticker"] = (
        events_df.get(ticker_col, pd.Series("UNKNOWN", index=events_df.index, dtype=object))
        .fillna("UNKNOWN")
        .astype(str)
    )
    parsed_dates = pd.to_datetime(
        events_df.get(event_date_col, pd.Series(index=events_df.index, dtype=object)),
        errors="coerce",
    )
    event_level["event_date"] = parsed_dates
    event_level["event_year"] = parsed_dates.dt.year

    event_level["valid_price_data"] = price_valid
    event_level["valid_options_data"] = options_valid
    event_level["valid_timing_classification"] = timing_valid

    event_level["missing_price_data"] = ~event_level["valid_price_data"]
    event_level["missing_options_data"] = ~event_level["valid_options_data"]
    event_level["missing_timing_classification"] = ~event_level["valid_timing_classification"]

    event_level["fully_valid_event"] = (
        event_level["valid_price_data"]
        & event_level["valid_options_data"]
        & event_level["valid_timing_classification"]
    )

    return event_level


def _build_summary_table(event_level: pd.DataFrame) -> pd.DataFrame:
    total_events = len(event_level)
    valid_price = int(event_level["valid_price_data"].sum())
    valid_options = int(event_level["valid_options_data"].sum())
    valid_timing = int(event_level["valid_timing_classification"].sum())
    fully_valid = int(event_level["fully_valid_event"].sum())

    rows = [
        {"metric": "total_events", "count": total_events, "pct_of_events": 1.0 if total_events else 0.0},
        {
            "metric": "events_with_valid_price_data",
            "count": valid_price,
            "pct_of_events": valid_price / total_events if total_events else 0.0,
        },
        {
            "metric": "events_with_valid_options_data",
            "count": valid_options,
            "pct_of_events": valid_options / total_events if total_events else 0.0,
        },
        {
            "metric": "events_with_valid_timing_classification",
            "count": valid_timing,
            "pct_of_events": valid_timing / total_events if total_events else 0.0,
        },
        {
            "metric": "fully_valid_events",
            "count": fully_valid,
            "pct_of_events": fully_valid / total_events if total_events else 0.0,
        },
    ]
    return pd.DataFrame(rows)


def _build_per_ticker_table(event_level: pd.DataFrame) -> pd.DataFrame:
    grouped = event_level.groupby("ticker", dropna=False).agg(
        num_events=("ticker", "size"),
        num_fully_valid_events=("fully_valid_event", "sum"),
        num_missing_options=("missing_options_data", "sum"),
        num_missing_timing=("missing_timing_classification", "sum"),
    )

    grouped["pct_valid_events"] = _safe_ratio(grouped["num_fully_valid_events"], grouped["num_events"])
    grouped["pct_missing_options"] = _safe_ratio(grouped["num_missing_options"], grouped["num_events"])
    grouped["pct_missing_timing"] = _safe_ratio(grouped["num_missing_timing"], grouped["num_events"])

    return grouped.reset_index().sort_values(["pct_valid_events", "num_events", "ticker"], ascending=[True, False, True])


def _build_per_year_table(event_level: pd.DataFrame) -> pd.DataFrame:
    known_years = event_level.dropna(subset=["event_year"]).copy()
    if known_years.empty:
        return pd.DataFrame(
            columns=[
                "event_year",
                "events_per_year",
                "missing_price_per_year",
                "missing_options_per_year",
                "missing_timing_per_year",
                "pct_missing_price",
                "pct_missing_options",
                "pct_missing_timing",
            ]
        )

    grouped = known_years.groupby("event_year", dropna=False).agg(
        events_per_year=("event_year", "size"),
        missing_price_per_year=("missing_price_data", "sum"),
        missing_options_per_year=("missing_options_data", "sum"),
        missing_timing_per_year=("missing_timing_classification", "sum"),
    )

    grouped["pct_missing_price"] = _safe_ratio(grouped["missing_price_per_year"], grouped["events_per_year"])
    grouped["pct_missing_options"] = _safe_ratio(grouped["missing_options_per_year"], grouped["events_per_year"])
    grouped["pct_missing_timing"] = _safe_ratio(grouped["missing_timing_per_year"], grouped["events_per_year"])

    return grouped.reset_index().sort_values("event_year")


def _write_outputs(report: CoverageReport, output_dir: Path, output_formats: tuple[str, ...]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "coverage_summary": report.summary,
        "coverage_per_ticker": report.per_ticker,
        "coverage_per_year": report.per_year,
        "coverage_event_level": report.event_level,
    }

    written_paths: list[Path] = []
    normalized_formats = tuple(fmt.lower() for fmt in output_formats)
    supported = {"csv", "parquet"}
    unsupported = sorted(set(normalized_formats) - supported)
    if unsupported:
        raise ValueError(f"Unsupported output format(s): {unsupported}. Supported formats: {sorted(supported)}")

    for name, table in tables.items():
        if "csv" in normalized_formats:
            csv_path = output_dir / f"{name}.csv"
            table.to_csv(csv_path, index=False)
            written_paths.append(csv_path)
        if "parquet" in normalized_formats:
            parquet_path = output_dir / f"{name}.parquet"
            table.to_parquet(parquet_path, index=False)
            written_paths.append(parquet_path)

    return written_paths


def build_data_coverage_report(
    events_df: pd.DataFrame,
    *,
    ticker_col: str = "ticker",
    event_date_col: str = "event_date",
    price_valid_col: str = "valid_price_data",
    options_valid_col: str = "valid_options_data",
    timing_valid_col: str = "valid_timing_classification",
    pre_close_col: str = "pre_close",
    post_open_col: str = "post_open",
    post_close_col: str = "post_close",
    timing_col: str = "announcement_timing",
    mapped_session_col: str = "mapped_session",
    output_dir: str | Path | None = None,
    output_formats: tuple[str, ...] = ("csv",),
) -> CoverageReport:
    """Build coverage tables for event-, ticker-, and year-level diagnostics.

    Parameters
    ----------
    events_df
        Earnings events DataFrame.
    output_dir
        Optional destination directory for writing tables to CSV/Parquet.
    output_formats
        Sequence containing any of ``"csv"`` and ``"parquet"``.

    Returns
    -------
    CoverageReport
        Structured report containing summary, per-ticker, per-year, and event-level tables.
    """
    if events_df is None:
        raise ValueError("events_df must be provided.")

    working = events_df.copy()

    price_valid = _infer_price_validity(
        working,
        price_valid_col=price_valid_col,
        pre_close_col=pre_close_col,
        post_open_col=post_open_col,
        post_close_col=post_close_col,
    )
    options_valid = _infer_options_validity(working, options_valid_col=options_valid_col)
    timing_valid = _infer_timing_validity(
        working,
        timing_valid_col=timing_valid_col,
        timing_col=timing_col,
        mapped_session_col=mapped_session_col,
    )

    event_level = _build_event_level_table(
        working,
        ticker_col=ticker_col,
        event_date_col=event_date_col,
        price_valid=price_valid,
        options_valid=options_valid,
        timing_valid=timing_valid,
    )

    report = CoverageReport(
        summary=_build_summary_table(event_level),
        per_ticker=_build_per_ticker_table(event_level),
        per_year=_build_per_year_table(event_level),
        event_level=event_level,
    )

    if output_dir is not None:
        _write_outputs(report, Path(output_dir), output_formats)

    return report


__all__: Iterable[str] = ["CoverageReport", "build_data_coverage_report"]
