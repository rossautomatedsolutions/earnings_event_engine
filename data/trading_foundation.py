"""Canonical Trading_Foundation historical-data adapters for earnings-event research."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import re
from typing import Iterable

import pandas as pd

from src.events import align_event_to_sessions
from src.events.timing import normalize_announcement_timing
from src.features.option_selection import select_event_options

_DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")
_OCC_PATTERN = re.compile(
    r"^(?P<underlying>[A-Z]{1,6})(?P<expiry>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$"
)

EOD_REQUIRED_COLUMNS = {"t", "open", "close"}
OPTION_SNAPSHOT_REQUIRED_COLUMNS = {"option_symbol", "bid_price", "ask_price"}
HISTORICAL_OPTION_EVENT_COLUMNS = [
    "event_id",
    "option_type",
    "strike",
    "underlying_price",
    "bid",
    "ask",
    "mid",
    "option_symbol",
    "expiration",
    "option_snapshot_quote_date",
    "option_snapshot_source_path",
]


def _validate_historical_data_root(historical_data_root: str | Path) -> Path:
    root = Path(historical_data_root)
    if not root.exists():
        raise FileNotFoundError(f"Trading_Foundation historical_data root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Trading_Foundation historical_data root is not a directory: {root}")
    return root


def _parse_timestamp_series(timestamp_series: pd.Series) -> pd.Series:
    try:
        parsed = pd.to_datetime(timestamp_series, errors="coerce", utc=True, format="mixed")
    except TypeError:
        parsed = pd.to_datetime(timestamp_series, errors="coerce", utc=True)

    missing_mask = parsed.isna()
    if not missing_mask.any():
        return parsed

    unresolved = timestamp_series[missing_mask]
    numeric = pd.to_numeric(unresolved, errors="coerce")
    numeric_mask = numeric.notna()
    if not numeric_mask.any():
        return parsed

    best_candidate = None
    best_success = -1
    for unit in ("s", "ms", "us", "ns"):
        candidate = pd.to_datetime(numeric[numeric_mask], errors="coerce", unit=unit, utc=True)
        success = int(candidate.notna().sum())
        if success > best_success:
            best_success = success
            best_candidate = candidate

    if best_candidate is not None and best_success > 0:
        parsed.loc[numeric[numeric_mask].index] = best_candidate
    return parsed


def _parse_occ_option_symbol(symbol: object) -> dict[str, object] | None:
    text = str(symbol or "").strip().upper()
    if not text:
        return None

    match = _OCC_PATTERN.match(text)
    if not match:
        return None

    try:
        expiration = pd.to_datetime(match.group("expiry"), format="%y%m%d", errors="coerce")
        if pd.isna(expiration):
            return None
        strike = int(match.group("strike")) / 1000.0
        option_type = "call" if match.group("cp") == "C" else "put"
    except Exception:
        return None

    return {
        "underlying": match.group("underlying"),
        "expiration": pd.Timestamp(expiration).normalize(),
        "strike": float(strike),
        "option_type": option_type,
    }


def _empty_historical_options_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=HISTORICAL_OPTION_EVENT_COLUMNS)


def _resolve_eod_path(root: Path, ticker: str) -> Path | None:
    exact = root / ticker / "eod" / f"{ticker}_1D.csv"
    if exact.exists():
        return exact

    candidates = sorted((root / ticker / "eod").glob("*_1D.csv"))
    return candidates[0] if candidates else None


def _load_local_eod_history(root: Path, ticker: str) -> pd.DataFrame:
    path = _resolve_eod_path(root, ticker)
    if path is None or not path.exists():
        return pd.DataFrame(columns=["open", "close"])

    history = pd.read_csv(path)
    missing = sorted(EOD_REQUIRED_COLUMNS.difference(history.columns))
    if missing:
        raise ValueError(f"Trading_Foundation EOD file {path} is missing required columns: {missing}")

    parsed_timestamps = _parse_timestamp_series(history["t"])
    if parsed_timestamps.isna().all():
        raise ValueError(f"Trading_Foundation EOD file {path} has no parseable timestamps in column 't'.")

    out = pd.DataFrame(
        {
            "date": parsed_timestamps.dt.tz_convert(None).dt.normalize(),
            "open": pd.to_numeric(history["open"], errors="coerce"),
            "close": pd.to_numeric(history["close"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last").sort_values("date")
    if out.empty:
        return pd.DataFrame(columns=["open", "close"])

    indexed = out.set_index("date")[["open", "close"]]
    indexed.index = pd.to_datetime(indexed.index, errors="raise")
    return indexed.sort_index()


def load_trading_foundation_eod_panel(
    historical_data_root: str | Path,
    tickers: Iterable[str],
) -> pd.DataFrame:
    """Load canonical Trading_Foundation EOD bars into the pipeline's price-panel shape."""
    root = _validate_historical_data_root(historical_data_root)
    frames: list[pd.DataFrame] = []
    seen: set[str] = set()

    for raw_ticker in tickers:
        ticker = str(raw_ticker).strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)

        history = _load_local_eod_history(root, ticker)
        if history.empty:
            continue

        frame = history.reset_index().rename(columns={"index": "date"})
        frame.insert(0, "ticker", ticker)
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "open", "close"])

    out = pd.concat(frames, ignore_index=True)
    out["ticker"] = out["ticker"].astype("string").str.upper().str.strip()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["open"] = pd.to_numeric(out["open"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    return out.dropna(subset=["ticker", "date"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def _resolve_option_snapshot_path(root: Path, ticker: str, snapshot_date: pd.Timestamp) -> Path | None:
    snapshot_dir = root / ticker / "option_snapshot"
    if not snapshot_dir.exists():
        return None

    iso_date = snapshot_date.date().isoformat()
    exact = snapshot_dir / f"{ticker}_options_snapshot_{iso_date}.csv"
    if exact.exists():
        return exact

    matches = sorted(snapshot_dir.glob(f"*options_snapshot*{iso_date}*.csv"))
    return matches[0] if matches else None


def _snapshot_date_from_path(path: Path) -> pd.Timestamp | pd.NaT:
    match = _DATE_PATTERN.search(path.name)
    if not match:
        return pd.NaT
    return pd.Timestamp(match.group(1)).normalize()


def _prepare_option_snapshot_chain(snapshot_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(snapshot_path)
    missing = sorted(OPTION_SNAPSHOT_REQUIRED_COLUMNS.difference(raw.columns))
    if missing:
        raise ValueError(
            f"Trading_Foundation option snapshot {snapshot_path} is missing required columns: {missing}"
        )

    deduped = raw.drop_duplicates(subset=["option_symbol"], keep="last").copy()
    parsed = deduped["option_symbol"].map(_parse_occ_option_symbol)
    valid_mask = parsed.notna()
    if not valid_mask.any():
        return pd.DataFrame(
            columns=[
                "option_symbol",
                "expiration",
                "strike",
                "option_type",
                "bid",
                "ask",
                "mid",
                "delta",
                "implied_volatility",
                "option_snapshot_quote_date",
                "option_snapshot_source_path",
            ]
        )

    working = deduped.loc[valid_mask].copy()
    parsed_df = parsed.loc[valid_mask].apply(pd.Series)
    working["expiration"] = pd.to_datetime(parsed_df["expiration"], errors="coerce").dt.normalize()
    working["strike"] = pd.to_numeric(parsed_df["strike"], errors="coerce")
    working["option_type"] = parsed_df["option_type"].astype("string").str.lower()
    working["bid"] = pd.to_numeric(working["bid_price"], errors="coerce")
    working["ask"] = pd.to_numeric(working["ask_price"], errors="coerce")
    working["mid"] = (
        pd.to_numeric(working["last_trade_price"], errors="coerce")
        if "last_trade_price" in working.columns
        else pd.NA
    )
    working["delta"] = pd.to_numeric(working["delta"], errors="coerce") if "delta" in working.columns else pd.NA
    working["implied_volatility"] = (
        pd.to_numeric(working["implied_volatility"], errors="coerce")
        if "implied_volatility" in working.columns
        else pd.NA
    )
    working["option_snapshot_quote_date"] = _snapshot_date_from_path(snapshot_path)
    working["option_snapshot_source_path"] = str(snapshot_path)
    return working.dropna(subset=["expiration", "strike", "option_type"]).reset_index(drop=True)


def _resolve_pre_event_snapshot_date(
    row: pd.Series,
    history: pd.DataFrame,
) -> pd.Timestamp | None:
    explicit_pre_event_date = pd.to_datetime(row.get("pre_event_date"), errors="coerce")
    if pd.notna(explicit_pre_event_date):
        return pd.Timestamp(explicit_pre_event_date).normalize()

    event_date = pd.to_datetime(row.get("event_date"), errors="coerce")
    if pd.isna(event_date) or history.empty:
        return None

    timing = normalize_announcement_timing(row.get("announcement_timing", row.get("timing")))
    if timing is None:
        return None

    alignment = align_event_to_sessions(
        event_date=pd.Timestamp(event_date),
        announcement_timing=timing,
        price_index=history.index,
    )
    if alignment.get("alignment_flag") != "ok":
        return None

    return pd.Timestamp(alignment["pre_event_date"]).normalize()


def _extract_selected_leg_rows(chain: pd.DataFrame, snapshot: dict[str, object]) -> tuple[pd.Series, pd.Series] | None:
    selected_expiry = pd.to_datetime(snapshot.get("selected_expiry"), errors="coerce")
    atm_strike = pd.to_numeric(pd.Series([snapshot.get("atm_strike")]), errors="coerce").iloc[0]
    if pd.isna(selected_expiry) or pd.isna(atm_strike):
        return None

    strike_group = chain.loc[
        (pd.to_datetime(chain["expiration"], errors="coerce").dt.normalize() == selected_expiry.normalize())
        & (pd.to_numeric(chain["strike"], errors="coerce") == float(atm_strike))
    ].copy()
    if strike_group.empty:
        return None

    call_rows = strike_group.loc[strike_group["option_type"].astype("string").str.lower() == "call"]
    put_rows = strike_group.loc[strike_group["option_type"].astype("string").str.lower() == "put"]
    if call_rows.empty or put_rows.empty:
        return None

    return call_rows.iloc[0], put_rows.iloc[0]


def load_trading_foundation_options_for_events(
    events_with_prices: pd.DataFrame,
    historical_data_root: str | Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Map canonical option_snapshot files into the repo's existing event-level option contract."""
    report = {
        "requested_events": int(len(events_with_prices)),
        "events_with_selected_options": 0,
        "emitted_option_rows": 0,
        "missing_snapshot_file_events": 0,
        "missing_eod_history_events": 0,
        "invalid_underlying_price_events": 0,
        "alignment_failed_events": 0,
        "empty_or_unparseable_snapshot_events": 0,
        "selection_failed_events": 0,
    }
    if events_with_prices is None or events_with_prices.empty:
        return _empty_historical_options_frame(), report

    required = {"event_id", "ticker", "event_date", "announcement_timing", "pre_close"}
    missing = sorted(required.difference(events_with_prices.columns))
    if missing:
        raise ValueError(
            "events_with_prices is missing required columns for Trading_Foundation option loading: "
            f"{missing}"
        )

    root = _validate_historical_data_root(historical_data_root)
    emitted_rows: list[dict[str, object]] = []
    history_cache: dict[str, pd.DataFrame] = {}
    snapshot_cache: dict[tuple[str, str], pd.DataFrame] = {}

    for _, row in events_with_prices.iterrows():
        ticker = str(row["ticker"]).strip().upper()
        history = history_cache.get(ticker)
        if history is None:
            history = _load_local_eod_history(root, ticker)
            history_cache[ticker] = history

        if history.empty:
            report["missing_eod_history_events"] += 1
            continue

        snapshot_date = _resolve_pre_event_snapshot_date(row, history)
        if snapshot_date is None:
            report["alignment_failed_events"] += 1
            continue

        underlying_price = pd.to_numeric(pd.Series([row.get("pre_close")]), errors="coerce").iloc[0]
        if pd.isna(underlying_price) or float(underlying_price) <= 0:
            if snapshot_date in history.index:
                underlying_price = pd.to_numeric(
                    pd.Series([history.loc[snapshot_date, "close"]]),
                    errors="coerce",
                ).iloc[0]
        if pd.isna(underlying_price) or float(underlying_price) <= 0:
            report["invalid_underlying_price_events"] += 1
            continue

        snapshot_path = _resolve_option_snapshot_path(root, ticker, snapshot_date)
        if snapshot_path is None:
            report["missing_snapshot_file_events"] += 1
            continue

        cache_key = (ticker, snapshot_path.name)
        chain = snapshot_cache.get(cache_key)
        if chain is None:
            chain = _prepare_option_snapshot_chain(snapshot_path)
            snapshot_cache[cache_key] = chain

        if chain.empty:
            report["empty_or_unparseable_snapshot_events"] += 1
            continue

        selection = select_event_options(
            chain_df=chain,
            underlying_price=float(underlying_price),
            event_date=row["event_date"],
        )
        selected_rows = _extract_selected_leg_rows(chain, selection["snapshot"])
        if selected_rows is None:
            report["selection_failed_events"] += 1
            continue

        call_row, put_row = selected_rows
        for leg_row in (call_row, put_row):
            emitted_rows.append(
                {
                    "event_id": row["event_id"],
                    "option_type": leg_row["option_type"],
                    "strike": float(leg_row["strike"]),
                    "underlying_price": float(underlying_price),
                    "bid": pd.to_numeric(pd.Series([leg_row.get("bid")]), errors="coerce").iloc[0],
                    "ask": pd.to_numeric(pd.Series([leg_row.get("ask")]), errors="coerce").iloc[0],
                    "mid": pd.to_numeric(pd.Series([leg_row.get("mid")]), errors="coerce").iloc[0],
                    "option_symbol": leg_row.get("option_symbol"),
                    "expiration": pd.to_datetime(leg_row.get("expiration"), errors="coerce"),
                    "option_snapshot_quote_date": pd.to_datetime(
                        leg_row.get("option_snapshot_quote_date"),
                        errors="coerce",
                    ),
                    "option_snapshot_source_path": leg_row.get("option_snapshot_source_path"),
                }
            )

        report["events_with_selected_options"] += 1

    report["emitted_option_rows"] = len(emitted_rows)
    if not emitted_rows:
        return _empty_historical_options_frame(), report

    options_df = pd.DataFrame(emitted_rows, columns=HISTORICAL_OPTION_EVENT_COLUMNS)
    return options_df, report
