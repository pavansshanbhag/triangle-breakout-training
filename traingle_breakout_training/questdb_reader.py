"""
questdb_reader.py — fetch candles and annotations from QuestDB via REST API.

QuestDB exposes a /exec endpoint that accepts SQL and returns JSON.
We use requests (no PG wire dependency) so it works with any QuestDB version.
"""

import logging
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlencode

import pandas as pd
import requests

from .config import QUESTDB_BASE_URL, INTERVAL

logger = logging.getLogger(__name__)


# ── Low-level query helper ────────────────────────────────────────────────────

def _query(sql: str) -> pd.DataFrame:
    """
    Execute a SQL query against QuestDB /exec and return a DataFrame.
    Raises on HTTP error or QuestDB error response.
    """
    params = {"query": sql}
    url = f"{QUESTDB_BASE_URL}/exec?{urlencode(params)}"

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    data = resp.json()

    if "error" in data:
        raise RuntimeError(f"QuestDB error: {data['error']}  SQL: {sql}")

    columns = [col["name"] for col in data.get("columns", [])]
    rows    = data.get("dataset", [])

    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows, columns=columns)

    # Parse timestamp columns
    for col in df.columns:
        if col in ("ts", "from_ts", "to_ts", "created_at"):
            df[col] = pd.to_datetime(df[col], utc=True)

    return df


# ── Candle fetcher ────────────────────────────────────────────────────────────

def fetch_candles(
    ticker: str,
    interval: str = INTERVAL,
    from_ts: Optional[datetime] = None,
    to_ts: Optional[datetime] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Return OHLCV candles for a ticker/interval, sorted ascending by ts.

    Columns: ts, ticker, interval, open, high, low, close, volume
    """
    where_clauses = [
        f"ticker = '{ticker}'",
        f"interval = '{interval}'",
        "ts IS NOT NULL",
    ]

    if from_ts:
        from_str = from_ts.strftime("%Y-%m-%dT%H:%M:%S.000000Z")
        where_clauses.append(f"ts >= '{from_str}'")

    if to_ts:
        to_str = to_ts.strftime("%Y-%m-%dT%H:%M:%S.000000Z")
        where_clauses.append(f"ts <= '{to_str}'")

    where = " AND ".join(where_clauses)
    limit_clause = f"LIMIT {limit}" if limit else ""

    sql = f"""
        SELECT ts, ticker, interval, open, high, low, close, volume
        FROM ohlcv
        WHERE {where}
        ORDER BY ts ASC
        {limit_clause}
    """.strip()

    df = _query(sql)
    logger.debug("fetch_candles(%s, %s): %d rows", ticker, interval, len(df))
    return df


def fetch_latest_candles(
    ticker: str,
    n: int,
    interval: str = INTERVAL,
) -> pd.DataFrame:
    """Return the most recent N candles for a ticker (ascending order)."""
    sql = f"""
        SELECT ts, ticker, interval, open, high, low, close, volume
        FROM ohlcv
        WHERE ticker = '{ticker}' AND interval = '{interval}'
        ORDER BY ts DESC
        LIMIT {n}
    """
    df = _query(sql)
    return df.sort_values("ts").reset_index(drop=True)


# ── Annotation fetcher ────────────────────────────────────────────────────────

def fetch_annotations(
    ticker: Optional[str] = None,
    interval: str = INTERVAL,
    label: Optional[str] = None,
) -> pd.DataFrame:
    """
    Return non-deleted annotations, optionally filtered by ticker and/or label.

    Columns: id, ticker, interval, from_ts, to_ts, label, notes, created_at
    """
    where_clauses = [
        f"interval = '{interval}'",
        "is_deleted = false",
    ]

    if ticker:
        where_clauses.append(f"ticker = '{ticker}'")

    if label:
        where_clauses.append(f"label = '{label}'")

    where = " AND ".join(where_clauses)

    sql = f"""
        SELECT id, ticker, interval, from_ts, to_ts, label, notes, created_at
        FROM annotations
        WHERE {where}
        ORDER BY from_ts ASC
    """.strip()

    df = _query(sql)
    logger.debug("fetch_annotations(ticker=%s, label=%s): %d rows", ticker, label, len(df))
    return df


def fetch_paired_examples(interval: str = INTERVAL) -> list[dict]:
    """
    Return all (zone, breakout) pairs across all tickers.

    Pairing logic:
      For each triangle_breakout annotation, find the most recent
      triangle_zone annotation for the same ticker where zone.to_ts < breakout.from_ts.

    Returns a list of dicts:
      {
        ticker, zone_from, zone_to, breakout_ts,
        zone_candles: DataFrame,
        breakout_candle: Series,
      }
    """
    zones     = fetch_annotations(interval=interval, label="triangle_zone")
    breakouts = fetch_annotations(interval=interval, label="triangle_breakout")

    if zones.empty or breakouts.empty:
        logger.warning("No paired examples found — zones=%d breakouts=%d", len(zones), len(breakouts))
        return []

    pairs = []

    for _, bo in breakouts.iterrows():
        # Find all zones for the same ticker that ended before this breakout
        candidate_zones = zones[
            (zones["ticker"] == bo["ticker"]) &
            (zones["to_ts"] < bo["from_ts"])
        ]

        if candidate_zones.empty:
            logger.debug("No matching zone for breakout id=%s ticker=%s", bo["id"], bo["ticker"])
            continue

        # Take the most recent zone (closest in time to the breakout)
        zone = candidate_zones.sort_values("to_ts").iloc[-1]

        # Fetch candles for the zone
        zone_candles = fetch_candles(
            ticker=bo["ticker"],
            interval=interval,
            from_ts=zone["from_ts"],
            to_ts=zone["to_ts"],
        )

        if zone_candles.empty:
            logger.warning("No candles found for zone ticker=%s %s→%s", bo["ticker"], zone["from_ts"], zone["to_ts"])
            continue

        # Fetch the breakout candle (from_ts of the breakout annotation)
        breakout_candles = fetch_candles(
            ticker=bo["ticker"],
            interval=interval,
            from_ts=bo["from_ts"],
            to_ts=bo["to_ts"],
        )

        if breakout_candles.empty:
            logger.warning("No candle found at breakout ts=%s ticker=%s", bo["from_ts"], bo["ticker"])
            continue

        # Use the last candle in the breakout range as the confirmed breakout candle
        breakout_candle = breakout_candles.iloc[-1]

        pairs.append({
            "ticker":          bo["ticker"],
            "zone_from":       zone["from_ts"],
            "zone_to":         zone["to_ts"],
            "breakout_ts":     breakout_candle["ts"],
            "zone_candles":    zone_candles.reset_index(drop=True),
            "breakout_candle": breakout_candle,
            "notes":           bo.get("notes", ""),
        })

    logger.info("Paired %d training examples", len(pairs))
    return pairs
