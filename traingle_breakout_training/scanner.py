"""
scanner.py — 15-minute triangle breakout scanner.

Calibrated for NSE 15-min candles (25 candles/trading day).
Triangles may span 3–70 trading days (~75–1,750 candles).

For each ticker in SCAN_TICKERS:
  1. Fetch last SCAN_LOOKBACK_CANDLES (1,750) of 15-min candles
  2. Run ZigZag on full window → extract swing highs/lows  [O(n)]
  3. Fit upper/lower trendlines on swing points
  4. Verify descending triangle geometry
  5. Check if the most recent candle broke the upper trendline
  6. Score via ML model (or rule-based fallback)
  7. Log alert if score >= threshold

Run directly:
  python scanner.py                   # scan all tickers once
  python scanner.py --ticker ROTO     # scan single ticker

Or call scan_all() from main.py on a schedule.
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    CANDLE_MINUTES,
    CANDLES_PER_DAY,
    INTERVAL,
    MIN_SWING_POINTS,
    MIN_VOLUME_RATIO,
    MIN_ZONE_CANDLES,
    ML_ALERT_THRESHOLD,
    RULE_ALERT_THRESHOLD,
    SCAN_LOOKBACK_CANDLES,
    SCAN_TICKERS,
    ZIGZAG_DEVIATIONS,
)
from .feature_extractor import (
    _best_zigzag,
    extract_features,
    explain_features,
    features_to_array,
    fit_trendlines_from_swings,
)
from .questdb_reader import fetch_candles, fetch_latest_candles
from .rule_based_scorer import explain as rule_explain
from .rule_based_scorer import is_alert as rule_is_alert
from .trainer import load_model

logger = logging.getLogger(__name__)


# ── Alert data class ──────────────────────────────────────────────────────────

@dataclass
class BreakoutAlert:
    ticker:              str
    breakout_ts:         datetime
    breakout_close:      float
    breakout_volume:     float
    zone_start_ts:       datetime
    zone_end_ts:         datetime
    zone_candle_count:   int
    zone_trading_days:   float
    triangle_type:       str          # "descending", "ascending", "symmetrical"
    score:               float
    mode:                str          # "ml" or "rule_based"
    direction:           str          # "bullish" or "bearish"
    upper_trendline_start_ts:  datetime
    upper_trendline_end_ts:    datetime
    lower_trendline_start_ts:  datetime
    lower_trendline_end_ts:    datetime
    upper_trendline_start_val: float
    upper_trendline_end_val:   float
    lower_trendline_start_val: float
    lower_trendline_end_val:   float
    upper_trendline_val:       float     # projected value at breakout candle
    lower_trendline_val:       float     # projected value at breakout candle
    volume_ratio:        float
    zigzag_deviation:    float
    n_swing_highs:       int
    n_swing_lows:        int
    features:            dict = field(repr=False)
    explanation:         str  = field(repr=False)

    def to_dict(self) -> dict:
        IST = timezone(timedelta(hours=5, minutes=30))

        def fmt(ts: datetime) -> str:
            return ts.astimezone(IST).strftime("%Y-%m-%d %H:%M IST")

        return {
            "ticker":        self.ticker,
            "breakout_ts":   fmt(self.breakout_ts),
            "breakout_close": self.breakout_close,
            "breakout_volume": self.breakout_volume,
            "triangle_type": self.triangle_type,
            "direction":     self.direction,
            "score":         round(self.score, 4),
            "mode":          self.mode,
            "volume_ratio":  round(self.volume_ratio, 3),
            "zone": {
                "start_ts":     fmt(self.zone_start_ts),
                "end_ts":       fmt(self.zone_end_ts),
                "candle_count": self.zone_candle_count,
                "trading_days": round(self.zone_trading_days, 1),
            },
            "upper_trendline": {
                "start_ts":    fmt(self.upper_trendline_start_ts),
                "start_price": round(self.upper_trendline_start_val, 2),
                "end_ts":      fmt(self.upper_trendline_end_ts),
                "end_price":   round(self.upper_trendline_end_val, 2),
                "at_breakout": round(self.upper_trendline_val, 2),
            },
            "lower_trendline": {
                "start_ts":    fmt(self.lower_trendline_start_ts),
                "start_price": round(self.lower_trendline_start_val, 2),
                "end_ts":      fmt(self.lower_trendline_end_ts),
                "end_price":   round(self.lower_trendline_end_val, 2),
                "at_breakout": round(self.lower_trendline_val, 2),
            },
            "zigzag": {
                "deviation_pct": round(self.zigzag_deviation * 100, 1),
                "n_swing_highs": self.n_swing_highs,
                "n_swing_lows":  self.n_swing_lows,
            },
            "features":    self.features,
            "explanation": self.explanation,
        }

    def log_summary(self):
        logger.info("BREAKOUT ALERT\n%s", json.dumps(self.to_dict(), indent=2))


# ── Backtest configuration ────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    """
    Tunable parameters for a backtest run. All fields have defaults that match
    the global config values so existing callers are unaffected.

    Example::

        from traangle_breakout_training.scanner import backtest, BacktestConfig

        cfg = BacktestConfig(min_volume_ratio=0.8, max_window_days=20)
        alerts = backtest("2024-06-01", "2024-09-30", ticker="ROTO", config=cfg)
    """
    # ── Zone detection ────────────────────────────────────────────────────────
    min_zone_candles:   int   = MIN_ZONE_CANDLES       # 75  — shortest valid triangle
    max_window_days:    int   = 12                     # 12  — sliding window cap (trading days)
    min_swing_points:   int   = MIN_SWING_POINTS       # 3   — per trendline
    zigzag_deviations:  list  = None                   # [1.5%, 2%, 2.5%] if None

    # ── Breakout confirmation ─────────────────────────────────────────────────
    min_volume_ratio:   float = MIN_VOLUME_RATIO       # 1.2 — breakout vol / zone avg vol
    ml_threshold:       float = ML_ALERT_THRESHOLD     # 0.60
    rule_threshold:     float = RULE_ALERT_THRESHOLD   # 0.55

    # ── Triangle geometry ─────────────────────────────────────────────────────
    flat_pct:           float = 0.02   # slope < this % of price over zone = "flat"
    trend_pct:          float = 0.01   # slope > this % of price over zone = "trending"

    def __post_init__(self):
        if self.zigzag_deviations is None:
            self.zigzag_deviations = list(ZIGZAG_DEVIATIONS)

    @property
    def max_window_candles(self) -> int:
        return self.max_window_days * CANDLES_PER_DAY


# ── Triangle detector (O(n) ZigZag-based) ─────────────────────────────────────

def detect_triangle_zone(candles: pd.DataFrame, cfg: BacktestConfig = None) -> Optional[dict]:
    """
    Detect a triangle zone in the candle window.

    Accepts four triangle/wedge types:
      Descending    : upper flat,    lower falling
      Ascending     : upper rising,  lower flat
      Symmetrical   : upper falling, lower rising
      Falling wedge : upper falling, lower also falling but less steeply (converging)

    The one geometry property all three share: the two trendlines converge,
    i.e. upper_slope < lower_slope (upper line rising less / falling more
    than lower line), so they meet at a future apex.

    Algorithm (O(n)):
      1. Run _best_zigzag on the full window (excluding last candle)
      2. Fit upper/lower trendlines on swing points
      3. Verify convergence: upper_slope < lower_slope AND apex in the future
      4. Verify zone width >= MIN_ZONE_CANDLES

    Returns dict with zone metadata, or None if no triangle found.
    """
    zone_window = candles.iloc[:-1]
    n_window    = len(zone_window)

    cfg = cfg or BacktestConfig()

    if n_window < cfg.min_zone_candles:
        return None

    avg_price = float(zone_window["close"].mean())

    dev, u_idx, u_prices, l_idx, l_prices, score = _best_zigzag(
        zone_window, cfg.zigzag_deviations
    )

    if dev is None:
        logger.debug("ZigZag produced no valid pivot set across all deviations")
        return None

    tl      = fit_trendlines_from_swings(u_idx, u_prices, l_idx, l_prices, n_window)
    upper_s = tl["upper_slope"]
    lower_s = tl["lower_slope"]

    # ── Universal three-type triangle check ──────────────────────────────────
    # Three valid patterns, each with a distinct slope signature:
    #
    #   Descending  : upper flat (< 2% drift over zone) + lower clearly falling
    #   Ascending   : lower flat (< 2% drift over zone) + upper clearly rising
    #   Symmetrical : upper_slope < lower_slope (lines converge to future apex)
    #
    # Thresholds are expressed relative to avg_price / zone_len so they scale
    # correctly across different price levels and zone durations.
    flat_thresh  = avg_price * cfg.flat_pct  / n_window
    trend_thresh = avg_price * cfg.trend_pct / n_window

    # For descending/ascending, also require a minimum *differential* between the
    # two slopes so that two parallel falling/rising lines don't qualify.
    # The non-flat line must be moving at least 1% more than the flat line.
    is_sym  = (upper_s < -trend_thresh                                        # upper clearly falling
               and lower_s > trend_thresh                                     # lower clearly rising
               and (lower_s - upper_s) > trend_thresh)                       # converging
    is_desc = (abs(upper_s) <= flat_thresh                                    # upper flat
               and lower_s < -trend_thresh                                    # lower clearly falling
               and (lower_s - upper_s) < -trend_thresh)                      # lower more negative than upper
    is_asc  = (abs(lower_s) <= flat_thresh                                    # lower flat
               and upper_s > trend_thresh                                     # upper clearly rising
               and (upper_s - lower_s) > trend_thresh)                       # upper more positive than lower
    is_wedge = (upper_s < -trend_thresh                                       # upper clearly falling
                and lower_s < -trend_thresh                                   # lower also falling
                and (lower_s - upper_s) > trend_thresh)                      # but lower falls less → converging

    if not (is_sym or is_desc or is_asc or is_wedge):
        logger.debug(
            "Not a triangle: upper_slope=%.5f  lower_slope=%.5f  "
            "flat_thresh=%.5f  trend_thresh=%.5f  "
            "(sym=%s desc=%s asc=%s)",
            upper_s, lower_s, flat_thresh, trend_thresh,
            is_sym, is_desc, is_asc,
        )
        return None

    triangle_type = ("symmetrical" if is_sym
                     else "descending" if is_desc
                     else "ascending" if is_asc
                     else "falling_wedge")

    # For all types the apex position is informational only.
    # The slope conditions above are the authoritative type gate.
    apex_x = tl["apex_x"]

    # ── Find zone start candle ────────────────────────────────────────────────
    # The zone starts at the earliest swing point on either line
    zone_start_idx = int(min(u_idx[0], l_idx[0]))

    # Zone must be at least MIN_ZONE_CANDLES long
    zone_len = n_window - zone_start_idx
    if zone_len < cfg.min_zone_candles:
        logger.debug(
            "Zone too short: %d candles from idx %d (need %d)",
            zone_len, zone_start_idx, cfg.min_zone_candles,
        )
        return None

    # Re-express swing indices relative to zone start for feature extraction
    u_idx_rel = u_idx - zone_start_idx
    l_idx_rel = l_idx - zone_start_idx

    # Keep only swing points within the zone (some may precede zone_start)
    u_mask = u_idx_rel >= 0
    l_mask = l_idx_rel >= 0
    if u_mask.sum() < cfg.min_swing_points or l_mask.sum() < cfg.min_swing_points:
        logger.debug("Not enough swing points within zone after trimming")
        return None

    trading_days = zone_len / CANDLES_PER_DAY

    logger.debug(
        "Triangle found: %s  start_idx=%d  zone=%d candles (%.1f days)  "
        "dev=%.1f%%  apex_x=%.0f  score=%.3f  swings: %dH/%dL",
        triangle_type, zone_start_idx, zone_len, trading_days,
        dev * 100, apex_x, score,
        u_mask.sum(), l_mask.sum(),
    )

    return {
        "zone_start_idx":  zone_start_idx,
        "zone_end_idx":    n_window - 1,
        "zone_len":        zone_len,
        "trading_days":    trading_days,
        "triangle_type":   triangle_type,
        "tl":              tl,
        "deviation":       dev,
        "zigzag_score":    score,
        "u_idx":           u_idx[u_mask],
        "u_prices":        u_prices[u_mask],
        "l_idx":           l_idx[l_mask],
        "l_prices":        l_prices[l_mask],
        "n_swing_highs":   int(u_mask.sum()),
        "n_swing_lows":    int(l_mask.sum()),
        "apex_x":          apex_x,
    }


# ── Breakout evaluator ────────────────────────────────────────────────────────

def evaluate_breakout(
    candles: pd.DataFrame,
    zone_info: dict,
    model_bundle: Optional[dict],
    cfg: BacktestConfig = None,
) -> Optional[BreakoutAlert]:
    """
    Given detected zone metadata and the latest (breakout candidate) candle,
    score the breakout.
    """
    zone_start  = zone_info["zone_start_idx"]
    zone_end    = zone_info["zone_end_idx"]
    tl          = zone_info["tl"]

    cfg = cfg or BacktestConfig()

    zone_candles    = candles.iloc[zone_start:zone_end + 1].copy()
    breakout_candle = candles.iloc[-1]   # always the very last candle

    # Project trendlines to the breakout candle's position.
    # The trendline x-coordinates are 0-based from the window start (absolute
    # positions within zone_window). zone_end_idx is the last zone candle, so
    # the breakout candle (one step beyond) sits at zone_end_idx + 1.
    n         = zone_info["zone_len"]
    bo_x      = float(zone_end + 1)   # = n_window: absolute position of breakout
    upper_at_bo = tl["upper_intercept"] + tl["upper_slope"] * bo_x
    lower_at_bo = tl["lower_intercept"] + tl["lower_slope"] * bo_x

    bo_close  = float(breakout_candle["close"])
    bo_volume = float(breakout_candle["volume"])
    zone_avg_vol = float(zone_candles["volume"].mean())
    volume_ratio = bo_volume / (zone_avg_vol + 1e-9)

    # ── Direction check ───────────────────────────────────────────────────────
    if bo_close > upper_at_bo:
        direction = "bullish"
    elif bo_close < lower_at_bo:
        direction = "bearish"
    else:
        logger.debug(
            "Candle inside triangle — not a breakout  close=%.2f  upper=%.2f  lower=%.2f",
            bo_close, upper_at_bo, lower_at_bo,
        )
        return None

    # ── Volume confirmation ───────────────────────────────────────────────────
    if volume_ratio < cfg.min_volume_ratio:
        logger.debug(
            "Volume not confirmed: ratio=%.2f < %.2f", volume_ratio, cfg.min_volume_ratio
        )
        return None

    # ── Feature extraction ────────────────────────────────────────────────────
    zone_to_ts = zone_candles.iloc[-1]["ts"]
    features = extract_features(
        zone_candles=zone_candles,
        breakout_candle=breakout_candle,
        zone_to_ts=zone_to_ts,
        upper_at_bo=upper_at_bo,
    )

    if features is None:
        logger.warning("Feature extraction returned None — skipping")
        return None

    # ── Scoring ───────────────────────────────────────────────────────────────
    if model_bundle is not None:
        scaler = model_bundle["scaler"]
        clf    = model_bundle["classifier"]
        X = features_to_array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prob  = float(clf.predict_proba(X_scaled)[0][1])
        fired = prob >= cfg.ml_threshold
        score = prob
        mode  = "ml"
        explanation = (
            f"ML probability: {prob:.3f}  (threshold: {cfg.ml_threshold})\n\n"
            + explain_features(features)
        )
    else:
        _, score, breakdown = rule_is_alert(features)
        fired = score >= cfg.rule_threshold
        mode  = "rule_based"
        explanation = rule_explain(features, breakdown, score)

    if not fired:
        logger.debug(
            "Score %.3f below threshold — no alert\n"
            "  Features:\n%s",
            score,
            "\n".join(f"    {k:<30s} {v:+.4f}" for k, v in features.items()),
        )
        return None

    # ── Build alert ───────────────────────────────────────────────────────────
    ticker = str(breakout_candle.get("ticker", "UNKNOWN"))

    return BreakoutAlert(
        ticker              = ticker,
        breakout_ts         = breakout_candle["ts"].to_pydatetime(),
        breakout_close      = bo_close,
        breakout_volume     = bo_volume,
        zone_start_ts       = zone_candles.iloc[0]["ts"].to_pydatetime(),
        zone_end_ts         = zone_candles.iloc[-1]["ts"].to_pydatetime(),
        zone_candle_count   = n,
        zone_trading_days   = zone_info["trading_days"],
        triangle_type       = zone_info["triangle_type"],
        score               = score,
        mode                = mode,
        direction           = direction,
        upper_trendline_start_ts  = candles.iloc[int(zone_info["u_idx"][0])]["ts"].to_pydatetime(),
        upper_trendline_end_ts    = candles.iloc[int(zone_info["u_idx"][-1])]["ts"].to_pydatetime(),
        lower_trendline_start_ts  = candles.iloc[int(zone_info["l_idx"][0])]["ts"].to_pydatetime(),
        lower_trendline_end_ts    = candles.iloc[int(zone_info["l_idx"][-1])]["ts"].to_pydatetime(),
        upper_trendline_start_val = float(zone_info["u_prices"][0]),
        upper_trendline_end_val   = float(zone_info["u_prices"][-1]),
        lower_trendline_start_val = float(zone_info["l_prices"][0]),
        lower_trendline_end_val   = float(zone_info["l_prices"][-1]),
        upper_trendline_val       = upper_at_bo,
        lower_trendline_val       = lower_at_bo,
        volume_ratio        = volume_ratio,
        zigzag_deviation    = zone_info["deviation"],
        n_swing_highs       = zone_info["n_swing_highs"],
        n_swing_lows        = zone_info["n_swing_lows"],
        features            = features,
        explanation         = explanation,
    )


# ── Single ticker scan ────────────────────────────────────────────────────────

def scan_ticker(
    ticker: str,
    model_bundle: Optional[dict],
    from_ts: Optional[datetime] = None,
    to_ts: Optional[datetime] = None,
) -> Optional[BreakoutAlert]:
    """
    Fetch 15-min candles for one ticker and check for a breakout.
    Returns a BreakoutAlert if one is detected, else None.

    If from_ts/to_ts are provided, fetches candles in that window.
    Otherwise fetches the most recent SCAN_LOOKBACK_CANDLES candles.
    """
    logger.debug("Scanning %s ...", ticker)

    if from_ts or to_ts:
        from .questdb_reader import fetch_candles
        candles = fetch_candles(ticker, interval=INTERVAL, from_ts=from_ts, to_ts=to_ts)
    else:
        candles = fetch_latest_candles(ticker, n=SCAN_LOOKBACK_CANDLES, interval=INTERVAL)

    if len(candles) < MIN_ZONE_CANDLES + 1:
        logger.debug("%s: insufficient candles (%d)", ticker, len(candles))
        return None

    zone_info = detect_triangle_zone(candles)

    if zone_info is None:
        logger.debug("%s: no descending triangle detected in %d candles (%.1f days)",
                     ticker, len(candles), len(candles) / CANDLES_PER_DAY)
        return None

    return evaluate_breakout(candles, zone_info, model_bundle)


# ── Continuous (historical) window scan ───────────────────────────────────────


def scan_ticker_continuous(
    ticker: str,
    model_bundle: Optional[dict],
    from_ts: datetime,
    to_ts: datetime,
    cfg: BacktestConfig = None,
) -> list[BreakoutAlert]:
    """
    Slide a growing/sliding window from from_ts to to_ts, detecting all breakouts.

    Window behaviour:
      - Left edge starts at from_ts (index 0).
      - Right edge (breakout candidate) starts at MIN_ZONE_CANDLES and advances
        one candle at a time.
      - Once the window exceeds 12 trading days (300 candles), the left edge
        advances so the window stays at exactly 300 candles.
      - After a breakout fires at position i, skip forward MIN_ZONE_CANDLES
        candles to avoid re-detecting the same event.
    """
    cfg = cfg or BacktestConfig()

    all_candles = fetch_candles(ticker, interval=INTERVAL, from_ts=from_ts, to_ts=to_ts)
    n = len(all_candles)

    if n < cfg.min_zone_candles + 1:
        logger.debug("%s: only %d candles in window — need %d", ticker, n, cfg.min_zone_candles + 1)
        return []

    logger.info(
        "%s: continuous scan  %s → %s  (%d candles, %.1f trading days)",
        ticker,
        all_candles.iloc[0]["ts"].strftime("%Y-%m-%d"),
        all_candles.iloc[-1]["ts"].strftime("%Y-%m-%d"),
        n, n / CANDLES_PER_DAY,
    )

    alerts: list[BreakoutAlert] = []
    left = 0
    i    = cfg.min_zone_candles   # right edge (breakout candidate index)

    while i < n:
        # Slide left edge if window exceeds max size
        if (i - left + 1) > cfg.max_window_candles:
            left = i - cfg.max_window_candles + 1

        window = all_candles.iloc[left : i + 1].reset_index(drop=True)

        zone_info = detect_triangle_zone(window, cfg)

        if zone_info is not None:
            alert = evaluate_breakout(window, zone_info, model_bundle, cfg)
            if alert:
                alert.log_summary()
                alerts.append(alert)
                # Abandon the old zone so the next window starts fresh,
                # but advance only 1 candle so a reversal breakout nearby
                # (e.g. bullish after a fake bearish) is not missed.
                left = i

        i += 1

    logger.info("%s: continuous scan complete — %d alert(s)", ticker, len(alerts))
    return alerts


def scan_all_continuous(
    from_ts: datetime,
    to_ts: datetime,
    tickers: list[str] = None,
    cfg: BacktestConfig = None,
) -> list[BreakoutAlert]:
    """Run continuous window scan across all configured tickers."""
    cfg = cfg or BacktestConfig()
    if tickers is None:
        tickers = SCAN_TICKERS

    logger.info("═" * 60)
    logger.info("  Continuous scan : %s → %s", from_ts.strftime("%Y-%m-%d"), to_ts.strftime("%Y-%m-%d"))
    logger.info("  Max window      : %d candles (%d trading days)", cfg.max_window_candles, cfg.max_window_days)
    logger.info("  Min volume ratio: %.2f", cfg.min_volume_ratio)
    logger.info("  ML threshold    : %.2f", cfg.ml_threshold)
    logger.info("  Tickers         : %s", ", ".join(tickers))
    logger.info("═" * 60)

    model_bundle = load_model()
    if model_bundle:
        logger.info("Mode: ML  (trained on %d examples)", model_bundle.get("n_training_examples", "?"))
    else:
        logger.info("Mode: RULE-BASED  (no trained model — run --train first)")

    all_alerts: list[BreakoutAlert] = []
    for ticker in tickers:
        try:
            alerts = scan_ticker_continuous(ticker, model_bundle, from_ts, to_ts, cfg)
            all_alerts.extend(alerts)
        except Exception as e:
            logger.error("Error scanning %s: %s", ticker, e, exc_info=True)

    logger.info("Continuous scan complete. %d total alert(s) across %d ticker(s).", len(all_alerts), len(tickers))
    return all_alerts


# ── Full scan (all tickers) ───────────────────────────────────────────────────

def scan_all(
    tickers: list[str] = None,
    from_ts: Optional[datetime] = None,
    to_ts: Optional[datetime] = None,
) -> list[BreakoutAlert]:
    """
    Run scanner across all configured tickers.
    Loads model once, reuses for all tickers.
    """
    if tickers is None:
        tickers = SCAN_TICKERS

    now = datetime.now(timezone.utc)
    logger.info("═" * 60)
    logger.info("  Scanner run : %s UTC", now.strftime("%Y-%m-%d %H:%M"))
    logger.info("  Interval    : %s  (%d candles/day, NSE)", INTERVAL, CANDLES_PER_DAY)
    if from_ts or to_ts:
        logger.info("  Window      : %s → %s",
                    from_ts.strftime("%Y-%m-%d") if from_ts else "start",
                    to_ts.strftime("%Y-%m-%d") if to_ts else "end")
    else:
        logger.info("  Lookback    : %d candles (%.0f trading days)",
                    SCAN_LOOKBACK_CANDLES, SCAN_LOOKBACK_CANDLES / CANDLES_PER_DAY)
    logger.info("  Tickers     : %s", ", ".join(tickers))
    logger.info("═" * 60)

    model_bundle = load_model()
    if model_bundle:
        logger.info("Mode: ML  (trained on %d examples)", model_bundle.get("n_training_examples", "?"))
    else:
        logger.info("Mode: RULE-BASED  (no trained model — run --train first)")

    alerts = []
    for ticker in tickers:
        try:
            alert = scan_ticker(ticker, model_bundle, from_ts=from_ts, to_ts=to_ts)
            if alert:
                alert.log_summary()
                alerts.append(alert)
        except Exception as e:
            logger.error("Error scanning %s: %s", ticker, e, exc_info=True)

    logger.info("Scan complete. %d alert(s) fired.", len(alerts))
    return alerts


# ── Public library API ───────────────────────────────────────────────────────

_IST = timezone(timedelta(hours=5, minutes=30))

def _parse_date(d, end_of_day: bool = False) -> Optional[datetime]:
    """
    Accept a datetime (returned as-is) or a string in one of these formats:
      'YYYY-MM-DD'           — date only, interpreted as IST
      'YYYY-MM-DD HH:MM'     — date + time, interpreted as IST
      'YYYY-MM-DD HH:MM:SS'  — date + time + seconds, interpreted as IST

    All string inputs are treated as IST (UTC+5:30) since candle data is
    stored in QuestDB as UTC converted from IST Kite timestamps.

    When end_of_day=True and a date-only string is given (no time component),
    the time is set to 23:59:59 so the full day's candles are included.
    When a time is explicitly provided it is always honoured as-is.
    """
    if d is None:
        return None
    if isinstance(d, datetime):
        return d if d.tzinfo else d.replace(tzinfo=_IST)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(d, fmt).replace(tzinfo=_IST)
        except ValueError:
            continue
    # Date-only string — apply end_of_day if requested
    try:
        dt = datetime.strptime(d, "%Y-%m-%d")
        if end_of_day:
            dt = dt.replace(hour=23, minute=59, second=59)
        return dt.replace(tzinfo=_IST)
    except ValueError:
        pass
    raise ValueError(f"Unrecognised date format: {d!r}  (expected YYYY-MM-DD or YYYY-MM-DD HH:MM)")


def scan(
    ticker: str = None,
    from_date=None,
    to_date=None,
) -> list[BreakoutAlert]:
    """
    Scan for triangle breakouts. Loads the model automatically.

    Args:
        ticker:    Single ticker symbol, or None to scan all configured tickers.
        from_date: Start of candle window — 'YYYY-MM-DD' string or datetime (UTC).
                   None = use default lookback (SCAN_LOOKBACK_CANDLES).
        to_date:   End of candle window   — 'YYYY-MM-DD' string or datetime (UTC).
                   None = up to the latest available candle.

    Returns:
        List of BreakoutAlert objects (may be empty).

    Examples::

        from traingle_breakout_training.scanner import scan, backtest

        alerts = scan()                                     # all tickers, latest window
        alerts = scan("ROTO")                              # single ticker
        alerts = scan("ROTO", from_date="2024-06-01",
                               to_date="2024-09-30")       # historical snapshot
    """
    from_ts = _parse_date(from_date)
    to_ts   = _parse_date(to_date, end_of_day=True)
    tickers = [ticker.upper()] if ticker else None
    return scan_all(tickers, from_ts=from_ts, to_ts=to_ts)


def backtest(
    from_date,
    to_date,
    ticker: str = None,
    config: BacktestConfig = None,
) -> list[BreakoutAlert]:
    """
    Sliding-window backtest — detect all breakouts between two dates.

    Args:
        from_date: Start date — 'YYYY-MM-DD' string or datetime (UTC).
        to_date:   End date   — 'YYYY-MM-DD' string or datetime (UTC).
        ticker:    Single ticker symbol, or None to scan all configured tickers.

    Returns:
        List of BreakoutAlert objects (may be empty).

    Examples::

        from traingle_breakout_training.scanner import backtest

        alerts = backtest("2024-06-01", "2024-09-30")
        alerts = backtest("2024-06-01", "2024-09-30", ticker="ROTO")
    """
    from_ts = _parse_date(from_date)
    to_ts   = _parse_date(to_date, end_of_day=True)
    tickers = [ticker.upper()] if ticker else None
    logger.info("Received backtest request for ticker[%s] , fromdate[%s] , todate[%s] ", tickers, from_ts, to_ts)
    return scan_all_continuous(from_ts, to_ts, tickers, config)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from traingle_breakout_training import logging_setup  # noqa

    parser = argparse.ArgumentParser(description="Triangle breakout scanner (15-min NSE)")
    parser.add_argument("--ticker",    help="Scan a single ticker instead of all")
    parser.add_argument("--from-date", help="Start of scan window (YYYY-MM-DD)")
    parser.add_argument("--to-date",   help="End of scan window (YYYY-MM-DD)")
    args = parser.parse_args()

    from_ts = datetime.strptime(args.from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.from_date else None
    to_ts   = datetime.strptime(args.to_date,   "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.to_date   else None

    if args.ticker:
        mb    = load_model()
        alert = scan_ticker(args.ticker.upper(), mb, from_ts=from_ts, to_ts=to_ts)
        if not alert:
            logger.info("No breakout detected for %s", args.ticker.upper())
    else:
        scan_all(from_ts=from_ts, to_ts=to_ts)
