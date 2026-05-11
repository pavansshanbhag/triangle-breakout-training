"""
feature_extractor.py — derive geometric and volume features from a
triangle zone (DataFrame of 15-min candles) and a single breakout candle.

Calibrated for 15-minute NSE candles spanning multi-week triangles.

Key design change vs 1hr version
──────────────────────────────────
Raw OHLCV candles are NOT fed directly into trendline regression.
A ZigZag filter first reduces the zone to meaningful swing highs/lows
(typically 15–25 pivots across a 55-day triangle).  Trendlines are then
fit through those pivot points, which gives geometrically clean lines
uncontaminated by intraday noise and wicks.

ZigZag deviation is auto-selected per zone (see extract_swing_points).

Slope units
───────────
Slopes from linregress are in price/swing_point.  We normalise to
%/trading_day for all logged explanations so they're human-readable,
but features stored for ML remain as %/swing_point (scale-invariant).

Feature vector (15 features — same names as 1hr version for model compatibility)
──────────────────────────────────────────────────────────────────────────────────
Trendline geometry (computed on swing-point series)
  1.  upper_slope_pct      — upper trendline slope per swing-point, % of first swing high
  2.  lower_slope_pct      — lower trendline slope per swing-point, % of first swing low
  3.  slope_ratio          — upper_slope / lower_slope
  4.  zone_width           — total candle count in the zone (not swing-point count)
  5.  apex_proximity       — zone_width / projected_apex_candle (0→start, 1→apex)
  6.  triangle_height_pct  — (first swing high − first swing low) / first close

Price action at breakout candle (unchanged)
  7.  breakout_close_vs_upper — close vs projected upper line at breakout position
  8.  breakout_body_pct       — |close−open| / (high−low)
  9.  breakout_wick_ratio     — upper wick / total range
  10. close_position          — (close−low) / (high−low)
  11. breakout_lag            — 15-min candles between zone end and breakout

Volume features (computed on raw candles — not swing points)
  12. volume_ratio            — breakout candle vol / zone avg vol
  13. volume_trend            — normalised slope of volume across zone
  14. zone_avg_volume_cv      — coefficient of variation of zone volume
  15. zone_close_trend        — normalised slope of closes across zone
"""

import logging
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress

from .config import (
    CANDLE_MINUTES,
    CANDLES_PER_DAY,
    MIN_SWING_POINTS,
    ZIGZAG_DEVIATIONS,
)

logger = logging.getLogger(__name__)

EPS = 1e-9

FEATURE_NAMES = [
    "upper_slope_pct",
    "lower_slope_pct",
    "slope_ratio",
    "zone_width",
    "apex_proximity",
    "triangle_height_pct",
    "breakout_close_vs_upper",
    "breakout_body_pct",
    "breakout_wick_ratio",
    "close_position",
    "breakout_lag",
    "volume_ratio",
    "volume_trend",
    "zone_avg_volume_cv",
    "zone_close_trend",
]


# ── ZigZag swing-point extractor ──────────────────────────────────────────────

def extract_swing_points(
    candles: pd.DataFrame,
    deviation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Single O(n) pass ZigZag filter.

    Tracks the running extreme (high or low) and records a pivot when price
    reverses by more than `deviation` fraction from that extreme.

    Parameters
    ----------
    candles   : DataFrame with columns high, low (15-min OHLCV, ascending ts)
    deviation : minimum reversal as a fraction of price, e.g. 0.02 = 2%

    Returns
    -------
    upper_idx    : candle indices of swing highs
    upper_prices : swing high prices  (for upper trendline)
    lower_idx    : candle indices of swing lows
    lower_prices : swing low prices   (for lower trendline)
    """
    highs = candles["high"].values.astype(float)
    lows  = candles["low"].values.astype(float)
    n     = len(highs)

    if n < 2:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Direction: +1 = currently in an upswing, -1 = downswing
    direction   = 1
    extreme_val = highs[0]
    extreme_idx = 0

    pivot_idx    = []
    pivot_prices = []
    pivot_dirs   = []   # +1 = swing high, -1 = swing low

    for i in range(1, n):
        if direction == 1:
            if highs[i] >= extreme_val:
                # New high — extend upswing
                extreme_val = highs[i]
                extreme_idx = i
            elif (extreme_val - lows[i]) / (extreme_val + EPS) >= deviation:
                # Reversal down — record the swing high
                pivot_idx.append(extreme_idx)
                pivot_prices.append(extreme_val)
                pivot_dirs.append(1)
                direction   = -1
                extreme_val = lows[i]
                extreme_idx = i
        else:
            if lows[i] <= extreme_val:
                # New low — extend downswing
                extreme_val = lows[i]
                extreme_idx = i
            elif (highs[i] - extreme_val) / (extreme_val + EPS) >= deviation:
                # Reversal up — record the swing low
                pivot_idx.append(extreme_idx)
                pivot_prices.append(extreme_val)
                pivot_dirs.append(-1)
                direction   = 1
                extreme_val = highs[i]
                extreme_idx = i

    # Record the final open pivot
    pivot_idx.append(extreme_idx)
    pivot_prices.append(extreme_val)
    pivot_dirs.append(direction)

    pivot_idx    = np.array(pivot_idx)
    pivot_prices = np.array(pivot_prices)
    pivot_dirs   = np.array(pivot_dirs)

    upper_mask = pivot_dirs == 1
    lower_mask = pivot_dirs == -1

    return (
        pivot_idx[upper_mask],
        pivot_prices[upper_mask],
        pivot_idx[lower_mask],
        pivot_prices[lower_mask],
    )


def _best_zigzag(
    candles: pd.DataFrame,
    deviations: list[float] = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Try each ZigZag deviation and pick the one with the best composite score.

    Composite score (0-1) balances three things:
      40% - Trendline quality : avg R2 of upper + lower fit
      35% - Pivot density     : how many swing points (more = more reliable)
      25% - Convergence       : are the two lines actually squeezing together?

    Convergence (upper_slope < lower_slope) is the one geometry property shared
    by ALL triangle types - descending, ascending, and symmetrical - so this
    criterion accepts all three without biasing toward any one.

    Note: maximising R2 alone biases toward large deviations because fewer pivot
    points always produce a tighter line fit. The density term counteracts this.

    Returns
    -------
    best_deviation, upper_idx, upper_prices, lower_idx, lower_prices, best_score
    """
    if deviations is None:
        deviations = ZIGZAG_DEVIATIONS

    best_score = -np.inf
    best       = None
    avg_price  = float(candles["close"].mean())

    TARGET_PIVOTS = 12  # benchmark pivot count per line for a 55-day zone

    for dev in deviations:
        u_idx, u_prices, l_idx, l_prices = extract_swing_points(candles, dev)

        if len(u_prices) < MIN_SWING_POINTS or len(l_prices) < MIN_SWING_POINTS:
            continue

        # 1. Trendline quality (R2) - 40%
        _, _, r_upper, _, _ = linregress(u_idx.astype(float), u_prices)
        _, _, r_lower, _, _ = linregress(l_idx.astype(float), l_prices)
        r2_score = (r_upper ** 2 + r_lower ** 2) / 2.0   # avg, 0-1

        # 2. Pivot density - 35%
        avg_pivots    = (len(u_prices) + len(l_prices)) / 2.0
        density_score = min(1.0, avg_pivots / TARGET_PIVOTS)

        # 3. Convergence / triangle shape — 25%
        # Mirror the same three-type check used in detect_triangle_zone.
        # Score 1.0 for a clean pattern, partial credit for borderline cases.
        u_slope, _ = _linreg(u_idx.astype(float), u_prices)
        l_slope, _ = _linreg(l_idx.astype(float), l_prices)
        n_zone      = len(candles)

        flat_thresh  = avg_price * 0.02 / n_zone
        trend_thresh = avg_price * 0.01 / n_zone

        is_sym   = (u_slope < -trend_thresh
                    and l_slope > trend_thresh
                    and (l_slope - u_slope) > trend_thresh)
        is_desc  = (abs(u_slope) <= flat_thresh
                    and l_slope < -trend_thresh
                    and (l_slope - u_slope) < -trend_thresh)
        is_asc   = (abs(l_slope) <= flat_thresh
                    and u_slope > trend_thresh
                    and (u_slope - l_slope) > trend_thresh)
        is_wedge = (u_slope < -trend_thresh
                    and l_slope < -trend_thresh
                    and (l_slope - u_slope) > trend_thresh)

        if is_sym or is_desc or is_asc or is_wedge:
            # Clean triangle — full score
            convergence_score = 1.0
        else:
            # Partial credit: how close are we to meeting any condition?
            # Score based on best near-miss
            # Near-desc: upper nearly flat
            desc_closeness = max(0.0, 1.0 - abs(u_slope) / (flat_thresh + EPS))
            # Near-asc: lower nearly flat
            asc_closeness  = max(0.0, 1.0 - abs(l_slope) / (flat_thresh + EPS))
            # Near-sym: how close are slopes to converging
            sym_closeness  = max(0.0, 1.0 - max(0.0, l_slope - u_slope) / (trend_thresh + EPS)) \
                             if l_slope > u_slope else 0.0
            convergence_score = max(desc_closeness, asc_closeness, sym_closeness) * 0.4

        composite = (
            0.40 * r2_score +
            0.35 * density_score +
            0.25 * convergence_score
        )

        if composite > best_score:
            best_score = composite
            best       = (dev, u_idx, u_prices, l_idx, l_prices)

    if best is None:
        return None, None, None, None, None, -np.inf

    return (*best, best_score)


# ── Trendline fitter (swing-point aware) ──────────────────────────────────────

def _linreg(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (slope, intercept) of OLS fit."""
    if len(x) < 2:
        return 0.0, float(y[0]) if len(y) else 0.0
    slope, intercept, *_ = linregress(x, y)
    return float(slope), float(intercept)


def _best_subset_line(
    idx: np.ndarray,
    prices: np.ndarray,
    min_points: int = MIN_SWING_POINTS,
    inlier_pct: float = 0.008,
    candle_x: np.ndarray = None,
    candle_closes: np.ndarray = None,
    violation_side: int = 1,
    max_violation_pct: float = 0.10,
) -> tuple[float, float, int, float]:
    """
    Exhaustive subset search: find the line through ≥ min_points swing points
    that maximises inlier count, then minimises RMSE among those inliers.

    For each subset of size k (k = min_points … N), an OLS line is fit and
    then tested against ALL swing points — a point is an inlier if its
    distance from the line is within inlier_pct of the average price
    (default 0.8%).  The best (inlier_count, -rmse) wins.

    Anchor enumeration is capped at 8 swing points regardless of N, keeping
    the worst-case subset count at C(8, 3..8) = 219.  Inliers are still
    counted against all N swing points so no information is lost.

    Optional candle_closes / candle_x filter:
      If provided, any candidate line where more than max_violation_pct of
      zone candles close on the wrong side is excluded.  violation_side=+1
      means the line is an upper bound (closes above it are violations);
      violation_side=-1 means a lower bound (closes below it are violations).

    Returns
    -------
    slope, intercept, inlier_count, inlier_rmse
    """
    n = len(idx)
    if n < 2:
        s, b = _linreg(idx.astype(float), prices)
        return s, b, n, 0.0

    x         = idx.astype(float)
    avg_price = float(np.mean(prices))
    eps_price = inlier_pct * avg_price

    # Cap anchor pool to 8 points — limits subsets to ≤ 219 regardless of N.
    # When N > 8 distribute slots evenly across the full range (always
    # including the last swing high) so late swing points can appear in
    # candidate subsets.  Lines are still scored against all N swing points.
    if n <= 8:
        anchor_pool = np.arange(n, dtype=np.intp)
    else:
        anchor_pool = np.unique(
            np.round(np.linspace(0, n - 1, 8)).astype(np.intp)
        )
    anchor_n = len(anchor_pool)

    # Batch OLS across all subsets of the same size in one numpy pass.
    # Avoids 219 individual scipy.linregress calls (which compute r/p/stderr
    # we discard) — replaces them with vectorised closed-form OLS formulas.
    all_slopes:      list[np.ndarray] = []
    all_intercepts:  list[np.ndarray] = []
    all_counts:      list[np.ndarray] = []
    all_rmses:       list[np.ndarray] = []
    all_max_x:       list[np.ndarray] = []
    all_last_inlier: list[np.ndarray] = []

    for size in range(min_points, anchor_n + 1):
        # idx_sets: (n_subsets, size) — pool-relative indices → map via anchor_pool
        idx_sets = np.array(list(combinations(range(anchor_n), size)), dtype=np.intp)

        sub_x = x[anchor_pool[idx_sets]]        # (n_subsets, size)
        sub_y = prices[anchor_pool[idx_sets]]   # (n_subsets, size)

        # Closed-form OLS: s = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)
        sx   = sub_x.sum(axis=1)
        sy   = sub_y.sum(axis=1)
        sxy  = (sub_x * sub_y).sum(axis=1)
        sx2  = (sub_x * sub_x).sum(axis=1)
        denom = size * sx2 - sx * sx

        ok    = np.abs(denom) > 1e-12
        d     = np.where(ok, denom, 1.0)
        slopes     = (size * sxy - sx * sy) / d
        intercepts = (sy - slopes * sx) / size

        # Score each candidate line against ALL n swing points: (n_subsets, n)
        predicted   = intercepts[:, None] + slopes[:, None] * x[None, :]
        abs_res     = np.abs(prices[None, :] - predicted)
        inlier_mask = abs_res <= eps_price
        counts      = inlier_mask.sum(axis=1)

        # Vectorised RMSE: zero non-inlier residuals, divide by inlier count
        sq_sum = (abs_res ** 2 * inlier_mask).sum(axis=1)
        rmses  = np.sqrt(sq_sum / np.maximum(counts, 1))

        # Max x-position of inliers: -1 sentinel for non-inliers so they
        # don't artificially inflate the max.
        x_inlier = np.where(inlier_mask, x[None, :], -1.0)
        max_x    = x_inlier.max(axis=1)

        # Whether the most recent swing high (x[-1]) is an inlier — used to
        # ensure the selected line reflects the current boundary, not just the
        # tightest historical cluster.
        last_inlier = inlier_mask[:, -1]

        # Candle violation filter: reject lines where too many zone candles
        # close on the wrong side (upper line cut below price, or lower line
        # cut above price).  Computed in a single vectorised pass.
        if candle_x is not None and candle_closes is not None:
            tl_c   = intercepts[:, None] + slopes[:, None] * candle_x[None, :]
            if violation_side > 0:
                viol = (candle_closes[None, :] > tl_c).sum(axis=1)
            else:
                viol = (candle_closes[None, :] < tl_c).sum(axis=1)
            viol_ok = viol <= int(max_violation_pct * len(candle_closes))
        else:
            viol_ok = np.ones(len(slopes), dtype=bool)

        keep = ok & (counts >= min_points) & viol_ok
        if keep.any():
            all_slopes.append(slopes[keep])
            all_intercepts.append(intercepts[keep])
            all_counts.append(counts[keep])
            all_rmses.append(rmses[keep])
            all_max_x.append(max_x[keep])
            all_last_inlier.append(last_inlier[keep])

    if not all_counts:
        s, b = _linreg(x, prices)
        return s, b, n, 0.0

    slopes_all      = np.concatenate(all_slopes)
    intercepts_all  = np.concatenate(all_intercepts)
    counts_all      = np.concatenate(all_counts)
    rmses_all       = np.concatenate(all_rmses)
    max_x_all       = np.concatenate(all_max_x)
    last_inlier_all = np.concatenate(all_last_inlier)

    # Selection strategy:
    # Prefer candidates where the most recent swing high is an inlier — a
    # trendline that doesn't reach the last swing high doesn't represent
    # where the boundary actually is.  Only if no such candidate exists
    # (e.g. the last point is a genuine outlier not caught by _trim) do we
    # fall back to the unconstrained pool.
    # Within the chosen pool, sort by: max inliers → most recent inliers → min RMSE.
    candidate_mask = last_inlier_all if last_inlier_all.any() else np.ones(len(counts_all), dtype=bool)
    sub_idx = np.where(candidate_mask)[0]
    sub_best = np.lexsort((
        rmses_all[candidate_mask],
        -max_x_all[candidate_mask],
        -counts_all[candidate_mask],
    ))[0]
    best = sub_idx[sub_best]
    return (
        float(slopes_all[best]),
        float(intercepts_all[best]),
        int(counts_all[best]),
        float(rmses_all[best]),
    )


def _trim_breakout_swings(
    idx: np.ndarray,
    prices: np.ndarray,
    min_points: int = MIN_SWING_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove trailing swing points that have broken out of the trendline.

    Breakout-phase candles (the rally/crash after the triangle resolves) show
    up as the last few swing points with a large residual from the regression
    line — the price has already left the channel.  Including them skews the
    slope and intercept so the drawn trendline no longer represents the
    triangle boundary.

    Algorithm: fit → compute residuals → if the last point's |residual| exceeds
    2 × residual std-dev, drop it and repeat.  Stops when the last point is
    within 2σ or min_points is reached.
    """
    while len(prices) > min_points:
        slope, intercept, *_ = linregress(idx.astype(float), prices)
        predicted  = intercept + slope * idx.astype(float)
        residuals  = prices - predicted
        res_std    = float(np.std(residuals))

        if res_std < EPS:
            break

        if abs(float(residuals[-1])) > 2.0 * res_std:
            idx    = idx[:-1]
            prices = prices[:-1]
        else:
            break

    return idx, prices


def fit_trendlines_from_swings(
    upper_idx: np.ndarray,
    upper_prices: np.ndarray,
    lower_idx: np.ndarray,
    lower_prices: np.ndarray,
    n_candles: int,
    zone_closes: np.ndarray = None,
) -> dict:
    """
    Fit trendlines through ZigZag pivot points (not raw candles).

    Uses exhaustive subset search (_best_subset_line) to find the line that
    touches the most swing points within 0.8% of price, breaking ties by RMSE.

    Indices are candle positions (0-based) within the zone.
    n_candles is the total candle count of the zone.
    zone_closes, if provided, enables the candle violation filter: subsets
    where >10% of closes breach the trendline are excluded.
    """
    candle_x = np.arange(n_candles, dtype=float) if zone_closes is not None else None

    upper_slope, upper_intercept, u_inliers, u_rmse = _best_subset_line(
        upper_idx, upper_prices,
        candle_x=candle_x, candle_closes=zone_closes, violation_side=1,
    )
    lower_slope, lower_intercept, l_inliers, l_rmse = _best_subset_line(
        lower_idx, lower_prices,
        candle_x=candle_x, candle_closes=zone_closes, violation_side=-1,
    )

    logger.debug(
        "Trendline fit — upper: %d/%d inliers rmse=%.3f  lower: %d/%d inliers rmse=%.3f",
        u_inliers, len(upper_prices), u_rmse,
        l_inliers, len(lower_prices), l_rmse,
    )

    # Project both lines at every candle position in the zone
    x_all      = np.arange(n_candles, dtype=float)
    upper_line = upper_intercept + upper_slope * x_all
    lower_line = lower_intercept + lower_slope * x_all

    # Apex: intersection of upper and lower trendlines (in candle units)
    denom = upper_slope - lower_slope
    if abs(denom) > EPS:
        apex_x = (lower_intercept - upper_intercept) / denom
    else:
        apex_x = float(n_candles) * 2   # parallel — apex far away

    return {
        "upper_slope":     upper_slope,
        "upper_intercept": upper_intercept,
        "lower_slope":     lower_slope,
        "lower_intercept": lower_intercept,
        "upper_line":      upper_line,
        "lower_line":      lower_line,
        "apex_x":          apex_x,
        "n":               n_candles,
        "x":               x_all,
        "upper_inliers":   u_inliers,
        "upper_rmse":      u_rmse,
        "lower_inliers":   l_inliers,
        "lower_rmse":      l_rmse,
    }


def fit_trendlines(zone_candles: pd.DataFrame) -> dict:
    """
    Public interface used by scanner.evaluate_breakout().
    Runs _best_zigzag internally and falls back to raw-candle regression
    if swing-point extraction fails (very short zone).
    """
    n = len(zone_candles)

    dev, u_idx, u_prices, l_idx, l_prices, r2 = _best_zigzag(zone_candles)

    if dev is not None:
        u_idx, u_prices = _trim_breakout_swings(u_idx, u_prices)
        l_idx, l_prices = _trim_breakout_swings(l_idx, l_prices)
        tl = fit_trendlines_from_swings(u_idx, u_prices, l_idx, l_prices, n,
                                        zone_closes=zone_candles["close"].values)
        tl["used_zigzag"]  = True
        tl["deviation"]    = dev
        tl["swing_r2"]     = r2
        tl["highs"] = zone_candles["high"].values.astype(float)
        tl["lows"]  = zone_candles["low"].values.astype(float)
        tl["closes"]= zone_candles["close"].values.astype(float)
        return tl

    # Fallback: raw candle regression (short zones < MIN_SWING_POINTS pivots)
    logger.debug("ZigZag produced insufficient pivots — falling back to raw regression")
    x      = np.arange(n, dtype=float)
    highs  = zone_candles["high"].values.astype(float)
    lows   = zone_candles["low"].values.astype(float)
    closes = zone_candles["close"].values.astype(float)

    upper_slope, upper_intercept = _linreg(x, highs)
    lower_slope, lower_intercept = _linreg(x, lows)
    upper_line = upper_intercept + upper_slope * x
    lower_line = lower_intercept + lower_slope * x
    denom = upper_slope - lower_slope
    apex_x = (lower_intercept - upper_intercept) / denom if abs(denom) > EPS else float(n) * 2

    return {
        "upper_slope": upper_slope, "upper_intercept": upper_intercept,
        "lower_slope": lower_slope, "lower_intercept": lower_intercept,
        "upper_line":  upper_line,  "lower_line":      lower_line,
        "apex_x":      apex_x,      "n":               n,
        "x":           x,           "highs":           highs,
        "lows":        lows,        "closes":          closes,
        "used_zigzag": False,       "deviation":       None,
        "swing_r2":    None,
    }


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(
    zone_candles: pd.DataFrame,
    breakout_candle: pd.Series,
    zone_to_ts=None,
    upper_at_bo: float = None,
    candles_per_day: int = CANDLES_PER_DAY,
) -> Optional[dict]:
    """
    Extract the 15-feature vector for one (zone, breakout) pair.

    zone_candles   : 15-min OHLCV DataFrame for the triangle zone
    breakout_candle: the candle that broke out (single row Series)
    zone_to_ts     : timestamp of the last candle in the zone (for lag calc)
    upper_at_bo    : projected upper trendline value at the breakout candle,
                     pre-computed by evaluate_breakout. When provided, this is
                     used directly for breakout_close_vs_upper so that the
                     feature is consistent with the direction check that already
                     confirmed the breakout. When None, it is re-projected here
                     (used during training where no direction check exists).

    Returns None if zone is too short or data is degenerate.
    """
    n_candles = len(zone_candles)
    if n_candles < 10:
        logger.debug("Zone too short (%d candles), skipping", n_candles)
        return None

    # ── Run ZigZag and fit trendlines ─────────────────────────────────────────
    tl = fit_trendlines(zone_candles)
    n  = tl["n"]

    # Anchor prices from first candle (not first swing point) for scale-invariance
    first_close = float(zone_candles["close"].iloc[0])
    first_high  = float(zone_candles["high"].iloc[0])
    first_low   = float(zone_candles["low"].iloc[0])

    if first_close < EPS or first_high < EPS or first_low < EPS:
        return None

    # ── 1. upper_slope_pct ────────────────────────────────────────────────────
    # Slope is price-per-candle; normalise by first high → dimensionless %
    upper_slope_pct = tl["upper_slope"] / (first_high + EPS) * 100

    # ── 2. lower_slope_pct ────────────────────────────────────────────────────
    lower_slope_pct = tl["lower_slope"] / (first_low + EPS) * 100

    # ── 3. slope_ratio ────────────────────────────────────────────────────────
    slope_ratio = upper_slope_pct / (abs(lower_slope_pct) + EPS)

    # ── 4. zone_width (trading days) ─────────────────────────────────────────
    zone_width = float(n) / candles_per_day

    # ── 5. apex_proximity ────────────────────────────────────────────────────
    apex_x = tl["apex_x"]
    if apex_x > 0:
        apex_proximity = min(n / (apex_x + EPS), 1.0)
    else:
        apex_proximity = 1.0

    # ── 6. triangle_height_pct ───────────────────────────────────────────────
    # Use first candle's range — captures the initial triangle amplitude
    zone_height = first_high - first_low
    triangle_height_pct = zone_height / (first_close + EPS) * 100

    # ── Breakout candle fields ────────────────────────────────────────────────
    bo_close  = float(breakout_candle["close"])
    bo_open   = float(breakout_candle["open"])
    bo_high   = float(breakout_candle["high"])
    bo_low    = float(breakout_candle["low"])
    bo_volume = float(breakout_candle["volume"])

    # ── 11. breakout_lag — in trading days ───────────────────────────────────
    if zone_to_ts is not None and "ts" in breakout_candle.index:
        try:
            delta = breakout_candle["ts"] - pd.Timestamp(zone_to_ts, tz="UTC")
            lag_candles = max(0.0, delta.total_seconds() / (CANDLE_MINUTES * 60))
            breakout_lag = lag_candles / candles_per_day
        except Exception:
            breakout_lag = 1.0 / candles_per_day
    else:
        breakout_lag = 1.0 / candles_per_day

    # ── 7. breakout_close_vs_upper ───────────────────────────────────────────
    # Use the caller-supplied projection when available (keeps this feature
    # consistent with the direction check in evaluate_breakout). Fall back to
    # re-projecting here only during training / standalone calls.
    if upper_at_bo is None:
        # n = len(zone_candles); trendline x=0 is first zone candle, so the
        # candle immediately after the zone is at x=n (not n+lag which skips ahead).
        bo_x = float(n)
        upper_at_bo = tl["upper_intercept"] + tl["upper_slope"] * bo_x
    breakout_close_vs_upper = (bo_close - upper_at_bo) / (upper_at_bo + EPS)

    # ── 8. breakout_body_pct ─────────────────────────────────────────────────
    candle_range = bo_high - bo_low
    breakout_body_pct = abs(bo_close - bo_open) / (candle_range + EPS)

    # ── 9. breakout_wick_ratio ───────────────────────────────────────────────
    upper_wick = bo_high - max(bo_open, bo_close)
    breakout_wick_ratio = upper_wick / (candle_range + EPS)

    # ── 10. close_position ───────────────────────────────────────────────────
    close_position = (bo_close - bo_low) / (candle_range + EPS)

    # ── 12. volume_ratio ─────────────────────────────────────────────────────
    zone_avg_vol = float(zone_candles["volume"].mean())
    volume_ratio = bo_volume / (zone_avg_vol + EPS)

    # ── 13. volume_trend ─────────────────────────────────────────────────────
    vols = zone_candles["volume"].values.astype(float)
    x_all = np.arange(n, dtype=float)
    vol_slope, _ = _linreg(x_all, vols)
    volume_trend = vol_slope / (zone_avg_vol + EPS)

    # ── 14. zone_avg_volume_cv ───────────────────────────────────────────────
    vol_std = float(zone_candles["volume"].std())
    zone_avg_volume_cv = vol_std / (zone_avg_vol + EPS)

    # ── 15. zone_close_trend ─────────────────────────────────────────────────
    closes = zone_candles["close"].values.astype(float)
    close_slope, _ = _linreg(x_all, closes)
    zone_close_trend = close_slope / (first_close + EPS) * 100

    features = {
        "upper_slope_pct":         upper_slope_pct,
        "lower_slope_pct":         lower_slope_pct,
        "slope_ratio":             slope_ratio,
        "zone_width":              zone_width,
        "apex_proximity":          apex_proximity,
        "triangle_height_pct":     triangle_height_pct,
        "breakout_close_vs_upper": breakout_close_vs_upper,
        "breakout_body_pct":       breakout_body_pct,
        "breakout_wick_ratio":     breakout_wick_ratio,
        "close_position":          close_position,
        "breakout_lag":            float(breakout_lag),
        "volume_ratio":            volume_ratio,
        "volume_trend":            volume_trend,
        "zone_avg_volume_cv":      zone_avg_volume_cv,
        "zone_close_trend":        zone_close_trend,
    }

    logger.debug(
        "Features extracted: zone=%d candles, zigzag=%s dev=%.1f%% r2=%.3f",
        n, tl.get("used_zigzag"), (tl.get("deviation") or 0) * 100, tl.get("swing_r2") or 0,
    )

    return features


def features_to_array(features: dict) -> np.ndarray:
    """Convert feature dict to numpy array in canonical FEATURE_NAMES order."""
    return np.array([features[k] for k in FEATURE_NAMES], dtype=float)


def explain_features(features: dict) -> str:
    """Human-readable summary calibrated for 15-min multi-week triangles."""
    lines = []

    upper_s  = features["upper_slope_pct"]
    lower_s  = features["lower_slope_pct"]
    days     = features["zone_width"]           # already in trading days
    lag_days = features["breakout_lag"]         # already in trading days

    # Trendline character
    # At 15-min, a "flat" upper line has slope < 0.002%/candle
    if upper_s < -0.01:
        lines.append(f"Upper trendline falling ({upper_s:.3f}%/candle) — symmetrical triangle")
    elif abs(upper_s) < 0.002:
        lines.append(f"Upper trendline flat ({upper_s:.4f}%/candle) — descending triangle")
    else:
        lines.append(f"Upper trendline: {upper_s:.3f}%/candle")

    lines.append(f"Lower trendline slope : {lower_s:.3f}%/candle")
    lines.append(f"Zone width            : {days:.1f} trading days")
    lines.append(f"Apex proximity        : {features['apex_proximity']:.0%} through triangle")
    lines.append(f"Breakout lag          : {lag_days:.2f} trading days after zone end")
    lines.append(f"Volume ratio          : {features['volume_ratio']:.1f}x zone avg")
    lines.append(f"Close vs upper line   : {features['breakout_close_vs_upper']*100:+.2f}%")
    lines.append(f"Body strength         : {features['breakout_body_pct']:.0%} of candle range")
    lines.append(f"Close position        : {features['close_position']:.0%} of range (top = bullish)")

    return "\n".join(lines)
