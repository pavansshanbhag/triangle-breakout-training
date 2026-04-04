"""
rule_based_scorer.py — deterministic breakout scorer used when there are
insufficient labeled examples to train the ML model.

Works for all three triangle types: descending, ascending, symmetrical.
The rules are type-agnostic — they score the breakout candle's behaviour
relative to whichever trendline it broke through, not the triangle shape.

Rules
─────
  1. Trendline convergence   — are the two lines actually squeezing together?
  2. Breakout vs trendline   — how cleanly did price exit the triangle?
  3. Volume surge            — volume >= MIN_VOLUME_RATIO × zone average
  4. Strong breakout body    — large body, close in direction of breakout
  5. Apex proximity          — 40–90% through the triangle is the sweet spot
  6. Breakout lag            — how many candles after zone end (fewer = cleaner)
"""

import logging

from .config import CANDLE_MINUTES, MIN_VOLUME_RATIO, RULE_ALERT_THRESHOLD
from .feature_extractor import FEATURE_NAMES, extract_features, features_to_array

logger = logging.getLogger(__name__)

RULE_WEIGHTS = {
    "trendline_convergence": 0.15,
    "breakout_vs_trendline": 0.25,
    "volume_surge":          0.20,
    "strong_body":           0.20,
    "apex_proximity":        0.12,
    "breakout_lag":          0.08,
}

assert abs(sum(RULE_WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum to 1"


def score(features: dict) -> tuple[float, dict]:
    """
    Score a feature dict using explicit rules.

    Returns
    -------
    score     : float in [0, 1]
    breakdown : dict mapping rule name -> weighted contribution
    """
    breakdown = {}

    # ── 1. Trendline convergence ──────────────────────────────────────────────
    # slope_ratio = upper_slope_pct / |lower_slope_pct|
    # For a descending triangle: upper~0, lower<0  -> ratio near 0  (converging)
    # For an ascending triangle: upper>0, lower~0  -> ratio large   (converging)
    # For a symmetrical:         upper<0, lower>0  -> ratio negative (converging)
    # In all cases slope_ratio < 1.0 means convergence. We reward low |ratio|.
    # A ratio of 0 = perfectly descending, ±inf = parallel lines (no convergence).
    slope_ratio = features["slope_ratio"]   # upper_slope_pct / |lower_slope_pct|
    abs_ratio   = abs(slope_ratio)
    # Score 1.0 when abs_ratio=0 (pure descending), 0.0 when abs_ratio>=2.0
    rule_score = max(0.0, 1.0 - abs_ratio / 2.0)
    breakdown["trendline_convergence"] = rule_score * RULE_WEIGHTS["trendline_convergence"]

    # ── 2. Breakout vs trendline ──────────────────────────────────────────────
    # breakout_close_vs_upper > 0 means close is above the upper trendline.
    # This is valid for bullish breakouts (above upper) and the same feature
    # captures bearish ones with a negative sign — the scanner already filters
    # direction before calling the scorer, so we always want this positive.
    close_vs = features["breakout_close_vs_upper"]
    if close_vs > 0:
        rule_score = min(1.0, close_vs / 0.02)   # saturates at +2% above line
    else:
        rule_score = 0.0
    breakdown["breakout_vs_trendline"] = rule_score * RULE_WEIGHTS["breakout_vs_trendline"]

    # ── 3. Volume surge ───────────────────────────────────────────────────────
    vol_ratio = features["volume_ratio"]
    if vol_ratio >= MIN_VOLUME_RATIO:
        rule_score = min(1.0, (vol_ratio - 1.0) / 2.0)   # saturates at 3x avg
    else:
        rule_score = 0.0
    breakdown["volume_surge"] = rule_score * RULE_WEIGHTS["volume_surge"]

    # ── 4. Strong breakout body ───────────────────────────────────────────────
    # Large body (body_pct) and close near the breakout extreme (close_position).
    # For bullish: close_position should be high (> 0.6).
    # For bearish: close_position should be low (< 0.4).
    # The scanner direction filter means we treat the feature as-is for bullish.
    body_pct  = features["breakout_body_pct"]
    close_pos = features["close_position"]
    # Score body strength; bonus if close is in the top 40% of the candle range
    body_score = body_pct
    if close_pos >= 0.6 or close_pos <= 0.4:
        body_score = min(1.0, body_score * 1.2)   # 20% bonus for decisive close
    breakdown["strong_body"] = body_score * RULE_WEIGHTS["strong_body"]

    # ── 5. Apex proximity ─────────────────────────────────────────────────────
    # 0 = at start of zone, 1 = at apex.
    # Ideal breakout range: 40–90% of the way to apex.
    prox = features["apex_proximity"]
    if 0.4 <= prox <= 0.9:
        rule_score = 1.0 - abs(prox - 0.65) / 0.35   # peak at 0.65
    elif prox < 0.4:
        rule_score = prox / 0.4 * 0.5                 # too early — lower confidence
    else:
        rule_score = max(0.0, (1.0 - prox) / 0.1 * 0.3)  # past apex — penalise
    breakdown["apex_proximity"] = rule_score * RULE_WEIGHTS["apex_proximity"]

    # ── 6. Breakout lag ───────────────────────────────────────────────────────
    # Fewer candles between zone end and breakout = cleaner signal.
    # 0–2 candles (0–30 min): full score. Degrades to 0 at MAX_BREAKOUT_LAG_CANDLES.
    lag = features["breakout_lag"]
    rule_score = max(0.0, 1.0 - lag / 8.0)   # 8 = MAX_BREAKOUT_LAG_CANDLES
    breakdown["breakout_lag"] = rule_score * RULE_WEIGHTS["breakout_lag"]

    total = sum(breakdown.values())
    return total, breakdown


def is_alert(features: dict) -> tuple[bool, float, dict]:
    """Returns (fire_alert, score, breakdown)."""
    total, breakdown = score(features)
    fired = total >= RULE_ALERT_THRESHOLD
    return fired, total, breakdown


def explain(features: dict, breakdown: dict, total: float) -> str:
    """Human-readable explanation of the rule-based score."""
    lines = [f"Rule-based score: {total:.2f}  (threshold: {RULE_ALERT_THRESHOLD})"]
    lines.append("")
    lines.append("Component scores:")
    for rule, contrib in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
        weight = RULE_WEIGHTS[rule]
        pct    = contrib / weight if weight > 0 else 0
        bar    = "█" * int(pct * 10) + "░" * (10 - int(pct * 10))
        lines.append(f"  {rule:<28} {bar}  {contrib:.3f} / {weight:.2f}")
    return "\n".join(lines)
