"""
config.py — central configuration for the triangle breakout system

Calibrated for 15-minute candles on NSE.

NSE session: 09:15 – 15:30 IST = 375 minutes = 25 candles/day
A 55-day triangle zone therefore spans ~1,375 candles (55 × 25).
The scanner lookback covers 70 trading days (~1,750 candles) to ensure
the full triangle is always within the fetch window.
"""

# ── QuestDB connection ────────────────────────────────────────────────────────
QUESTDB_HOST = "localhost"
QUESTDB_PORT = 9000          # HTTP/REST port (not PG wire)
QUESTDB_BASE_URL = f"http://{QUESTDB_HOST}:{QUESTDB_PORT}"

# ── Data settings ─────────────────────────────────────────────────────────────
INTERVAL = "15m"

# NSE session constants (used for lag/duration calculations)
CANDLES_PER_DAY = 25         # 375-min session ÷ 15-min candles
CANDLE_MINUTES  = 15

# Tickers to scan during live inference (add all tickers you trade)
SCAN_TICKERS = [
    "ROTO",
    "ESCORTS",
    "SONACOMS",
    "EQUITASBNK",
    "NUVAMA",
    "PRAJIND",
    "VGUARD",
]

# ── Training ──────────────────────────────────────────────────────────────────
import os as _os
_MODELS_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "models")

MIN_EXAMPLES_FOR_ML = 8       # below this, fall back to rule-based scorer
MODEL_PATH = _os.path.join(_MODELS_DIR, "triangle_breakout.pkl")
FEATURE_IMPORTANCE_PATH = _os.path.join(_MODELS_DIR, "feature_importance.json")

# How many 15-min candles after zone end we allow a breakout annotation to sit.
# "a couple of candles far" at 15-min = up to ~2 hrs = 8 candles
MAX_BREAKOUT_LAG_CANDLES = 8

# Minimum candles for a valid zone.
# Shortest meaningful triangle at 15-min = ~3 trading days = 75 candles.
MIN_ZONE_CANDLES = 75

# ── ZigZag swing-point detector ───────────────────────────────────────────────
# ZigZag filters raw OHLCV down to meaningful swing highs/lows before fitting
# trendlines.  At 15-min granularity, intraday noise requires a slightly higher
# deviation than daily charts to avoid over-detecting micro-swings.
#
# Three candidate deviations are tried per zone; the one producing the best
# triangle R² is selected.  Sweet spot for NSE 15-min multi-week triangles:
#   1.5% — catches tighter consolidations
#   2.0% — standard for most triangles
#   2.5% — filters aggressive intraday noise in volatile stocks
ZIGZAG_DEVIATIONS = [0.015, 0.020, 0.025]   # tried in order; best R² wins

# Minimum swing points required on each trendline after ZigZag reduction.
# Fewer than this and the trendline fit is geometrically meaningless.
MIN_SWING_POINTS = 3

# ── Inference thresholds ──────────────────────────────────────────────────────
# ML mode: minimum probability score to fire an alert
ML_ALERT_THRESHOLD = 0.60

# Rule-based mode: minimum score (0–1 composite) to fire an alert
RULE_ALERT_THRESHOLD = 0.55

# Volume confirmation: breakout candle volume must be >= this multiple of zone avg
# Slightly lower than 1hr equivalent — 15-min breakout candles are naturally
# noisier, so we accept a more modest volume surge.
MIN_VOLUME_RATIO = 1.2

# ── Scanner schedule ──────────────────────────────────────────────────────────
# Cron: fire 2 minutes after each 15-min candle closes.
# 15-min candles close at :00, :15, :30, :45 — so we run at :02, :17, :32, :47.
SCANNER_CRON = "2,17,32,47 * * * *"

# How many 15-min candles to fetch per ticker for the live scan.
# 70 trading days × 25 candles/day = 1,750 candles — covers a 55-day triangle
# with 15 days of headroom for detecting the zone start cleanly.
SCAN_LOOKBACK_CANDLES = 1750

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_PATH  = "logs/scanner.log"
LOG_LEVEL = _os.getenv("SCANNER_LOG_LEVEL", "INFO")
