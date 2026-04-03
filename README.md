# Triangle Breakout Detection System

Learns descending triangle breakout patterns from your QuestDB annotations
and scans live 1hr candles to fire alerts when a new breakout is detected.

---

## Architecture

```
questdb_reader.py     — fetch candles + annotations via QuestDB REST /exec
feature_extractor.py  — 15 geometric/volume features per (zone, breakout) pair
trainer.py            — XGBoost classifier, LOOCV eval, persists model to disk
rule_based_scorer.py  — deterministic fallback when < 8 labeled examples
scanner.py            — hourly job: detect triangle → score breakout → alert
main.py               — CLI entry point
config.py             — all tunable parameters
```

---

## Setup

```bash
pip install -r requirements.txt
```

Edit `config.py`:
- Set `QUESTDB_HOST` / `QUESTDB_PORT` to your QuestDB instance
- Set `SCAN_TICKERS` to the list of tickers you want to monitor

---

## Usage

### 1. Train the model
Reads all `triangle_zone` + `triangle_breakout` annotation pairs from QuestDB
and trains an XGBoost classifier.

```bash
python main.py --train
```

Expected output with 20 examples:
```
LOOCV → acc=0.82  prec=0.80  recall=0.85  (TP=17 FP=4 FN=3 TN=36)
Model saved → models/triangle_breakout.pkl
```

If fewer than 8 pairs are found, the system automatically uses the rule-based
scorer until more examples are annotated.

### 2. Run scanner once
```bash
python main.py --scan               # all tickers in SCAN_TICKERS
python main.py --scan --ticker ROTO # single ticker
```

### 3. Start hourly scheduler (blocking)
Runs at HH:02 every hour (configurable via `SCANNER_CRON` in config.py).
```bash
python main.py --schedule
```

For production, run via systemd or supervisor rather than nohup:

```ini
# /etc/systemd/system/triangle-scanner.service
[Unit]
Description=Triangle Breakout Scanner

[Service]
WorkingDirectory=/path/to/triangle_breakout
ExecStart=/path/to/venv/bin/python main.py --schedule
Restart=always

[Install]
WantedBy=multi-user.target
```

### 4. Check model status
```bash
python main.py --status
```

### 5. Train then scan immediately
```bash
python main.py --train --scan
```

---

## Features extracted per example

| Feature | Description |
|---|---|
| `upper_slope_pct` | Upper trendline slope as % of first high per candle |
| `lower_slope_pct` | Lower trendline slope as % of first low per candle |
| `slope_ratio` | upper/lower — how "pure" a descending triangle it is |
| `zone_width` | Number of candles in the consolidation zone |
| `apex_proximity` | How far through the triangle (0=start, 1=apex) |
| `triangle_height_pct` | Height of triangle as % of price |
| `breakout_close_vs_upper` | How far close is above the upper trendline |
| `breakout_body_pct` | Candle body / total range |
| `breakout_wick_ratio` | Upper wick / range (rejection signal) |
| `close_position` | Where close sits in candle range |
| `breakout_lag` | Candles between zone end and breakout candle |
| `volume_ratio` | Breakout volume / zone average volume |
| `volume_trend` | Volume trend direction across zone |
| `zone_avg_volume_cv` | Volume consistency across zone |
| `zone_close_trend` | Price drift direction within zone |

---

## Annotation pairing logic

For each `triangle_breakout` annotation:
1. Find all `triangle_zone` annotations for the **same ticker**
2. Keep only zones where `zone.to_ts < breakout.from_ts`
3. Take the **most recent** zone — that's the paired training example
4. Fetch candles for the zone range and the breakout candle range
5. Use the **last candle** in the breakout range as the confirmed breakout

---

## Thresholds (config.py)

| Parameter | Default | Description |
|---|---|---|
| `ML_ALERT_THRESHOLD` | 0.60 | Minimum ML probability to fire alert |
| `RULE_ALERT_THRESHOLD` | 0.55 | Minimum rule score to fire alert |
| `MIN_VOLUME_RATIO` | 1.3 | Breakout volume must be ≥ 1.3× zone avg |
| `MIN_ZONE_CANDLES` | 8 | Minimum candles for a valid zone |
| `MAX_BREAKOUT_LAG_CANDLES` | 5 | Max candles between zone end and breakout |
| `MIN_EXAMPLES_FOR_ML` | 8 | Below this, use rule-based fallback |

---

## Re-training

Re-run `--train` any time you add new annotations. The model file is overwritten.
The scanner always loads the latest model from disk at the start of each run,
so no restart is needed after re-training.

---

## Log output

Alerts are logged to both console and `logs/scanner.log` (rotating, 5 MB × 3).

Sample alert output:
```
────────────────────────────────────────────────────────
  🔺 BREAKOUT ALERT: ROTO
────────────────────────────────────────────────────────
  Timestamp  : 2025-03-20 10:00 UTC
  Close      : 412.50
  Direction  : BULLISH
  Score      : 0.847  [mode: ml]
  Volume     : 145200  (2.8x zone avg)
  Zone       : 34 candles  02-10 09:00 → 03-19 18:00
  Upper line : 408.20   Lower line: 395.10

  ML probability: 0.847  (threshold: 0.60)

  Upper trendline flat (0.01%/candle) — descending triangle
  Lower trendline slope: -0.18%/candle
  Zone width: 34 candles
  Apex proximity: 71% through triangle
  Volume ratio at breakout: 2.8x zone avg
  Close vs upper trendline: +1.1%
  Breakout body strength: 72% of candle range
────────────────────────────────────────────────────────
```
