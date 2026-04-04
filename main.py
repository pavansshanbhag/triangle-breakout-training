"""
main.py — entry point for the triangle breakout system.

Usage
─────
  python main.py --train              # (re)train the model from QuestDB annotations
  python main.py --scan               # run scanner once across all configured tickers
  python main.py --scan --ticker ROTO # scan a single ticker
  python main.py --scan --from-date 2024-06-01 --to-date 2024-09-30  # snapshot at end of window
  python main.py --backtest --from-date 2024-06-01 --to-date 2024-09-30          # all breakouts in range
  python main.py --backtest --from-date 2024-06-01 --to-date 2024-09-30 --ticker ROTO
  python main.py --schedule           # start the hourly scheduler (blocking)
  python main.py --status             # print model status + feature importances
  python main.py --train --scan       # train then immediately scan
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from traingle_breakout_training import logging_setup  # noqa — must be first, sets up handlers

from traingle_breakout_training.config import MODEL_PATH, FEATURE_IMPORTANCE_PATH, SCAN_TICKERS, SCANNER_CRON
from traingle_breakout_training.trainer import train, load_model

logger = logging.getLogger(__name__)


# ── Scheduler (simple cron-like, no external dependency) ─────────────────────

def _parse_cron_minute(cron: str) -> int:
    """Extract the minute field from a cron expression like '2 * * * *'."""
    return int(cron.split()[0])


def run_scheduler(tickers: list[str]):
    """
    Blocking hourly scheduler.
    Waits until HH:MM (SCANNER_CRON minute) and fires scan_all().
    """
    from traingle_breakout_training.scanner import scan_all

    target_minute = _parse_cron_minute(SCANNER_CRON)
    logger.info("Scheduler started — will scan at HH:%02d each hour", target_minute)
    logger.info("Tickers: %s", ", ".join(tickers))

    last_run_hour = -1

    while True:
        now = datetime.now(timezone.utc)

        if now.minute == target_minute and now.hour != last_run_hour:
            logger.info("Scheduler firing at %s", now.strftime("%Y-%m-%d %H:%M UTC"))
            try:
                alerts = scan_all(tickers)
            except Exception as e:
                logger.error("Scheduler scan failed: %s", e, exc_info=True)
            last_run_hour = now.hour

        time.sleep(15)   # check every 15 seconds


# ── Status printer ────────────────────────────────────────────────────────────

def print_status():
    model_path = Path(MODEL_PATH)
    imp_path   = Path(FEATURE_IMPORTANCE_PATH)

    logger.info("═" * 56)
    logger.info("  TRIANGLE BREAKOUT SYSTEM — STATUS")
    logger.info("═" * 56)

    if not model_path.exists():
        logger.info("  Model      : NOT TRAINED  (run --train)")
    else:
        bundle = load_model()
        if bundle:
            cv = bundle.get("cv_results", {})
            logger.info("  Model      : %s", model_path)
            logger.info("  Examples   : %d", bundle.get("n_training_examples", "?"))
            logger.info("  LOOCV acc  : %.2f", cv.get("accuracy", 0))
            logger.info("  LOOCV prec : %.2f", cv.get("precision", 0))
            logger.info("  LOOCV rec  : %.2f", cv.get("recall", 0))
            logger.info("  Confusion  : TP=%d FP=%d FN=%d TN=%d",
                        cv.get("tp",0), cv.get("fp",0), cv.get("fn",0), cv.get("tn",0))

    if imp_path.exists():
        logger.info("")
        logger.info("  Feature importances (top 10):")
        with open(imp_path) as f:
            imps = json.load(f)
        for feat, score in list(imps.items())[:10]:
            bar = "█" * int(score * 40)
            logger.info("    %-35s %s  %.4f", feat, bar, score)

    logger.info("═" * 56)
    logger.info("  Configured tickers: %s", ", ".join(SCAN_TICKERS))
    logger.info("  Scanner cron      : %s  (HH:%02d)", SCANNER_CRON, _parse_cron_minute(SCANNER_CRON))
    logger.info("═" * 56)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Triangle Breakout Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--train",      action="store_true", help="Train/retrain the ML model")
    parser.add_argument("--scan",       action="store_true", help="Run scanner once (latest candles)")
    parser.add_argument("--backtest",   action="store_true", help="Continuous window scan between --from-date and --to-date")
    parser.add_argument("--schedule",   action="store_true", help="Start hourly scheduler (blocking)")
    parser.add_argument("--status",     action="store_true", help="Print model status")
    parser.add_argument("--ticker",     type=str,            help="Limit scan/backtest to one ticker")
    parser.add_argument("--from-date",  type=str,            help="Start date (YYYY-MM-DD) for --scan or --backtest")
    parser.add_argument("--to-date",    type=str,            help="End date   (YYYY-MM-DD) for --scan or --backtest")

    args = parser.parse_args()

    if not any([args.train, args.scan, args.backtest, args.schedule, args.status]):
        parser.print_help()
        sys.exit(0)

    tickers = [args.ticker.upper()] if args.ticker else SCAN_TICKERS

    from_ts = datetime.strptime(args.from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.from_date else None
    to_ts   = datetime.strptime(args.to_date,   "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.to_date   else None

    # ── Train ──────────────────────────────────────────────────────────────
    if args.train:
        result = train()
        logger.info("Training result: mode=%s  positives=%d  negatives=%d",
                    result.mode, result.n_positive, result.n_negative)
        if result.mode == "ml":
            logger.info("LOOCV  acc=%.2f  prec=%.2f  recall=%.2f",
                        result.loocv_accuracy, result.loocv_precision, result.loocv_recall)

    # ── Scan once ──────────────────────────────────────────────────────────
    if args.scan:
        from traingle_breakout_training.scanner import scan_all, scan_ticker
        model_bundle = load_model()

        if args.ticker:
            alert = scan_ticker(args.ticker.upper(), model_bundle, from_ts=from_ts, to_ts=to_ts)
            if not alert:
                logger.info("No breakout detected for %s", args.ticker.upper())
        else:
            scan_all(tickers, from_ts=from_ts, to_ts=to_ts)

    # ── Backtest (continuous sliding window) ───────────────────────────────
    if args.backtest:
        if not from_ts or not to_ts:
            logger.error("--backtest requires both --from-date and --to-date")
            sys.exit(1)

        from traingle_breakout_training.scanner import scan_all_continuous, scan_ticker_continuous
        model_bundle = load_model()

        if args.ticker:
            scan_ticker_continuous(args.ticker.upper(), model_bundle, from_ts, to_ts)
        else:
            scan_all_continuous(from_ts, to_ts, tickers)

    # ── Status ─────────────────────────────────────────────────────────────
    if args.status:
        print_status()

    # ── Scheduler (blocking — must be last) ────────────────────────────────
    if args.schedule:
        run_scheduler(tickers)


if __name__ == "__main__":
    main()
