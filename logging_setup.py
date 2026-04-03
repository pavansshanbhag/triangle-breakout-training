"""
logging_setup.py — configure logging for the triangle breakout system.

Import this module at the top of any entry point (main.py, scanner.py CLI).
It sets up:
  - Coloured console output (INFO+)
  - Rotating file handler → logs/scanner.log
"""

import logging
import logging.handlers
import sys
from pathlib import Path

from config import LOG_LEVEL, LOG_PATH


class _ColourFormatter(logging.Formatter):
    """Minimal ANSI colour formatter for terminal output."""

    GREY    = "\x1b[38;5;245m"
    CYAN    = "\x1b[36m"
    YELLOW  = "\x1b[33m"
    RED     = "\x1b[31m"
    BOLD_RED= "\x1b[1;31m"
    GREEN   = "\x1b[32m"
    RESET   = "\x1b[0m"

    FMT = "%(asctime)s  %(levelname)-8s  %(name)-22s  %(message)s"
    DATEFMT = "%H:%M:%S"

    LEVEL_COLOURS = {
        logging.DEBUG:    GREY,
        logging.INFO:     CYAN,
        logging.WARNING:  YELLOW,
        logging.ERROR:    RED,
        logging.CRITICAL: BOLD_RED,
    }

    def format(self, record):
        colour = self.LEVEL_COLOURS.get(record.levelno, self.RESET)
        formatter = logging.Formatter(
            fmt    = colour + self.FMT + self.RESET,
            datefmt= self.DATEFMT,
        )
        return formatter.format(record)


def setup(level: str = None):
    level = level or LOG_LEVEL
    numeric = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    if root.handlers:
        return   # already configured

    root.setLevel(numeric)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(numeric)
    console.setFormatter(_ColourFormatter())
    root.addHandler(console)

    # File handler (rotating, 5 MB × 3 files)
    log_path = Path(LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(numeric)
    file_handler.setFormatter(logging.Formatter(
        fmt    = "%(asctime)s  %(levelname)-8s  %(name)-22s  %(message)s",
        datefmt= "%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in ("urllib3", "requests", "xgboost"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# Auto-setup on import
setup()
