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

from .config import LOG_LEVEL, LOG_PATH


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

    # Attach the file handler to the package-level logger, not the root logger.
    # When running inside FastAPI, main.py sets propagate=False on this logger
    # (to avoid double console output via uvicorn), which means scanner messages
    # never reach the root logger.  Handlers on the package logger fire regardless
    # of propagate, so attaching here works in both standalone and FastAPI modes.
    pkg_logger = logging.getLogger(__name__.split('.')[0])  # "traingle_breakout_training"

    # Guard: skip if our file handler is already attached.
    target_path = str(Path(LOG_PATH).resolve())
    for h in pkg_logger.handlers:
        if (isinstance(h, logging.handlers.BaseRotatingHandler)
                and getattr(h, 'baseFilename', None) == target_path):
            return

    pkg_logger.setLevel(numeric)

    # Console handler on root — only in standalone mode (no other framework running).
    # In FastAPI mode uvicorn owns the console; we leave root alone.
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(numeric)
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
    pkg_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in ("urllib3", "requests", "xgboost"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# Auto-setup on import
setup()
