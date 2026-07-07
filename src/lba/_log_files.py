"""Log-file path and logger construction for LBA runs."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from time import strftime


def default_log_dir(cwd: Path | None = None) -> Path:
    """Return the default LBA log directory."""

    if cwd is not None:
        return cwd / ".lba" / "logs"
    return Path.home() / ".lba" / "logs"


def create_run_logger(log_dir: str | Path | None = None) -> tuple[logging.Logger, Path]:
    """Create a per-run file logger."""

    directory = Path(log_dir) if log_dir is not None else default_log_dir()
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / f"lba-{strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.log"

    logger = logging.getLogger(f"lba.{id(log_path)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger, log_path


def event_log_path_for(log_path: Path) -> Path:
    """Return the structured-event path next to a human log file."""

    return log_path.with_suffix(".jsonl")
