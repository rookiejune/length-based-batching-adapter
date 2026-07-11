"""Log-file path and logger construction for LBA runs."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


def default_log_dir(cwd: Optional[Path] = None) -> Path:
    """Return the default LBA log directory."""

    if cwd is not None:
        return cwd / ".lba" / "logs"
    return Path.home() / ".lba" / "logs"


def create_run_logger(
    log_dir: Optional[Union[str, Path]] = None,
) -> tuple[logging.Logger, Path]:
    """Create a per-run file logger."""

    directory = Path(log_dir) if log_dir is not None else default_log_dir()
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    log_path = directory / f"lba-{timestamp}-{os.getpid()}.log"

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
