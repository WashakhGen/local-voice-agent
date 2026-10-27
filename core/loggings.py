import logging
import pathlib
from typing import Any

from core.config import CONFIGS

MAIN_LOG_FILE = f"{CONFIGS.LOG_DIR}/api-main.log"


FORMATTER = logging.Formatter(
    "[%(asctime)s.%(msecs)03d000] %(message)s",
    "%Y-%m-%d %H:%M:%S",
)


def setup_logger(
    name: str,
    level: int | str,
    log_file: str | pathlib.Path | None = None,
    stream: bool = True,
    formatter: logging.Formatter | None = FORMATTER,
    recreate: bool = False,
) -> logging.Logger:
    "Creates or retrieves the requested logger. Will not configure an extant logger."
    logger = logging.getLogger(name)

    if logger.hasHandlers() and not recreate:  # Logger already configured
        return logger
    elif recreate:  # Remove if overwriting
        [logger.removeHandler(h) for h in reversed(logger.handlers)]

    logger.setLevel(level)

    if log_file is not None:
        pathlib.Path(log_file).resolve().parent.mkdir(parents=True, exist_ok=True)
        file_handle = logging.FileHandler(log_file, mode="a")
        file_handle.setFormatter(formatter)
        logger.addHandler(file_handle)

    if stream:
        stream_handle = logging.StreamHandler()
        stream_handle.setFormatter(formatter)
        logger.addHandler(stream_handle)

    return logger


def main_logger(recreate: bool = False) -> logging.Logger:
    return setup_logger("log_main", logging.INFO, MAIN_LOG_FILE, recreate=recreate)


def log_main(*messages: object, extra: dict[str, Any] | None = None) -> None:
    for message in messages:
        main_logger().info(message, extra=extra)
