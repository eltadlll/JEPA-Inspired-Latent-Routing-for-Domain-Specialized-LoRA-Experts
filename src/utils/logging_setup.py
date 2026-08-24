"""
Centralized logging configuration using loguru.

Every module in the pipeline imports `get_logger(__name__)` rather than
configuring logging itself. `configure_logging` is called exactly once,
by the CLI entrypoint, based on the validated `LoggingConfig`.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from src.config.schema import LoggingConfig

_CONFIGURED = False


def configure_logging(logging_config: LoggingConfig, log_dir: Path) -> None:
    """Configure the global loguru logger. Safe to call once at startup."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    logger.remove()  # drop default handler

    logger.add(
        sys.stderr,
        level=logging_config.level.value,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        ),
    )

    if logging_config.log_to_file:
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_dir / logging_config.log_file_name),
            level=logging_config.level.value,
            rotation=logging_config.rotation,
            retention=logging_config.retention,
            serialize=logging_config.serialize_json,
            enqueue=True,
            backtrace=True,
            diagnose=False,
        )

    _CONFIGURED = True


def get_logger(name: str):
    """Return a loguru logger bound with a module name for context."""
    return logger.bind(module=name)
