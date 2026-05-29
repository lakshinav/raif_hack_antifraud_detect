from __future__ import annotations

import sys
import typing

from loguru import logger

from app.settings import load_settings

LOG_FORMAT: typing.Final = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

is_logging_configured = False


def load_logging() -> None:
    global is_logging_configured  # noqa: PLW0603
    if is_logging_configured:
        return

    logger.remove()
    logger.add(
        sys.stdout,
        level=load_settings().log_level.upper(),
        format=LOG_FORMAT,
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
    is_logging_configured = True


load_logging()

app_logger = logger
