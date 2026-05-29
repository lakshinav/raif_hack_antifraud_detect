from __future__ import annotations

import logging
import sys

import structlog
import structlog.typing

from app.settings import load_settings

APP_LOGGER_NAME = "red_flag_detector"
STD_LOGGER_NAMES = ("uvicorn", "uvicorn.error", "uvicorn.access", "httpx", "httpcore")

is_logging_configured = False
app_logger: structlog.stdlib.BoundLogger = structlog.get_logger(APP_LOGGER_NAME)


def load_logging() -> None:
    global app_logger, is_logging_configured  # noqa: PLW0603
    if is_logging_configured:
        return

    log_level = load_settings().log_level.upper()
    shared_processors: tuple[structlog.typing.Processor, ...] = (
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(stream_handler)
    root_logger.setLevel(log_level)

    for one_logger_name in STD_LOGGER_NAMES:
        standard_logger = logging.getLogger(one_logger_name)
        standard_logger.handlers.clear()
        standard_logger.propagate = True
        standard_logger.setLevel(log_level)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )

    app_logger = structlog.get_logger(APP_LOGGER_NAME)
    is_logging_configured = True


load_logging()
