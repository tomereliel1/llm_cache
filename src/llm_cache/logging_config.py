from __future__ import annotations

import logging
import os

from llm_cache.request_context import get_current_request_id

DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(service_name)s | %(levelname)s | request_id=%(request_id)s | %(message)s"
)


class RequestContextFilter(logging.Filter):
    def __init__(self, service_name: str) -> None:
        super().__init__()
        self._service_name = service_name

    def filter(self, record: logging.LogRecord) -> bool:
        record.service_name = self._service_name
        record.request_id = get_current_request_id() or "-"
        return True


def configure_logging(service_name: str) -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler()
    handler.addFilter(RequestContextFilter(service_name))
    handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
