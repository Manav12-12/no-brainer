from __future__ import annotations

import logging
import re

_KEY_PATTERN = re.compile(r"(?i)(bearer\s+|apikey_)[A-Za-z0-9_.-]+")


def redact(value: object) -> str:
    return _KEY_PATTERN.sub(lambda match: match.group(1) + "[REDACTED]", str(value))


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = ()
        return True


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.addFilter(RedactingFilter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
