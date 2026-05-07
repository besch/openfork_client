import logging
import re
import threading
from collections import deque
from typing import List, Optional

_DEFAULT_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
_lock = threading.Lock()
_recent_logs: deque[str] = deque(maxlen=1000)
_installed = False
_handler: Optional[logging.Handler] = None
_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization:\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)(x-dgn-api-key[=:]\s*)[^\s,;]+"),
    re.compile(r"(?i)\b(dgn_)[a-f0-9]{32,}\b"),
    re.compile(r"(?i)\b([a-z0-9_]*(?:api[_-]?key|secret|token|password)[a-z0-9_]*\s*[=:]\s*)[^\s,;]+"),
)


def _redact_secrets(message: str) -> str:
    redacted = message
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(r"\1[REDACTED]", redacted)
    return redacted


class RecentLogsHandler(logging.Handler):
    """Keeps a rolling in-memory buffer of recent log lines."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()

        with _lock:
            _recent_logs.append(_redact_secrets(message))


def install_recent_logs_handler(max_lines: int = 1000) -> None:
    """Attach a root logger handler once so failures can persist recent logs."""
    global _installed, _handler, _recent_logs

    if _installed:
        return

    _recent_logs = deque(maxlen=max(100, max_lines))

    handler = RecentLogsHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))

    logging.getLogger().addHandler(handler)

    _handler = handler
    _installed = True


def get_recent_logs_tail(lines: int = 100) -> List[str]:
    with _lock:
        if lines <= 0:
            return []
        return list(_recent_logs)[-lines:]
