import logging
import threading
from collections import deque
from typing import List, Optional

_DEFAULT_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
_lock = threading.Lock()
_recent_logs: deque[str] = deque(maxlen=1000)
_installed = False
_handler: Optional[logging.Handler] = None


class RecentLogsHandler(logging.Handler):
    """Keeps a rolling in-memory buffer of recent log lines."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()

        with _lock:
            _recent_logs.append(message)


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
