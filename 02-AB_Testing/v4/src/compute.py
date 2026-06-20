from __future__ import annotations

import threading
import time
from typing import Any, Callable

_RUN_ID_LOCK = threading.Lock()
_RUN_ID_COUNTER = 0


def _next_run_id() -> str:
    global _RUN_ID_COUNTER
    with _RUN_ID_LOCK:
        _RUN_ID_COUNTER += 1
        return f"run-{_RUN_ID_COUNTER}"


class CancelToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def event(self) -> threading.Event:
        return self._event


class ProgressTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}

    def init(self, run_id: str) -> None:
        with self._lock:
            self._data[run_id] = {
                "phase": "",
                "pct": 0,
                "status": "running",
                "start_time": time.time(),
                "end_time": None,
            }

    def update(self, run_id: str, phase: str, pct: int) -> None:
        with self._lock:
            if run_id in self._data:
                self._data[run_id]["phase"] = phase
                self._data[run_id]["pct"] = pct

    def get(self, run_id: str) -> dict:
        with self._lock:
            return dict(self._data.get(run_id, {}))

    def set_status(self, run_id: str, status: str) -> None:
        with self._lock:
            if run_id in self._data:
                self._data[run_id]["status"] = status

    def finish(self, run_id: str) -> None:
        with self._lock:
            if run_id in self._data:
                end = time.time()
                self._data[run_id].update(status="done", pct=100, end_time=end)

    def remove(self, run_id: str) -> None:
        with self._lock:
            self._data.pop(run_id, None)


_tracker = ProgressTracker()


def get_tracker() -> ProgressTracker:
    return _tracker


class RunResult:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict | None = None

    def set(self, data: dict) -> None:
        with self._lock:
            self._data = data

    def get(self) -> dict | None:
        with self._lock:
            return self._data

    def clear(self) -> None:
        with self._lock:
            self._data = None


def run_async(
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> tuple[str, CancelToken]:
    run_id = _next_run_id()
    cancel_token = CancelToken()
    tracker = _tracker
    tracker.init(run_id)

    def wrapper() -> None:
        try:
            fn(tracker, cancel_token, run_id, *args, **kwargs)
            status = tracker.get(run_id).get("status", "running")
            if status != "error":
                tracker.finish(run_id)
        except Exception:
            tracker.set_status(run_id, "error")
            raise

    t = threading.Thread(target=wrapper, daemon=True)
    t.start()
    return run_id, cancel_token
