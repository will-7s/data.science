import uuid
from threading import Lock

_cache: dict[str, object] = {}
_lock = Lock()
_MAX_ENTRIES = 6


def store(obj: object) -> str:
    key = uuid.uuid4().hex
    with _lock:
        _cache[key] = obj
        if len(_cache) > _MAX_ENTRIES:
            _cache.pop(next(iter(_cache)), None)
    return key


def get(key: str) -> object | None:
    with _lock:
        return _cache.get(key)


def remove(key: str) -> None:
    with _lock:
        _cache.pop(key, None)
