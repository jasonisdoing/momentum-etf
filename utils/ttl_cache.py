"""TTL 메모리 캐시 — 프로세스 내 계산 결과를 짧게 재사용하는 공용 구현.

TTL 값은 직접 적지 않고 ``config`` 의 ``CACHE_TTL_*`` 중 성격에 맞는 것을 골라 넘긴다.
같은 키를 동시에 요청하면 하나만 계산하고 나머지는 그 결과를 받는다(중복 계산 방지).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Hashable
from copy import deepcopy
from threading import Lock
from time import monotonic
from typing import Any


def _normalize(part: Any) -> Any:
    """캐시 키 조각을 해시 가능한 값으로 바꾼다 — dict/list 는 정렬된 JSON 문자열."""
    if isinstance(part, (dict, list, tuple, set)):
        return json.dumps(part, sort_keys=True, default=str, ensure_ascii=False)
    return part


class TtlCache:
    """키별로 TTL 동안 값을 보관하는 캐시. 반환값은 항상 복사본이다.

    ``max_entries`` 를 주면 그 수를 넘을 때 가장 오래 담긴 항목부터 버린다. 값 하나가
    무거운 캐시(가격 프레임 등)에 쓴다 — 만료는 그 키를 다시 볼 때만 걷히므로, 상한이
    없으면 아무도 안 찾는 옛 항목이 TTL 이 지나도 메모리에 남는다.
    """

    def __init__(self, ttl_seconds: float, *, name: str = "", max_entries: int | None = None) -> None:
        self.ttl_seconds = float(ttl_seconds)
        self.name = name
        self.max_entries = max_entries
        self._entries: dict[Hashable, tuple[float, Any]] = {}
        self._lock = Lock()
        self._inflight_locks: dict[Hashable, Lock] = {}

    @staticmethod
    def make_key(*parts: Any) -> tuple[Any, ...]:
        """dict 를 포함한 인자들로 캐시 키를 만든다."""
        return tuple(_normalize(part) for part in parts)

    def get(self, key: Hashable) -> Any | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            cached_at, payload = entry
            if monotonic() - cached_at > self.ttl_seconds:
                self._entries.pop(key, None)
                return None
            return deepcopy(payload)

    def set(self, key: Hashable, payload: Any) -> None:
        with self._lock:
            # 다시 넣는 키는 맨 뒤로 — dict 는 삽입 순서를 지키므로 그게 곧 나이 순서다.
            self._entries.pop(key, None)
            self._entries[key] = (monotonic(), deepcopy(payload))
            if self.max_entries is None:
                return
            now = monotonic()
            for stale in [k for k, (at, _) in self._entries.items() if now - at > self.ttl_seconds]:
                self._entries.pop(stale, None)
            while len(self._entries) > self.max_entries:
                self._entries.pop(next(iter(self._entries)))

    def invalidate(self, predicate: Callable[[Any], bool] | None = None) -> None:
        """캐시를 비운다. ``predicate`` 를 주면 참인 키만 지운다."""
        with self._lock:
            if predicate is None:
                self._entries.clear()
                return
            for key in [key for key in self._entries if predicate(key)]:
                self._entries.pop(key, None)

    def _inflight_lock(self, key: Hashable) -> Lock:
        with self._lock:
            lock = self._inflight_locks.get(key)
            if lock is None:
                lock = Lock()
                self._inflight_locks[key] = lock
            return lock

    def get_or_compute(self, key: Hashable, factory: Callable[[], Any]) -> Any:
        """캐시가 있으면 그대로, 없으면 계산해서 채운다.

        같은 키를 동시에 요청하면 앞선 계산이 끝날 때까지 기다렸다가 그 결과를 쓴다.
        """
        cached = self.get(key)
        if cached is not None:
            return cached

        with self._inflight_lock(key):
            cached = self.get(key)
            if cached is not None:
                return cached
            payload = factory()
            self.set(key, payload)
            return deepcopy(payload)
