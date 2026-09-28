"""가격 캐시 쓰기와 소유자 삭제를 같은 파일 잠금으로 직렬화한다."""

from __future__ import annotations

import fcntl
import hashlib
import os
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def cache_owner_lock(owner_id: str) -> Iterator[None]:
    """소유자 삭제와 캐시 저장이 서로 지나쳐 고아 캐시를 만들지 않게 한다."""
    token = str(owner_id).strip().lower()
    if not token:
        raise ValueError("가격 캐시 소유자 ID가 필요합니다.")
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    path = f"/tmp/momentum_etf_cache_owner_{digest}.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
