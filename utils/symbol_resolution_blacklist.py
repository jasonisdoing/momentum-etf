"""심볼 해석 실패 블랙리스트.

토스/야후 등 외부 심볼 매핑에 실패한 티커를 일정 시간 동안 skip 하기 위한 파일 캐시.

- 위치: data/yahoo_resolve_blacklist.json
- TTL: 1시간 (지나면 자동 휘발 후 재시도)
- 동시성: 같은 프로세스 내에서는 threading.Lock 으로 보호.
  여러 프로세스가 동시에 쓸 가능성은 cron 스케줄상 거의 없음 (충돌 시 최악의 경우 한 번의 항목 손실 — 다음 실행에서 다시 마킹됨).
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BLACKLIST_PATH = Path(__file__).resolve().parents[1] / "data" / "yahoo_resolve_blacklist.json"
_TTL = timedelta(hours=1)
_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _load_raw() -> dict[str, dict[str, Any]]:
    if not _BLACKLIST_PATH.exists():
        return {}
    try:
        payload = json.loads(_BLACKLIST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("심볼 블랙리스트 로드 실패: %s", exc)
        return {}
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, dict):
        return {}
    return entries


def _filter_active(entries: dict[str, dict[str, Any]], now: datetime) -> dict[str, dict[str, Any]]:
    """TTL 만료 항목 제거."""
    active: dict[str, dict[str, Any]] = {}
    for symbol, info in entries.items():
        if not isinstance(info, dict):
            continue
        failed_at = _parse_iso(info.get("failed_at"))
        if failed_at is None:
            continue
        if now - failed_at >= _TTL:
            continue
        active[symbol] = info
    return active


def _save(entries: dict[str, dict[str, Any]]) -> None:
    try:
        _BLACKLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "entries": entries, "updated_at": _now_iso()}
        _BLACKLIST_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("심볼 블랙리스트 저장 실패: %s", exc)


def is_blacklisted(symbol: str) -> bool:
    """주어진 심볼이 활성 블랙리스트에 있는지 확인."""
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        return False
    now = datetime.now(timezone.utc)
    with _LOCK:
        entries = _filter_active(_load_raw(), now)
        return normalized in entries


def get_active_blacklist() -> set[str]:
    """현재 활성 블랙리스트(만료 후 제거된 상태)의 심볼 set 을 반환."""
    now = datetime.now(timezone.utc)
    with _LOCK:
        entries = _filter_active(_load_raw(), now)
        return set(entries.keys())


def mark_failed(symbol: str, *, source: str, reason: str = "") -> None:
    """심볼을 블랙리스트에 추가/갱신한다.

    Args:
        symbol: 실패한 심볼.
        source: 'toss' 또는 'yahoo' 등 호출자.
        reason: 사람이 읽을 실패 사유.
    """
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        return
    now = datetime.now(timezone.utc)
    with _LOCK:
        entries = _filter_active(_load_raw(), now)
        entries[normalized] = {
            "failed_at": _now_iso(),
            "source": source,
            "reason": reason,
        }
        _save(entries)
