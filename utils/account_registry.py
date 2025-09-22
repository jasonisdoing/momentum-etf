"""계좌 메타데이터를 로드하고 조회하기 위한 헬퍼 함수 모음입니다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ACCOUNTS_FILE = Path(__file__).resolve().parent.parent / "data" / "accounts.json"

_accounts_cache: List[Dict[str, Any]] = []
_account_map: Dict[str, Dict[str, Any]] = {}
_warned_once = False


def _refresh_cache() -> None:
    """디스크에 저장된 계좌 메타데이터를 메모리 캐시에 다시 로드합니다."""

    global _accounts_cache, _account_map, _warned_once

    try:
        with ACCOUNTS_FILE.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, list):
            raise TypeError("accounts.json 파일은 리스트 구조여야 합니다")
        normalized: List[Dict[str, Any]] = []
        mapping: Dict[str, Dict[str, Any]] = {}
        for entry in data:
            if not isinstance(entry, dict):
                continue
            account_code = str(entry.get("account") or "").strip()
            if not account_code:
                continue
            # 이후 단계에서 원본 데이터를 변경해도 캐시에 영향을 주지 않도록 얕은 복사본을 저장합니다.
            item = dict(entry)
            normalized.append(item)
            mapping[account_code] = item
        _accounts_cache = normalized
        _account_map = mapping
    except FileNotFoundError:
        if not _warned_once:
            print("경고: data/accounts.json 파일을 찾을 수 없습니다. 계좌 매핑이 비어있습니다.")
            _warned_once = True
        _accounts_cache = []
        _account_map = {}
    except Exception as exc:  # noqa: BLE001
        if not _warned_once:
            print(f"경고: 계좌 정보를 불러오지 못했습니다: {exc}")
            _warned_once = True
        _accounts_cache = []
        _account_map = {}


def load_accounts(force_reload: bool = False) -> List[Dict[str, Any]]:
    """디스크에서 필요 시 다시 로드하여 사용 가능한 계좌 목록을 반환합니다."""

    if force_reload or not _account_map:
        _refresh_cache()
    return list(_accounts_cache)


def get_account_info(account: Optional[str]) -> Optional[Dict[str, Any]]:
    """등록된 계좌 코드라면 해당 계좌의 메타데이터를 반환합니다."""

    if not account:
        return None
    load_accounts()
    return _account_map.get(account)


def get_accounts_by_country(country: Optional[str]) -> List[Dict[str, Any]]:
    """지정한 국가 코드에 속한 모든 계좌 정보를 반환합니다."""

    if not country:
        return []
    country = str(country).strip()
    return [item for item in load_accounts() if item.get("country") == country]


def get_country_for_account(
    account: Optional[str], *, fallback_to_account: bool = True
) -> Optional[str]:
    """계좌 코드에 대응하는 국가 코드를 반환합니다.

    ``fallback_to_account``가 True이고 알 수 없는 계좌라면, 기존 코드가 동작하도록
    원본 ``account`` 값을 그대로 반환합니다.
    """

    if not account:
        return None
    info = get_account_info(account)
    if info and info.get("country"):
        return str(info["country"]).strip()
    return account if fallback_to_account else None


def reload_accounts() -> None:
    """CLI나 테스트에서 사용할 때 계좌 메타데이터를 강제로 다시 로드합니다."""

    _refresh_cache()


def get_all_account_codes() -> List[str]:
    """등록된 모든 계좌 코드 목록을 반환합니다."""

    return [item.get("account") for item in load_accounts() if item.get("account")]


def iter_account_info() -> Iterable[Dict[str, Any]]:
    """계좌 메타데이터를 하나씩 읽기 전용 형태로 순회합니다."""

    yield from load_accounts()
