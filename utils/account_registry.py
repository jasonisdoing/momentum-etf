"""계정(국가) 설정 메타데이터 로더."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from utils.settings_loader import CountrySettingsError, get_country_settings


_SETTINGS_DIR = Path(__file__).resolve().parent.parent / "settings" / "country"
_ICON_FALLBACKS: Dict[str, str] = {
    "kor": "🇰🇷",
    "aus": "🇦🇺",
}


def _normalize_code(value: Any, fallback: str) -> str:
    text = str(value or "").strip().lower()
    return text or fallback


def _resolve_order(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def load_account_configs() -> List[Dict[str, Any]]:
    """`settings/country`에 정의된 계정 정보를 정렬된 리스트로 반환합니다."""

    accounts: List[Dict[str, Any]] = []

    if not _SETTINGS_DIR.exists():
        print(f"경고: 계정 설정 디렉터리를 찾을 수 없습니다: {_SETTINGS_DIR}")
        return accounts

    for path in sorted(_SETTINGS_DIR.glob("*.json")):
        if not path.is_file():
            continue

        account_id = path.stem
        try:
            settings = get_country_settings(account_id)
        except CountrySettingsError as exc:
            print(f"[WARN] 계정 설정 로딩 실패({account_id}): {exc}")
            continue

        country_code = _normalize_code(settings.get("country_code"), account_id)
        name = settings.get("name") or account_id.upper()
        icon = settings.get("icon") or _ICON_FALLBACKS.get(country_code, "")
        is_default = bool(settings.get("default", False))
        order = _resolve_order(settings.get("order"))

        accounts.append(
            {
                "account_id": account_id,
                "country_code": country_code,
                "name": name,
                "icon": icon,
                "is_default": is_default,
                "order": order,
                "settings": settings,
            }
        )

    accounts.sort(key=lambda acc: (acc["order"], acc["name"]))
    return accounts


def pick_default_account(accounts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """기본으로 선택할 계정 설정을 결정합니다."""

    if not accounts:
        raise ValueError("선택 가능한 계정이 없습니다.")

    for account in accounts:
        if account.get("is_default"):
            return account

    for account in accounts:
        if account.get("country_code") == "kor":
            return account

    return accounts[0]


def get_icon_fallback(country_code: str) -> str:
    return _ICON_FALLBACKS.get((country_code or "").strip().lower(), "")


def build_account_meta(accounts: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    meta: Dict[str, Dict[str, str]] = {}
    for account in accounts:
        account_id = account["account_id"]
        meta[account_id] = {
            "label": account["name"],
            "icon": account.get("icon", ""),
            "country_code": account.get("country_code", account_id),
        }
    return meta


__all__ = [
    "load_account_configs",
    "pick_default_account",
    "build_account_meta",
    "get_icon_fallback",
]
