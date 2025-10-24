"""계정 설정 메타데이터 로더."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from utils.logger import get_app_logger
from utils.settings_loader import load_common_settings

from utils.settings_loader import (
    AccountSettingsError,
    get_account_precision,
    get_account_settings,
    get_account_slack_channel,
    get_account_strategy,
    get_account_strategy_sections,
    get_strategy_rules,
)


_SETTINGS_DIR = Path(__file__).resolve().parent.parent / "data" / "settings" / "account"
_ICON_FALLBACKS: Dict[str, str] = {
    "kor": "🇰🇷",
    "aus": "🇦🇺",
}

logger = get_app_logger()


def _normalize_code(value: Any, fallback: str) -> str:
    text = str(value or "").strip().lower()
    return text or fallback


def _resolve_order(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def list_available_accounts() -> List[str]:
    """`data/settings/account`에 존재하는 계정 ID 목록을 반환합니다."""

    if not _SETTINGS_DIR.exists():
        logger.warning("계정 설정 디렉터리를 찾을 수 없습니다: %s", _SETTINGS_DIR)
        return []

    return [path.stem.lower() for path in sorted(_SETTINGS_DIR.glob("*.json")) if path.is_file() and path.suffix.lower() == ".json"]


def load_account_configs() -> List[Dict[str, Any]]:
    """`data/settings/account`에 정의된 계정 정보를 정렬된 리스트로 반환합니다."""

    configs: List[Dict[str, Any]] = []

    for account_id in list_available_accounts():
        try:
            settings = get_account_settings(account_id)
        except AccountSettingsError as exc:
            logger.warning("계정 설정 로딩 실패(%s): %s", account_id, exc)
            continue

        country_code = _normalize_code(settings.get("country_code"), account_id)
        base_name = settings.get("name") or account_id.upper()

        # PORTFOLIO_TOPN을 이름에 추가
        portfolio_topn = None
        strategy = settings.get("strategy", {})
        if isinstance(strategy, dict):
            tuning = strategy.get("tuning", {})
            if isinstance(tuning, dict):
                portfolio_topn = tuning.get("PORTFOLIO_TOPN")

        if portfolio_topn is not None:
            name = f"{base_name}({portfolio_topn} 종목 포트폴리오)"
        else:
            name = base_name

        icon = settings.get("icon") or _ICON_FALLBACKS.get(country_code, "")
        is_default = bool(settings.get("default", False))
        order = _resolve_order(settings.get("order"))

        configs.append(
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

    configs.sort(key=lambda acc: (acc["order"], acc["name"]))
    return configs


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


def iter_accounts() -> Iterable[str]:
    """계정 ID 이터레이터."""

    yield from list_available_accounts()


def get_common_file_settings() -> dict[str, Any]:
    """config.py의 공통 설정을 딕셔너리로 반환합니다."""

    try:
        return load_common_settings()
    except Exception as exc:
        logger.warning("공통 설정 로드 실패: %s", exc)
        return {}


__all__ = [
    "AccountSettingsError",
    "list_available_accounts",
    "iter_accounts",
    "load_account_configs",
    "pick_default_account",
    "build_account_meta",
    "get_icon_fallback",
    "get_account_settings",
    "get_account_strategy",
    "get_account_strategy_sections",
    "get_account_precision",
    "get_account_slack_channel",
    "get_strategy_rules",
    "get_common_file_settings",
]
