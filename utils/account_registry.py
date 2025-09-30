"""국가 기반 설정 로더 (구 계좌 헬퍼 대체)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from utils.settings_loader import (
    CountrySettingsError,
    get_country_precision,
    get_country_settings,
    get_country_slack_channel,
    get_country_strategy,
    get_strategy_rules,
)


__all__ = [
    "CountrySettingsError",
    "list_available_countries",
    "iter_countries",
    "get_country_settings",
    "get_country_strategy",
    "get_country_precision",
    "get_country_slack_channel",
    "get_strategy_rules",
    "deprecated",
]


_SETTINGS_DIR = Path(__file__).resolve().parent.parent / "settings" / "country"


def list_available_countries() -> List[str]:
    """`data/settings/` 폴더에 존재하는 국가 코드 목록을 반환합니다."""

    if not _SETTINGS_DIR.exists():
        return []

    countries: List[str] = []
    for path in sorted(_SETTINGS_DIR.glob("*.json")):
        countries.append(path.stem.lower())
    return countries


def iter_countries() -> Iterable[str]:
    """국가 코드 이터레이터."""

    yield from list_available_countries()


def deprecated(*_args: object, **_kwargs: object) -> None:
    """계좌 기반 함수 호출 시 명확한 에러를 발생시킵니다."""

    raise RuntimeError("계좌 기반 헬퍼는 더 이상 지원되지 않습니다. 국가 기반 헬퍼를 사용하세요.")


# ---- 이하: 이전 API 호환을 위한 에러 가드 ----


def load_accounts(*args: object, **kwargs: object) -> List[Dict[str, Any]]:  # type: ignore[override]
    deprecated(args, kwargs)
    return []


def get_account_info(_account: Optional[str]) -> Optional[Dict[str, Any]]:  # type: ignore[override]
    deprecated(_account)
    return None


def get_accounts_by_country(_country: Optional[str]) -> List[Dict[str, Any]]:  # type: ignore[override]
    deprecated(_country)
    return []


def get_country_for_account(
    _account: Optional[str], *, fallback_to_account: bool = True
) -> Optional[str]:
    deprecated(_account, fallback_to_account)
    return None


def reload_accounts() -> None:
    deprecated()


def get_all_account_codes() -> List[str]:
    deprecated()
    return []


def iter_account_info() -> Iterable[Dict[str, Any]]:  # type: ignore[override]
    deprecated()
    return []


def get_strategy_rules_for_account(_account: str):  # type: ignore[override]
    deprecated(_account)


def get_strategy_dict_for_account(_account: str) -> Dict[str, Any]:  # type: ignore[override]
    deprecated(_account)
    return {}


def get_all_accounts_sorted_by_order() -> List[Dict[str, Any]]:  # type: ignore[override]
    deprecated()
    return []


def get_coin_min_holding_cost(_account: str) -> Optional[float]:  # type: ignore[override]
    deprecated(_account)
    return None
