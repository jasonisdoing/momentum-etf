"""국가 기반 헬퍼에 대한 하위 호환 래퍼.

새로운 계정 기반 구현(`settings_loader`, `account_registry`)을 감싸 기존 코드를 지원합니다.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from utils.account_registry import (
    build_account_meta as build_country_meta,
)
from utils.account_registry import (
    get_icon_fallback,
    pick_default_account,
)
from utils.account_registry import (
    list_available_accounts as list_available_countries,
)
from utils.settings_loader import (
    AccountSettingsError as CountrySettingsError,
)
from utils.settings_loader import (
    get_account_precision as get_country_precision,
)
from utils.settings_loader import (
    get_account_settings as get_country_settings,
)
from utils.settings_loader import (
    get_account_strategy as get_country_strategy,
)
from utils.settings_loader import (
    get_account_strategy_sections as get_country_strategy_sections,
)
from utils.settings_loader import (
    get_slack_channel as get_country_slack_channel,
)
from utils.settings_loader import (
    get_strategy_rules,
)


def iter_countries() -> Iterable[str]:  # pragma: no cover - 호환 함수
    yield from list_available_countries()


def get_common_file_settings() -> dict[str, Any]:
    """config.py 설정을 딕셔너리로 반환합니다."""

    return {
        "REALTIME_PRICE_ENABLED": True,
    }


__all__ = [
    "CountrySettingsError",
    "list_available_countries",
    "iter_countries",
    "get_country_settings",
    "get_country_strategy",
    "get_country_strategy_sections",
    "get_country_precision",
    "get_country_slack_channel",
    "get_strategy_rules",
    "get_common_file_settings",
    "build_country_meta",
    "get_icon_fallback",
    "pick_default_account",
]
