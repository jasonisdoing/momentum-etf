"""국가 기반 설정 로더 (구 계좌 헬퍼 대체)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List

from utils.settings_loader import (
    CountrySettingsError,
    get_country_precision,
    get_country_settings,
    get_country_slack_channel,
    get_country_strategy,
    get_strategy_rules,
)
from settings.common import (
    MARKET_REGIME_FILTER_ENABLED,
    MARKET_REGIME_FILTER_TICKER,
    MARKET_REGIME_FILTER_MA_PERIOD,
    REALTIME_PRICE_ENABLED,
    MAX_PER_CATEGORY,
    HOLDING_STOP_LOSS_PCT,
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
    "get_common_file_settings",
]


_SETTINGS_DIR = Path(__file__).resolve().parent.parent / "settings" / "country"


def list_available_countries() -> List[str]:
    """`settings/country/` 폴더에 존재하는 국가 코드 목록을 반환합니다."""
    if not _SETTINGS_DIR.exists():
        print(f"경고: 설정 디렉토리를 찾을 수 없습니다: {_SETTINGS_DIR}")
        return []

    countries = []
    for path in sorted(_SETTINGS_DIR.glob("*.json")):
        if path.is_file() and path.suffix.lower() == ".json":
            country_code = path.stem.lower()
            countries.append(country_code)

    return countries


def iter_countries() -> Iterable[str]:
    """국가 코드 이터레이터."""
    yield from list_available_countries()


def get_common_file_settings() -> dict[str, Any]:
    """settings/common.py의 설정을 딕셔너리로 반환합니다."""

    return {
        "MARKET_REGIME_FILTER_ENABLED": MARKET_REGIME_FILTER_ENABLED,
        "MARKET_REGIME_FILTER_TICKER": MARKET_REGIME_FILTER_TICKER,
        "MARKET_REGIME_FILTER_MA_PERIOD": MARKET_REGIME_FILTER_MA_PERIOD,
        "REALTIME_PRICE_ENABLED": REALTIME_PRICE_ENABLED,
        "MAX_PER_CATEGORY": MAX_PER_CATEGORY,
        "HOLDING_STOP_LOSS_PCT": HOLDING_STOP_LOSS_PCT,
    }
