"""국가별 설정을 파일에서 로드하기 위한 헬퍼 모듈."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional


class CountrySettingsError(RuntimeError):
    """설정 로딩 중 발생하는 예외."""


SETTINGS_ROOT = Path(__file__).resolve().parents[1] / "settings"
COUNTRY_SETTINGS_DIR = SETTINGS_ROOT / "country"
COMMON_SETTINGS_PATH = SETTINGS_ROOT / "common.py"
SCHEDULE_CONFIG_PATH = SETTINGS_ROOT / "schedule_config.json"


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:  # noqa: PERF203
        raise CountrySettingsError(f"설정 파일을 찾을 수 없습니다: {path}") from exc
    except OSError as exc:  # noqa: PERF203
        raise CountrySettingsError(f"설정 파일을 읽을 수 없습니다: {path}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:  # noqa: PERF203
        raise CountrySettingsError(f"설정 파일이 올바른 JSON 형식이 아닙니다: {path}") from exc

    if not isinstance(data, dict):
        raise CountrySettingsError(f"설정 파일의 루트는 객체(JSON object)여야 합니다: {path}")
    return data


@lru_cache(maxsize=None)
def get_country_settings(country: str) -> Dict[str, Any]:
    """`settings/country/{country}.json` 파일을 로드합니다."""

    country = (country or "").strip().lower()
    if not country:
        raise CountrySettingsError("국가 코드를 지정해야 합니다.")

    path = COUNTRY_SETTINGS_DIR / f"{country}.json"
    settings = _load_json(path)
    settings.setdefault("country", country)
    return settings


def get_country_strategy(country: str) -> Dict[str, Any]:
    """전략 설정(dict)을 반환합니다."""

    settings = get_country_settings(country)
    strategy = settings.get("strategy")
    if not isinstance(strategy, dict):
        raise CountrySettingsError(f"'{country}' 설정에서 'strategy' 항목이 누락되었거나 잘못되었습니다.")
    return strategy


def get_country_precision(country: str) -> Dict[str, Any]:
    """표시/계산 정밀도 설정을 반환합니다."""

    settings = get_country_settings(country)
    precision = settings.get("precision") or {}
    if not isinstance(precision, dict):
        raise CountrySettingsError(f"'{country}' 설정에서 'precision' 항목이 잘못되었습니다. dict 이어야 합니다.")
    return precision


def get_country_slack_channel(country: str) -> Optional[str]:
    """슬랙 채널 ID(없으면 None)를 반환합니다."""

    settings = get_country_settings(country)
    channel = settings.get("slack_channel")
    return str(channel) if isinstance(channel, str) and channel.strip() else None


def get_strategy_rules(country: str):
    """국가별 전략 설정을 `StrategyRules` 객체로 반환합니다."""

    from logic.strategies.momentum.rules import StrategyRules

    strategy = get_country_strategy(country)
    return StrategyRules.from_mapping(strategy)
