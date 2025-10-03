"""계정별 설정을 파일에서 로드하기 위한 헬퍼 모듈."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class AccountSettingsError(RuntimeError):
    """계정 설정 로딩 중 발생하는 예외."""


SETTINGS_ROOT = Path(__file__).resolve().parents[1] / "settings"
ACCOUNT_SETTINGS_DIR = SETTINGS_ROOT / "account"
COMMON_SETTINGS_PATH = SETTINGS_ROOT / "common.py"
SCHEDULE_CONFIG_PATH = SETTINGS_ROOT / "schedule_config.json"


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:  # noqa: PERF203
        raise AccountSettingsError(f"설정 파일을 찾을 수 없습니다: {path}") from exc
    except OSError as exc:  # noqa: PERF203
        raise AccountSettingsError(f"설정 파일을 읽을 수 없습니다: {path}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:  # noqa: PERF203
        raise AccountSettingsError(f"설정 파일이 올바른 JSON 형식이 아닙니다: {path}") from exc

    if not isinstance(data, dict):
        raise AccountSettingsError(f"설정 파일의 루트는 객체(JSON object)여야 합니다: {path}")
    return data


@lru_cache(maxsize=None)
def get_account_settings(account_id: str) -> Dict[str, Any]:
    """`settings/account/{account}.json` 파일을 로드합니다."""

    account = (account_id or "").strip().lower()
    if not account:
        raise AccountSettingsError("계정 식별자를 지정해야 합니다.")

    path = ACCOUNT_SETTINGS_DIR / f"{account}.json"
    print(path)
    settings = _load_json(path)
    settings.setdefault("account", account)
    settings.setdefault("country_code", settings.get("country_code") or account)
    return settings


def _split_strategy_sections(strategy: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    tuning_raw = strategy.get("tuning")
    static_raw = strategy.get("static")

    if tuning_raw is None:
        tuning = {}
    elif isinstance(tuning_raw, dict):
        tuning = dict(tuning_raw)
    else:
        raise AccountSettingsError("'strategy.tuning' 항목은 객체(dict)여야 합니다.")

    if static_raw is None:
        static = {}
    elif isinstance(static_raw, dict):
        static = dict(static_raw)
    else:
        raise AccountSettingsError("'strategy.static' 항목은 객체(dict)여야 합니다.")

    return tuning, static


def get_account_strategy_sections(account_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """계정 전략 설정을 (튜닝용, 고정값)으로 분리해 반환합니다."""

    settings = get_account_settings(account_id)
    strategy = settings.get("strategy")
    if not isinstance(strategy, dict):
        raise AccountSettingsError(f"'{account_id}' 설정에서 'strategy' 항목이 누락되었거나 잘못되었습니다.")

    if "tuning" in strategy or "static" in strategy:
        return _split_strategy_sections(strategy)

    # 이전 포맷과의 호환성: 모든 값을 튜닝 영역으로 간주
    return dict(strategy), {}


def get_account_strategy(account_id: str) -> Dict[str, Any]:
    """전략 설정(dict)을 반환합니다.

    새 포맷에서는 tuning/static을 병합하여 상위 키 접근을 계속 지원합니다.
    """

    tuning, static = get_account_strategy_sections(account_id)
    merged: Dict[str, Any] = dict(static)
    merged.update(tuning)
    return merged


def get_account_precision(account_id: str) -> Dict[str, Any]:
    """표시/계산 정밀도 설정을 반환합니다."""

    settings = get_account_settings(account_id)
    precision = settings.get("precision") or {}
    if not isinstance(precision, dict):
        raise AccountSettingsError(f"'{account_id}' 설정에서 'precision' 항목이 잘못되었습니다. dict 이어야 합니다.")
    return precision


def get_account_slack_channel(account_id: str) -> Optional[str]:
    """슬랙 채널 ID(없으면 None)를 반환합니다."""

    settings = get_account_settings(account_id)
    channel = settings.get("slack_channel")
    return str(channel) if isinstance(channel, str) and channel.strip() else None


def get_strategy_rules(account_id: str):
    """계정별 전략 설정을 `StrategyRules` 객체로 반환합니다."""

    from logic.entry_point import StrategyRules

    tuning, _ = get_account_strategy_sections(account_id)
    return StrategyRules.from_mapping(tuning)


# ---------------------------------------------------------------------------
# 하위 호환 래퍼 (다음 단계에서 제거 예정)
# ---------------------------------------------------------------------------


class CountrySettingsError(AccountSettingsError):
    """기존 코드 호환을 위한 예외 alias."""


def get_country_settings(country: str) -> Dict[str, Any]:  # pragma: no cover - 호환용
    return get_account_settings(country)


def get_country_strategy_sections(
    country: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:  # pragma: no cover
    return get_account_strategy_sections(country)


def get_country_strategy(country: str) -> Dict[str, Any]:  # pragma: no cover
    return get_account_strategy(country)


def get_country_precision(country: str) -> Dict[str, Any]:  # pragma: no cover
    return get_account_precision(country)


def get_country_slack_channel(country: str) -> Optional[str]:  # pragma: no cover
    return get_account_slack_channel(country)
