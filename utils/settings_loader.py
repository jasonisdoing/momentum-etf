"""계정별 설정을 파일에서 로드하기 위한 헬퍼 모듈."""

from __future__ import annotations

import json
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

from utils.logger import get_app_logger


class AccountSettingsError(RuntimeError):
    """계정 설정 로딩 중 발생하는 예외."""


SETTINGS_ROOT = Path(__file__).resolve().parents[1] / "zaccounts"
ACCOUNT_SETTINGS_DIR = SETTINGS_ROOT  # Backward compatibility alias
COMMON_SETTINGS_PATH = SETTINGS_ROOT / "common.py"
SCHEDULE_CONFIG_PATH = SETTINGS_ROOT / "schedule_config.json"
logger = get_app_logger()


def list_available_accounts() -> list[str]:
    """
    zaccounts 디렉토리 하위의 유효한 계정(디렉토리 내 config.json 존재) 목록을 반환합니다.
    """
    accounts = []
    if not SETTINGS_ROOT.exists():
        return []

    for item in SETTINGS_ROOT.iterdir():
        if item.is_dir() and not item.name.startswith(".") and not item.name.startswith("_"):
            config_path = item / "config.json"
            if config_path.exists():
                accounts.append(item.name)

    return sorted(accounts)


def _load_json(path: Path) -> dict[str, Any]:
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


def get_tune_month_configs(account_id: str = None) -> list[dict[str, Any]]:
    """튜닝용 MONTHS_RANGE 설정을 반환합니다.

    계정별 strategy.MONTHS_RANGE를 사용합니다.
    """
    normalized: list[dict[str, Any]] = []

    def _append(months_raw: Any, *, weight: float = 1.0, source: Any = None) -> None:
        try:
            months_range = int(months_raw)
        except (TypeError, ValueError):
            return
        if months_range <= 0:
            return
        normalized.append(
            {
                "months_range": months_range,
                "weight": float(weight),
                "source": source,
            }
        )

    # 계정별 strategy.MONTHS_RANGE 사용
    if account_id:
        try:
            account_settings = get_account_settings(account_id)
            strategy = account_settings.get("strategy", {})
            account_months = strategy.get("MONTHS_RANGE")
            if account_months is not None:
                _append(account_months, weight=1.0, source=f"account_{account_id}")
        except Exception:
            pass

    if not normalized:
        return []

    seen: dict[int, dict[str, Any]] = {}
    for entry in normalized:
        months_range = entry["months_range"]
        if months_range not in seen:
            seen[months_range] = entry

    return list(seen.values())


@cache
def get_account_settings(account_id: str) -> dict[str, Any]:
    """`zaccounts/{account}/config.json` 파일을 로드합니다."""

    account = (account_id or "").strip().lower()
    if not account:
        raise AccountSettingsError("계정 식별자를 지정해야 합니다.")

    # New: zaccounts/<account>/config.json
    path = SETTINGS_ROOT / account / "config.json"
    logger.debug("계정 설정 로드: %s", path)

    settings = _load_json(path)
    settings.setdefault("account", account)

    if not settings.get("country_code"):
        raise AccountSettingsError(f"'{path}' 설정 파일에 필수 항목 'country_code'가 누락되었습니다.")

    return settings


def _split_strategy_sections(
    strategy: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
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


def resolve_strategy_params(strategy_cfg: Any) -> dict[str, Any]:
    """전략 설정에서 실제 파라미터(dict)를 추출합니다.

    최신 포맷은 strategy 하위에 바로 값을 두고, 구 포맷은 strategy.tuning에 둡니다.
    """

    if not isinstance(strategy_cfg, dict):
        return {}

    tuning = strategy_cfg.get("tuning")
    if isinstance(tuning, dict) and tuning:
        return dict(tuning)

    return dict(strategy_cfg)


def get_account_strategy_sections(
    account_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """계정 전략 설정을 (튜닝용, 고정값)으로 분리해 반환합니다.

    설정을 반환하기 전에 모든 필수 항목을 검증합니다.
    """
    from utils.strategy_validator import validate_strategy_settings

    settings = get_account_settings(account_id)
    strategy = settings.get("strategy")
    if not isinstance(strategy, dict):
        raise AccountSettingsError(f"'{account_id}' 설정에서 'strategy' 항목이 누락되었거나 잘못되었습니다.")

    if "tuning" in strategy or "static" in strategy:
        tuning, static = _split_strategy_sections(strategy)
    else:
        # 이전 포맷과의 호환성: 모든 값을 튜닝 영역으로 간주
        tuning, static = dict(strategy), {}

    # 전략 설정 검증 (한 번만 수행)
    validate_strategy_settings(tuning, account_id)

    return tuning, static


def get_account_strategy(account_id: str) -> dict[str, Any]:
    """전략 설정(dict)을 반환합니다.

    새 포맷에서는 tuning/static을 병합하여 상위 키 접근을 계속 지원합니다.
    """

    tuning, static = get_account_strategy_sections(account_id)
    merged: dict[str, Any] = dict(static)
    merged.update(tuning)
    return merged


def get_account_precision(account_id: str) -> dict[str, Any]:
    """표시/계산 정밀도 설정을 반환합니다."""

    settings = get_account_settings(account_id)
    country_code = (settings.get("country_code") or account_id).strip().lower()
    if country_code in ("us", "usa"):
        return {
            "currency": "USD",
            "qty_precision": 0,
            "price_precision": 2,
            "fx_rate_to_krw": 1.0,
        }

    if country_code not in ("kor", "kr"):
        raise AccountSettingsError(f"지원하지 않는 국가 코드입니다: {country_code}")

    return {
        "currency": "KRW",
        "qty_precision": 0,
        "price_precision": 0,
        "fx_rate_to_krw": 1.0,
    }


def get_account_slack_channel(account_id: str) -> str | None:
    """슬랙 채널 ID(없으면 None)를 반환합니다."""

    settings = get_account_settings(account_id)
    channel_value: str | None = None

    if isinstance(settings.get("slack"), dict):
        channel_field = settings["slack"].get("channel")
        if isinstance(channel_field, str) and channel_field.strip():
            channel_value = channel_field.strip()

    if not channel_value:
        legacy_channel = settings.get("slack_channel")
        if isinstance(legacy_channel, str) and legacy_channel.strip():
            channel_value = legacy_channel.strip()

    return channel_value


@lru_cache(maxsize=1)
def load_common_settings() -> dict[str, Any]:
    """config.py 모듈에서 공통 설정을 추출해 딕셔너리로 반환합니다."""

    try:
        import importlib

        config_module = importlib.import_module("config")
    except ModuleNotFoundError as exc:
        raise AccountSettingsError("공통 설정 모듈(config.py)을 찾을 수 없습니다.") from exc
    except Exception as exc:
        raise AccountSettingsError(f"공통 설정을 로드하지 못했습니다: {exc}") from exc

    data = {key: getattr(config_module, key) for key in dir(config_module) if key.isupper() and not key.startswith("_")}
    return data


def get_strategy_rules(account_id: str):
    """계정별 전략 설정을 `StrategyRules` 객체로 반환합니다."""

    from strategies.maps.rules import StrategyRules

    tuning, _ = get_account_strategy_sections(account_id)
    return StrategyRules.from_mapping(tuning)


# ---------------------------------------------------------------------------
# 하위 호환 래퍼 (다음 단계에서 제거 예정)
# ---------------------------------------------------------------------------


class CountrySettingsError(AccountSettingsError):
    """기존 코드 호환을 위한 예외 alias."""


def get_country_settings(country: str) -> dict[str, Any]:  # pragma: no cover - 호환용
    return get_account_settings(country)


def get_country_strategy_sections(
    country: str,
) -> tuple[dict[str, Any], dict[str, Any]]:  # pragma: no cover
    return get_account_strategy_sections(country)


def get_country_strategy(country: str) -> dict[str, Any]:  # pragma: no cover
    return get_account_strategy(country)


def get_country_precision(country: str) -> dict[str, Any]:  # pragma: no cover
    return get_account_precision(country)


def get_country_slack_channel(country: str) -> str | None:  # pragma: no cover
    return get_account_slack_channel(country)
