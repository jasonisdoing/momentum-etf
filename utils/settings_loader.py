"""계정별 설정을 파일에서 로드하기 위한 헬퍼 모듈."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Mapping

from utils.logger import get_app_logger


class AccountSettingsError(RuntimeError):
    """계정 설정 로딩 중 발생하는 예외."""


SETTINGS_ROOT = Path(__file__).resolve().parents[1] / "data" / "settings"
ACCOUNT_SETTINGS_DIR = SETTINGS_ROOT / "account"
COMMON_SETTINGS_PATH = SETTINGS_ROOT / "common.py"
SCHEDULE_CONFIG_PATH = SETTINGS_ROOT / "schedule_config.json"
PRECISION_SETTINGS_PATH = SETTINGS_ROOT / "precision.json"
BACKTEST_SETTINGS_PATH = SETTINGS_ROOT / "backtest.json"
TUNE_SETTINGS_PATH = SETTINGS_ROOT / "tune.json"
logger = get_app_logger()


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


@lru_cache(maxsize=1)
def _load_precision_settings() -> Dict[str, Any]:
    try:
        raw = PRECISION_SETTINGS_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise AccountSettingsError(f"정밀도 설정 파일을 찾을 수 없습니다: {PRECISION_SETTINGS_PATH}") from exc
    except OSError as exc:
        raise AccountSettingsError(f"정밀도 설정 파일을 읽을 수 없습니다: {PRECISION_SETTINGS_PATH}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AccountSettingsError(f"정밀도 설정 파일이 올바른 JSON 형식이 아닙니다: {PRECISION_SETTINGS_PATH}") from exc

    if not isinstance(data, dict):
        raise AccountSettingsError(f"정밀도 설정 파일의 루트는 객체(JSON object)여야 합니다: {PRECISION_SETTINGS_PATH}")

    return data


@lru_cache(maxsize=1)
def _load_backtest_settings() -> Dict[str, Any]:
    try:
        return _load_json(BACKTEST_SETTINGS_PATH)
    except AccountSettingsError:
        return {}
    except Exception:
        return {}


def get_backtest_settings() -> Dict[str, Any]:
    return dict(_load_backtest_settings())


def get_backtest_months_range(default: int = 36) -> int:
    settings = _load_backtest_settings()
    value = settings.get("MONTHS_RANGE")
    try:
        months = int(value)
        if months > 0:
            return months
    except (TypeError, ValueError):
        pass
    return int(default)


def get_backtest_initial_capital(default: float = 100_000_000) -> float:
    settings = _load_backtest_settings()
    value = settings.get("INITIAL_CAPITAL_KRW")
    try:
        capital = float(value)
        if capital > 0:
            return capital
    except (TypeError, ValueError):
        pass
    return float(default)


@lru_cache(maxsize=1)
def _load_tune_settings() -> Dict[str, Any]:
    try:
        return _load_json(TUNE_SETTINGS_PATH)
    except AccountSettingsError:
        return {}
    except Exception:
        return {}


def get_tune_month_configs() -> List[Dict[str, Any]]:
    settings = _load_tune_settings()
    root = settings.get("COMMON_CONSTANTS")
    if not isinstance(root, dict):
        return []

    entries = root.get("MONTHS_CONFIG")
    if not isinstance(entries, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue

        months_raw = item.get("MONTHS_RANGE")
        weight_raw = item.get("weight", 0)
        source = item.get("source")

        try:
            months_range = int(months_raw)
        except (TypeError, ValueError):
            continue

        if months_range <= 0:
            continue

        try:
            weight = float(weight_raw)
        except (TypeError, ValueError):
            weight = 0.0

        normalized.append(
            {
                "months_range": months_range,
                "weight": weight,
                "source": source,
            }
        )

    return normalized


@lru_cache(maxsize=None)
def get_account_settings(account_id: str) -> Dict[str, Any]:
    """`data/settings/account/{account}.json` 파일을 로드합니다."""

    account = (account_id or "").strip().lower()
    if not account:
        raise AccountSettingsError("계정 식별자를 지정해야 합니다.")

    path = ACCOUNT_SETTINGS_DIR / f"{account}.json"
    logger.debug("계정 설정 로드: %s", path)
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
    country_code = (settings.get("country_code") or account_id).strip().lower()

    precision_map = _load_precision_settings()
    precision = precision_map.get(country_code)

    if precision is None or not isinstance(precision, dict):
        raise AccountSettingsError(f"'{account_id}'(국가 코드: {country_code})에 대한 정밀도 설정을 찾을 수 없습니다.")

    return dict(precision)


def get_account_slack_channel(account_id: str) -> Optional[str]:
    """슬랙 채널 ID(없으면 None)를 반환합니다."""

    settings = get_account_settings(account_id)
    channel_value: Optional[str] = None

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
def load_common_settings() -> Dict[str, Any]:
    """data/settings/common.py 모듈을 로드하여 딕셔너리 형태로 반환합니다."""

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "settings_common",
            (SETTINGS_ROOT / "common.py"),
        )
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    except FileNotFoundError:
        raise AccountSettingsError(f"공통 설정 파일이 없습니다: {SETTINGS_ROOT / 'common.py'}")
    except Exception as exc:
        raise AccountSettingsError(f"공통 설정을 로드하지 못했습니다: {exc}") from exc

    data = {key: getattr(module, key) for key in dir(module) if key.isupper() and not key.startswith("_")}
    return data


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


def get_market_regime_settings(common_settings: Optional[Mapping[str, Any]] = None) -> Tuple[str, int, str]:
    """공통 설정에서 메인 시장 레짐 필터 설정을 반환합니다."""

    if isinstance(common_settings, Mapping):
        settings_view: Mapping[str, Any] = common_settings
    else:
        settings_view = load_common_settings()

    ticker_raw = settings_view.get("MARKET_REGIME_FILTER_TICKER_MAIN")
    ticker = str(ticker_raw or "").strip()
    if not ticker:
        raise AccountSettingsError("공통 설정에 'MARKET_REGIME_FILTER_TICKER_MAIN' 값이 필요합니다.")

    ma_raw = settings_view.get("MARKET_REGIME_FILTER_MA_PERIOD")
    if ma_raw is None:
        raise AccountSettingsError("공통 설정에 'MARKET_REGIME_FILTER_MA_PERIOD' 값이 필요합니다.")

    try:
        ma_period = int(ma_raw)
    except (TypeError, ValueError) as exc:  # noqa: PERF203
        raise AccountSettingsError("'MARKET_REGIME_FILTER_MA_PERIOD' 값은 정수여야 합니다.") from exc

    if ma_period <= 0:
        raise AccountSettingsError("'MARKET_REGIME_FILTER_MA_PERIOD' 값은 0보다 커야 합니다.")

    country_raw = settings_view.get("MARKET_REGIME_FILTER_COUNTRY")
    country = str(country_raw or "us").strip().lower() or "us"

    return ticker, ma_period, country


def get_market_regime_aux_tickers(common_settings: Optional[Mapping[str, Any]] = None) -> List[str]:
    """공통 설정에 정의된 보조 레짐 필터 티커 목록을 반환합니다."""

    if isinstance(common_settings, Mapping):
        settings_view: Mapping[str, Any] = common_settings
    else:
        settings_view = load_common_settings()

    aux_raw = settings_view.get("MARKET_REGIME_FILTER_TICKERS_AUX", [])
    tickers: List[str] = []
    if isinstance(aux_raw, (list, tuple)):
        for value in aux_raw:
            ticker = str(value or "").strip()
            if ticker:
                tickers.append(ticker)
    return tickers
