"""계좌 메타데이터를 로드하고 조회하기 위한 헬퍼 함수 모음입니다."""

from __future__ import annotations

import json
import importlib
import importlib.util
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from datetime import datetime


# NOTE: StrategyRules는 순환 의존성(circular import)을 피하기 위해
# 함수 내부에서 지연 임포트합니다.


ACCOUNTS_FILE = (
    Path(__file__).resolve().parent.parent / "data" / "settings" / "country_mapping.json"
)

_accounts_cache: List[Dict[str, Any]] = []
_account_map: Dict[str, Dict[str, Any]] = {}
_warned_once = False


_common_file_settings_cache: Dict[str, Any] = {}


def get_common_file_settings() -> Dict[str, Any]:
    """
    공통 설정 파일('data/common/settings.py')에서 전역 설정을 동적으로 로드합니다.
    """
    global _common_file_settings_cache
    if _common_file_settings_cache:
        return _common_file_settings_cache

    settings: Dict[str, Any] = {}
    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / "data" / "settings" / "common.py"
    module_name = "common_settings"

    if not file_path.is_file():
        raise SystemExit(f"오류: 공통 설정 파일({file_path})을 찾을 수 없습니다. 이 파일은 필수입니다.")

    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec from {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 필수 설정 로드
        settings["MARKET_REGIME_FILTER_ENABLED"] = getattr(module, "MARKET_REGIME_FILTER_ENABLED")
        settings["MARKET_REGIME_FILTER_TICKER"] = getattr(module, "MARKET_REGIME_FILTER_TICKER")
        settings["MARKET_REGIME_FILTER_MA_PERIOD"] = getattr(
            module, "MARKET_REGIME_FILTER_MA_PERIOD"
        )
        settings["HOLDING_STOP_LOSS_PCT"] = getattr(module, "HOLDING_STOP_LOSS_PCT")
        settings["REALTIME_PRICE_ENABLED"] = bool(getattr(module, "REALTIME_PRICE_ENABLED", True))

        # 유효성 검사
        if not isinstance(settings["MARKET_REGIME_FILTER_ENABLED"], bool):
            raise ValueError("MARKET_REGIME_FILTER_ENABLED는 True 또는 False여야 합니다.")
        if (
            not isinstance(settings["MARKET_REGIME_FILTER_TICKER"], str)
            or not settings["MARKET_REGIME_FILTER_TICKER"]
        ):
            raise ValueError("MARKET_REGIME_FILTER_TICKER는 비어있지 않은 문자열이어야 합니다.")
        if (
            not isinstance(settings["MARKET_REGIME_FILTER_MA_PERIOD"], int)
            or settings["MARKET_REGIME_FILTER_MA_PERIOD"] <= 0
        ):
            raise ValueError("MARKET_REGIME_FILTER_MA_PERIOD는 0보다 큰 정수여야 합니다.")
        if not isinstance(settings["HOLDING_STOP_LOSS_PCT"], (int, float)):
            raise ValueError("HOLDING_STOP_LOSS_PCT는 숫자여야 합니다.")
        if not isinstance(settings["REALTIME_PRICE_ENABLED"], bool):
            raise ValueError("REALTIME_PRICE_ENABLED는 True 또는 False여야 합니다.")

    except (AttributeError, ValueError, TypeError, ImportError) as e:
        raise SystemExit(f"오류: 공통 설정 파일({file_path})에 문제가 있습니다: {e}")

    _common_file_settings_cache = settings
    return settings


_account_settings_cache: Dict[str, Dict[str, Any]] = {}


def get_account_file_settings(account: str) -> Dict[str, Any]:
    """country_mapping.json에 저장된 계좌별 설정을 반환합니다."""

    cache_key = f"account_{account}"
    if cache_key in _account_settings_cache:
        return _account_settings_cache[cache_key]

    account_info = get_account_info(account)
    if not account_info:
        raise SystemExit(f"오류: 등록되지 않은 계좌입니다: {account}")

    settings_cfg = account_info.get("account_settings")
    if not isinstance(settings_cfg, dict):
        raise SystemExit(f"오류: 계좌 '{account}' 설정에 account_settings 항목이 없습니다.")

    try:
        initial_capital = float(settings_cfg["initial_capital_krw"])
    except (KeyError, TypeError, ValueError) as exc:  # noqa: PERF203
        raise SystemExit(f"오류: 계좌 '{account}'의 initial_capital_krw 값이 올바르지 않습니다.") from exc

    date_value = settings_cfg.get("initial_date")
    if not date_value:
        raise SystemExit(f"오류: 계좌 '{account}'의 initial_date 값이 누락되었습니다.")

    if isinstance(date_value, datetime):
        initial_date = date_value
    else:
        try:
            initial_date = datetime.strptime(str(date_value), "%Y-%m-%d")
        except ValueError as exc:  # noqa: PERF203
            raise SystemExit(f"오류: 계좌 '{account}'의 initial_date 값이 YYYY-MM-DD 형식이 아닙니다.") from exc

    slack_webhook_url = settings_cfg.get("slack_webhook_url")
    if slack_webhook_url is not None and not isinstance(slack_webhook_url, str):
        raise SystemExit(f"오류: 계좌 '{account}'의 slack_webhook_url 값은 문자열이어야 합니다.")

    cooldown_days_raw = settings_cfg.get("cooldown_days", 0)
    try:
        cooldown_days = int(cooldown_days_raw)
        if cooldown_days < 0:
            raise ValueError
    except (ValueError, TypeError):
        raise SystemExit(f"오류: 계좌 '{account}'의 cooldown_days 값이 0 이상의 정수가 아닙니다.")

    settings: Dict[str, Any] = {
        "initial_capital_krw": initial_capital,
        "initial_date": initial_date,
        "slack_webhook_url": slack_webhook_url,
        "cooldown_days": cooldown_days,
    }

    _account_settings_cache[cache_key] = settings
    return settings


def _refresh_cache() -> None:
    """디스크에 저장된 계좌 메타데이터를 메모리 캐시에 다시 로드합니다."""

    global _accounts_cache, _account_map, _warned_once

    try:
        with ACCOUNTS_FILE.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, list):
            raise TypeError("country_mapping.json 파일은 리스트 구조여야 합니다")
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
            strategy_cfg = item.get("strategy")
            if not strategy_cfg:
                raise SystemExit(f"계좌 '{account_code}' 설정에 'strategy' 항목이 필요합니다.")
            try:
                # 지연 임포트로 순환 의존성 회피
                from logic.strategies.momentum.rules import StrategyRules  # type: ignore

                strategy_rules = StrategyRules.from_mapping(strategy_cfg)
            except ValueError as exc:
                raise SystemExit(f"계좌 '{account_code}'의 전략 설정이 올바르지 않습니다: {exc}") from exc

            item["strategy_rules"] = strategy_rules
            strategy_dict = strategy_rules.to_dict()
            item["strategy"] = strategy_dict
            item["ma_period"] = strategy_rules.ma_period
            item["portfolio_topn"] = strategy_rules.portfolio_topn
            item["replace_threshold"] = strategy_rules.replace_threshold

            precision_cfg = item.get("precision")
            if isinstance(precision_cfg, dict):
                item["currency"] = precision_cfg.get("currency", item.get("currency"))
                item["amt_precision"] = precision_cfg.get(
                    "amt_precision", item.get("amt_precision")
                )
                item["qty_precision"] = precision_cfg.get(
                    "qty_precision", item.get("qty_precision")
                )
            normalized.append(item)
            mapping[account_code] = item
        _accounts_cache = normalized
        _account_map = mapping
    except FileNotFoundError:
        if not _warned_once:
            print("경고: data/accounts/country_mapping.json 파일을 찾을 수 없습니다. 계좌 매핑이 비어있습니다.")
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


def get_strategy_rules_for_account(account: str):
    info = get_account_info(account)
    if not info or "strategy_rules" not in info:
        raise ValueError(f"'{account}' 계좌의 전략 설정을 찾을 수 없습니다.")
    return info["strategy_rules"]


def get_strategy_dict_for_account(account: str) -> Dict[str, Any]:
    return get_strategy_rules_for_account(account).to_dict()


def get_all_accounts_sorted_by_order() -> List[Dict[str, Any]]:
    """order 순으로 정렬된 모든 활성 계좌 목록을 반환합니다."""
    all_accounts = load_accounts()
    # 활성 계좌만 필터링하고 order 순으로 정렬
    active_accounts = [
        account
        for account in all_accounts
        if account.get("is_active", True) and account.get("account") is not None
    ]
    return sorted(active_accounts, key=lambda x: x.get("order", 999))


def get_coin_min_holding_cost(account: str) -> Optional[float]:
    return get_strategy_rules_for_account(account).coin_min_holding_cost_krw
