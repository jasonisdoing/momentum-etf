"""계좌 메타데이터를 로드하고 조회하기 위한 헬퍼 함수 모음입니다."""

from __future__ import annotations

import json
import importlib
import importlib.util
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from datetime import datetime


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
        settings["COOLDOWN_DAYS"] = getattr(module, "COOLDOWN_DAYS")

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
        if not isinstance(settings["COOLDOWN_DAYS"], int) or settings["COOLDOWN_DAYS"] < 0:
            raise ValueError("COOLDOWN_DAYS는 0 이상의 정수여야 합니다.")

    except (AttributeError, ValueError, TypeError, ImportError) as e:
        raise SystemExit(f"오류: 공통 설정 파일({file_path})에 문제가 있습니다: {e}")

    _common_file_settings_cache = settings
    return settings


_account_file_settings_cache: Dict[str, Dict[str, Any]] = {}


def get_country_file_settings(country: str) -> Dict[str, Any]:
    """
    국가별 전략 설정 파일(예: 'data/settings/country/kor.py')에서
    전략 파라미터를 동적으로 로드합니다.
    """
    cache_key = f"country_{country}"
    if cache_key in _account_file_settings_cache:
        return _account_file_settings_cache[cache_key]

    settings: Dict[str, Any] = {}
    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / "data" / "settings" / "country" / f"{country}.py"
    module_name = f"country_settings_{country}"

    if not file_path.is_file():
        raise SystemExit(f"오류: 국가 설정 파일({file_path})을 찾을 수 없습니다. 이 파일은 필수입니다.")

    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec from {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 필수 설정 로드
        ma_period = getattr(module, "MA_PERIOD")
        portfolio_topn = getattr(module, "PORTFOLIO_TOPN")
        replace_weaker = getattr(module, "REPLACE_WEAKER_STOCK")
        replace_threshold = getattr(module, "REPLACE_SCORE_THRESHOLD")
        min_buy_score = getattr(module, "MIN_BUY_SCORE", 0.0)  # 없으면 0.0으로 폴백
        coin_min_cost = getattr(module, "COIN_MIN_HOLDING_COST_KRW", None)

        # 유효성 검사
        if not isinstance(ma_period, int) or ma_period <= 0:
            raise ValueError("MA_PERIOD는 0보다 큰 정수여야 합니다.")
        if not isinstance(portfolio_topn, int) or portfolio_topn <= 0:
            raise ValueError("PORTFOLIO_TOPN은 0보다 큰 정수여야 합니다.")
        if not isinstance(replace_weaker, bool):
            raise ValueError("REPLACE_WEAKER_STOCK은 True 또는 False여야 합니다.")
        if not isinstance(replace_threshold, (int, float)):
            raise ValueError("REPLACE_SCORE_THRESHOLD는 숫자여야 합니다.")
        if not isinstance(min_buy_score, (int, float)):
            raise ValueError("MIN_BUY_SCORE는 숫자여야 합니다.")
        if coin_min_cost is not None and (
            not isinstance(coin_min_cost, (int, float)) or coin_min_cost < 0
        ):
            raise ValueError("COIN_MIN_HOLDING_COST_KRW는 0 이상 숫자여야 합니다.")

        settings["ma_period"] = ma_period
        settings["portfolio_topn"] = portfolio_topn
        settings["replace_weaker_stock"] = replace_weaker
        settings["replace_threshold"] = replace_threshold
        settings["min_buy_score"] = min_buy_score
        if coin_min_cost is not None:
            settings["coin_min_holding_cost_krw"] = float(coin_min_cost)

    except (AttributeError, ValueError, TypeError, ImportError) as e:
        raise SystemExit(f"오류: 국가 설정 파일({file_path})에 문제가 있습니다: {e}")

    _account_file_settings_cache[cache_key] = settings
    return settings


def get_account_file_settings(account: str) -> Dict[str, Any]:
    """
    계좌별 설정 파일(예: 'data/accounts/settings/kor_m1.py')에서 초기 자본금,
    기준일 및 전략 파라미터를 동적으로 로드합니다.
    """
    cache_key = f"account_{account}"
    if cache_key in _account_file_settings_cache:
        return _account_file_settings_cache[cache_key]

    settings: Dict[str, Any] = {}
    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / "data" / "settings" / "accounts" / f"{account}.py"
    module_name = f"account_settings_{account}"

    if not file_path.is_file():
        raise SystemExit(f"오류: 계좌 설정 파일({file_path})을 찾을 수 없습니다. 이 파일은 필수입니다.")

    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec from {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 필수 설정 로드
        initial_capital_krw = getattr(module, "INITIAL_CAPITAL_KRW")
        date_str = getattr(module, "INITIAL_DATE")

        # 유효성 검사
        if not isinstance(initial_capital_krw, (int, float)) or initial_capital_krw <= 0:
            raise ValueError("INITIAL_CAPITAL_KRW 0보다 큰 숫자여야 합니다.")

        settings["initial_capital_krw"] = initial_capital_krw
        settings["initial_date"] = datetime.strptime(date_str, "%Y-%m-%d")

        # (선택) 슬랙 웹훅 URL
        slack_webhook_url = getattr(module, "SLACK_WEBHOOK_URL", None)
        if slack_webhook_url and not isinstance(slack_webhook_url, str):
            raise ValueError("SLACK_WEBHOOK_URL은 문자열이어야 합니다.")
        settings["slack_webhook_url"] = slack_webhook_url

    except (AttributeError, ValueError, TypeError, ImportError) as e:
        raise SystemExit(f"오류: 계좌 설정 파일({file_path})에 문제가 있습니다: {e}")

    _account_file_settings_cache[cache_key] = settings
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
