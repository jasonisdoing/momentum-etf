"""계정/종목풀 설정을 파일에서 로드하기 위한 헬퍼 모듈."""

from __future__ import annotations

import json
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

from utils.logger import get_app_logger


class AccountSettingsError(RuntimeError):
    """계정 설정 로딩 중 발생하는 예외."""


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACCOUNT_SETTINGS_PATH = PROJECT_ROOT / "accounts.json"
POOL_SETTINGS_PATH = PROJECT_ROOT / "pools.json"
logger = get_app_logger()


def _load_accounts_payload() -> dict[str, Any]:
    return _load_json(ACCOUNT_SETTINGS_PATH)


def _load_pools_payload() -> dict[str, Any]:
    return _load_json(POOL_SETTINGS_PATH)


@cache
def _load_account_configs() -> list[dict[str, Any]]:
    payload = _load_accounts_payload()
    accounts = payload.get("accounts")
    if not isinstance(accounts, list):
        raise AccountSettingsError(f"'accounts.json'의 'accounts'는 배열이어야 합니다: {ACCOUNT_SETTINGS_PATH}")

    loaded: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for raw_entry in accounts:
        if not isinstance(raw_entry, dict):
            raise AccountSettingsError(f"'accounts' 항목은 객체여야 합니다: {ACCOUNT_SETTINGS_PATH}")

        account_id = str(raw_entry.get("account_id") or "").strip().lower()
        if not account_id:
            raise AccountSettingsError(f"'account_id'는 필수입니다: {ACCOUNT_SETTINGS_PATH}")
        if account_id in seen_ids:
            raise AccountSettingsError(f"중복된 account_id가 있습니다: {account_id}")

        order = raw_entry.get("order")
        if not isinstance(order, int):
            raise AccountSettingsError(f"계정 '{account_id}'의 'order'는 정수여야 합니다.")

        country_code = str(raw_entry.get("country_code") or "").strip().lower()
        if country_code not in {"kor", "au", "us"}:
            raise AccountSettingsError(f"계정 '{account_id}'의 country_code는 kor, au, us만 허용합니다: {country_code}")

        loaded.append(
            {
                **raw_entry,
                "account_id": account_id,
                "order": order,
                "country_code": country_code,
            }
        )
        seen_ids.add(account_id)

    return sorted(loaded, key=lambda item: (int(item["order"]), str(item["account_id"])))


@cache
def _load_pool_configs() -> list[dict[str, Any]]:
    payload = _load_pools_payload()
    pools = payload.get("pools")
    if not isinstance(pools, list):
        raise AccountSettingsError(f"'pools.json'의 'pools'는 배열이어야 합니다: {POOL_SETTINGS_PATH}")

    loaded: list[dict[str, Any]] = []
    seen_types: set[str] = set()

    for raw_entry in pools:
        if not isinstance(raw_entry, dict):
            raise AccountSettingsError(f"'pools' 항목은 객체여야 합니다: {POOL_SETTINGS_PATH}")

        ticker_type = str(raw_entry.get("ticker_type") or "").strip().lower()
        if not ticker_type:
            raise AccountSettingsError(f"'ticker_type'은 필수입니다: {POOL_SETTINGS_PATH}")
        if ticker_type in seen_types:
            raise AccountSettingsError(f"중복된 ticker_type이 있습니다: {ticker_type}")

        order = raw_entry.get("order")
        if not isinstance(order, int):
            raise AccountSettingsError(f"종목풀 '{ticker_type}'의 'order'는 정수여야 합니다.")

        country_code = str(raw_entry.get("country_code") or "").strip().lower()
        if country_code not in {"kor", "au", "us"}:
            raise AccountSettingsError(f"종목풀 '{ticker_type}'의 country_code는 kor, au, us만 허용합니다: {country_code}")

        icon = str(raw_entry.get("icon") or "").strip()
        if not icon:
            raise AccountSettingsError(f"종목풀 '{ticker_type}' 설정에 icon이 필요합니다.")

        loaded.append(
            {
                **raw_entry,
                "ticker_type": ticker_type,
                "order": order,
                "country_code": country_code,
                "icon": icon,
            }
        )
        seen_types.add(ticker_type)

    return sorted(loaded, key=lambda item: (int(item["order"]), str(item["ticker_type"])))


@cache
def get_all_pool_settings() -> dict[str, Any]:
    """pools.json의 0. 전체 가상 종목풀 설정을 로드합니다."""

    payload = _load_pools_payload()
    all_settings = payload.get("all")
    if not isinstance(all_settings, dict):
        raise AccountSettingsError(f"'pools.json'의 'all'은 객체여야 합니다: {POOL_SETTINGS_PATH}")

    required_keys = ["TOP_N_HOLD", "HOLDING_BONUS_SCORE", "MA_TYPE", "MA_MONTHS", "RSI_LIMIT", "include"]
    missing_keys = [key for key in required_keys if key not in all_settings]
    if missing_keys:
        raise AccountSettingsError(f"'all' 설정에 필수 항목이 누락되었습니다: {', '.join(missing_keys)}")

    include_raw = all_settings.get("include")
    if not isinstance(include_raw, list):
        raise AccountSettingsError("'all.include'는 배열이어야 합니다.")

    include: list[str] = []
    for raw_ticker_type in include_raw:
        ticker_type = str(raw_ticker_type or "").strip().lower()
        if not ticker_type:
            raise AccountSettingsError("'all.include'에는 빈 종목풀 식별자를 넣을 수 없습니다.")
        if ticker_type in include:
            raise AccountSettingsError(f"'all.include'에 중복된 종목풀이 있습니다: {ticker_type}")
        include.append(ticker_type)

    if not include:
        raise AccountSettingsError("'all.include'에는 하나 이상의 종목풀이 필요합니다.")

    available_types = set(list_available_ticker_types())
    unknown_types = [ticker_type for ticker_type in include if ticker_type not in available_types]
    if unknown_types:
        raise AccountSettingsError(f"'all.include'에 알 수 없는 종목풀이 있습니다: {', '.join(unknown_types)}")

    return {
        **all_settings,
        "include": include,
    }


def list_available_ticker_types() -> list[str]:
    """pools.json에 정의된 유효한 종목타입 목록을 반환합니다."""
    return [str(item["ticker_type"]) for item in _load_pool_configs()]


def list_available_accounts() -> list[str]:
    """
    accounts.json에 정의된 유효한 계정 목록을 반환합니다.
    """
    return [str(item["account_id"]) for item in _load_account_configs()]


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


def get_account_order(account_id: str) -> int:
    """논리 계정 ID에 대응하는 계정 순번을 반환합니다."""

    return int(get_account_settings(account_id)["order"])


@cache
def get_account_settings(account_id: str) -> dict[str, Any]:
    """accounts.json에 정의된 개별 계정 설정을 로드합니다."""

    account = (account_id or "").strip().lower()
    if not account:
        raise AccountSettingsError("계정 식별자를 지정해야 합니다.")

    logger.debug("계정 설정 로드: %s (%s)", ACCOUNT_SETTINGS_PATH, account)
    for settings in _load_account_configs():
        if settings["account_id"] == account:
            return dict(settings)
    raise AccountSettingsError(f"계정 '{account}'에 해당하는 설정을 찾을 수 없습니다.")

@cache
def get_ticker_type_settings(ticker_type: str) -> dict[str, Any]:
    """pools.json에 정의된 개별 종목풀 설정을 로드합니다."""
    t_id = (ticker_type or "").strip().lower()
    if not t_id:
        raise AccountSettingsError("종목타입을 지정해야 합니다.")

    for settings in _load_pool_configs():
        if settings["ticker_type"] == t_id:
            return dict(settings)
    raise AccountSettingsError(f"종목타입 '{t_id}'에 해당하는 설정을 찾을 수 없습니다.")


def get_account_precision(account_id: str) -> dict[str, Any]:
    """표시/계산 정밀도 설정을 반환합니다."""

    settings = get_account_settings(account_id)
    country_code = (settings.get("country_code") or account_id).strip().lower()
    if country_code == "au":
        return {
            "currency": "AUD",
            "qty_precision": 0,
            "price_precision": 2,
        }

    if country_code == "us":
        return {
            "currency": "USD",
            "qty_precision": 0,
            "price_precision": 2,
        }

    if country_code != "kor":
        raise AccountSettingsError(f"지원하지 않는 국가 코드입니다: {country_code}")

    return {
        "currency": "KRW",
        "qty_precision": 0,
        "price_precision": 0,
    }


def get_slack_channel() -> str | None:
    """공통 슬랙 채널 ID를 반환합니다. config.SLACK_CHANNEL을 사용합니다."""

    try:
        import config

        channel = getattr(config, "SLACK_CHANNEL", None)
        if isinstance(channel, str) and channel.strip():
            return channel.strip()
    except Exception:
        pass

    return None


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


def get_country_precision(country: str) -> dict[str, Any]:  # pragma: no cover
    country_code = (country or "").strip().lower()
    if country_code == "au":
        return {
            "currency": "AUD",
            "qty_precision": 0,
            "price_precision": 2,
        }
    if country_code == "us":
        return {
            "currency": "USD",
            "qty_precision": 0,
            "price_precision": 2,
        }
    if country_code == "kor":
        return {
            "currency": "KRW",
            "qty_precision": 0,
            "price_precision": 0,
        }
    raise AccountSettingsError(f"지원하지 않는 국가 코드입니다: {country}")


def get_country_slack_channel(country: str) -> str | None:  # pragma: no cover
    return get_slack_channel()
