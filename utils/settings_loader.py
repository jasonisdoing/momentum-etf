"""계정/종목풀 설정 로더."""

from __future__ import annotations

from functools import cache, lru_cache
from typing import Any

from utils.logger import get_app_logger


class AccountSettingsError(RuntimeError):
    """계정 설정 로딩 중 발생하는 예외."""


logger = get_app_logger()


def _load_account_configs() -> list[dict[str, Any]]:
    """계좌 설정을 DB(account_settings) 단일 소스에서 읽는다 (store 가 TTL 캐시 담당).

    문서가 없으면 명시적 에러.
    """
    from utils.account_settings_store import AccountSettingsStoreError, load_account_docs

    try:
        return load_account_docs()
    except AccountSettingsStoreError as exc:
        raise AccountSettingsError(str(exc)) from exc


@cache
def _load_pool_configs() -> list[dict[str, Any]]:
    """DB(pool_settings)에 정의된 활성 종목풀 목록을 반환한다."""
    from utils.pool_settings_store import PoolSettingsError, load_pool_definitions

    try:
        return load_pool_definitions()
    except PoolSettingsError as exc:
        raise AccountSettingsError(str(exc)) from exc


def list_available_ticker_types() -> list[str]:
    """DB(pool_settings)에 정의된 활성 종목타입 목록을 반환합니다."""
    return [str(item["ticker_type"]) for item in _load_pool_configs()]


def list_available_accounts() -> list[str]:
    """
    DB(account_settings)에 정의된 유효한 계정 목록을 반환합니다.
    """
    return [str(item["account_id"]) for item in _load_account_configs()]


def get_account_order(account_id: str) -> int:
    """논리 계정 ID에 대응하는 계정 순번을 반환합니다."""

    return int(get_account_settings(account_id)["order"])


def get_account_settings(account_id: str) -> dict[str, Any]:
    """DB(account_settings)에 정의된 개별 계정 설정을 로드합니다.

    캐시는 두지 않는다 — 아래 `_load_account_configs` 가 store 의 TTL 캐시(30초)를 타므로
    DB 를 매번 때리지 않으면서 다른 프로세스의 변경도 자동으로 반영된다.
    예전에는 여기에 만료 없는 `@cache` 가 있어서, 저장한 프로세스만 최신이 되고 나머지는
    재시작 전까지 옛 값을 봤다. 같은 프로세스 안에서도 이 함수를 타는 화면(`/assets` 현금
    통화 목록)과 TTL 을 직접 읽는 화면(`/account-settings`)의 답이 갈렸다.
    """

    account = (account_id or "").strip().lower()
    if not account:
        raise AccountSettingsError("계정 식별자를 지정해야 합니다.")

    for settings in _load_account_configs():
        if settings["account_id"] == account:
            return dict(settings)
    raise AccountSettingsError(f"계정 '{account}'에 해당하는 설정을 찾을 수 없습니다.")


def get_ticker_type_settings(ticker_type: str) -> dict[str, Any]:
    """DB(pool_settings)에 정의된 개별 종목풀 설정을 로드합니다."""

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
