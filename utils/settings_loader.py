"""계정별 설정을 파일에서 로드하기 위한 헬퍼 모듈."""

from __future__ import annotations

import json
import re
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
ACCOUNT_DIR_PATTERN = re.compile(r"^(?P<order>\d+)_(?P<account>[a-z0-9_]+)$")


def parse_account_dir_name(dir_name: str) -> tuple[int, str]:
    """`<order>_<account>` 형식의 디렉토리명에서 순번과 계정 코드를 추출합니다."""

    normalized = (dir_name or "").strip().lower()
    match = ACCOUNT_DIR_PATTERN.fullmatch(normalized)
    if not match:
        raise AccountSettingsError(f"계정 디렉토리명은 '<order>_<account>' 형식이어야 합니다: {dir_name}")
    return int(match.group("order")), match.group("account")


def _iter_account_dirs() -> list[tuple[str, Path]]:
    account_dirs: dict[str, Path] = {}
    if not SETTINGS_ROOT.exists():
        return []

    for item in SETTINGS_ROOT.iterdir():
        if not item.is_dir() or item.name.startswith(".") or item.name.startswith("_"):
            continue

        config_path = item / "config.json"
        if not config_path.exists():
            continue

        config_data = _load_json(config_path)
        _, account_id = parse_account_dir_name(item.name)
        configured = str(config_data.get("account") or "").strip().lower()
        if configured and configured != account_id:
            raise AccountSettingsError(
                f"계정 디렉토리명과 config.json account 값이 다릅니다: {item.name} / {configured}"
            )
        account_dirs[account_id] = item

    return sorted(account_dirs.items(), key=lambda pair: parse_account_dir_name(pair[1].name))


def list_available_accounts() -> list[str]:
    """
    zaccounts 디렉토리 하위의 유효한 계정(디렉토리 내 config.json 존재) 목록을 반환합니다.
    """
    return [account_id for account_id, _ in _iter_account_dirs()]


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


def _normalize_pool_ids(raw_value: Any, *, context: str) -> list[str]:
    """계좌 설정의 pool 값을 정규화하고 검증합니다."""

    if not isinstance(raw_value, list):
        raise AccountSettingsError(f"{context}의 'pool'은 문자열이 아니라 리스트여야 합니다.")
    if not raw_value:
        raise AccountSettingsError(f"{context}의 'pool'은 최소 1개 이상의 종목풀 ID를 가져야 합니다.")

    normalized: list[str] = []
    seen: set[str] = set()
    for value in raw_value:
        pool_id = str(value or "").strip().lower()
        if not pool_id:
            raise AccountSettingsError(f"{context}의 'pool' 목록에는 빈 값이 올 수 없습니다.")
        if pool_id in seen:
            continue
        seen.add(pool_id)
        normalized.append(pool_id)

    if not normalized:
        raise AccountSettingsError(f"{context}의 'pool'에서 유효한 종목풀 ID를 찾지 못했습니다.")

    from utils.pool_registry import list_available_pools

    available_pools = set(list_available_pools())
    invalid_pools = [pool_id for pool_id in normalized if pool_id not in available_pools]
    if invalid_pools:
        invalid_text = ", ".join(invalid_pools)
        raise AccountSettingsError(f"{context}의 'pool'에 존재하지 않는 종목풀이 포함되어 있습니다: {invalid_text}")

    return normalized


def get_tune_month_configs(account_id: str = None) -> list[dict[str, Any]]:
    """튜닝용 시작일 설정을 반환합니다.

    계정별 strategy.TUNE_MONTHS를 사용합니다.
    """
    normalized: list[dict[str, Any]] = []

    # 계정별 strategy.TUNE_MONTHS 사용
    if account_id:
        account_settings = get_account_settings(account_id)
        strategy_cfg = account_settings.get("strategy", {})
        strategy = resolve_strategy_params(strategy_cfg)
        tune_months = strategy.get("TUNE_MONTHS")
        if tune_months is None:
            raise AccountSettingsError(f"{account_id} 계좌의 필수 설정이 누락되었습니다: strategy.TUNE_MONTHS")

        import pandas as pd

        try:
            months_back = int(tune_months)
        except (TypeError, ValueError) as exc:
            raise AccountSettingsError(
                f"{account_id} 계좌의 strategy.TUNE_MONTHS는 정수여야 합니다: {tune_months}"
            ) from exc
        if months_back < 1:
            raise AccountSettingsError(f"{account_id} 계좌의 strategy.TUNE_MONTHS는 1 이상이어야 합니다: {months_back}")

        start_dt = pd.Timestamp.today().normalize() - pd.DateOffset(months=months_back)
        backtest_start_date = start_dt.strftime("%Y-%m-%d")

        normalized.append(
            {
                "backtest_start_date": str(backtest_start_date),
                "weight": 1.0,
                "source": f"account_{account_id}",
            }
        )

    if not normalized:
        return []

    # 중복 제거 (날짜 기준)
    seen: dict[str, dict[str, Any]] = {}
    for entry in normalized:
        start_date = entry["backtest_start_date"]
        if start_date not in seen:
            seen[start_date] = entry

    return list(seen.values())


@cache
def get_account_dir(account_id: str) -> Path:
    """논리 계정 ID에 대응하는 실제 zaccounts 디렉토리를 반환합니다."""

    account = (account_id or "").strip().lower()
    if not account:
        raise AccountSettingsError("계정 식별자를 지정해야 합니다.")

    account_dirs = dict(_iter_account_dirs())
    path = account_dirs.get(account)
    if path is None:
        raise AccountSettingsError(f"계정 '{account}'에 해당하는 설정 디렉토리를 찾을 수 없습니다.")
    return path


def get_account_order(account_id: str) -> int:
    """논리 계정 ID에 대응하는 디렉토리명의 순번을 반환합니다."""

    return parse_account_dir_name(get_account_dir(account_id).name)[0]


@cache
def get_account_settings(account_id: str) -> dict[str, Any]:
    """`zaccounts/{account}/config.json` 파일을 로드합니다."""

    account = (account_id or "").strip().lower()
    if not account:
        raise AccountSettingsError("계정 식별자를 지정해야 합니다.")

    path = get_account_dir(account) / "config.json"
    logger.debug("계정 설정 로드: %s", path)

    settings = _load_json(path)
    settings["account"] = account

    if not settings.get("country_code"):
        raise AccountSettingsError(f"'{path}' 설정 파일에 필수 항목 'country_code'가 누락되었습니다.")
    settings["pool"] = _normalize_pool_ids(settings.get("pool"), context=str(path))

    return settings


def get_account_pool_ids(account_id: str) -> list[str]:
    """계좌에 연결된 종목풀 ID 목록을 반환합니다."""

    settings = get_account_settings(account_id)
    return list(settings["pool"])


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

    # 기본 포맷: COMMON + 상위 키를 병합한다.
    common_raw = strategy_cfg.get("COMMON")
    if isinstance(common_raw, dict):
        common = dict(common_raw)
        merged = {key: value for key, value in strategy_cfg.items() if key != "COMMON"}
        merged.update(common)
        merged.pop("STRATEGY", None)
        # 레거시 전략 전용 블록은 키 이름과 무관하게 제거
        for key in list(merged.keys()):
            if key == "STRATEGY":
                continue
            if not isinstance(key, str):
                continue
            value = strategy_cfg.get(key)
            if key.isupper() and isinstance(value, dict):
                merged.pop(key, None)
        return merged

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
        # 신규/기존 포맷 공통 처리
        tuning, static = resolve_strategy_params(strategy), {}

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
        }

    if country_code in ("au", "aus"):
        return {
            "currency": "AUD",
            "qty_precision": 0,
            "price_precision": 2,
        }

    if country_code not in ("kor", "kr"):
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


def get_strategy_rules(account_id: str):
    """계정별 전략 설정을 `StrategyRules` 객체로 반환합니다."""

    from core.strategy.rules import StrategyRules
    from utils.pool_registry import get_pool_dir

    tuning, _ = get_account_strategy_sections(account_id)
    normalized_tuning = dict(tuning)

    if normalized_tuning.get("MA_MONTH") is None and normalized_tuning.get("ma_month") is None:
        try:
            settings = get_account_settings(account_id)
            pool_ids = settings.get("pool") or []
            if pool_ids:
                pool_dir = get_pool_dir(str(pool_ids[0]))
                pool_config = _load_json(pool_dir / "config.json")
                rank_cfg = pool_config.get("rank") or {}
                if isinstance(rank_cfg, dict):
                    months = rank_cfg.get("months")
                    ma_type = rank_cfg.get("ma_type")
                    if months is not None:
                        normalized_tuning["MA_MONTH"] = months
                    if (
                        ma_type
                        and normalized_tuning.get("MA_TYPE") is None
                        and normalized_tuning.get("ma_type") is None
                    ):
                        normalized_tuning["MA_TYPE"] = ma_type
        except Exception:
            pass

    return StrategyRules.from_mapping(normalized_tuning)


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
    country_code = (country or "").strip().lower()
    if country_code in ("us", "usa"):
        return {
            "currency": "USD",
            "qty_precision": 0,
            "price_precision": 2,
        }
    if country_code in ("au", "aus"):
        return {
            "currency": "AUD",
            "qty_precision": 0,
            "price_precision": 2,
        }
    if country_code in ("kor", "kr"):
        return {
            "currency": "KRW",
            "qty_precision": 0,
            "price_precision": 0,
        }
    raise AccountSettingsError(f"지원하지 않는 국가 코드입니다: {country}")


def get_country_slack_channel(country: str) -> str | None:  # pragma: no cover
    return get_slack_channel()
