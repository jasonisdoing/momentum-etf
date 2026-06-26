"""레버리지 전략 설정·상태 조회 서비스 (UI/API 용).

설정·상태의 단일 소스는 MongoDB(`leverage/config_store.py`). 여기서는 화면 표시에
필요한 형태로 묶어 반환한다.
"""

from __future__ import annotations

from typing import Any

from leverage.config_store import (
    load_leverage_config_raw,
    load_leverage_state,
    save_leverage_config_raw,
)
from leverage.engine.backtest.settings import normalize_settings
from leverage.holding import count_holding_trading_days
from leverage.tuning_config import validate_tuning_section


def load_leverage_settings(profile: str = "switch") -> dict[str, Any]:
    """레버리지 설정(편집 대상) + 직전 추천 상태(읽기 전용)를 함께 반환한다."""
    state = load_leverage_state(profile)
    if state and state.get("holding_start_date"):
        state["holding_days"] = count_holding_trading_days(state.get("target", ""), state["holding_start_date"])

    return {
        "profile": profile,
        "config": load_leverage_config_raw(profile),
        "state": state,
    }


def _validate_leverage_config(config: dict[str, Any]) -> None:
    """저장 전 검증 (실패 시 ValueError → 400). 정상값만 DB 에 들어가게 한다."""
    if not isinstance(config, dict):
        raise ValueError("설정 형식이 올바르지 않습니다.")

    for key in ("signal", "offense", "defense"):
        asset = config.get(key)
        if not isinstance(asset, dict) or not str(asset.get("ticker") or "").strip():
            raise ValueError(f"'{key}' 자산의 티커가 필요합니다.")

    for key in ("drawdown_buy_cutoff", "drawdown_sell_cutoff", "slippage"):
        value = config.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ValueError(f"'{key}' 는 0 이상의 숫자여야 합니다.")

    months_range = config.get("months_range")
    has_months = isinstance(months_range, (int, float)) and not isinstance(months_range, bool) and months_range > 0
    if not config.get("start_date") and not has_months:
        raise ValueError("'months_range'(0보다 큰 수) 또는 'start_date' 가 필요합니다.")

    benchmarks = config.get("benchmarks")
    if not isinstance(benchmarks, list) or len(benchmarks) == 0:
        raise ValueError("벤치마크가 1개 이상 필요합니다.")
    for entry in benchmarks:
        if not isinstance(entry, dict) or not str(entry.get("ticker") or "").strip():
            raise ValueError("벤치마크 항목에 티커가 필요합니다.")

    # 튜닝 탐색 공간(있으면) 검증 — tune.py 와 동일한 공통 검증기를 사용.
    if "tuning" in config:
        validate_tuning_section(config.get("tuning"))

    # 엔진 정규화로 추가 검증 (사본으로 — 파생 키가 저장값에 섞이지 않게).
    normalize_settings(dict(config))


def save_leverage_settings(profile: str, config: dict[str, Any]) -> dict[str, Any]:
    """검증 후 설정을 DB 에 저장하고, 갱신된 설정+상태를 반환한다."""
    _validate_leverage_config(config)
    save_leverage_config_raw(profile, config)
    return load_leverage_settings(profile)


def resolve_pool_ticker(ticker: str) -> dict[str, Any]:
    """종목풀(stock_meta)에서 해당 티커를 가진 활성 종목을 찾아 종목명을 반환합니다."""
    from utils.db_manager import get_db_connection

    ticker_norm = str(ticker or "").strip().upper()
    if not ticker_norm:
        raise ValueError("조회할 티커가 필요합니다.")

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")

    # active stocks 중 일치하는 종목 조회 (is_deleted가 참이 아닌 것)
    doc = db.stock_meta.find_one(
        {
            "ticker": ticker_norm,
            "is_deleted": {"$ne": True}
        },
        {"ticker": 1, "name": 1, "ticker_type": 1}
    )

    if doc is None:
        # fallback: 삭제 상태이더라도 종목풀에 등록된 적이 있는 종목 검색
        doc = db.stock_meta.find_one(
            {"ticker": ticker_norm},
            {"ticker": 1, "name": 1, "ticker_type": 1}
        )

    if doc is None:
        raise ValueError(f"종목풀에서 티커 '{ticker_norm}'를 찾을 수 없습니다.")

    return {
        "ticker": doc["ticker"],
        "name": doc["name"],
        "ticker_type": doc["ticker_type"]
    }
