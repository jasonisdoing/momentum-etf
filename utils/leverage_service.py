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


def load_leverage_settings(profile: str = "switch") -> dict[str, Any]:
    """레버리지 설정(편집 대상) + 직전 추천 상태(읽기 전용)를 함께 반환한다."""
    return {
        "profile": profile,
        "config": load_leverage_config_raw(profile),
        "state": load_leverage_state(profile),
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

    # 엔진 정규화로 추가 검증 (사본으로 — 파생 키가 저장값에 섞이지 않게).
    normalize_settings(dict(config))


def save_leverage_settings(profile: str, config: dict[str, Any]) -> dict[str, Any]:
    """검증 후 설정을 DB 에 저장하고, 갱신된 설정+상태를 반환한다."""
    _validate_leverage_config(config)
    save_leverage_config_raw(profile, config)
    return load_leverage_settings(profile)
