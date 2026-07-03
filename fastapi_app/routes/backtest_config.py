"""백테스트 탐색공간(backtest_config) 조회/저장 API.

풀별 BACKTEST_MONTHS(개월수) + BENCHMARK + TOP_N_HOLD/HOLDING_BONUS_SCORE/
TREND_WEIGHT_RATIO/MA_TYPE/MA_MONTHS/RSI_LIMIT(리스트)을 DB 에서 조회·저장한다
(단일 소스: utils.backtest_config_store).
모멘텀-설정 화면에서 편집한다.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from config import ALLOWED_MA_TYPES
from fastapi_app.dependencies import require_internal_token
from utils.backtest_config_store import (
    get_backtest_config_updated_at,
    list_backtest_pools,
    load_backtest_config,
    save_backtest_config,
)
from utils.settings_loader import get_ticker_type_settings
from utils.ticker_registry import load_ticker_type_configs

router = APIRouter(prefix="/internal/backtest-config", tags=["backtest-config"])


class BacktestConfigUpdatePayload(BaseModel):
    pool_id: str
    config: dict[str, Any]


def _ordered_pools(db_pools: set[str]) -> list[str]:
    """pools.json 에 등록된 종목풀만 그 순서대로 나열한다 (DB 의 옛/미등록 문서는 표시하지 않음)."""
    order = [str(c["ticker_type"]) for c in load_ticker_type_configs()]
    return [p for p in order if p in db_pools]


@router.get("")
def get_backtest_configs(_: None = Depends(require_internal_token)) -> dict[str, object]:
    """풀별 백테스트 탐색공간 + 입력 제약(MA 타입)을 반환한다."""
    db_pools = set(list_backtest_pools())
    name_by_type = {str(c["ticker_type"]): str(c["name"]) for c in load_ticker_type_configs()}

    pools: list[dict[str, Any]] = []
    for pid in _ordered_pools(db_pools):
        name = name_by_type.get(pid, pid)
        # TOP_N_HOLD 미저장 풀의 UI 초기값용 — 종목풀 설정(pool_settings)의 라이브 N.
        try:
            live_top_n = int(get_ticker_type_settings(pid)["TOP_N_HOLD"])
        except Exception:
            live_top_n = None
        pools.append(
            {
                "pool_id": pid,
                "name": name,
                "config": load_backtest_config(pid),
                "live_top_n_hold": live_top_n,
                "updated_at": get_backtest_config_updated_at(pid),
            }
        )

    return {"pools": pools, "constraints": {"ma_types": ALLOWED_MA_TYPES}}


@router.put("")
def put_backtest_config(
    payload: BacktestConfigUpdatePayload, _: None = Depends(require_internal_token)
) -> dict[str, object]:
    """풀의 백테스트 탐색공간을 검증 후 저장한다. 검증 실패 → 400."""
    try:
        save_backtest_config(payload.pool_id, payload.config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": True,
        "pool_id": payload.pool_id,
        "config": load_backtest_config(payload.pool_id),
        "updated_at": get_backtest_config_updated_at(payload.pool_id),
    }
