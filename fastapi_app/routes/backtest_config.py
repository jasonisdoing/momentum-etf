"""백테스트 탐색공간(backtest_config) 조회/저장 API.

풀별 BENCHMARK + HOLDING_BONUS_SCORE/MA_TYPE/MA_MONTHS/RSI_LIMIT(리스트)을 DB 에서
조회·저장한다(단일 소스: utils.backtest_config_store). 모멘텀-설정 화면에서 편집한다.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from config import ALLOWED_MA_TYPES
from fastapi_app.dependencies import require_internal_token
from utils.backtest_config_store import (
    list_backtest_pools,
    load_backtest_config,
    save_backtest_config,
)
from utils.ticker_registry import load_ticker_type_configs

router = APIRouter(prefix="/internal/backtest-config", tags=["backtest-config"])

_ALL_POOL_ID = "all"


class BacktestConfigUpdatePayload(BaseModel):
    pool_id: str
    config: dict[str, Any]


def _ordered_pools(db_pools: set[str]) -> list[str]:
    """all 을 맨 앞에, 그다음 ticker_type 순서로 정렬한다."""
    order = [_ALL_POOL_ID] + [str(c["ticker_type"]) for c in load_ticker_type_configs()]
    ordered = [p for p in order if p in db_pools]
    ordered += [p for p in sorted(db_pools) if p not in order]
    return ordered


@router.get("")
def get_backtest_configs(_: None = Depends(require_internal_token)) -> dict[str, object]:
    """풀별 백테스트 탐색공간 + 입력 제약(MA 타입)을 반환한다."""
    db_pools = set(list_backtest_pools())
    name_by_type = {str(c["ticker_type"]): str(c["name"]) for c in load_ticker_type_configs()}

    pools: list[dict[str, Any]] = []
    for pid in _ordered_pools(db_pools):
        name = "전체 (가상 종목풀)" if pid == _ALL_POOL_ID else name_by_type.get(pid, pid)
        pools.append({"pool_id": pid, "name": name, "config": load_backtest_config(pid)})

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
    return {"ok": True, "pool_id": payload.pool_id, "config": load_backtest_config(payload.pool_id)}
