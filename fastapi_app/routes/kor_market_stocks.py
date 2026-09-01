from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from fastapi_app.dependencies import require_internal_token
from utils.kor_stock_market_service import load_kor_stock_market

router = APIRouter(prefix="/internal/kor-market-stocks", tags=["kor-market-stocks"])


@router.get("")
def get_kor_market_stocks(
    market: Annotated[str, Query(pattern="^(KOSPI|KOSDAQ|KOSPI200|KOSDAQ150)$")],
    limit: Annotated[int, Query(ge=1, le=200)],
    min_market_cap_jo: Annotated[int, Query(ge=0)],
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    try:
        return load_kor_stock_market(market=market, limit=limit, min_market_cap_jo=min_market_cap_jo)
    except LookupError as exc:
        # 지수 구성종목 배치가 아직 안 돌았다는 뜻 — 서버 오류가 아니라 준비 안 됨이다.
        raise HTTPException(status_code=404, detail=str(exc)) from exc
