from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from fastapi_app.dependencies import require_internal_token
from utils.rank_service import load_rank_data, load_rank_toolbar_data

router = APIRouter(prefix="/internal/rank", tags=["rank"])


@router.get("/toolbar")
def get_rank_toolbar_data(
    ticker_type: str | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    return load_rank_toolbar_data(ticker_type=ticker_type)


@router.get("")
def get_rank_data(
    ticker_type: str | None = Query(default=None),
    as_of_date: str | None = Query(default=None),
    short_ma_days: int | None = Query(default=None),
    long_ma_days: int | None = Query(default=None),
    # 종목 수·업종 상한 — 화면 상단에서 바꿔 보는 값. 생략하면 종목풀 저장값.
    # 업종 상한의 -1 은 '제한 없음' 이다.
    top_n: int | None = Query(default=None),
    max_per_industry: int | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    ma_rule_override: dict[str, object] | None = None
    if short_ma_days is not None or long_ma_days is not None:
        ma_rule_override = {
            "short_ma_days": short_ma_days,
            "long_ma_days": long_ma_days,
        }
    return load_rank_data(
        ticker_type=ticker_type,
        ma_rule_override=ma_rule_override,
        as_of_date=as_of_date,
        top_n_override=top_n,
        max_per_industry_override=max_per_industry,
    )
