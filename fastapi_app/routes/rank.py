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
    main_ma_days: int | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    ma_rule_override: dict[str, object] | None = None
    if short_ma_days is not None or main_ma_days is not None:
        ma_rule_override = {
            "short_ma_days": short_ma_days,
            "main_ma_days": main_ma_days,
        }
    return load_rank_data(
        ticker_type=ticker_type,
        ma_rule_override=ma_rule_override,
        as_of_date=as_of_date,
    )
