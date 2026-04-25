from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request

from fastapi_app.dependencies import require_internal_token
from utils.rank_service import load_rank_data

router = APIRouter(prefix="/internal/rank", tags=["rank"])

@router.get("")
def get_rank_data(
    request: Request,
    ticker_type: str | None = Query(default=None),
    as_of_date: str | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    ma_type = request.query_params.get("ma_type")
    ma_months_raw = request.query_params.get("ma_months")
    ma_rule_override: dict[str, object] | None = None
    if ma_type is not None or ma_months_raw is not None:
        ma_rule_override = {
            "ma_type": ma_type or "",
            "ma_months": int(ma_months_raw) if ma_months_raw is not None else 0,
        }
    raw_held_bonus_score = request.query_params.get("held_bonus_score")
    held_bonus_score: int | None = None
    if raw_held_bonus_score is not None:
        try:
            held_bonus_score = int(raw_held_bonus_score)
        except ValueError as exc:
            raise ValueError(f"보유보너스점수 형식이 올바르지 않습니다: {raw_held_bonus_score}") from exc

    return load_rank_data(
        ticker_type=ticker_type,
        ma_rule_override=ma_rule_override,
        as_of_date=as_of_date,
        held_bonus_score=held_bonus_score,
    )
