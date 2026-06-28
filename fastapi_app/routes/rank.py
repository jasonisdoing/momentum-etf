from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request

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

    raw_ath_bonus = request.query_params.get("ath_bonus")
    ath_bonus: int | None = None
    if raw_ath_bonus is not None:
        try:
            ath_bonus = int(raw_ath_bonus)
        except ValueError as exc:
            raise ValueError(f"ATH보너스 형식이 올바르지 않습니다: {raw_ath_bonus}") from exc

    return load_rank_data(
        ticker_type=ticker_type,
        ma_rule_override=ma_rule_override,
        as_of_date=as_of_date,
        held_bonus_score=held_bonus_score,
        ath_bonus=ath_bonus,
    )
