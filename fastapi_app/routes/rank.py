from __future__ import annotations

import re

from fastapi import APIRouter, Depends, Query, Request

from fastapi_app.dependencies import require_internal_token
from utils.rank_service import load_rank_data

router = APIRouter(prefix="/internal/rank", tags=["rank"])


def _parse_ma_rule_overrides(request: Request) -> list[dict[str, object]]:
    overrides: dict[int, dict[str, object]] = {}
    for key, value in request.query_params.multi_items():
        match = re.fullmatch(r"rule(\d+)_(ma_type|ma_months)", key)
        if not match:
            continue
        order = int(match.group(1))
        field = match.group(2)
        entry = overrides.setdefault(order, {"order": order})
        entry[field] = value
    return [overrides[order] for order in sorted(overrides)]


@router.get("")
def get_rank_data(
    request: Request,
    ticker_type: str | None = Query(default=None),
    as_of_date: str | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    ma_rule_overrides = _parse_ma_rule_overrides(request)
    raw_held_bonus_score = request.query_params.get("held_bonus_score")
    if raw_held_bonus_score is None:
        raise ValueError("보유보너스점수 값이 필요합니다.")
    try:
        held_bonus_score = int(raw_held_bonus_score)
    except ValueError as exc:
        raise ValueError(f"보유보너스점수 형식이 올바르지 않습니다: {raw_held_bonus_score}") from exc

    return load_rank_data(
        ticker_type=ticker_type,
        ma_rule_overrides=ma_rule_overrides,
        as_of_date=as_of_date,
        held_bonus_score=held_bonus_score,
    )
