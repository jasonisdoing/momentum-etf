from __future__ import annotations

from fastapi import APIRouter, Body, Depends, Query

from config import HOLDING_CHART_MONTHS
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
    short_ma_days: int | None = Query(default=None),
    long_ma_days: int | None = Query(default=None),
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
    )


@router.post("/charts")
def post_rank_charts(
    payload: dict = Body(default={}),
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    """순위 화면 차트 — 일봉 + 그 풀의 판정 이평선. body: ``{"ticker_type": "...", "tickers": [...]}``.

    티커는 화면이 보고 있는 순서 그대로 넘긴다(정렬·필터를 반영한 순서). 여기서 순위를
    다시 매기면 표와 차트 순서가 갈린다. 이평선은 화면이 임시로 바꿀 수 있어(툴바) 값을
    받으면 그걸 쓰고, 없으면 풀 저장값을 쓴다.
    """
    from utils.holding_chart_service import holding_charts
    from utils.settings_loader import get_ticker_type_settings

    ticker_type = str(payload.get("ticker_type") or "").strip()
    if not ticker_type:
        raise ValueError("'ticker_type' 이 필요합니다.")
    tickers = payload.get("tickers")
    if not isinstance(tickers, list):
        raise ValueError("'tickers' 는 목록이어야 합니다.")

    pool_settings = get_ticker_type_settings(ticker_type) or {}
    short = payload.get("short_ma_days") or pool_settings["SHORT_MA_DAYS"]
    long = payload.get("long_ma_days") or pool_settings["LONG_MA_DAYS"]
    return {
        "charts": holding_charts(ticker_type, [str(ticker) for ticker in tickers], [int(short), int(long)]),
        # 화면 안내 문구("최근 N개월 일봉입니다")가 쓰는 값 — 프론트에 복사본을 두지 않는다.
        "months": HOLDING_CHART_MONTHS,
        "short_ma_days": int(short),
        "long_ma_days": int(long),
    }
