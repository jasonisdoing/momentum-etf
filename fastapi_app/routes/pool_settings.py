"""종목풀 편집 가능 설정(pool_settings) 조회/저장 API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from config import TOP_N_HOLD_OPTIONS
from fastapi_app.dependencies import require_internal_token
from utils.ma_options import ma_options_by_country
from utils.market_breadth_service import MARKET_BY_INDEX_TICKER, SELF_POOL_REGIME_TICKER
from utils.market_trend_service import INDICES
from utils.pool_settings_store import (
    POOL_EDITABLE_KEYS,
    SLIPPAGE_PCT_OPTIONS,
    STOPLOSS_PCT_OPTIONS,
    PoolSettingsError,
    create_pool,
    delete_pool,
    get_pool_delete_impact,
    load_pool_definitions,
    save_pool_settings,
    update_pool,
)

router = APIRouter(prefix="/internal/pool-settings", tags=["pool-settings"])


class PoolSettingsUpdatePayload(BaseModel):
    pool_id: str
    values: dict[str, Any]
    # 저장 경로 — 화면의 「마지막 저장」 배지에 그대로 뜬다.
    # 사용자(사람이 화면에서 저장) / 모멘텀 전략(전략 화면·튜닝 적용)
    save_method: str = "사용자"


class PoolDefinitionPayload(BaseModel):
    """풀 생성·수정 — 둘 다 사람이 화면에서 한 일이라 저장 경로는 「사용자」다."""

    values: dict[str, Any]
    save_method: str = "사용자"


def _editable(settings: dict[str, Any]) -> dict[str, Any]:
    """편집 가능한 키의 현재(DB) 값을 반환한다."""
    return {key: {"value": settings.get(key)} for key in POOL_EDITABLE_KEYS}


def _pool_payload(settings: dict[str, Any], stock_counts: dict[str, int]) -> dict[str, Any]:
    return {
        # 그 풀에 담긴 종목 수 — 설정이 아니라 현황이라 편집할 수 없다.
        "stock_count": stock_counts.get(str(settings["ticker_type"]).strip().lower(), 0),
        "ticker_type": settings["ticker_type"],
        "name": settings["name"],
        "icon": settings["icon"],
        "order": settings["order"],
        "country_code": settings["country_code"],
        "currency": settings["currency"],
        # 풀 성격(stock/etf) — 미설정이면 None (화면이 '미설정' 으로 보여준다).
        "pool_kind": settings.get("pool_kind"),
        "type_source": settings.get("type_source"),
        "settings": _editable(settings),
        "updated_at": settings.get("updated_at"),
        "save_method": settings.get("save_method"),
    }


@router.get("")
def get_pool_settings(_: None = Depends(require_internal_token)) -> dict[str, object]:
    """풀별 편집 가능 설정과 입력 제약을 반환한다."""
    from utils.stock_list_io import count_stocks_by_pool

    try:
        stock_counts = count_stocks_by_pool()
        pools = [_pool_payload(settings, stock_counts) for settings in load_pool_definitions()]
    except PoolSettingsError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    from utils.pool_backtest_store import load_results

    return {
        "pools": pools,
        # 전략별 12개월 백테스트 — 「백테스트」로 계산해 저장해 둔 값. 설정이 바뀌면 지워진다.
        "backtests": load_results(),
        "constraints": {
            "ma_options_by_country": ma_options_by_country(),
            "top_n_hold_options": list(TOP_N_HOLD_OPTIONS),
            "slippage_pct_options": list(SLIPPAGE_PCT_OPTIONS),
            "stoploss_pct_options": list(STOPLOSS_PCT_OPTIONS),
            "editable_keys": list(POOL_EDITABLE_KEYS),
            # ADR 기준 후보 — 지수 4개(코스피·코스닥·S&P500·나스닥100) + 그 종목풀 자신.
            # 종목풀은 지수와 구성이 달라(us_stock 은 S&P100+나스닥100 조합) 매매하는
            # 종목의 폭을 그대로 본다. 값은 예약 티커라 저장 스키마는 그대로다.
            "market_indices": [
                *(
                    {"ticker": item["yf_ticker"], "name": item["name"]}
                    for item in INDICES
                    if item["yf_ticker"] in MARKET_BY_INDEX_TICKER
                ),
                {"ticker": SELF_POOL_REGIME_TICKER, "name": "종목풀"},
            ],
        },
    }


@router.put("")
def put_pool_settings(
    payload: PoolSettingsUpdatePayload, _: None = Depends(require_internal_token)
) -> dict[str, object]:
    """편집한 값을 저장한다 (pool_id = ticker_type)."""
    try:
        saved = save_pool_settings(payload.pool_id, payload.values, save_method=payload.save_method)
    except PoolSettingsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "pool_id": payload.pool_id, "saved": saved}


@router.post("/pools")
def post_pool_definition(
    payload: PoolDefinitionPayload, _: None = Depends(require_internal_token)
) -> dict[str, object]:
    """신규 종목풀을 생성한다."""
    try:
        saved = create_pool(payload.values, save_method=payload.save_method)
    except PoolSettingsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "pool": saved}


@router.patch("/pools/{pool_id}")
def patch_pool_definition(
    pool_id: str, payload: PoolDefinitionPayload, _: None = Depends(require_internal_token)
) -> dict[str, object]:
    """기존 종목풀의 메타/설정을 수정한다. ticker_type 은 변경 불가."""
    try:
        saved = update_pool(pool_id, payload.values, save_method=payload.save_method)
    except PoolSettingsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "pool_id": pool_id, "saved": saved}


@router.get("/pools/{pool_id}/delete-impact")
def get_pool_delete_impact_route(pool_id: str, _: None = Depends(require_internal_token)) -> dict[str, object]:
    """종목풀 삭제 영향도를 반환한다."""
    try:
        return get_pool_delete_impact(pool_id)
    except PoolSettingsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/pools/{pool_id}")
def delete_pool_definition(pool_id: str, _: None = Depends(require_internal_token)) -> dict[str, object]:
    """계좌 연결이 없는 종목풀을 하드 삭제한다."""
    try:
        deleted = delete_pool(pool_id)
    except PoolSettingsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "deleted": deleted}


@router.get("/backtest")
def get_pool_strategy_backtest(
    pool: str,
    strategy: str,
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    """그 종목풀·전략의 저장 설정으로 12개월 백테스트를 돌려 요약만 돌려준다.

    종목풀 설정 화면이 전략별 성과를 나란히 보기 위해 쓴다. 풀×전략을 한꺼번에 돌리면
    수십 초가 걸려서, 화면이 버튼을 눌렀을 때 하나씩만 부른다.
    설정이 없는 전략은 값을 만들어 내지 않고 빈 결과(None)를 돌려준다.
    """
    from utils.mix_sleeve import MOMENTUM, NEW_HIGH, PORTFOLIO, normalize_strategy, settings_map

    months = 12
    name = normalize_strategy(strategy)
    stored = settings_map(name).get(pool)
    if not stored:
        from utils.pool_backtest_store import save_result

        # 설정이 없어 못 돌린 것과 「아직 안 돌린 것」은 화면에서 구분돼야 한다 — 그 사실을 저장한다.
        empty = {"cagr_pct": None, "mdd_pct": None, "sortino": None, "no_settings": True}
        return {"pool": pool, "strategy": name, "result": save_result(pool, name, empty)}

    settings = {**stored, "pool": pool}
    try:
        if name == MOMENTUM:
            from utils.momentum_backtest import run_backtest

            result = run_backtest(months, settings)
        elif name == NEW_HIGH:
            from utils.new_high_backtest import run_backtest as nh_backtest

            result = nh_backtest(months, settings)
        else:
            from utils.portfolio_backtest import run_backtest as portfolio_backtest

            assert name == PORTFOLIO
            result = portfolio_backtest(months, settings)
    except Exception as error:  # noqa: BLE001 - 한 전략 실패가 화면을 막지 않게 한다
        raise HTTPException(status_code=400, detail=str(error)) from error

    from utils.pool_backtest_store import save_result

    saved = save_result(
        pool,
        name,
        {
            "cagr_pct": result.get("strategy_cagr_pct"),
            "mdd_pct": result.get("strategy_mdd_pct"),
            "sortino": result.get("strategy_sortino"),
        },
    )
    return {"pool": pool, "strategy": name, "result": saved}
