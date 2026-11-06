"""종목풀별 **전략 사용 여부** — 어떤 전략을 어떤 종목풀에서 쓸지.

전략 화면(`/strategy-momentum` 등)의 종목풀 셀렉트는 이 플래그로 걸러진다. 예전에는 모든
종목풀이 다 나와서, 쓰지도 않는 풀이 목록을 채웠다.

**설정 존재 여부와 분리한다.** 켜기만 하고 설정은 나중에 그 전략 화면에서 채울 수 있어야
하므로, 「설정이 있으면 사용 중」으로 볼 수 없다. 끌 때는 그 전략의 설정과 저장된 백테스트
결과를 함께 지운다 — 다시 켰을 때 옛 설정이 되살아나면 혼란스럽다.
"""

from __future__ import annotations

from typing import Any

from utils.db_manager import get_db_connection
from utils.logger import get_app_logger
from utils.mix_sleeve import STRATEGY_OPTIONS, STRATEGY_TRADE

logger = get_app_logger()

COLLECTION = "pool_settings"
FIELD = "STRATEGY_USE"

# 종목풀에서 켜고 끄는 전략 — 합성 슬리브 전략 목록(`utils.mix_sleeve.STRATEGY_OPTIONS`)과 같다.
# 사고팔기는 국장 ETF 풀 전용이고 풀 단위 백테스트 결과가 없다.
STRATEGIES: tuple[str, ...] = STRATEGY_OPTIONS


def _db():
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")
    return db


def load_usage() -> dict[str, dict[str, bool]]:
    """{종목풀: {전략: 사용}} — 화면이 토글을 그리는 데 쓴다. 기록이 없으면 꺼짐."""
    usage: dict[str, dict[str, bool]] = {}
    for doc in _db()[COLLECTION].find({}, {"_id": 1, FIELD: 1}):
        pool = str(doc.get("_id") or "")
        if not pool or pool.startswith("__"):
            continue
        stored = doc.get(FIELD) or {}
        usage[pool] = {name: bool(stored.get(name)) for name in STRATEGIES}
    return usage


def trade_eligible_pools() -> list[str]:
    """사고팔기를 켤 수 있는 활성 종목풀 — 화면이 토글을 그릴 풀과 저장 검증이 같은 목록을 쓴다."""
    from utils.settings_loader import _load_pool_configs
    from utils.strategy_trade_config import is_eligible_pool

    return [str(config["ticker_type"]) for config in _load_pool_configs() if is_eligible_pool(config)]


def pool_options(pools: list[str]) -> list[dict[str, Any]]:
    """화면 셀렉트용 종목풀 정보 — 공용 `formatPoolLabel` 이 쓰는 필드(ticker_type·name·icon·order)와
    화면이 국가·통화·풀 성격으로 분기하는 값. order 순.

    전략 화면(모멘텀·신고가·포트폴리오·합성·사고팔기)이 모두 이 함수를 쓴다.
    """
    from utils.settings_loader import get_ticker_type_settings

    options: list[dict[str, Any]] = []
    for pool in pools:
        try:
            settings = get_ticker_type_settings(pool) or {}
        except Exception:
            settings = {}
        options.append(
            {
                "ticker_type": pool,
                "name": str(settings.get("name") or "").strip() or pool,
                "icon": str(settings.get("icon") or "").strip(),
                "order": settings.get("order"),
                "country_code": str(settings.get("country_code") or "").strip().lower(),
                "currency": str(settings.get("currency") or "").strip().upper(),
                "pool_kind": str(settings.get("pool_kind") or "").strip().lower(),
            }
        )
    return sorted(options, key=lambda item: (item["order"] is None, item["order"]))


def pools_using(strategy: str) -> list[str]:
    """그 전략을 쓰기로 켠 **활성** 종목풀 — 전략 화면의 종목풀 목록이 이 값이다.

    순서는 종목풀 order 순이다(셀렉트 번호와 같은 순서) — 활성 풀 목록
    (`list_available_ticker_types`, order 정렬)을 기준으로 켠 풀만 남긴다.
    """
    name = str(strategy or "").strip().lower()
    if name not in STRATEGIES:
        raise ValueError(f"알 수 없는 전략입니다: {strategy}")
    used = {
        str(doc["_id"])
        for doc in _db()[COLLECTION].find({f"{FIELD}.{name}": True}, {"_id": 1})
        if not str(doc["_id"]).startswith("__")
    }
    from utils.settings_loader import list_available_ticker_types

    return [pool for pool in list_available_ticker_types() if pool in used]


def set_usage(pool: str, strategy: str, used: bool) -> None:
    """사용 여부를 저장한다. **끄면 그 전략의 설정과 백테스트 결과를 지운다.**"""
    name = str(strategy or "").strip().lower()
    if name not in STRATEGIES:
        raise ValueError(f"알 수 없는 전략입니다: {strategy}")
    if name == STRATEGY_TRADE and pool not in trade_eligible_pools():
        raise ValueError(f"{pool} 종목풀에서는 사고팔기를 쓸 수 없습니다 (한국 ETF 종목풀만 가능).")

    _db()[COLLECTION].update_one({"_id": pool}, {"$set": {f"{FIELD}.{name}": bool(used)}}, upsert=True)
    if used:
        return

    _delete_strategy_settings(pool, name)
    if name != STRATEGY_TRADE:
        from utils.pool_backtest_store import clear_result

        # 그 전략 결과만 지운다 — 풀 전체를 지우면 다른 전략의 백테스트까지 사라진다.
        clear_result(pool, name)


def _delete_strategy_settings(pool: str, strategy: str) -> None:
    """그 전략이 이 풀에 저장해 둔 설정을 지운다 — 전략마다 저장 위치가 다르다."""
    try:
        if strategy == "momentum":
            from utils.momentum_service import delete_settings

            delete_settings(pool)
        elif strategy == "new_high":
            from utils.new_high_service import delete_settings

            delete_settings(pool)
        elif strategy == STRATEGY_TRADE:
            from utils.strategy_trade_service import delete_pool_strategies

            delete_pool_strategies(pool)
        else:
            from utils.portfolio_service import delete_settings

            delete_settings(pool)
    except Exception as exc:  # noqa: BLE001 - 플래그는 이미 껐으므로 여기서 막지 않는다
        logger.warning("전략 설정 삭제 실패 (%s/%s): %s", pool, strategy, exc)
