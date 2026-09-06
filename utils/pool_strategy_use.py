"""종목풀별 **전략 사용 여부** — 어떤 전략을 어떤 종목풀에서 쓸지.

전략 화면(`/strategy-momentum` 등)의 종목풀 셀렉트는 이 플래그로 걸러진다. 예전에는 모든
종목풀이 다 나와서, 쓰지도 않는 풀이 목록을 채웠다.

**설정 존재 여부와 분리한다.** 켜기만 하고 설정은 나중에 그 전략 화면에서 채울 수 있어야
하므로, 「설정이 있으면 사용 중」으로 볼 수 없다. 끌 때는 그 전략의 설정과 저장된 백테스트
결과를 함께 지운다 — 다시 켰을 때 옛 설정이 되살아나면 혼란스럽다.
"""

from __future__ import annotations

from utils.db_manager import get_db_connection
from utils.logger import get_app_logger

logger = get_app_logger()

COLLECTION = "pool_settings"
FIELD = "STRATEGY_USE"

# 값은 `utils.mix_sleeve.STRATEGY_OPTIONS` 와 같은 이름을 쓴다.
STRATEGIES: tuple[str, ...] = ("momentum", "new_high", "portfolio")


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


def pools_using(strategy: str) -> list[str]:
    """그 전략을 쓰기로 켠 종목풀 — 전략 화면의 종목풀 목록이 이 값이다."""
    name = str(strategy or "").strip().lower()
    if name not in STRATEGIES:
        raise ValueError(f"알 수 없는 전략입니다: {strategy}")
    pools = [
        str(doc["_id"])
        for doc in _db()[COLLECTION].find({f"{FIELD}.{name}": True}, {"_id": 1})
        if not str(doc["_id"]).startswith("__")
    ]
    return sorted(pools)


def set_usage(pool: str, strategy: str, used: bool) -> None:
    """사용 여부를 저장한다. **끄면 그 전략의 설정과 백테스트 결과를 지운다.**"""
    name = str(strategy or "").strip().lower()
    if name not in STRATEGIES:
        raise ValueError(f"알 수 없는 전략입니다: {strategy}")

    _db()[COLLECTION].update_one({"_id": pool}, {"$set": {f"{FIELD}.{name}": bool(used)}}, upsert=True)
    if used:
        return

    from utils.pool_backtest_store import clear_pool

    _delete_strategy_settings(pool, name)
    clear_pool(pool)


def _delete_strategy_settings(pool: str, strategy: str) -> None:
    """그 전략이 이 풀에 저장해 둔 설정을 지운다 — 전략마다 저장 위치가 다르다."""
    try:
        if strategy == "momentum":
            from utils.momentum_service import delete_settings

            delete_settings(pool)
        elif strategy == "new_high":
            from utils.new_high_service import delete_settings

            delete_settings(pool)
        else:
            from utils.portfolio_service import delete_settings

            delete_settings(pool)
    except Exception as exc:  # noqa: BLE001 - 플래그는 이미 껐으므로 여기서 막지 않는다
        logger.warning("전략 설정 삭제 실패 (%s/%s): %s", pool, strategy, exc)
