"""종목풀·전략별 12개월 백테스트 요약 저장소.

종목풀 설정 화면이 전략 성과를 나란히 보여주는 데 쓴다. 풀×전략을 화면 열 때마다 돌리면
수십 초가 걸려서, 사용자가 「백테스트」를 눌렀을 때 계산해 여기 저장하고 이후에는 읽기만 한다.

**그 풀의 설정이 바뀌면 지운다.** 옛 설정으로 낸 성과가 새 설정 옆에 남으면 잘못 읽는다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from utils.db_manager import get_db_connection
from utils.logger import get_app_logger

logger = get_app_logger()

COLLECTION = "pool_strategy_backtest"


def _db():
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")
    return db


def save_result(pool: str, strategy: str, result: dict[str, Any]) -> dict[str, Any]:
    """한 조합의 결과를 저장하고 저장 시각을 붙여 돌려준다."""
    updated_at = datetime.now(timezone.utc)
    doc = {**result, "updated_at": updated_at}
    _db()[COLLECTION].update_one(
        {"pool": pool, "strategy": strategy},
        {"$set": {"pool": pool, "strategy": strategy, **doc}},
        upsert=True,
    )
    return {**result, "updated_at": updated_at.isoformat()}


def load_results() -> dict[str, dict[str, dict[str, Any]]]:
    """{종목풀: {전략: 결과}} — 화면이 한 번에 받아 표에 채운다."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for doc in _db()[COLLECTION].find({}, {"_id": 0}):
        pool = str(doc.get("pool") or "")
        strategy = str(doc.get("strategy") or "")
        if not pool or not strategy:
            continue
        updated_at = doc.get("updated_at")
        out.setdefault(pool, {})[strategy] = {
            "cagr_pct": doc.get("cagr_pct"),
            "mdd_pct": doc.get("mdd_pct"),
            "sortino": doc.get("sortino"),
            "no_settings": bool(doc.get("no_settings")),
            "updated_at": updated_at.isoformat() if isinstance(updated_at, datetime) else updated_at,
        }
    return out


def clear_pool(pool: str) -> None:
    """그 종목풀의 저장 결과를 모두 지운다 — 설정이 바뀌면 옛 성과를 남기지 않는다."""
    try:
        _db()[COLLECTION].delete_many({"pool": pool})
    except Exception as exc:  # noqa: BLE001 - 저장은 이미 끝났으므로 여기서 막지 않는다
        logger.warning("종목풀 백테스트 결과 삭제 실패 (%s): %s", pool, exc)
