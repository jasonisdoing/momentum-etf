"""종목 메모 — 종목(티커)에 붙는 한 줄 메모의 단일 소스.

저장 위치는 종목 관리 원본인 ``stock_meta.memo`` 다. **계좌가 아니라 종목에 붙는다** —
예전에는 계좌 보유 항목(`portfolio_master.accounts[].holdings[].memo`)에 있어서 종목을
전량 매도하면 메모가 사라지고 계좌마다 따로 적어야 했다.

자산 관리(`/assets`)의 매입단가 옆 칸과 종목풀 순위(`/pools-rank`)의 종목명 옆 칸이
같은 값을 읽고 쓴다. 같은 티커가 여러 풀에 있으면 모든 문서에 같은 값을 써서 한 종목의
메모가 하나로 유지된다.
"""

from __future__ import annotations

from datetime import datetime, timezone

from utils.logger import get_app_logger

logger = get_app_logger()

MAX_LENGTH = 100


def _collection():
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    return db["stock_meta"] if db is not None else None


def get_stock_memos(tickers: list[str] | set[str] | tuple[str, ...]) -> dict[str, str]:
    """티커 → 메모. 메모가 없는 종목은 결과에서 빠진다."""
    wanted = {str(t).strip() for t in tickers if str(t or "").strip()}
    coll = _collection()
    if not wanted or coll is None:
        return {}
    memos: dict[str, str] = {}
    for doc in coll.find({"ticker": {"$in": sorted(wanted)}, "memo": {"$nin": [None, ""]}}, {"ticker": 1, "memo": 1}):
        ticker = str(doc.get("ticker") or "").strip()
        memo = str(doc.get("memo") or "").strip()
        if ticker and memo:
            memos[ticker] = memo
    return memos


def set_stock_memo(ticker: str, memo: str | None) -> bool:
    """종목 메모를 저장한다(빈 값이면 지운다). 종목이 없으면 False."""
    ticker_norm = str(ticker or "").strip()
    coll = _collection()
    if not ticker_norm or coll is None:
        return False
    text = str(memo or "").strip()[:MAX_LENGTH]
    update = {"$set": {"memo": text, "updated_at": datetime.now(timezone.utc)}} if text else {"$unset": {"memo": ""}}
    try:
        # 같은 티커가 여러 풀에 있으면 모두 갱신한다 — 종목 하나에 메모 하나.
        result = coll.update_many({"ticker": ticker_norm}, update)
        if result.matched_count == 0:
            return False
    except Exception as exc:
        logger.warning("종목 메모 저장 실패 %s: %s", ticker_norm, exc)
        return False
    # 종목 목록 캐시가 메모를 그대로 실어 나르므로 비워서 다음 조회에 반영되게 한다.
    # 순위 캐시(5분)도 메모를 실어 나르므로 함께 지운다 — 목록 캐시만 지우면
    # `/assets` 에서 고친 메모가 `/pools-rank` 에 최대 5분간 옛 값으로 남는다.
    # 어느 풀인지는 티커만으로 알 수 없어(같은 티커가 여러 풀에 있을 수 있다) 전체를 지운다.
    try:
        from utils.cache_invalidation import invalidate_pool_caches
        from utils.stock_list_io import invalidate_ticker_type_cache

        invalidate_ticker_type_cache()  # 인자 없으면 전체 무효화
        invalidate_pool_caches()
    except Exception:
        pass
    return True
