"""종목 메타 캐시(stock_cache_meta) 컬렉션을 읽고 쓰는 유틸리티."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from utils.db_manager import get_db_connection
from utils.logger import get_app_logger

logger = get_app_logger()

_COLLECTION_NAME = "stock_cache_meta"
# 종목별 **직전 영업일 스냅샷 1건**만 보관한다(티커당 문서 1개).
#
# 예전에는 `stock_cache_meta_history` 에 날짜별 스냅샷을 전부 쌓았다. 오늘 것까지
# 들어가다 보니 "가장 최근 1건"이 오늘인지 어제인지 알 수 없어 조회를 2번 했고,
# 휴장일이면 최대 7번까지 거슬러 올라갔다. 데이터는 무한히 늘어 193MB(DB의 55%)가
# 되어 mongodump 가 끊겼다.
#
# 지금은 **현재 값은 stock_cache_meta, 직전 값은 여기** 로 역할을 나눈다.
# 비교가 늘 (현재 vs 직전) 한 쌍이라 조회 1회면 되고, 크기도 티커 수로 고정된다.
_PREVIOUS_COLLECTION_NAME = "previous_stock_cache_meta"
_INDEX_ENSURED = False


def _get_collection():
    """stock_cache_meta 컬렉션 핸들을 반환하고, 최초 호출 시 인덱스를 보장한다."""
    global _INDEX_ENSURED
    db = get_db_connection()
    if db is None:
        return None

    coll = db[_COLLECTION_NAME]
    if not _INDEX_ENSURED:
        try:
            coll.create_index(
                [("ticker_type", 1), ("ticker", 1)],
                unique=True,
                name="ticker_type_ticker_unique",
                background=True,
            )
            coll.create_index(
                [("country_code", 1), ("ticker", 1)],
                name="country_code_ticker_lookup",
                background=True,
            )
            _INDEX_ENSURED = True
        except Exception:
            pass
    return coll


def ensure_stock_cache_meta_readable() -> None:
    """stock_cache_meta 컬렉션을 읽을 수 없으면 즉시 예외를 발생시킨다."""
    coll = _get_collection()
    if coll is None:
        raise RuntimeError("MongoDB 연결 실패 — stock_cache_meta 컬렉션을 읽을 수 없습니다.")


def get_stock_cache_meta_doc(ticker_type: str, ticker: str) -> dict[str, Any] | None:
    """종목 메타 캐시 문서 1건을 반환한다."""
    type_norm = (ticker_type or "").strip().lower()
    ticker_norm = str(ticker or "").strip().upper()
    if not type_norm:
        raise ValueError("ticker_type must be provided")
    if not ticker_norm:
        raise ValueError("ticker must be provided")

    coll = _get_collection()
    if coll is None:
        raise RuntimeError("MongoDB 연결 실패 — stock_cache_meta 컬렉션을 읽을 수 없습니다.")

    doc = coll.find_one({"ticker_type": type_norm, "ticker": ticker_norm}, {"_id": 0})
    return dict(doc) if isinstance(doc, dict) else None


def get_stock_cache_meta_docs(ticker_type: str, tickers: list[str]) -> dict[str, dict[str, Any]]:
    """종목 메타 캐시 문서를 티커 기준 맵으로 반환한다."""
    type_norm = (ticker_type or "").strip().lower()
    normalized_tickers = sorted({str(ticker or "").strip().upper() for ticker in tickers if str(ticker or "").strip()})
    if not type_norm:
        raise ValueError("ticker_type must be provided")
    if not normalized_tickers:
        return {}

    coll = _get_collection()
    if coll is None:
        raise RuntimeError("MongoDB 연결 실패 — stock_cache_meta 컬렉션을 읽을 수 없습니다.")

    docs = coll.find(
        {"ticker_type": type_norm, "ticker": {"$in": normalized_tickers}},
        {"_id": 0},
    )
    result: dict[str, dict[str, Any]] = {}
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        ticker_norm = str(doc.get("ticker") or "").strip().upper()
        if ticker_norm:
            result[ticker_norm] = dict(doc)
    return result


def get_previous_stock_cache_meta(ticker_type: str, ticker: str) -> dict[str, Any] | None:
    """직전 영업일 스냅샷 1건. 없으면 None(비교 기준이 없다는 뜻).

    ``date`` 는 저장 시점에 이미 거래일로 정해져 있어 호출자가 휴장일 보정을 할 필요가 없다.
    """
    type_norm = (ticker_type or "").strip().lower()
    ticker_norm = str(ticker or "").strip().upper()

    db = get_db_connection()
    if db is None:
        return None

    doc = db[_PREVIOUS_COLLECTION_NAME].find_one(
        {"ticker_type": type_norm, "ticker": ticker_norm},
        projection={"_id": 0},
    )
    return dict(doc) if isinstance(doc, dict) else None


def _resolve_snapshot_date() -> str:
    """스냅샷 귀속 거래일 — 9시 이후이고 오늘이 거래일이면 오늘, 아니면 직전 거래일."""
    from utils.data_loader import get_trading_days

    now_kst = datetime.now(ZoneInfo("Asia/Seoul"))
    today_str = now_kst.strftime("%Y-%m-%d")
    trading_days = get_trading_days((now_kst - timedelta(days=14)).strftime("%Y-%m-%d"), today_str, "kor")
    trading_days_str = [d.strftime("%Y-%m-%d") for d in trading_days]

    if now_kst.hour >= 9 and today_str in trading_days_str:
        return today_str
    past_days = [d for d in trading_days_str if d < today_str]
    return past_days[-1] if past_days else today_str


def upsert_stock_cache_meta_doc(
    ticker_type: str,
    ticker: str,
    *,
    country_code: str,
    name: str,
    meta_cache: dict[str, Any] | None = None,
    holdings_cache: dict[str, Any] | None = None,
) -> None:
    """종목 메타 캐시 문서를 upsert한다."""
    type_norm = (ticker_type or "").strip().lower()
    ticker_norm = str(ticker or "").strip().upper()
    country_norm = str(country_code or "").strip().lower()
    name_norm = str(name or "").strip()
    if not type_norm:
        raise ValueError("ticker_type must be provided")
    if not ticker_norm:
        raise ValueError("ticker must be provided")
    if not country_norm:
        raise ValueError("country_code must be provided")
    if not name_norm:
        raise ValueError("name must be provided")

    coll = _get_collection()
    if coll is None:
        raise RuntimeError("MongoDB 연결 실패 — stock_cache_meta 컬렉션에 쓸 수 없습니다.")

    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "ticker_type": type_norm,
        "ticker": ticker_norm,
        "country_code": country_norm,
        "name": name_norm,
        "updated_at": now,
    }
    if meta_cache is not None:
        payload["meta_cache"] = meta_cache
    if holdings_cache is not None:
        payload["holdings_cache"] = holdings_cache

    snapshot_date = _resolve_snapshot_date()
    payload["snapshot_date"] = snapshot_date

    db = get_db_connection()
    # 오늘 값을 덮기 전에, 저장돼 있던 값이 **다른 날짜의 것**이면 직전값으로 옮긴다.
    # 같은 날 여러 번 돌아도 직전값은 그대로다(귀속 날짜가 같으므로).
    if db is not None:
        current = coll.find_one(
            {"ticker_type": type_norm, "ticker": ticker_norm},
            projection={"_id": 0, "created_at": 0},
        )
        if current and str(current.get("snapshot_date") or "") not in ("", snapshot_date):
            db[_PREVIOUS_COLLECTION_NAME].update_one(
                {"ticker_type": type_norm, "ticker": ticker_norm},
                {"$set": {**current, "date": current["snapshot_date"]}},
                upsert=True,
            )

    coll.update_one(
        {"ticker_type": type_norm, "ticker": ticker_norm},
        {
            "$set": payload,
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )


def update_stock_portfolio_change_cache_doc(
    ticker_type: str,
    ticker: str,
    portfolio_change_cache: dict[str, Any],
) -> None:
    """종목 메타 캐시 문서에 포트폴리오 변동 계산 캐시만 갱신한다."""
    type_norm = (ticker_type or "").strip().lower()
    ticker_norm = str(ticker or "").strip().upper()
    if not type_norm:
        raise ValueError("ticker_type must be provided")
    if not ticker_norm:
        raise ValueError("ticker must be provided")
    if not isinstance(portfolio_change_cache, dict):
        raise ValueError("portfolio_change_cache must be a dict")

    coll = _get_collection()
    if coll is None:
        raise RuntimeError("MongoDB 연결 실패 — stock_cache_meta 컬렉션에 쓸 수 없습니다.")

    now = datetime.now(timezone.utc)
    result = coll.update_one(
        {"ticker_type": type_norm, "ticker": ticker_norm},
        {
            "$set": {
                "portfolio_change_cache": portfolio_change_cache,
                "portfolio_change_cache_updated_at": now,
                "updated_at": now,
            },
        },
    )
    if result.matched_count == 0:
        raise RuntimeError(f"[{type_norm}/{ticker_norm}] 포트폴리오 변동 캐시를 저장할 메타 문서가 없습니다.")


def delete_stock_cache_meta_doc(ticker_type: str, ticker: str) -> None:
    """종목 메타 캐시 문서 1건을 삭제한다."""
    type_norm = (ticker_type or "").strip().lower()
    ticker_norm = str(ticker or "").strip().upper()
    if not type_norm:
        raise ValueError("ticker_type must be provided")
    if not ticker_norm:
        raise ValueError("ticker must be provided")

    coll = _get_collection()
    if coll is None:
        raise RuntimeError("MongoDB 연결 실패 — stock_cache_meta 컬렉션에 쓸 수 없습니다.")

    coll.delete_one({"ticker_type": type_norm, "ticker": ticker_norm})


def set_stock_cache_meta_field(ticker_type: str, ticker: str, key: str, value: Any) -> bool:
    """``meta_cache`` 의 한 필드만 바꾼다(문서 전체를 덮지 않는다). ``value`` 가 None 이면 필드를 지운다.

    배치 B 의 종목별 갱신 뒤에 붙는 파생값(시총 순위 등)용.

    값이 있으면 문서가 없어도 만든다. 예전에는 "배치 B 가 먼저 문서를 만든다"고 보고
    만들지 않았는데, 한국 **개별주** 풀은 ETF 상세가 없어 `stock_cache_meta` 문서 자체가
    생기지 않는다. 그래서 시총 순위가 한 건도 안 써졌고 화면의 시총 컬럼이 통째로 비었다.

    Returns:
        실제로 문서가 바뀌었는지. 호출부가 "몇 건 기록"을 정확히 셀 수 있게 돌려준다
        (반복 횟수로 세면 한 건도 안 써져도 성공처럼 보인다).
    """
    type_norm = (ticker_type or "").strip().lower()
    ticker_norm = str(ticker or "").strip().upper()
    if not type_norm or not ticker_norm or not key:
        raise ValueError("ticker_type, ticker, key must be provided")
    coll = _get_collection()
    if coll is None:
        raise RuntimeError("MongoDB 연결 실패 — stock_cache_meta 컬렉션에 쓸 수 없습니다.")
    field = f"meta_cache.{key}"
    if value is None:
        # 지우는 쪽은 문서를 만들지 않는다 — 빈 문서만 남는다.
        result = coll.update_one({"ticker_type": type_norm, "ticker": ticker_norm}, {"$unset": {field: ""}})
    else:
        result = coll.update_one(
            {"ticker_type": type_norm, "ticker": ticker_norm},
            {"$set": {field: value}, "$setOnInsert": {"ticker_type": type_norm, "ticker": ticker_norm}},
            upsert=True,
        )
    return bool(result.modified_count or getattr(result, "upserted_id", None))
