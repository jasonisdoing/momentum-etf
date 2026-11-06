"""종목 메타 캐시(stock_cache_meta) 컬렉션을 읽고 쓰는 유틸리티."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from utils.db_manager import get_db_connection
from utils.logger import get_app_logger
from utils.market_cap_rank import META_FIELD as MARKET_CAP_RANK_FIELD
from utils.normalization import to_timestamp_iso

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
    if holdings_cache is not None:
        payload["holdings_cache"] = {
            **holdings_cache,
            "updated_at": to_timestamp_iso(
                holdings_cache.get("updated_at"), naive_timezone=datetime.now().astimezone().tzinfo
            ),
        }

    # 집계 파이프라인 업데이트 — 값은 $literal 로 감싸 `$` 로 시작하는 문자열이 식으로 읽히지 않게 한다.
    stage: dict[str, Any] = {key: {"$literal": value} for key, value in payload.items()}
    stage["created_at"] = {"$ifNull": ["$created_at", now]}
    if meta_cache is not None:
        # 메타 갱신은 meta_cache 를 통째로 바꾼다. 그 뒤 따로 적히는 시총 순위는 그 단계가 실패해도
        # 지워지지 않게 기존 값을 이어 붙인다(없으면 키 자체가 생기지 않는다).
        stage["meta_cache"] = {
            "$mergeObjects": [
                {MARKET_CAP_RANK_FIELD: f"$meta_cache.{MARKET_CAP_RANK_FIELD}"},
                {"$literal": meta_cache},
            ]
        }
    pipeline: list[dict[str, Any]] = [{"$set": stage}]
    if holdings_cache is not None:
        pipeline.append({"$unset": ["portfolio_change_cache", "portfolio_change_cache_updated_at"]})

    coll.update_one(
        {"ticker_type": type_norm, "ticker": ticker_norm},
        pipeline,
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
