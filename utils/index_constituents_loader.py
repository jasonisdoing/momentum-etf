"""주요 인덱스 구성종목 저장소 (MongoDB).

파일(`data/*_tickers.json`)로 두다가 DB 로 옮겼다. 배치는 로컬에서만 자동 실행되는데
서버 컨테이너의 `data/` 는 읽기 전용 마운트라, 파일로 두면 서버 데이터가 갱신되지 않고
사람이 직접 올려야 했다. DB 는 서버·로컬이 함께 보므로 어디서 돌든 결과가 공유된다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

COLLECTION = "index_constituents"
SUPPORTED_INDICES = {"SP500", "NDX100", "ASX200", "KOSPI200"}


def _normalize_index(index: str) -> str:
    key = str(index or "").strip().upper()
    if key not in SUPPORTED_INDICES:
        raise ValueError(f"지원하지 않는 인덱스입니다: {index} (지원: {', '.join(sorted(SUPPORTED_INDICES))})")
    return key


def _load_document(index: str) -> dict[str, Any] | None:
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패해 인덱스 구성종목을 읽을 수 없습니다.")
    return db[COLLECTION].find_one({"_id": _normalize_index(index)})


def load_index_constituents(index: str) -> list[dict[str, Any]]:
    """지정한 인덱스의 구성종목 목록을 반환한다. ticker, name, sector, industry 포함."""
    key = _normalize_index(index)
    doc = _load_document(key)
    if not doc:
        raise LookupError(
            f"{key} 구성종목이 DB 에 없습니다 (컬렉션 {COLLECTION}).\n"
            "미국은 scripts/update_us_market_stocks.py, 호주는 scripts/update_aus_market_stocks.py, "
            "한국은 scripts/update_kor_dividend_stocks.py 를 실행해 저장하세요."
        )
    return list(doc.get("tickers") or [])


def load_index_meta(index: str) -> dict[str, Any]:
    """updated_at, source, count 등 메타 정보를 반환한다. 없으면 빈 dict."""
    doc = _load_document(_normalize_index(index))
    if not doc:
        return {}
    return {k: v for k, v in doc.items() if k not in ("tickers", "_id")}


def save_index_constituents(index: str, tickers: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    """구성종목과 메타를 통째로 교체 저장한다 (배치 전용)."""
    from utils.db_manager import get_db_connection

    key = _normalize_index(index)
    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패해 인덱스 구성종목을 저장할 수 없습니다.")
    db[COLLECTION].replace_one(
        {"_id": key},
        {"_id": key, **meta, "count": len(tickers), "saved_at": datetime.now().isoformat(), "tickers": tickers},
        upsert=True,
    )
