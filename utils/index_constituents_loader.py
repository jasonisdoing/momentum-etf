"""주요 인덱스 구성종목 저장소 (MongoDB).

파일(`data/*_tickers.json`)로 두다가 DB 로 옮겼다. 배치는 로컬에서만 자동 실행되는데
서버 컨테이너의 `data/` 는 읽기 전용 마운트라, 파일로 두면 서버 데이터가 갱신되지 않고
사람이 직접 올려야 했다. DB 는 서버·로컬이 함께 보므로 어디서 돌든 결과가 공유된다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

COLLECTION = "index_constituents"
SUPPORTED_INDICES = {"SP500", "NDX100", "ASX200", "KOSPI200", "KOSDAQ150"}

# 한국 지수는 공식 구성종목 API 가 없어 **추종 ETF 의 보유종목**을 명단으로 쓴다.
# 배치가 하루 한 번 여기 목록대로 적재하고, 화면(`/kor-market-stock`·`/kor-dividend`)은
# 저장된 명단만 읽는다. 다른 ETF 로 대체하지 않는다 — 조회가 깨지면 배치가 죽어야 한다.
#   market: 시세·시총을 붙일 때 넘길 네이버 시장 코드(구성종목이 그 시장에 상장돼 있다).
KOR_INDEX_SOURCES: dict[str, dict[str, Any]] = {
    "KOSPI200": {"etf_ticker": "069500", "etf_name": "KODEX 200", "min_count": 150, "market": "KOSPI"},
    "KOSDAQ150": {"etf_ticker": "229200", "etf_name": "KODEX 코스닥150", "min_count": 120, "market": "KOSDAQ"},
}


# 지수별 적재 배치 — 명단이 없을 때 "무엇을 돌려야 하는지" 를 정확히 알려주기 위한 것이다.
# 배치를 옮기면 여기도 함께 고쳐야 한다(안 그러면 화면이 없는 스크립트를 안내한다).
_REFRESH_SCRIPT_BY_INDEX: dict[str, str] = {
    "SP500": "scripts/update_us_market_stocks.py",
    "NDX100": "scripts/update_us_market_stocks.py",
    "ASX200": "scripts/update_aus_market_stocks.py",
    "KOSPI200": "scripts/update_kor_market_stocks.py",
    "KOSDAQ150": "scripts/update_kor_market_stocks.py",
}


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
        script = _REFRESH_SCRIPT_BY_INDEX.get(key, "해당 시장의 구성종목 배치")
        raise LookupError(f"{key} 구성종목이 DB 에 없습니다 (컬렉션 {COLLECTION}). {script} 를 실행해 저장하세요.")
    return list(doc.get("tickers") or [])


def load_index_meta(index: str) -> dict[str, Any]:
    """updated_at, source, count 등 메타 정보를 반환한다. 없으면 빈 dict."""
    doc = _load_document(_normalize_index(index))
    if not doc:
        return {}
    return {k: v for k, v in doc.items() if k not in ("tickers", "_id")}


def refresh_kor_index_from_etf(index: str) -> dict[str, Any]:
    """추종 ETF 의 보유종목을 읽어 그 지수의 구성종목으로 저장한다 (한국 지수 공용).

    Returns:
        {"index": 지수 키, "count": 저장 종목 수, "as_of_date": ETF 보유종목 기준일|None}

    Raises:
        RuntimeError: 보유종목을 못 가져왔거나 수가 비정상일 때. 배치는 여기서 죽어야 한다
            — 조용히 넘기면 명단이 낡은 채로 화면에 계속 쓰인다.
    """
    from datetime import date

    from services.etf_holdings_service import fetch_korean_etf_holdings_from_naver

    key = _normalize_index(index)
    source = KOR_INDEX_SOURCES.get(key)
    if source is None:
        raise ValueError(f"ETF 기반 적재를 지원하지 않는 지수입니다: {index}")

    payload = fetch_korean_etf_holdings_from_naver(source["etf_ticker"])
    holdings = (payload or {}).get("holdings") or []

    tickers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in holdings:
        ticker = str(item.get("ticker") or "").strip().upper()
        # 현금·선물 등 종목코드가 아닌 항목이 섞여 오므로 6자리 코드만 남긴다.
        if len(ticker) != 6 or ticker in seen:
            continue
        seen.add(ticker)
        tickers.append(
            {
                "ticker": ticker,
                "name": str(item.get("name") or ticker).strip(),
                # 지수 내 비중 — 규모·유동성의 대리 지표라 화면 필터에 쓴다.
                "weight": float(item["weight"]) if item.get("weight") is not None else None,
            }
        )

    if len(tickers) < int(source["min_count"]):
        raise RuntimeError(
            f"{source['etf_name']}({source['etf_ticker']}) 보유종목이 {len(tickers)}개뿐입니다 "
            f"(최소 {source['min_count']}개 기대). 네이버 응답이 바뀌었거나 조회에 실패했습니다."
        )

    as_of_date = (payload or {}).get("as_of_date")
    save_index_constituents(
        key,
        tickers,
        {
            "updated_at": date.today().isoformat(),
            "source": f"{source['etf_name']}({source['etf_ticker']}) ETF 보유종목",
            "as_of_date": as_of_date,
        },
    )
    return {"index": key, "count": len(tickers), "as_of_date": as_of_date}


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
