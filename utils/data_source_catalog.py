"""데이터 소스 카탈로그 — 어떤 데이터를 어디서 가져오는지 한곳에 정리한다.

화면(`/data_source`)이 읽는 단일 소스다. 새 외부 소스를 붙이거나 기존 소스를
교체하면 여기도 함께 고친다. 목록은 사람이 관리하는 정적 정의이고, 호주 ETF
구성종목 수집 현황만 실제 캐시에서 집계해 붙인다(발행사마다 성공/실패가 갈리므로).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from utils.logger import get_app_logger

logger = get_app_logger()

# 구성종목 캐시의 source 값 → 사람이 읽는 이름. 배치가 저장한 값과 1:1 로 맞춘다.
HOLDINGS_SOURCE_LABELS: dict[str, str] = {
    "betashares_csv": "BetaShares 공식 CSV",
    "vanguard_au_api": "Vanguard AU 공식 API",
    "yfinance_holdings": "yfinance (상위 10종목)",
    "naver_etf_component": "네이버 ETF 구성종목",
}

# 데이터 영역별 소스 정의.
# provider: 실제 데이터 제공처 / endpoint: 호출 대상 / usage: 이 시스템에서 쓰는 용도
DATA_SOURCES: list[dict[str, Any]] = [
    # ── 시세 (실시간) ────────────────────────────────────────────────
    {
        "category": "실시간 시세",
        "country": "kor",
        "provider": "네이버 증권",
        "endpoint": "stock.naver.com/api/polling/domestic/stock",
        "usage": "한국 개별주 실시간 가격·등락률",
        "code_ref": "utils/data_loader.py",
        "note": "ETF 는 iNAV API 를 먼저 쓰고, 없으면 이 API 로 보완한다.",
    },
    {
        "category": "실시간 시세",
        "country": "kor",
        "provider": "네이버 증권",
        "endpoint": "stock.naver.com/api/domestic/detail/{ticker}/ETFBase",
        "usage": "한국 ETF iNAV·괴리율",
        "code_ref": "services/etf_meta_service.py",
        "note": None,
    },
    {
        "category": "실시간 시세",
        "country": "us",
        "provider": "토스증권",
        "endpoint": "wts-info-api.tossinvest.com",
        "usage": "미국 주식·ETF 실시간 가격",
        "code_ref": "utils/data_loader.py",
        "note": "심볼→상품코드 매핑을 먼저 조회한 뒤 50개씩 묶어 시세를 받는다.",
    },
    {
        "category": "실시간 시세",
        "country": "au",
        "provider": "Market Index (QuoteAPI)",
        "endpoint": "quoteapi.com/api/v5/symbols",
        "usage": "호주 주식·ETF 실시간 가격",
        "code_ref": "utils/data_loader.py",
        "note": "티커는 ASX: 접두사를 벗기고 소문자 + .asx 형식으로 보낸다.",
    },
    {
        "category": "실시간 시세",
        "country": "global",
        "provider": "네이버 증권",
        "endpoint": "stock.naver.com/api/polling/worldstock/stock",
        "usage": "해외 상장 종목(미국 외) 가격 보완",
        "code_ref": "utils/data_loader.py",
        "note": "ETF 구성종목 중 유럽·일본 등 미국 외 종목에 쓴다.",
    },
    # ── 시세 (과거) ──────────────────────────────────────────────────
    {
        "category": "과거 시세",
        "country": "all",
        "provider": "yfinance (Yahoo Finance)",
        "endpoint": "yfinance 라이브러리",
        "usage": "일별 OHLCV 캐시 — 순위·백테스트·비교 화면의 기반 데이터",
        "code_ref": "utils/data_loader.py",
        "note": "호주는 ASX:NAB → NAB.AX, 한국은 069500.KS 형태로 변환해 조회한다.",
    },
    {
        "category": "과거 시세",
        "country": "kor",
        "provider": "네이버 금융",
        "endpoint": "fchart.stock.naver.com/sise.nhn",
        "usage": "한국 종목 차트 데이터 보완",
        "code_ref": "config.py",
        "note": None,
    },
    {
        "category": "과거 시세",
        "country": "global",
        "provider": "Yahoo Finance",
        "endpoint": "KRW=X / AUDKRW=X 등 환율 심볼",
        "usage": "원화 환산용 환율 시계열",
        "code_ref": "utils/data_loader.py",
        "note": None,
    },
    # ── 종목 마스터 ──────────────────────────────────────────────────
    {
        "category": "종목 마스터",
        "country": "kor",
        "provider": "한국투자증권(KIS)",
        "endpoint": "new.real.download.dws.co.kr (kospi/kosdaq_code.mst)",
        "usage": "국내 상장 종목·ETF 마스터 파일",
        "code_ref": "utils/kis_market.py",
        "note": None,
    },
    {
        "category": "종목 마스터",
        "country": "kor",
        "provider": "네이버 금융",
        "endpoint": "finance.naver.com/api/sise/etfItemList.nhn",
        "usage": "한국 ETF 목록",
        "code_ref": "config.py",
        "note": None,
    },
    {
        "category": "종목 마스터",
        "country": "kor",
        "provider": "pykrx",
        "endpoint": "pykrx 라이브러리 (KRX)",
        "usage": "한국 종목 기초 정보·시가총액",
        "code_ref": "services/etf_holdings_service.py",
        "note": None,
    },
    {
        "category": "종목 마스터",
        "country": "kor",
        "provider": "네이버 증권",
        "endpoint": "m.stock.naver.com/api/stocks/marketValue/{market}",
        "usage": "코스피·코스닥 시가총액 순위",
        "code_ref": "utils/kor_stock_market_service.py",
        "note": None,
    },
    # ── 업종 분류 ────────────────────────────────────────────────────
    # 섹터는 쓰지 않는다. 11종뿐이라 5~10종목 포트폴리오에서는 해상도가 낮고,
    # 집중도 관리는 업종 상한(max_per_industry)이 담당한다.
    {
        "category": "업종 분류",
        "country": "kor",
        "provider": "네이버 증권",
        "endpoint": "m.stock.naver.com/api/stock/{ticker}/integration + /api/stocks/industry/{code}",
        "usage": "한국 종목 업종 — 업종 상한 그룹핑·화면 표시",
        "code_ref": "services/naver_industry_service.py",
        "note": (
            "yfinance 는 국내 종목 911개 중 312개(34%)에 분류가 없어 그만큼 업종 상한에서 빠졌다. "
            "네이버는 한국어 원본이라 번역이 필요 없고 용어도 사업 실체에 가깝다(가비아=IT서비스)."
        ),
    },
    {
        "category": "업종 분류",
        "country": "us",
        "provider": "yfinance (Yahoo Finance)",
        "endpoint": "지수 구성종목 수집 시 함께 저장",
        "usage": "미국 종목 업종 — 업종 상한 그룹핑·화면 표시",
        "code_ref": "scripts/update_us_market_stocks.py",
        "note": "영문 원본을 그대로 쓴다(번역하지 않는다).",
    },
    {
        "category": "업종 분류",
        "country": "au",
        "provider": "yfinance (Yahoo Finance)",
        "endpoint": "yfinance info",
        "usage": "호주 종목 업종",
        "code_ref": "utils/stock_meta_updater.py",
        "note": "영문 원본을 그대로 쓴다(번역하지 않는다).",
    },
    # ── ETF 상세 (구성종목·배당·보수) ────────────────────────────────
    {
        "category": "ETF 상세",
        "country": "kor",
        "provider": "네이버 증권",
        "endpoint": "stock.naver.com/api/domestic/detail/{ticker}/ETFComponent",
        "usage": "한국 ETF 구성종목 전체",
        "code_ref": "services/etf_holdings_service.py",
        "note": None,
    },
    {
        "category": "ETF 상세",
        "country": "kor",
        "provider": "네이버 증권",
        "endpoint": "stock.naver.com/api/domestic/detail/{ticker}/ETFDividend",
        "usage": "한국 ETF 배당수익률·배당 이력",
        "code_ref": "services/etf_meta_service.py",
        "note": None,
    },
    {
        "category": "ETF 상세",
        "country": "us",
        "provider": "yfinance (Yahoo Finance)",
        "endpoint": "yfinance funds_data / info",
        "usage": "미국 ETF 구성종목(상위 10)·보수·배당·순자산",
        "code_ref": "utils/stock_meta_updater.py",
        "note": "미국은 보수(netExpenseRatio)가 정상 제공된다.",
    },
    # 호주 ETF 구성종목은 발행사마다 경로가 갈리므로 아래에서 실제 캐시를 보고 동적으로 만든다.
    # ── 지표·기타 ────────────────────────────────────────────────────
    {
        "category": "시장 지표",
        "country": "us",
        "provider": "CNN Business",
        "endpoint": "production.dataviz.cnn.io/index/fearandgreed/graphdata",
        "usage": "공포·탐욕 지수",
        "code_ref": "services/fear_greed_service.py",
        "note": None,
    },
    {
        "category": "시장 지표",
        "country": "global",
        "provider": "Hyperliquid",
        "endpoint": "api.hyperliquid.xyz/info",
        "usage": "24시간 선물 시세 (나스닥·개별주 야간 참고가)",
        "code_ref": "utils/live_24h_service.py",
        "note": None,
    },
]

# 호주 ETF 구성종목 수집 순서 — 실제 코드(_refresh_overseas_etf_meta_cache)와 일치시킨다.
AU_HOLDINGS_FALLBACK_ORDER: list[str] = ["betashares_csv", "vanguard_au_api", "yfinance_holdings"]

# ETF 이름 → 발행사(운용사). 호주는 통합 소스가 없어 발행사별로 수집 경로가 갈리므로
# 발행사를 기준으로 정리한다. 이름 앞부분에 발행사명이 들어가는 업계 관행을 이용하되,
# 매칭 키워드를 명시해 둔다(추정하지 않는다 — 목록에 없으면 "기타"로 남긴다).
AU_ISSUER_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("Vanguard", ("vanguard",)),
    ("BetaShares", ("betashares", "beta shares")),
    ("iShares (BlackRock)", ("ishares",)),
    ("Global X", ("global x",)),
    ("VanEck", ("vaneck", "van eck")),
    ("SPDR (State Street)", ("spdr", "state street")),
    ("Fidelity", ("fidelity",)),
]
AU_ISSUER_UNKNOWN = "기타"


def resolve_au_issuer(name: str) -> str:
    """호주 ETF 이름에서 발행사를 판별한다. 목록에 없으면 '기타'."""
    lowered = str(name or "").strip().lower()
    if not lowered:
        return AU_ISSUER_UNKNOWN
    for issuer, keywords in AU_ISSUER_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return issuer
    return AU_ISSUER_UNKNOWN


def _load_au_etf_rows() -> list[dict[str, Any]]:
    """호주 종목풀의 ETF 별 구성종목·보수 수집 현황을 읽는다."""
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결 실패 — 데이터 소스 현황을 읽을 수 없습니다.")

    rows: list[dict[str, Any]] = []
    cursor = db.stock_cache_meta.find(
        {"ticker_type": "aus"},
        {"_id": 0, "ticker": 1, "name": 1, "holdings_cache": 1, "meta_cache": 1},
    )
    for doc in cursor:
        holdings_cache = doc.get("holdings_cache") or {}
        meta_cache = doc.get("meta_cache") or {}
        source = holdings_cache.get("source")
        rows.append(
            {
                "ticker": str(doc.get("ticker") or ""),
                "name": str(doc.get("name") or ""),
                "holdings_source": source,
                "holdings_source_label": HOLDINGS_SOURCE_LABELS.get(str(source or ""), None),
                "holdings_count": len(holdings_cache.get("items") or []),
                "reference_date": holdings_cache.get("reference_date"),
                "expense_ratio": meta_cache.get("expense_ratio"),
                "dividend_yield_ttm": meta_cache.get("dividend_yield_ttm"),
            }
        )
    rows.sort(key=lambda row: row["ticker"])
    return rows


# 발행사별 행에 붙일 호출 대상·코드 위치 (수집 소스 기준).
_SOURCE_ENDPOINTS: dict[str, tuple[str, str]] = {
    "betashares_csv": (
        "betashares.com.au/files/csv/{ticker}_Portfolio_Holdings.csv",
        "utils/stock_meta_updater.py",
    ),
    "vanguard_au_api": (
        "vanguard.com.au/adviser/api/data/products/holdings/{portId}",
        "services/vanguard_au_service.py",
    ),
    "yfinance_holdings": ("yfinance funds_data", "utils/stock_meta_updater.py"),
}
_SOURCE_NOTES: dict[str, str] = {
    "betashares_csv": "CSV 에 Currency·Country·Asset Class 열이 있어 구성종목의 상장 국가를 정확히 안다.",
    "vanguard_au_api": "ASX 티커가 아닌 내부 portId 로 조회한다. 운용보수(MER)도 이 API 로 함께 받는다.",
    "yfinance_holdings": "공식 소스가 없어 폴백. 상위 10종목까지만 나오고 운용보수는 제공되지 않는다.",
}


def _build_au_issuer_sources(au_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """호주 ETF 구성종목을 발행사별로 묶어 소스 카탈로그 행으로 만든다."""
    by_issuer: dict[str, list[dict[str, Any]]] = {}
    for row in au_rows:
        by_issuer.setdefault(resolve_au_issuer(row["name"]), []).append(row)

    entries: list[dict[str, Any]] = []
    for issuer, rows in by_issuer.items():
        # 발행사의 대표 수집 경로 = 가장 많이 쓰인 소스. 실패분은 따로 센다.
        source_counter = Counter(str(row["holdings_source"] or "") for row in rows)
        missing_tickers = [row["ticker"] for row in rows if not row["holdings_source"]]
        primary_source = next(
            (source for source, _ in source_counter.most_common() if source),
            "",
        )
        endpoint, code_ref = _SOURCE_ENDPOINTS.get(primary_source, ("-", "utils/stock_meta_updater.py"))
        source_label = HOLDINGS_SOURCE_LABELS.get(primary_source, "수집 경로 없음")

        entries.append(
            {
                "category": "ETF 상세",
                "country": "au",
                "provider": issuer,
                "endpoint": endpoint,
                "usage": f"{source_label} — ETF {len(rows)}종",
                "code_ref": code_ref,
                "note": _SOURCE_NOTES.get(primary_source),
                "etf_count": len(rows),
                "source": primary_source or None,
                "source_label": source_label,
                "missing_count": len(missing_tickers),
                "missing_tickers": missing_tickers,
            }
        )

    # 공식 소스를 쓰는 발행사를 위에, 그 다음 ETF 수 많은 순.
    source_rank = {source: index for index, source in enumerate(AU_HOLDINGS_FALLBACK_ORDER)}
    entries.sort(key=lambda entry: (source_rank.get(entry["source"] or "", 99), -entry["etf_count"]))
    return entries


def build_data_source_payload() -> dict[str, Any]:
    """화면이 쓰는 데이터 소스 카탈로그. 호주 ETF 상세는 실제 캐시를 보고 발행사별로 만든다."""
    au_rows = _load_au_etf_rows()
    au_issuer_sources = _build_au_issuer_sources(au_rows)

    # "ETF 상세" 정적 항목 뒤에 호주 발행사별 행을 이어 붙인다.
    sources: list[dict[str, Any]] = list(DATA_SOURCES)
    last_detail_index = max(
        (index for index, row in enumerate(sources) if row["category"] == "ETF 상세"),
        default=None,
    )
    if last_detail_index is None:
        sources.extend(au_issuer_sources)
    else:
        sources[last_detail_index + 1 : last_detail_index + 1] = au_issuer_sources

    return {
        "sources": sources,
        "au_holdings_order": AU_HOLDINGS_FALLBACK_ORDER,
        "au_total": len(au_rows),
        "au_expense_ratio_count": sum(1 for row in au_rows if row["expense_ratio"] is not None),
    }
