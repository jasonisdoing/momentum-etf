from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests
import yfinance as yf
from pykrx import stock
from pykrx.website.comm import webio
from pykrx.website.krx.etx.core import PDF
from pykrx.website.krx.etx.wrap import get_etx_isin

from utils.db_manager import get_db_connection
from utils.data_loader import get_trading_days
from utils.logger import get_app_logger

logger = get_app_logger()

_COLLECTION_NAME = "etf_holdings_cache"
_INDEX_ENSURED = False

LOGIN_PAGE_URL = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001.cmd"
LOGIN_IFRAME_URL = "https://data.krx.co.kr/contents/MDC/COMS/client/view/login.jsp?site=mdc"
LOGIN_POST_URL = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001D1.cmd"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
_YAHOO_IDENTITY_CACHE: dict[str, dict[str, str | None]] = {}


def _get_collection():
    global _INDEX_ENSURED
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결 실패 — etf_holdings_cache 컬렉션을 사용할 수 없습니다.")

    collection = db[_COLLECTION_NAME]
    if not _INDEX_ENSURED:
        collection.create_index(
            [("ticker", 1), ("as_of_date", -1), ("source", 1)],
            unique=True,
            name="ticker_as_of_date_source_unique",
            background=True,
        )
        collection.create_index(
            [("ticker", 1), ("as_of_date", -1)],
            name="ticker_as_of_date_desc",
            background=True,
        )
        _INDEX_ENSURED = True
    return collection


def _normalize_ticker(ticker: str) -> str:
    return str(ticker or "").strip().upper().replace("ASX:", "")


def _normalize_date(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.replace(",", "").strip()
        if not normalized or normalized == "-":
            return None
        value = normalized
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return None
    return float(parsed)


def _to_int(value: Any) -> int | None:
    parsed = _to_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _install_requests_session(session: requests.Session) -> None:
    # pykrx는 requests.get/post를 직접 호출하므로 동일 세션으로 교체한다.
    webio.requests.get = session.get
    webio.requests.post = session.post


def _is_korean_stock_code(code: str) -> bool:
    return code.isdigit() and len(code) == 6


def _should_resolve_yahoo_identity(raw_code: str) -> bool:
    normalized = str(raw_code or "").strip().upper()
    return normalized.startswith("US") and len(normalized) >= 10


def _resolve_yahoo_identity(raw_code: str, raw_name: str) -> dict[str, str | None]:
    cached = _YAHOO_IDENTITY_CACHE.get(raw_code)
    if cached is not None:
        return cached

    search = yf.Search(
        raw_code,
        max_results=5,
        news_count=0,
        lists_count=0,
        recommended=0,
        raise_errors=True,
    )
    quotes = getattr(search, "quotes", []) or []
    equity_quote = next(
        (
            quote
            for quote in quotes
            if str(quote.get("quoteType") or "").strip().upper() == "EQUITY"
            and str(quote.get("symbol") or "").strip()
        ),
        None,
    )
    if equity_quote is None:
        raise RuntimeError(f"Yahoo 심볼 검색 실패: {raw_code}")

    normalized = {
        "ticker": str(equity_quote.get("symbol") or "").strip().upper() or raw_code,
        "name": (
            str(equity_quote.get("longname") or "").strip()
            or str(equity_quote.get("shortname") or "").strip()
            or raw_name
        ),
        "yahoo_symbol": str(equity_quote.get("symbol") or "").strip().upper() or None,
    }
    _YAHOO_IDENTITY_CACHE[raw_code] = normalized
    return normalized


def resolve_krx_login_credentials() -> tuple[str, str]:
    login_id = str(os.environ.get("KRX_LOGIN_ID") or "").strip()
    login_password = str(os.environ.get("KRX_LOGIN_PASSWORD") or "").strip()
    if not login_id or not login_password:
        raise RuntimeError(
            "KRX 로그인 정보가 필요합니다. KRX_LOGIN_ID와 KRX_LOGIN_PASSWORD 환경변수를 설정하세요."
        )
    return login_id, login_password


def login_krx_session() -> requests.Session:
    login_id, login_password = resolve_krx_login_credentials()
    session = requests.Session()
    _install_requests_session(session)

    session.get(LOGIN_PAGE_URL, headers={"User-Agent": DEFAULT_USER_AGENT}, timeout=15)
    session.get(
        LOGIN_IFRAME_URL,
        headers={"User-Agent": DEFAULT_USER_AGENT, "Referer": LOGIN_PAGE_URL},
        timeout=15,
    )

    payload = {
        "mbrNm": "",
        "telNo": "",
        "di": "",
        "certType": "",
        "mbrId": login_id,
        "pw": login_password,
    }
    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Referer": LOGIN_PAGE_URL,
        "X-Requested-With": "XMLHttpRequest",
    }

    response = session.post(LOGIN_POST_URL, data=payload, headers=headers, timeout=15)
    data = response.json()
    error_code = str(data.get("_error_code") or "")

    if error_code == "CD011":
        payload["skipDup"] = "Y"
        response = session.post(LOGIN_POST_URL, data=payload, headers=headers, timeout=15)
        data = response.json()
        error_code = str(data.get("_error_code") or "")

    if error_code != "CD001":
        error_message = str(data.get("_error_message") or "알 수 없는 로그인 오류").strip()
        raise RuntimeError(f"KRX 로그인 실패: {error_code} {error_message}".strip())

    return session


def fetch_korean_etf_holdings_from_krx(ticker: str, as_of_date: str) -> list[dict[str, Any]]:
    normalized_ticker = _normalize_ticker(ticker)
    normalized_date = _normalize_date(as_of_date)
    if not normalized_ticker:
        raise ValueError("ETF 티커가 필요합니다.")
    if not normalized_date:
        raise ValueError("구성종목 조회 기준일(as_of_date)이 필요합니다.")

    isin = get_etx_isin(normalized_ticker)
    df = PDF().fetch(normalized_date, isin)
    if df is None or df.empty:
        raise RuntimeError(f"{normalized_ticker} ETF 구성종목 조회 결과가 비어 있습니다. ({normalized_date})")

    working_df = df.copy()
    working_df["raw_code"] = working_df["COMPST_ISU_CD"].map(lambda value: str(value or "").strip().upper())
    working_df["weight_value"] = working_df["COMPST_RTO"].map(_to_float)
    working_df = working_df.sort_values(
        by="weight_value",
        ascending=False,
        na_position="last",
    )

    records: list[dict[str, Any]] = []
    for _, row in working_df.iterrows():
        raw_code = str(row.get("raw_code") or "").strip().upper()
        raw_name = str(row.get("COMPST_ISU_NM") or "").strip()
        contracts = _to_float(row.get("COMPST_ISU_CU1_SHRS"))
        amount = _to_float(row.get("VALU_AMT"))
        market_cap = _to_float(row.get("COMPST_AMT"))
        weight = _to_float(row.get("COMPST_RTO"))
        display_ticker = raw_code
        display_name = raw_name
        yahoo_symbol: str | None = None
        if _should_resolve_yahoo_identity(raw_code):
            try:
                yahoo_identity = _resolve_yahoo_identity(raw_code, raw_name)
            except Exception as exc:
                logger.warning("해외 구성종목 Yahoo 매핑 실패: code=%s name=%s error=%s", raw_code, raw_name, exc)
            else:
                display_ticker = str(yahoo_identity.get("ticker") or raw_code)
                display_name = str(yahoo_identity.get("name") or raw_name)
                yahoo_symbol = str(yahoo_identity.get("yahoo_symbol") or "").strip().upper() or None
        records.append(
            {
                "ticker": display_ticker,
                "name": display_name,
                "raw_code": raw_code,
                "raw_name": raw_name,
                "yahoo_symbol": yahoo_symbol,
                "contracts": int(contracts) if contracts is not None else None,
                "amount": int(amount) if amount is not None else None,
                "market_cap": int(market_cap) if market_cap is not None else None,
                "weight": round(weight, 2) if weight is not None else None,
            }
        )

    return records


def upsert_korean_etf_holdings_cache(
    *,
    ticker: str,
    etf_name: str,
    as_of_date: str,
    holdings: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized_ticker = _normalize_ticker(ticker)
    normalized_date = _normalize_date(as_of_date)
    if not normalized_ticker:
        raise ValueError("ETF 티커가 필요합니다.")
    if not normalized_date:
        raise ValueError("구성종목 저장 기준일(as_of_date)이 필요합니다.")

    collection = _get_collection()
    document = {
        "ticker": normalized_ticker,
        "country_code": "kor",
        "source": "krx_pdf",
        "as_of_date": normalized_date,
        "etf_name": str(etf_name or "").strip(),
        "holdings_count": len(holdings),
        "holdings": holdings,
        "fetched_at": datetime.now(timezone.utc),
    }
    collection.update_one(
        {"ticker": normalized_ticker, "as_of_date": normalized_date, "source": "krx_pdf"},
        {"$set": document},
        upsert=True,
    )
    return document


def load_korean_etf_holdings_cache(ticker: str) -> dict[str, Any] | None:
    normalized_ticker = _normalize_ticker(ticker)
    if not normalized_ticker:
        raise ValueError("ETF 티커가 필요합니다.")

    collection = _get_collection()
    document = collection.find_one(
        {"ticker": normalized_ticker, "country_code": "kor", "source": "krx_pdf"},
        sort=[("as_of_date", -1)],
        projection={"_id": 0},
    )
    if document is None:
        return None
    return document


def fetch_korean_stock_price_snapshot(tickers: list[str], as_of_date: str) -> dict[str, dict[str, Any]]:
    normalized_date = _normalize_date(as_of_date)
    if not normalized_date:
        raise ValueError("현재가 조회 기준일(as_of_date)이 필요합니다.")

    normalized_tickers = [_normalize_ticker(ticker) for ticker in tickers if _normalize_ticker(ticker)]
    if not normalized_tickers:
        return {}

    target_date = pd.Timestamp(normalized_date)
    trading_days = get_trading_days(
        (target_date - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
        target_date.strftime("%Y-%m-%d"),
        "kor",
    )
    normalized_trading_days = [pd.Timestamp(day).strftime("%Y%m%d") for day in trading_days]
    if normalized_date not in normalized_trading_days:
        raise RuntimeError(f"한국 거래일이 아닌 날짜입니다: {normalized_date}")

    target_index = normalized_trading_days.index(normalized_date)
    if target_index == 0:
        raise RuntimeError(f"전일 거래일을 계산할 수 없습니다: {normalized_date}")
    previous_date = normalized_trading_days[target_index - 1]

    result: dict[str, dict[str, Any]] = {}
    for ticker in normalized_tickers:
        if not ticker.isdigit() or len(ticker) != 6:
            continue
        df = stock.get_market_ohlcv_by_date(previous_date, normalized_date, ticker)
        if df is None or df.empty:
            continue
        working_df = df.copy().sort_index()
        if len(working_df) < 2:
            continue

        previous_close = _to_int(working_df.iloc[-2].get("종가"))
        current_price = _to_int(working_df.iloc[-1].get("종가"))
        if previous_close is None or current_price is None or previous_close == 0:
            continue

        change_pct = round(((current_price / previous_close) - 1.0) * 100.0, 2)
        result[ticker] = {
            "current_price": current_price,
            "previous_close": previous_close,
            "change_pct": change_pct,
        }

    return result


def sync_korean_etf_holdings_cache(*, ticker: str, etf_name: str, as_of_date: str) -> dict[str, Any]:
    holdings = fetch_korean_etf_holdings_from_krx(ticker, as_of_date)
    document = upsert_korean_etf_holdings_cache(
        ticker=ticker,
        etf_name=etf_name,
        as_of_date=as_of_date,
        holdings=holdings,
    )
    logger.info(
        "한국 ETF 구성종목 캐시 저장 완료: ticker=%s as_of_date=%s count=%s",
        document["ticker"],
        document["as_of_date"],
        document["holdings_count"],
    )
    return document
