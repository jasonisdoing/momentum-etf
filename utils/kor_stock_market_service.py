"""한국 개별주 시가총액 리스트 — 네이버 금융 API 기반."""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd
import requests

from config import CACHE_TTL_COMPUTE, NAVER_FINANCE_HEADERS
from services.price_service import get_realtime_snapshot
from utils.industry_map import industry_map_for_country
from utils.market_service import load_ticker_pool_map, load_ticker_pool_type_map
from utils.naver_chart import fetch_naver_daily_ohlc
from utils.portfolio_io import load_all_holding_tickers

logger = logging.getLogger(__name__)

_NAVER_STOCK_LIST_URL = "https://m.stock.naver.com/api/stocks/marketValue"


def _parse_number(value: str | None) -> int | None:
    """쉼표가 포함된 숫자 문자열을 int로 변환한다."""
    if not value:
        return None
    try:
        return int(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _parse_float(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _fetch_market_value_page(market: str, page: int, page_size: int) -> dict[str, Any]:
    url = f"{_NAVER_STOCK_LIST_URL}/{market}?page={page}&pageSize={page_size}"

    try:
        resp = requests.get(url, headers=NAVER_FINANCE_HEADERS, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.error("네이버 주식 리스트 조회 실패 (market=%s, page=%s): %s", market, page, exc)
        raise RuntimeError(f"네이버 주식 리스트 조회에 실패했습니다: {exc}") from exc


def is_individual_stock_item(item: dict[str, Any]) -> bool:
    """네이버 주식 리스트 항목이 **개별주**인지 — ETF·ETN 은 제외.

    시장 화면(`/kor-market-stock`)과 시총 순위(`market_cap_rank`)가 같은 분모를 쓰도록
    여기서만 정의한다. 예전에는 화면에만 이 규칙이 있어서, 순위표에는 ETF 가 섞인 채로
    번호가 매겨져 두 화면의 시총 순위가 달랐다.

    `stockEndType` 이 1차 기준이고, 이름의 ETF/ETN 키워드는 유형이 잘못 온 항목까지
    걸러 내는 보조 조건이다.
    """
    if str(item.get("stockEndType") or "").strip().lower() != "stock":
        return False
    name = str(item.get("stockName") or "").upper()
    return not any(keyword in name for keyword in ("ETF", "ETN"))


def _build_row(item: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """네이버 시총 항목 → 화면 행. 시세·시총이 없으면 호출부가 걸러야 한다."""
    ticker = item.get("itemCode", "")
    change_ratio = _parse_float(item.get("fluctuationsRatio"))
    compare_code = (item.get("compareToPreviousPrice") or {}).get("code", "")
    # code "5"=하락 → 등락률을 음수로
    if compare_code == "5" and change_ratio is not None and change_ratio > 0:
        change_ratio = -change_ratio

    return {
        "rank": 0,
        "ticker": ticker,
        "name": item.get("stockName", ""),
        "industry": context["industry_by"].get(ticker, ""),
        "ticker_pools": ", ".join(context["ticker_pool_map"].get(ticker, [])),
        "ticker_pool_types": context["ticker_pool_type_map"].get(ticker, []),
        "is_held": ticker in context["held_tickers"],
        "current_price": _parse_number(item.get("closePrice")),
        "change_pct": change_ratio,
        "volume": _parse_number(item.get("accumulatedTradingVolume")),
        # 네이버 marketValue는 이미 억 단위다.
        "market_cap": _parse_number(item.get("marketValue")),
    }


def _lookup_context() -> dict[str, Any]:
    """행에 붙일 공통 정보(종목풀·보유·업종)를 한 번만 읽는다."""
    return {
        "ticker_pool_map": load_ticker_pool_map(),
        # 화면이 "이미 이 풀에 있는 종목"을 걸러내려면 이름이 아니라 풀 id 가 필요하다.
        "ticker_pool_type_map": load_ticker_pool_type_map(),
        "held_tickers": load_all_holding_tickers(),
        # 업종 — 신고가·순위 화면과 같은 소스를 쓰되, 이 화면은 종목풀이 아니라 시장 전체
        # 상위 종목을 받아 오므로 **한국 풀 전체**를 합쳐 읽는다. 풀 하나를 박아 두면 그
        # 풀을 지울 때 화면이 깨지고, 다른 풀에만 있는 종목은 업종이 비어 버린다.
        # 분류가 없는 종목은 빈 문자열로 두고 임의 값으로 묶지 않는다.
        "industry_by": industry_map_for_country("kor"),
    }


def _load_kospi200(min_market_cap_eok: int) -> dict[str, Any]:
    """KOSPI200 구성종목 전체. 시총 순위와 달리 **상위 N 을 자르지 않는다**.

    구성종목 명단은 배치가 KODEX 200 보유종목으로 적재해 둔 것(`index_constituents`)이고,
    시세·시총은 코스피 전체 페이지에서 찾아 붙인다. 시총 상위 200 페이지만 보면 지수에
    들어 있는 중형주가 빠지기 때문에 명단을 다 채울 때까지 페이지를 넘긴다.
    """
    from utils.kor_dividend_service import load_universe

    constituents = load_universe()
    wanted = {str(item["ticker"]).strip().upper() for item in constituents}
    context = _lookup_context()

    page_size = 100
    payload = _fetch_market_value_page("KOSPI", page=1, page_size=page_size)
    total_count = int(payload.get("totalCount") or 0)
    total_pages = max(1, math.ceil(total_count / page_size)) if total_count > 0 else 1

    rows: list[dict[str, Any]] = []
    found: set[str] = set()
    for page in range(1, total_pages + 1):
        for item in payload.get("stocks") or []:
            ticker = str(item.get("itemCode") or "").strip().upper()
            if ticker not in wanted or ticker in found:
                continue
            row = _build_row(item, context)
            if row["market_cap"] is None or row["market_cap"] < min_market_cap_eok:
                found.add(ticker)  # 조건 미달도 '찾음' — 다음 페이지에서 또 보지 않는다
                continue
            found.add(ticker)
            rows.append(row)
        if len(found) >= len(wanted) or page >= total_pages:
            break
        payload = _fetch_market_value_page("KOSPI", page=page + 1, page_size=page_size)

    _apply_kor_realtime_overlay(rows)
    _apply_kor_history_metrics(rows)
    rows.sort(key=lambda row: -(row["market_cap"] or 0))
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    return {"market": "KOSPI200", "total_count": len(wanted), "count": len(rows), "rows": rows}


def load_kor_stock_market(
    market: str,
    limit: int,
    min_market_cap_jo: int,
) -> dict[str, Any]:
    """네이버 API에서 시가총액 상위 종목 리스트를 가져온다.

    Args:
        market: "KOSPI" · "KOSDAQ" · "KOSPI200"
        limit: 가져올 종목 수 (최대 200). KOSPI200 은 구성종목 전체라 무시한다.
        min_market_cap_jo: 최소 시가총액(조)
    """
    if market not in ("KOSPI", "KOSDAQ", "KOSPI200"):
        raise ValueError(f"지원하지 않는 마켓입니다: {market}")
    if limit <= 0:
        raise ValueError(f"가져올 종목 수는 1 이상이어야 합니다: {limit}")
    if min_market_cap_jo < 0:
        raise ValueError(f"최소 시가총액은 음수일 수 없습니다: {min_market_cap_jo}")

    min_market_cap_eok = min_market_cap_jo * 10000
    if market == "KOSPI200":
        return _load_kospi200(min_market_cap_eok)

    page_size = 100
    first_payload = _fetch_market_value_page(market, page=1, page_size=page_size)
    total_count = int(first_payload.get("totalCount") or 0)
    total_pages = max(1, math.ceil(total_count / page_size)) if total_count > 0 else 1

    context = _lookup_context()

    target_count = min(limit, 200)
    rows: list[dict[str, Any]] = []
    payload = first_payload
    for page in range(1, total_pages + 1):
        stocks = payload.get("stocks") or []
        for item in stocks:
            # 개별주만 — ETF·ETN 제외(시총 순위와 같은 기준).
            if not is_individual_stock_item(item):
                continue
            row = _build_row(item, context)
            if row["market_cap"] is None or row["market_cap"] < min_market_cap_eok:
                continue
            rows.append(row)
            if len(rows) >= target_count:
                break
        if len(rows) >= target_count:
            break
        if page >= total_pages:
            break
        payload = _fetch_market_value_page(market, page=page + 1, page_size=page_size)

    rows = rows[:target_count]
    _apply_kor_realtime_overlay(rows)
    _apply_kor_history_metrics(rows)
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    return {
        "market": market,
        "total_count": total_count,
        "count": len(rows),
        "rows": rows,
    }


def _apply_kor_realtime_overlay(rows: list[dict[str, Any]]) -> None:
    """한국 개별주 리스트에 네이버 실시간/장전 가격을 반영한다."""
    tickers = [str(row.get("ticker") or "").strip().upper() for row in rows if str(row.get("ticker") or "").strip()]
    if not tickers:
        return

    try:
        snapshot = get_realtime_snapshot("kor", tickers)
    except Exception as exc:
        logger.warning("한국 개별주 실시간 가격 오버레이 실패: %s", exc)
        return

    for row in rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        realtime = snapshot.get(ticker)
        if not realtime:
            continue

        now_val = realtime.get("nowVal")
        if now_val is not None:
            row["current_price"] = now_val

        change_rate = realtime.get("changeRate")
        if change_rate is not None:
            row["change_pct"] = change_rate

        volume = realtime.get("volume")
        if volume is not None:
            row["volume"] = volume


def _calculate_period_return(close: pd.Series, latest_price: float, months: int) -> float | None:
    target_date = pd.Timestamp.today().normalize() - pd.DateOffset(months=months)
    base_candidates = close[close.index >= target_date]
    if close.empty:
        return None
    base_date = close.index.min() if base_candidates.empty else base_candidates.index.min()
    base_price = float(close.loc[base_date])
    if base_price <= 0:
        return None
    return round(((latest_price / base_price) - 1.0) * 100.0, 4)


def _calculate_mdd(close: pd.Series, months: int) -> float | None:
    target_date = pd.Timestamp.today().normalize() - pd.DateOffset(months=months)
    period = close[close.index >= target_date]
    if period.empty:
        period = close
    if period.empty:
        return None
    drawdown = (period / period.cummax() - 1.0) * 100.0
    return round(float(drawdown.min()), 4)


def _apply_kor_history_metrics(rows: list[dict[str, Any]]) -> None:
    """한국 개별주 일봉으로 기간 수익률과 MDD를 보강한다."""
    for row in rows:
        ticker = str(row.get("ticker") or "").strip()
        if not ticker:
            continue
        history = fetch_naver_daily_ohlc(ticker, count=280)
        if history is None or history.empty or "Close" not in history.columns:
            row.update({"return_1m_pct": None, "return_3m_pct": None, "return_12m_pct": None, "mdd_12m_pct": None})
            continue
        close = pd.to_numeric(history["Close"], errors="coerce").dropna()
        close = close[close > 0]
        if close.empty:
            row.update({"return_1m_pct": None, "return_3m_pct": None, "return_12m_pct": None, "mdd_12m_pct": None})
            continue
        latest_price = float(row.get("current_price") or close.iloc[-1])
        row["return_1m_pct"] = _calculate_period_return(close, latest_price, 1)
        row["return_3m_pct"] = _calculate_period_return(close, latest_price, 3)
        row["return_12m_pct"] = _calculate_period_return(close, latest_price, 12)
        row["mdd_12m_pct"] = _calculate_mdd(close, 12)


# 티커→시가총액 맵은 화면 재방문마다 페이지 순회를 반복하지 않게 짧은 TTL 로 캐시한다.
_MARKET_CAP_CACHE: dict[str, tuple[float, dict[str, int]]] = {}
_MARKET_CAP_CACHE_TTL_SEC = CACHE_TTL_COMPUTE


def load_kor_market_caps(tickers: list[str]) -> dict[str, int]:
    """티커 → 시가총액(억 원) 맵 — 네이버 marketValue 리스트(위 화면과 같은 소스).

    KOSPI·KOSDAQ 를 시총 상위부터 순회하며 요청 티커를 찾는다. 요청 티커를 모두
    찾으면 조기 종료한다. 목록에 없는 티커(순위권 밖)는 맵에서 빠진다 — 값이 없으면
    화면은 '-' 로 둔다(임의 보정 없음).
    """
    import time as _time

    wanted = {str(t or "").strip() for t in tickers if str(t or "").strip()}
    if not wanted:
        return {}

    now = _time.monotonic()
    cached = _MARKET_CAP_CACHE.get("kor")
    if cached is not None and now - cached[0] < _MARKET_CAP_CACHE_TTL_SEC:
        return {t: cap for t, cap in cached[1].items() if t in wanted}

    caps: dict[str, int] = {}
    page_size = 100
    for market in ("KOSPI", "KOSDAQ"):
        try:
            first = _fetch_market_value_page(market, page=1, page_size=page_size)
        except RuntimeError:
            continue  # 한 시장 실패가 다른 시장까지 막지 않게 — 실패분은 맵에서 빠진다.
        total_count = int(first.get("totalCount") or 0)
        total_pages = max(1, math.ceil(total_count / page_size)) if total_count > 0 else 1
        payload = first
        for page in range(1, total_pages + 1):
            for item in payload.get("stocks") or []:
                ticker = str(item.get("itemCode") or "").strip()
                cap = _parse_number(item.get("marketValue"))
                if ticker and cap is not None:
                    caps[ticker] = cap
            if wanted <= set(caps):
                break
            if page < total_pages:
                payload = _fetch_market_value_page(market, page=page + 1, page_size=page_size)
        if wanted <= set(caps):
            break

    _MARKET_CAP_CACHE["kor"] = (now, caps)
    return {t: cap for t, cap in caps.items() if t in wanted}
