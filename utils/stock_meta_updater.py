"""계좌 종목 메타데이터를 업데이트합니다."""

import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests  # noqa: F401  # 타입 힌트/하위 호환을 위해 유지
import yfinance as yf

import numpy as np
from services.etf_holdings_service import fetch_korean_etf_holdings_from_naver
from services.etf_meta_service import fetch_korean_etf_info_from_naver
from services.stock_cache_service import get_stock_cache_meta_map, refresh_stock_cache
from utils.data_loader import (
    _YF_SESSION,
    fetch_naver_kor_market,
    fetch_naver_kor_stock_map,
    fetch_pykrx_name,
    fetch_ohlcv,
)
from utils.http_session import shared_session
from utils.kis_market import refresh_kis_domestic_etf_master_cache
from utils.logger import get_app_logger
from utils.settings_loader import get_ticker_type_settings, list_available_ticker_types
from utils.indicators import calculate_moving_average_signals
from utils.perf_metrics import curve_metrics


def _simulate_single_stock_ma_strategy(close_prices: pd.Series, lookback_months: int) -> dict[str, Any]:
    """단일 종목에 대해 지정 개월수(lookback_months) 동안의 단순 보유(Buy & Hold) 성과 지표(수익률, MDD, Sortino)를 구합니다.
    상장일이 시작일보다 뒤에 있는 경우, 가용한 전체 기간으로 계산하고 is_partial=True 플래그를 반환합니다.
    """
    if close_prices.empty:
        return {"cagr": 0.0, "mdd": 0.0, "sortino": 0.0, "is_partial": False}

    try:
        last_date = close_prices.index[-1]
        start_date = last_date - pd.DateOffset(months=lookback_months)
        
        # 상장일이 시작일보다 나중인지 여부 판정
        first_price_date = close_prices.index[0]
        is_partial = first_price_date > start_date

        target_series = close_prices.loc[start_date:]
        if len(target_series) < 2:
            return {"cagr": 0.0, "mdd": 0.0, "sortino": 0.0, "is_partial": is_partial}

        start_val = float(target_series.iloc[0])
        values = target_series.iloc[1:].to_numpy()

        metrics = curve_metrics(start_val, values)
        return {
            "cagr": round(float(metrics.get("total_return_pct", 0.0)), 2),  # CAGR 대신 단순 누적 수익률(%) 저장
            "mdd": round(float(metrics.get("mdd_pct", 0.0)), 2),
            "sortino": round(float(metrics.get("sortino", 0.0)), 2),
            "is_partial": is_partial,
        }
    except Exception:
        return {"cagr": 0.0, "mdd": 0.0, "sortino": 0.0, "is_partial": False}


# 랭킹 표시용 백테스트 기준 기간(개월). MDD·소르티노 모두 이 기간으로 계산한다.
# 상장 기간이 이보다 짧으면 is_partial=True 로 표시(프론트에서 다른 색으로 강조).
RANK_BACKTEST_MONTHS = 12

# 가격 파생 지표(거래량·기간수익률·backtest_stats) 필드 목록 — 가격지표 배치가 담당하는 필드.
PRICE_METRIC_FIELDS = (
    "1_week_avg_volume",
    "volume",
    "1_week_earn_rate",
    "2_week_earn_rate",
    "1_month_earn_rate",
    "3_month_earn_rate",
    "6_month_earn_rate",
    "12_month_earn_rate",
    "backtest_stats",
)


def compute_price_metrics(frame: "pd.DataFrame | None") -> dict[str, Any]:
    """OHLCV 프레임에서 거래량·기간수익률(1주~12개월)·backtest_stats 를 계산한다.

    가격지표 배치와 단일 종목 추가가 **같은 로직**을 쓰도록 공용화한 함수다.
    프레임이 없거나 Close 가 없으면 빈 dict 를 반환한다(호출부가 부분 갱신).
    """
    result: dict[str, Any] = {}
    if frame is None or frame.empty or "Close" not in frame.columns:
        return result

    if "Volume" in frame.columns:
        avg_volume = frame.tail(5)["Volume"].mean()
        if pd.notna(avg_volume):
            result["1_week_avg_volume"] = int(avg_volume)
        non_empty_vols = frame["Volume"].dropna()
        if not non_empty_vols.empty:
            result["volume"] = int(non_empty_vols.iloc[-1])

    def calc_rate_safe(df: pd.DataFrame, days_lookback: int) -> float | None:
        if len(df) < days_lookback + 1 or "Close" not in df.columns:
            return None
        subset = df.tail(days_lookback + 1)
        if len(subset) < 2:
            return None
        start_price = subset.iloc[0]["Close"]
        end_price = subset.iloc[-1]["Close"]
        if pd.notna(start_price) and pd.notna(end_price) and start_price > 0:
            return round(((end_price - start_price) / start_price) * 100, 4)
        return None

    result["1_week_earn_rate"] = calc_rate_safe(frame, 5)
    result["2_week_earn_rate"] = calc_rate_safe(frame, 10)
    result["1_month_earn_rate"] = calc_rate_safe(frame, 21)
    result["3_month_earn_rate"] = calc_rate_safe(frame, 63)
    result["6_month_earn_rate"] = calc_rate_safe(frame, 126)
    result["12_month_earn_rate"] = calc_rate_safe(frame, 252)

    bt_close = frame["Close"].dropna()
    if not bt_close.empty:
        result["backtest_stats"] = _simulate_single_stock_ma_strategy(bt_close, RANK_BACKTEST_MONTHS)
    return result

# -------------------------------------------------------------------------
# 배치 단위 공유 캐시 (메타 업데이트 1회 진입 시 1회만 빌드, 풀들 간 공유)
# `update_stock_metadata` 진입 시 `_reset_batch_caches()` 로 초기화한다.
# -------------------------------------------------------------------------
_BATCH_NAVER_ETF_NAMES_MAP: dict[str, str] | None = None


def _reset_batch_caches() -> None:
    """메타 업데이트 배치 진입 시 풀 간 공유 캐시를 초기화한다."""
    global _BATCH_NAVER_ETF_NAMES_MAP
    _BATCH_NAVER_ETF_NAMES_MAP = None


def _get_cached_naver_etf_names_map() -> dict[str, str]:
    """배치 동안 1회만 ETF 이름 맵을 빌드해 풀들에 공유."""
    global _BATCH_NAVER_ETF_NAMES_MAP
    if _BATCH_NAVER_ETF_NAMES_MAP is None:
        _BATCH_NAVER_ETF_NAMES_MAP = fetch_naver_etf_names_map()
    return _BATCH_NAVER_ETF_NAMES_MAP


def fetch_naver_etf_names_map() -> dict[str, str]:
    """
    네이버 ETF API를 호출하여 전체 ETF 종목의 {코드: 이름} 맵을 반환합니다.
    """
    from config import NAVER_FINANCE_ETF_API_URL, NAVER_FINANCE_HEADERS

    url = NAVER_FINANCE_ETF_API_URL
    names_map = {}

    try:
        response = shared_session.get(url, headers=NAVER_FINANCE_HEADERS, timeout=5)
        response.raise_for_status()
        data = response.json()

        items = data.get("result", {}).get("etfItemList", [])
        for item in items:
            code = str(item.get("itemcode", "")).strip()
            name = str(item.get("itemname", "")).strip()
            if code and name:
                names_map[code] = name

        return names_map
    except Exception as e:
        logger = get_app_logger()
        logger.warning(f"네이버 ETF 목록 조회 실패: {e}")
        return {}


def _fetch_naver_listing_date(ticker: str) -> str | None:
    """
    네이버 차트 API에서 한국 ETF의 실제 상장일을 가져옵니다.

    Args:
        ticker: 종목 코드 (예: 379800)

    Returns:
        상장일 문자열 (YYYY-MM-DD) 또는 None
    """
    from config import NAVER_FINANCE_CHART_API_URL

    logger = get_app_logger()
    url = f"{NAVER_FINANCE_CHART_API_URL}?symbol={ticker}&timeframe=day&count=1&requestType=0"

    try:
        response = shared_session.get(url, timeout=5)
        response.raise_for_status()

        # 네이버 차트 API는 EUC-KR 기반 XML을 반환하므로 명시적으로 디코딩
        try:
            text = response.content.decode("euc-kr")
        except Exception:
            text = response.text

        # XML 파싱
        root = ET.fromstring(text)
        chartdata = root.find("chartdata")

        if chartdata is not None:
            origintime = chartdata.get("origintime")
            if origintime and len(origintime) == 8:  # YYYYMMDD 형식
                # YYYYMMDD -> YYYY-MM-DD 변환
                listing_date = f"{origintime[:4]}-{origintime[4:6]}-{origintime[6:8]}"
                logger.debug(f"[네이버 API] {ticker} 상장일: {listing_date}")
                return listing_date

    except Exception as e:
        logger.info(f"[네이버 API] {ticker} 상장일 조회 실패: {e}")

    return None


def _try_parse_float(value: Any) -> float | None:
    """문자열/숫자 값을 안전하게 float로 변환합니다."""
    if value is None:
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


from collections.abc import Callable

from utils.stock_list_io import bulk_update_stocks, get_all_etfs_including_deleted

# ETF 메타(info) 캐시 TTL — 이 시간 안에 갱신된 메타가 있으면 네이버 info 호출을 스킵.
# 사용자 정책: info 는 변경 빈도가 낮으므로 1일 TTL, holdings 는 항상 갱신.
_ETF_INFO_CACHE_TTL = timedelta(days=1)


def _is_meta_cache_fresh(existing_doc: dict[str, Any] | None) -> bool:
    """기존 메타 캐시 문서의 updated_at 이 TTL 이내인지 판정."""
    if not isinstance(existing_doc, dict):
        return False
    meta = existing_doc.get("meta_cache")
    if not isinstance(meta, dict) or not meta:
        return False
    updated_at = existing_doc.get("updated_at")
    if not isinstance(updated_at, datetime):
        return False
    # MongoDB datetime 은 naive(UTC) 일 수 있어 tz-aware 화
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return (now - updated_at) < _ETF_INFO_CACHE_TTL


def _refresh_korean_etf_meta_cache(
    ticker_type: str,
    ticker: str,
    name: str,
    *,
    existing_cache_doc: dict[str, Any] | None = None,
    backtest_stats: dict[str, Any] | None = None,
) -> None:
    """한국 ETF 메타/구성종목 캐시를 네이버 기준으로 갱신한다.

    info(저빈도) 는 TTL 이내면 네이버 호출을 스킵하고 holdings 만 갱신한다.
    """
    ticker_type_norm = str(ticker_type or "").strip().lower()
    ticker_norm = str(ticker or "").strip().upper()
    name_norm = str(name or "").strip() or ticker_norm
    if not ticker_type_norm or not ticker_norm:
        raise ValueError("ticker_type과 ticker가 필요합니다.")

    # 1) info 캐시 TTL 판정
    skip_info = _is_meta_cache_fresh(existing_cache_doc)

    meta_cache: dict[str, Any] | None = None
    if not skip_info:
        etf_info = fetch_korean_etf_info_from_naver(ticker_norm)

        # 실시간 iNAV/괴리율 추가 획득 (글로벌 캐시라 비용 거의 없음)
        from utils.data_loader import fetch_naver_etf_inav_snapshot
        inav_snapshot = fetch_naver_etf_inav_snapshot([ticker_norm]).get(ticker_norm, {})

        meta_cache = {
            "source": str(etf_info.get("source") or "naver_etf_meta"),
            "updated_at": str(etf_info.get("fetched_at") or ""),
            "nav": inav_snapshot.get("nav"),
            "deviation": inav_snapshot.get("deviation"),
            "reference_date": etf_info.get("reference_date"),
            "listed_date": etf_info.get("listed_date"),
            "dividend_yield_ttm": etf_info.get("dividend_yield_ttm"),
            "dividend_per_share_ttm": etf_info.get("dividend_per_share_ttm"),
            "recent_ex_dividend_at": etf_info.get("recent_ex_dividend_at"),
            "dividend_history": etf_info.get("dividend_history") or [],
            "expense_ratio": etf_info.get("expense_ratio"),
            "total_net_assets": etf_info.get("total_net_assets"),
            "issue_name": etf_info.get("issue_name"),
            "base_index": etf_info.get("base_index"),
        }

    if meta_cache is not None:
        meta_cache["backtest_stats"] = backtest_stats

    # 2) holdings 는 항상 갱신 (TTL 미적용)
    holdings_info = fetch_korean_etf_holdings_from_naver(ticker_norm)
    holdings_cache = {
        "source": str(holdings_info.get("source") or "naver_etf_component"),
        "updated_at": str(holdings_info.get("fetched_at") or ""),
        "reference_date": holdings_info.get("as_of_date"),
        "holdings_count": holdings_info.get("holdings_count"),
        "items": list(holdings_info.get("holdings") or []),
    }

    refresh_stock_cache(
        ticker_type_norm,
        ticker_norm,
        country_code="kor",
        name=name_norm,
        meta_cache=meta_cache,  # None 이면 meta 부분은 미변경
        holdings_cache=holdings_cache,
    )


def _format_iso_date(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return pd.Timestamp(text).strftime("%Y-%m-%d")
    except Exception:
        return text[:10] if len(text) >= 10 else text


def _refresh_us_stock_meta_cache(
    ticker_type: str,
    ticker: str,
    name: str,
    naver_entry: dict[str, Any],
    backtest_stats: dict[str, Any] | None = None,
) -> None:
    """미국 개별주 메타 캐시를 네이버 미국 종목 API 기준으로 갱신한다."""
    ticker_type_norm = str(ticker_type or "").strip().lower()
    ticker_norm = str(ticker or "").strip().upper()
    name_norm = str(name or "").strip() or ticker_norm
    if not ticker_type_norm or not ticker_norm:
        raise ValueError("ticker_type과 ticker가 필요합니다.")

    meta_cache = {
        "source": "naver_us_market_stock",
        "updated_at": datetime.now().isoformat(),
        "listed_date": _format_iso_date(naver_entry.get("listing_date")),
        "dividend_yield_ttm": naver_entry.get("dividend_yield_ttm"),
        "dividend_per_share_ttm": naver_entry.get("dividend_per_share_ttm"),
        "expense_ratio": None,
        "total_net_assets": naver_entry.get("market_cap"),
        "issue_name": name_norm,
        "market": naver_entry.get("market"),
        "industry": naver_entry.get("industry"),
        "backtest_stats": backtest_stats,
    }

    refresh_stock_cache(
        ticker_type_norm,
        ticker_norm,
        country_code="us",
        name=name_norm,
        meta_cache=meta_cache,
    )


def fetch_betashares_holdings(ticker: str) -> dict[str, Any] | None:
    """BetaShares 공식 홈페이지에서 portfolio holdings CSV를 긁어서 구성종목 딕셔너리를 반환합니다."""
    import urllib.request
    import csv
    from datetime import datetime

    logger = get_app_logger()
    ticker_clean = str(ticker).strip().upper()
    url = f"https://www.betashares.com.au/files/csv/{ticker_clean}_Portfolio_Holdings.csv"

    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req, timeout=12) as response:
            content = response.read().decode('utf-8', errors='ignore')

        lines = content.splitlines()

        # 날짜(Date) 파싱 시도 (보통 4번째 라인: "Date,2026-06-29")
        reference_date = None
        for line in lines[:8]:
            if line.startswith("Date,"):
                parts = line.split(",")
                if len(parts) > 1 and parts[1].strip():
                    reference_date = parts[1].strip()
                    break
        if not reference_date:
            reference_date = datetime.now().strftime("%Y-%m-%d")

        # "Ticker," 로 시작하는 헤더 찾기
        header_idx = -1
        for idx, line in enumerate(lines[:12]):
            if line.startswith("Ticker,"):
                header_idx = idx
                break

        if header_idx == -1:
            logger.warning(f"[BetaShares] {ticker_clean} CSV 헤더 시작부(Ticker,)를 찾지 못했습니다.")
            return None

        # 데이터 파싱
        reader = csv.reader(lines[header_idx:])
        headers = next(reader)

        try:
            ticker_col = headers.index("Ticker")
            name_col = headers.index("Name")
            weight_col = headers.index("Weight (%)")
        except ValueError as e:
            logger.warning(f"[BetaShares] {ticker_clean} CSV 필수 열 매핑 실패: {e}")
            return None

        items = []
        for row in reader:
            if not row or len(row) <= max(ticker_col, name_col, weight_col):
                continue
            t_val = str(row[ticker_col]).strip()
            if not t_val or t_val.lower() == "ticker":
                continue
            # Ticker 정제: "AVGO UW" -> "AVGO" (공백 뒤 거래소 구분자 제거)
            t_clean = t_val.split()[0].upper()
            name_val = str(row[name_col]).strip()

            try:
                # weight는 백분율(%) 그대로 적재 (예: 4.16)
                weight_val = float(row[weight_col])
            except ValueError:
                continue

            items.append({
                "ticker": t_clean,
                "name": name_val,
                "weight": weight_val,
            })

        if not items:
            return None

        # 가중치 순으로 정렬
        items.sort(key=lambda x: x.get("weight") or 0.0, reverse=True)

        return {
            "source": "betashares_csv",
            "fetched_at": datetime.now().isoformat(),
            "as_of_date": reference_date,
            "holdings_count": len(items),
            "holdings": items,
        }
    except Exception as exc:
        logger.warning(f"[BetaShares] {ticker_clean} CSV 수집 실패: {exc}")
        return None


def fetch_yfinance_holdings(ticker: str, is_australian: bool = False) -> dict[str, Any] | None:
    """yfinance를 활용해 Top 10 구성종목 정보를 가져옵니다."""
    import yfinance as yf
    from datetime import datetime

    logger = get_app_logger()
    symbol = f"{ticker.upper()}.AX" if is_australian else ticker.upper()
    try:
        t_data = yf.Ticker(symbol)
        funds_data = getattr(t_data, "funds_data", None)
        if funds_data is None:
            return None
        top_h = getattr(funds_data, "top_holdings", None)
        if top_h is None or top_h.empty:
            return None

        items = []
        for idx, row in top_h.iterrows():
            t_code = str(idx).strip().upper()
            t_clean = t_code.split(".")[0].split()[0]
            name_val = str(row.get("Name") or "").strip() or t_clean
            try:
                # yfinance의 Holding Percent 값(예: 0.0493)을 백분율(4.93)로 변환
                weight_val = float(row.get("Holding Percent") or 0.0) * 100.0
            except ValueError:
                weight_val = 0.0

            items.append({
                "ticker": t_clean,
                "name": name_val,
                "weight": weight_val,
            })

        if not items:
            return None

        items.sort(key=lambda x: x.get("weight") or 0.0, reverse=True)

        return {
            "source": "yfinance_holdings",
            "fetched_at": datetime.now().isoformat(),
            "as_of_date": datetime.now().strftime("%Y-%m-%d"),
            "holdings_count": len(items),
            "holdings": items,
        }
    except Exception as exc:
        logger.warning(f"[yfinance] {symbol} 수집 실패: {exc}")
        return None


def _refresh_overseas_etf_meta_cache(
    ticker_type: str,
    ticker: str,
    name: str,
    country_code: str,
    backtest_stats: dict[str, Any] | None = None,
) -> None:
    """호주/미국 등 해외 ETF 메타와 holdings 캐시를 수집 및 저장한다."""
    from datetime import datetime
    logger = get_app_logger()
    ticker_type_norm = str(ticker_type or "").strip().lower()
    ticker_norm = str(ticker or "").strip().upper()
    name_norm = str(name or "").strip() or ticker_norm
    country_norm = str(country_code or "").strip().lower()

    holdings_info = None

    # 호주 ETF인 경우 BetaShares CSV 우선 시도 후 yfinance로 fallback
    if country_norm == "au":
        holdings_info = fetch_betashares_holdings(ticker_norm)
        if not holdings_info:
            holdings_info = fetch_yfinance_holdings(ticker_norm, is_australian=True)
    else:
        holdings_info = fetch_yfinance_holdings(ticker_norm, is_australian=False)

    if not holdings_info:
        logger.warning(f"[{ticker_type_norm.upper()}/{ticker_norm}] 해외 ETF holdings 수집 실패 (건너뜀)")
        return

    holdings_cache = {
        "source": holdings_info["source"],
        "updated_at": holdings_info["fetched_at"],
        "reference_date": holdings_info["as_of_date"],
        "holdings_count": holdings_info["holdings_count"],
        "items": holdings_info["holdings"],
    }

    meta_cache = {
        "source": "overseas_etf_meta",
        "updated_at": datetime.now().isoformat(),
        "listed_date": None,
        "dividend_yield_ttm": None,
        "expense_ratio": None,
        "total_net_assets": None,
        "issue_name": name_norm,
        "backtest_stats": backtest_stats,
    }

    # yfinance를 통해 추가적인 ETF 메타 정보 보완
    try:
        import yfinance as yf
        symbol = f"{ticker_norm}.AX" if country_norm == "au" else ticker_norm
        t_data = yf.Ticker(symbol)
        t_info = getattr(t_data, "info", {})
        if t_info:
            meta_cache["dividend_yield_ttm"] = t_info.get("trailingAnnualDividendYield") or t_info.get("dividendYield")
            meta_cache["expense_ratio"] = t_info.get("feesExpensesTotal") or t_info.get("expenseRatio")
            meta_cache["total_net_assets"] = t_info.get("totalAssets") or t_info.get("marketCap")
            if t_info.get("firstTradeDateEpochUtc") or t_info.get("startDate"):
                raw_start = t_info.get("firstTradeDateEpochUtc") or t_info.get("startDate")
                meta_cache["listed_date"] = _format_iso_date(raw_start)
    except Exception as e:
        logger.debug(f"yfinance 추가 메타데이터 수집 건너뜀: {e}")

    refresh_stock_cache(
        ticker_type_norm,
        ticker_norm,
        country_code=country_norm,
        name=name_norm,
        meta_cache=meta_cache,
        holdings_cache=holdings_cache,
    )
    logger.info(f"[{ticker_type_norm.upper()}/{ticker_norm}] 해외 ETF 구성종목 캐시 갱신 완료 ({holdings_info['holdings_count']}개 종목)")


def update_ticker_type_metadata(
    ticker_type: str, progress_callback: Callable[[int, int, str], None] | None = None
):
    """지정된 종목타입의 모든 종목 메타데이터를 업데이트합니다."""
    logger = get_app_logger()
    type_norm = (ticker_type or "").strip().lower()

    try:
        settings = get_ticker_type_settings(type_norm)
        country_code = str(settings.get("country_code") or "").strip().lower()
    except Exception as e:
        logger.error(f"종목타입 설정을 로드할 수 없습니다 ({type_norm}): {e}")
        return

    # 삭제된 종목 포함하여 모든 종목 로드
    stock_data = get_all_etfs_including_deleted(type_norm)
    
    if not stock_data:
        logger.warning(f"'{type_norm}' 종목타입의 종목 데이터가 비어있습니다.")
        return

    updated_count = 0
    ticker_entries: list[dict[str, Any]] = []

    for stock in stock_data:
        ticker_entries.append(stock)

    total_count = len(ticker_entries)
    logger.info(f"[{type_norm.upper()}] 메타데이터 업데이트 시작 (총 {total_count}개 종목)")

    if progress_callback:
        progress_callback(0, total_count, "데이터 준비 중...")

    # [KOR] 전체 종목(일반주/ETF) 맵 구성하여 루프 내 호출 최소화
    naver_etf_map: dict[str, str] = {}
    if country_code == "kor":
        # 풀 간 공유 캐시 사용 (한국 종목풀은 같은 API 결과를 공유)
        logger.info("네이버 API에서 한국 ETF/종목 정보를 수집합니다 (배치 캐시)...")
        naver_etf_map = dict(_get_cached_naver_etf_names_map())
        fetch_naver_kor_stock_map()  # 캐시 워밍 (일반주/ETN 이름·시장 조회 대비)

    # 한국 종목풀: 기존 메타 캐시 문서를 1회 일괄 로드해 ETF 상세 TTL 판정에 사용한다.
    existing_meta_cache_map: dict[str, dict[str, Any]] = {}
    if country_code == "kor":
        all_tickers_for_pool = [
            str(stock.get("ticker") or "").strip().upper()
            for stock in ticker_entries
            if str(stock.get("ticker") or "").strip()
        ]
        try:
            existing_meta_cache_map = get_stock_cache_meta_map(type_norm, all_tickers_for_pool)
            logger.info(
                f"[{type_norm.upper()}] 기존 메타 캐시 문서 {len(existing_meta_cache_map)}건 로드 (TTL 판정용)"
            )
        except Exception as exc:
            logger.warning(f"[{type_norm.upper()}] 기존 메타 캐시 로드 실패 — 전체 갱신으로 진행: {exc}")
            existing_meta_cache_map = {}

    naver_us_stock_map: dict[str, dict[str, Any]] = {}
    if country_code == "us":
        from utils.us_stock_market_service import fetch_naver_us_stock_info_map

        us_tickers = {
            str(stock.get("ticker") or "").strip().upper()
            for stock in ticker_entries
            if str(stock.get("ticker") or "").strip()
        }
        logger.info(f"[{type_norm.upper()}] 네이버 미국 종목 업종 맵을 구성합니다...")
        naver_us_stock_map = fetch_naver_us_stock_info_map(us_tickers)
        logger.info(f"[{type_norm.upper()}] 네이버 미국 종목 업종 {len(naver_us_stock_map)}건 수집")

    # 업데이트 사항을 모아두기 위한 리스트
    updates_for_db = []

    for idx, stock in enumerate(ticker_entries, start=1):
        ticker = stock.get("ticker")
        if not ticker:
            continue

        try:
            update_single_stock_metadata(
                stock,
                country_code,
                naver_etf_map,
                type_norm,
                naver_us_stock_map=naver_us_stock_map,
            )

            name = stock.get("name") or "-"
            logger.info(f"  -> 메타데이터 획득 중: {idx}/{total_count} - {name}({ticker})")

            # 가격 파생 지표(거래량·기간수익률·backtest_stats) — OHLCV(60개월) 1회 로드로 계산.
            price_metrics: dict[str, Any] = {}
            try:
                df = fetch_ohlcv(ticker, country=country_code, months_back=60, ticker_type=type_norm)
                price_metrics = compute_price_metrics(df)
            except Exception as exc:
                logger.warning(f"  -> [{ticker}] 가격 지표 연산 중 오류 발생: {exc}")
            backtest_stats = price_metrics.get("backtest_stats") or {}

            # 저장할 필드 = 식별(stock) + 가격지표(price_metrics)
            update_doc: dict[str, Any] = {"ticker": ticker, **price_metrics}
            identity_fields = [
                "name",
                "listing_date",
                "market",
                "is_etf",
                "etf_category",
                "dividend_yield_ttm",
                "market_cap",
            ]
            for f in identity_fields:
                if f in stock:
                    update_doc[f] = stock[f]

            # 한국 종목풀은 ETF 상세 캐시 갱신을 시도한다. 개별주/비ETF는 서비스 오류를 경고로 건너뛴다.
            if country_code == "kor":
                try:
                    existing_doc = existing_meta_cache_map.get(str(ticker).strip().upper())
                    _refresh_korean_etf_meta_cache(
                        type_norm,
                        str(ticker),
                        str(name),
                        existing_cache_doc=existing_doc,
                        backtest_stats=backtest_stats,
                    )
                except Exception as e:
                    logger.warning(f"[{type_norm.upper()}/{ticker}] ETF 상세 캐시 갱신 건너뜀: {e}")
            elif country_code == "au":
                # 호주 ETF
                try:
                    _refresh_overseas_etf_meta_cache(type_norm, str(ticker), str(name), country_code, backtest_stats=backtest_stats)
                except Exception as e:
                    logger.warning(f"[{type_norm.upper()}/{ticker}] 호주 ETF 상세 캐시 갱신 실패: {e}")
            elif country_code == "us":
                # 미국 개별주
                naver_entry = naver_us_stock_map.get(str(ticker).strip().upper(), {})
                if naver_entry:
                    try:
                        _refresh_us_stock_meta_cache(type_norm, str(ticker), str(name), naver_entry, backtest_stats=backtest_stats)
                    except Exception as e:
                        logger.warning(f"[{type_norm.upper()}/{ticker}] 미국 개별주 메타 캐시 갱신 건너뜀: {e}")

            updates_for_db.append(update_doc)

            # 중간 저장 (100개 단위로 MongoDB round-trip 절감)
            if len(updates_for_db) >= 100:
                try:
                    modified = bulk_update_stocks(type_norm, updates_for_db)
                    logger.info(f"[{type_norm.upper()}] 중간 저장 완료 ({idx}/{total_count}, {modified}건)")
                    updates_for_db.clear()
                except Exception as e:
                    logger.error(f"[{type_norm.upper()}] 중간 저장 실패: {e}")

            if progress_callback:
                progress_callback(idx, total_count, f"{name}({ticker})")
            updated_count += 1
            time.sleep(0.02)  # 속도 조절 (외부 API 보호용 미소 슬립)

        except Exception as e:
            logger.error(f"[{type_norm.upper()}/{ticker}] 메타데이터 업데이트 실패: {e}")

    # 남아있는 업데이트 저장
    try:
        if updates_for_db:
            modified = bulk_update_stocks(type_norm, updates_for_db)
            logger.info(f"[{type_norm.upper()}] 최종 메타데이터 변경사항 저장 완료 ({modified}건)")
    except Exception as e:
        logger.error(f"'{type_norm}' 최종 저장 실패: {e}")


def update_stock_metadata(ticker_type: str | None = None):
    """
    모든 종목타입 또는 특정 종목타입의 메타데이터를 업데이트합니다.
    """
    logger = get_app_logger()

    # 풀 간 공유 캐시 초기화 — 이번 배치에서 ETF 이름 맵을 1회만 빌드한다.
    _reset_batch_caches()

    ticker_types_to_update: list[str] = []
    available_ticker_types = list_available_ticker_types()

    if ticker_type:
        type_norm = ticker_type.strip().lower()
        if type_norm in available_ticker_types:
            ticker_types_to_update.append(type_norm)
        else:
            logger.error(f"대상 종목타입 '{ticker_type}'를 찾을 수 없습니다.")
            return
    else:
        ticker_types_to_update = available_ticker_types.copy()
        try:
            logger.info("KIS 국내 ETF 마스터 캐시 갱신을 시작합니다.")
            refreshed_count = refresh_kis_domestic_etf_master_cache()
            logger.info("KIS 국내 ETF 마스터 캐시 갱신 완료: %d건", refreshed_count)
        except Exception as exc:
            logger.error("KIS 국내 ETF 마스터 캐시 갱신 실패: %s", exc)

    logger.info(f"메타데이터 업데이트 대상 종목타입: {ticker_types_to_update}")

    for type_norm in ticker_types_to_update:
        update_ticker_type_metadata(type_norm)

    logger.info("모든 메타데이터 업데이트 작업이 완료되었습니다.")


def fetch_stock_info(ticker: str, country_code: str) -> dict[str, Any] | None:
    """
    단일 종목의 이름과 메타데이터를 조회합니다.
    UI에서 '조회' 버튼 클릭 시 사용합니다.
    """
    country_norm = (country_code or "").lower().strip()
    ticker = str(ticker).strip()
    if not ticker:
        return None

    # 기본 반환 구조
    result = {"ticker": ticker, "name": "", "listing_date": None}
    logger = get_app_logger()

    try:
        if country_norm == "kor":
            # 1. Pykrx로 이름 조회 시도 (가장 빠름)
            try:
                name = fetch_pykrx_name(ticker)
                if name:
                    result["name"] = name
            except Exception:
                pass

            # 2. 이름 못 찾으면 Naver ETF 이름 맵 시도 (신규 상장 ETF 대응)
            if not result["name"]:
                try:
                    naver_map = fetch_naver_etf_names_map()
                    if ticker in naver_map:
                        result["name"] = naver_map[ticker]
                except Exception:
                    pass

            # 3. 상장일 조회
            try:
                ld = _fetch_naver_listing_date(ticker)
                if ld:
                    result["listing_date"] = ld
            except Exception:
                pass

        elif country_norm in ("us", "au"):
            yf_ticker = ticker
            # Strip exchange prefix (e.g., "ASX:VGS" → "VGS")
            if ":" in yf_ticker:
                yf_ticker = yf_ticker.split(":")[-1]
            if country_norm == "au" and not yf_ticker.endswith(".AX"):
                yf_ticker = f"{yf_ticker}.AX"

            t = yf.Ticker(yf_ticker)
            try:
                info = t.info
                # 이름
                name = info.get("longName") or info.get("shortName")
                if name:
                    result["name"] = name
            except Exception:
                pass

            # 상장일
            try:
                hist = t.history(period="max")
                if not hist.empty:
                    result["listing_date"] = hist.index.min().strftime("%Y-%m-%d")
            except Exception:
                pass

        # 이름이라도 찾았으면 성공
        if result["name"]:
            return result
        # 이름 못 찾았어도 상장일 있으면 반환? (일단 이름이 중요)
        return result if result["name"] or result["listing_date"] else None

    except Exception as e:
        logger.warning(f"종목 정보 조회 실패 ({ticker}, {country_norm}): {e}")
        return None


def update_single_ticker_metadata(ticker_type: str, ticker: str) -> None:
    """단일 종목의 메타데이터를 조회하고 DB에 저장합니다."""
    logger = get_app_logger()
    type_norm = (ticker_type or "").strip().lower()
    ticker_norm = (ticker or "").strip().upper()

    if not type_norm or not ticker_norm:
        return

    try:
        settings = get_ticker_type_settings(type_norm)
        country_code = str(settings.get("country_code") or "").strip().lower()
    except Exception as exc:
        raise RuntimeError(f"[{type_norm.upper()}/{ticker_norm}] 종목타입 설정 로드 실패: {exc}") from exc

    naver_etf_map: dict[str, str] = {}
    naver_us_stock_map: dict[str, dict[str, Any]] = {}
    if country_code == "kor":
        naver_etf_map = dict(_get_cached_naver_etf_names_map())
    if country_code == "us":
        from utils.us_stock_market_service import fetch_naver_us_stock_info_map

        naver_us_stock_map = fetch_naver_us_stock_info_map({ticker_norm})

    stock: dict[str, Any] = {"ticker": ticker_norm}

    # 기존 메타 로드
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is not None:
        existing = db.stock_meta.find_one(
            {"ticker_type": type_norm, "ticker": ticker_norm},
            {"name": 1, "listing_date": 1},
        )
        if existing:
            stock["name"] = existing.get("name") or ""
            stock["listing_date"] = existing.get("listing_date")

    update_single_stock_metadata(
        stock,
        country_code,
        naver_etf_map,
        type_norm,
        naver_us_stock_map=naver_us_stock_map,
    )

    # 한국 종목풀은 ETF 상세 캐시 갱신을 시도한다. 개별주/비ETF는 서비스 오류를 경고로 건너뛴다.
    if country_code == "kor":
        try:
            _refresh_korean_etf_meta_cache(type_norm, ticker_norm, str(stock.get("name") or ticker_norm))
        except Exception as meta_cache_error:
            logger.warning(f"[{type_norm.upper()}/{ticker_norm}] ETF 메타 캐시 갱신 건너뜀: {meta_cache_error}")
    elif country_code == "au":
        try:
            _refresh_overseas_etf_meta_cache(type_norm, ticker_norm, str(stock.get("name") or ticker_norm), country_code)
        except Exception as meta_cache_error:
            logger.error(f"[{type_norm.upper()}/{ticker_norm}] 호주 ETF 상세 캐시 갱신 실패: {meta_cache_error}")
    elif country_code == "us":
        naver_entry = naver_us_stock_map.get(ticker_norm, {})
        if naver_entry:
            try:
                _refresh_us_stock_meta_cache(type_norm, ticker_norm, str(stock.get("name") or ticker_norm), naver_entry)
            except Exception as meta_cache_error:
                logger.error(f"[{type_norm.upper()}/{ticker_norm}] 미국 개별주 메타 캐시 갱신 실패: {meta_cache_error}")

    # 가격 파생 지표 — 종목 추가 즉시 랭킹 지표가 채워지도록 배치와 동일하게 계산한다.
    price_metrics: dict[str, Any] = {}
    try:
        df = fetch_ohlcv(ticker_norm, country=country_code, months_back=60, ticker_type=type_norm)
        price_metrics = compute_price_metrics(df)
    except Exception as exc:
        logger.warning(f"[{type_norm.upper()}/{ticker_norm}] 가격 지표 연산 실패: {exc}")

    update_doc: dict[str, Any] = {"ticker": ticker_norm, **price_metrics}
    identity_fields = [
        "name",
        "listing_date",
        "market",
        "is_etf",
        "etf_category",
        "dividend_yield_ttm",
        "market_cap",
    ]
    for f in identity_fields:
        if f in stock:
            update_doc[f] = stock[f]

    try:
        modified = bulk_update_stocks(type_norm, [update_doc])
        logger.info(f"[{type_norm.upper()}/{ticker_norm}] 메타데이터 업데이트 완료 ({modified}건)")
    except Exception as e:
        logger.error(f"[{type_norm.upper()}/{ticker_norm}] 메타데이터 저장 실패: {e}")


def update_single_stock_metadata(
    stock: dict[str, Any],
    country_code: str,
    naver_etf_map: dict[str, str],
    account_norm: str = "",
    *,
    naver_us_stock_map: dict[str, Any] | None = None,
):
    """단일 종목의 메타데이터를 업데이트합니다."""
    logger = get_app_logger()
    ticker = stock.get("ticker")
    if not ticker:
        return

    if country_code == "kor":
        yfinance_ticker = f"{ticker}.KS"
    elif country_code == "au" and not ticker.endswith(".AX"):
        yfinance_ticker = f"{ticker}.AX"
    else:
        yfinance_ticker = ticker

    listing_date_str = None
    data = None

    # 한국 주식인 경우
    if country_code == "kor":
        # 1. 상장일 조회
        if not stock.get("listing_date"):
            listing_date_str = _fetch_naver_listing_date(ticker)
            if listing_date_str and account_norm:
                logger.debug(f"[{account_norm.upper()}/{ticker}] 네이버 API에서 상장일 획득: {listing_date_str}")
        else:
            listing_date_str = stock.get("listing_date")

        # 2. 종목명 조회 및 업데이트
        new_name = naver_etf_map.get(ticker)
        if new_name:
            stock["name"] = new_name
            stock["is_etf"] = True
        elif not stock.get("name") or stock.get("name") == ticker:
            stock["is_etf"] = False
            # 일반주/ETN은 네이버 marketValue 통합 맵 → pykrx 폴백 순서로 조회
            try:
                fetched_name = fetch_pykrx_name(ticker)
                if fetched_name:
                    stock["name"] = fetched_name
                    if account_norm:
                        logger.info(f"[{account_norm.upper()}/{ticker}] 종목명 획득: {fetched_name}")
            except Exception as e:
                logger.warning(f"[{account_norm.upper()}/{ticker}] 종목명 조회 실패: {e}")
        else:
            stock["is_etf"] = False

        # 3. 마켓(KOSPI/KOSDAQ) 조회 — 네이버 marketValue 통합 맵 사용
        if not stock.get("market"):
            try:
                market = fetch_naver_kor_market(ticker)
                if market:
                    stock["market"] = market
                    if account_norm:
                        logger.info(f"[{account_norm.upper()}/{ticker}] 네이버에서 마켓 정보 획득: {market}")
            except Exception as e:
                logger.warning(f"[{account_norm.upper()}/{ticker}] 마켓 정보 조회 실패: {e}")

    elif country_code in ("us", "au"):
        # 해외 주식 플래그 설정
        if country_code == "au":
            stock["is_etf"] = True
        else:
            stock["is_etf"] = False

        if country_code == "us" and naver_us_stock_map:
            naver_entry = naver_us_stock_map.get(str(ticker).strip().upper(), {})
            industry = str(naver_entry.get("industry") or "").strip()
            if industry:
                stock["etf_category"] = industry
            market = str(naver_entry.get("market") or "").strip().upper()
            if market:
                stock["market"] = market
            if not stock.get("name"):
                fetched_name = str(naver_entry.get("name") or "").strip()
                if fetched_name:
                    stock["name"] = fetched_name
            if naver_entry.get("dividend_yield_ttm") is not None:
                stock["dividend_yield_ttm"] = naver_entry.get("dividend_yield_ttm")
            if naver_entry.get("market_cap") is not None:
                stock["market_cap"] = naver_entry.get("market_cap")

        try:
            # HTTP keep-alive: 가능하면 공유 curl_cffi Session 사용
            t = (
                yf.Ticker(yfinance_ticker, session=_YF_SESSION)
                if _YF_SESSION is not None
                else yf.Ticker(yfinance_ticker)
            )

            if not stock.get("name") or stock.get("name") == ticker:
                try:
                    info = t.info
                    fetched_name = info.get("longName") or info.get("shortName")
                    if fetched_name:
                        stock["name"] = fetched_name
                        if account_norm:
                            logger.debug(f"[{account_norm.upper()}/{ticker}] 종목명 업데이트: {fetched_name}")
                except Exception as e:
                    logger.warning(f"[{account_norm.upper()}/{ticker}] 종목명 조회 실패: {e}")

            # 상장일이 이미 있으면 history 호출 자체를 생략
            if not stock.get("listing_date"):
                try:
                    hist = t.history(period="max", auto_adjust=True)
                    if hist is not None and not hist.empty:
                        first_date = hist.index.min()
                        listing_date_str = first_date.strftime("%Y-%m-%d")
                        if account_norm:
                            logger.debug(
                                f"[{account_norm.upper()}/{ticker}] yfinance 상장일 획득: {listing_date_str}"
                            )
                        # period='max' 결과를 아래 수익률 계산에도 재사용 (yf.download 중복 호출 방지)
                        if isinstance(hist, pd.DataFrame):
                            data = hist.tail(520).copy()
                except Exception as e:
                    logger.warning(f"[{account_norm.upper()}/{ticker}] yfinance history 조회 실패: {e}")
        except Exception as e:
            logger.warning(f"[{account_norm.upper()}/{ticker}] yfinance 메타데이터 조회 조회 실패: {e}")

    if not listing_date_str:
        listing_date_str = stock.get("listing_date")

    if listing_date_str:
        stock["listing_date"] = listing_date_str

    # 가격 파생 지표(거래량·기간수익률·backtest_stats)는 이 함수에서 계산하지 않는다.
    # 호출부가 OHLCV 를 1회 로드해 compute_price_metrics() 로 계산·병합한다(가격지표 배치와 공유).
    stock.pop("1_month_avg_volume", None)
    stock.pop("1_week_avg_turnover", None)
    stock.pop("1_month_avg_turnover", None)
