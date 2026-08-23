"""계좌 종목 메타데이터를 업데이트합니다."""

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests  # noqa: F401  # 타입 힌트/하위 호환을 위해 유지
import yfinance as yf

from config import CACHE_TTL_META
from services.etf_holdings_service import fetch_korean_etf_holdings_from_naver
from services.etf_meta_service import fetch_korean_etf_info_from_naver
from services.stock_cache_service import get_stock_cache_meta_map, refresh_stock_cache
from services.vanguard_au_service import fetch_vanguard_au_expense_ratio_pct, fetch_vanguard_au_holdings
from utils.asx_ticker import (
    ensure_asx_prefix,
    from_yahoo_symbol,
    normalize_ticker,
    strip_asx_prefix,
    to_yahoo_symbol,
)
from utils.data_loader import (
    _YF_SESSION,
    fetch_naver_kor_market,
    fetch_naver_kor_stock_map,
    fetch_ohlcv,
    fetch_pykrx_name,
)
from utils.http_session import shared_session
from utils.kis_market import refresh_kis_domestic_etf_master_cache
from utils.logger import get_app_logger
from utils.settings_loader import get_ticker_type_settings, list_available_ticker_types

# 가격 파생 지표(거래량·기간수익률) 필드 목록 — 가격지표 배치가 담당하는 필드.
# MDD·소르티노(backtest_stats)는 순위 조회 시 장기 이평선 파생 기간으로 실시간 계산한다(utils/rankings.py).
PRICE_METRIC_FIELDS = (
    "1_week_avg_volume",
    "volume",
    "1_week_earn_rate",
    "2_week_earn_rate",
    "1_month_earn_rate",
    "3_month_earn_rate",
    "6_month_earn_rate",
    "12_month_earn_rate",
)


def compute_price_metrics(frame: "pd.DataFrame | None") -> dict[str, Any]:
    """OHLCV 프레임에서 거래량·기간수익률(1주~12개월)을 계산한다.

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
    return result


# -------------------------------------------------------------------------
# 배치 B 종목 병렬 워커 수 — 종목당 작업이 네이버/야후 HTTP 대기 위주라 병렬 이득이 크다.
# 사전 벤치마크: 네이버 x16·yfinance x12 까지 실패 0. 보수적으로 8을 쓴다.
# 문제가 생기면 1 로 되돌리면 기존 직렬 동작이다. (pykrx 폴백은 데이터로더에서 락 직렬화)
META_FETCH_WORKERS = 8

# 배치 단위 공유 캐시 (메타 업데이트 1회 진입 시 1회만 빌드, 풀들 간 공유)
# `update_stock_reference_metadata` 진입 시 `_reset_batch_caches()` 로 초기화한다.
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

    # 한국 종목코드는 6자리(숫자 포함)다. 미국·호주 알파벳 티커(예: IOO, SCHD)는 국내 조회 대상이 아니므로
    # 네이버 API 를 호출하지 않는다(호출하면 비 XML 응답으로 파싱 실패 로그만 남는다).
    ticker_norm = str(ticker or "").strip().upper()
    if len(ticker_norm) != 6 or not any(ch.isdigit() for ch in ticker_norm):
        return None

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
_ETF_INFO_CACHE_TTL = timedelta(seconds=CACHE_TTL_META)


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


def _yf_dividend_yield_pct(info: dict[str, Any]) -> float | None:
    """yfinance .info 의 배당수익률을 '퍼센트' 단위로 정규화한다(화면 formatPercent 가 값을 그대로 %로 표기).

    ETF 는 ``yield`` 가 정확하다(소수. 예: SPMO 0.0065=0.65%). ``trailingAnnualDividendYield`` 는
    ETF 에서 0/누락이 잦아 뒤로 미룬다. ``dividendYield`` 는 버전에 따라 소수/퍼센트가 섞여 있어
    1 미만이면 소수로 보고 ×100 한다.
    """
    v = info.get("yield")  # ETF 배당수익률(소수)
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
        return round(float(v) * 100.0, 2)
    v = info.get("trailingAnnualDividendYield")
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
        return round(float(v) * 100.0, 2)
    v = info.get("dividendYield")
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
        v = float(v)
        return round(v * 100.0 if v < 1.0 else v, 2)
    return None


def _yf_expense_ratio_pct(info: dict[str, Any]) -> float | None:
    """yfinance .info 의 운용보수를 '퍼센트' 단위로 정규화한다.

    ETF 는 ``netExpenseRatio`` 가 이미 퍼센트(0.13=0.13%). 구버전 필드(annualReportExpenseRatio,
    feesExpensesTotal)는 소수라 1 미만이면 ×100 한다.
    """
    v = info.get("netExpenseRatio")
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
        return round(float(v), 4)
    for key in ("annualReportExpenseRatio", "feesExpensesTotal", "expenseRatio"):
        v = info.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
            v = float(v)
            return round(v * 100.0 if v < 1.0 else v, 4)
    return None


def _refresh_us_stock_meta_cache(
    ticker_type: str,
    ticker: str,
    name: str,
    naver_entry: dict[str, Any],
    *,
    is_etf: bool = False,
) -> None:
    """미국 종목 메타 캐시를 갱신한다. 네이버 미국 개별주 API 를 기본으로 하되, 네이버에 없는 종목
    (예: 미국 ETF ENFR 등)은 yfinance .info 로 배당수익률·순자산(시가총액)을 보완한다.

    ``is_etf`` 이면 구성종목(holdings)도 yfinance 로 수집해 저장한다(개별주는 구성종목이 없다).
    """
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
    }

    # yfinance .info 보완 — ETF 는 네이버 배당값이 부정확해(개별주 기준) yfinance 를 우선한다.
    # 개별주는 네이버 값이 있으면 유지하고 빈 값만 채운다. 보수·상장일은 네이버에 없어 항상 보완.
    needs_yf = (
        is_etf
        or meta_cache["dividend_yield_ttm"] is None
        or meta_cache["total_net_assets"] is None
        or meta_cache["listed_date"] is None
    )
    if needs_yf:
        try:
            info = getattr(yf.Ticker(ticker_norm), "info", {}) or {}
            if is_etf or meta_cache["dividend_yield_ttm"] is None:
                yf_yield = _yf_dividend_yield_pct(info)
                if yf_yield is not None:
                    meta_cache["dividend_yield_ttm"] = yf_yield
            if meta_cache["expense_ratio"] is None:
                meta_cache["expense_ratio"] = _yf_expense_ratio_pct(info)
            if meta_cache["total_net_assets"] is None:
                assets = info.get("totalAssets") or info.get("marketCap")  # ETF=순자산 / 개별주=시총
                if isinstance(assets, (int, float)) and not isinstance(assets, bool) and assets > 0:
                    meta_cache["total_net_assets"] = float(assets)
            if meta_cache["listed_date"] is None:
                epoch = info.get("firstTradeDateEpochUtc") or info.get("fundInceptionDate")
                if isinstance(epoch, (int, float)) and not isinstance(epoch, bool) and epoch > 0:
                    meta_cache["listed_date"] = datetime.fromtimestamp(float(epoch), tz=timezone.utc).strftime(
                        "%Y-%m-%d"
                    )
            if not meta_cache.get("market"):
                meta_cache["market"] = str(info.get("exchange") or "").strip().upper() or None
            if not meta_cache.get("industry"):
                meta_cache["industry"] = str(info.get("category") or info.get("industry") or "").strip() or None
            meta_cache["source"] = "naver_us_market_stock+yfinance"
        except Exception as exc:
            get_app_logger().debug(f"[{ticker_type_norm.upper()}/{ticker_norm}] yfinance 메타 보완 건너뜀: {exc}")

    # 미국 ETF 는 구성종목도 함께 저장한다(개별주는 holdings 개념이 없어 건너뜀).
    holdings_cache: dict[str, Any] | None = None
    if is_etf:
        holdings_info = fetch_yfinance_holdings(ticker_norm, is_australian=False)
        if holdings_info:
            holdings_cache = {
                "source": holdings_info["source"],
                "updated_at": holdings_info["fetched_at"],
                "reference_date": holdings_info["as_of_date"],
                "holdings_count": holdings_info["holdings_count"],
                "items": holdings_info["holdings"],
            }
        else:
            get_app_logger().warning(
                f"[{ticker_type_norm.upper()}/{ticker_norm}] 미국 ETF holdings 수집 실패 (메타만 저장)"
            )

    refresh_stock_cache(
        ticker_type_norm,
        ticker_norm,
        country_code="us",
        name=name_norm,
        meta_cache=meta_cache,
        holdings_cache=holdings_cache,
    )


def fetch_betashares_holdings(ticker: str) -> dict[str, Any] | None:
    """BetaShares 공식 홈페이지에서 portfolio holdings CSV를 긁어서 구성종목 딕셔너리를 반환합니다."""
    import csv
    import urllib.request
    from datetime import datetime

    logger = get_app_logger()
    # BetaShares CSV 는 접두사 없는 티커를 쓴다(ASX:ACDC → ACDC).
    ticker_clean = strip_asx_prefix(ticker)
    url = f"https://www.betashares.com.au/files/csv/{ticker_clean}_Portfolio_Holdings.csv"

    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        with urllib.request.urlopen(req, timeout=12) as response:
            content = response.read().decode("utf-8", errors="ignore")

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
            # 비 BetaShares ETF(예: Fidelity FEMX)는 CSV 형식이 없거나 달라 실패 — yfinance 폴백이 있으니 debug.
            logger.debug(f"[BetaShares] {ticker_clean} CSV 헤더 시작부(Ticker,)를 찾지 못했습니다.")
            return None

        # 데이터 파싱
        reader = csv.reader(lines[header_idx:])
        headers = next(reader)

        try:
            ticker_col = headers.index("Ticker")
            name_col = headers.index("Name")
            weight_col = headers.index("Weight (%)")
        except ValueError as e:
            logger.debug(f"[BetaShares] {ticker_clean} CSV 필수 열 매핑 실패: {e}")
            return None

        # 상장 통화/국가/자산군 열은 CSV 에 있을 때만 읽는다(없으면 해당 항목을 비워 둔다).
        currency_col = headers.index("Currency") if "Currency" in headers else None
        country_col = headers.index("Country") if "Country" in headers else None
        asset_class_col = headers.index("Asset Class") if "Asset Class" in headers else None

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

            # 티커만 남기면 상장 국가를 알 수 없어 호주 종목이 미국으로 분류된다.
            # CSV 가 알려주는 상장 통화/국가를 그대로 보존한다(추정하지 않는다).
            currency_val = ""
            if currency_col is not None and len(row) > currency_col:
                currency_val = str(row[currency_col]).strip().upper()
            country_val = ""
            if country_col is not None and len(row) > country_col:
                country_val = str(row[country_col]).strip()
            asset_class_val = ""
            if asset_class_col is not None and len(row) > asset_class_col:
                asset_class_val = str(row[asset_class_col]).strip()

            # 현금 포지션(`AUD - AUSTRALIA DOLLAR`)은 상장 종목이 아니므로 ASX: 를 붙이지 않는다.
            is_cash = asset_class_val.lower() == "cash"
            # 호주 상장 종목은 시스템 표준 표기(ASX:NAB)로 저장한다.
            item: dict[str, Any] = {
                "ticker": ensure_asx_prefix(t_clean) if (currency_val == "AUD" and not is_cash) else t_clean,
                "name": name_val,
                "weight": weight_val,
            }
            if asset_class_val:
                item["asset_class"] = asset_class_val
            if currency_val:
                item["listing_currency"] = currency_val
            if country_val:
                item["listing_country"] = country_val
            items.append(item)

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
        # 모든 호주 ETF에 BetaShares 를 먼저 시도하므로 비 BetaShares(404 등)는 흔하다 — yfinance 폴백이 있어 debug.
        logger.debug(f"[BetaShares] {ticker_clean} CSV 수집 실패: {exc}")
        return None


def fetch_yfinance_holdings(ticker: str, is_australian: bool = False) -> dict[str, Any] | None:
    """yfinance를 활용해 Top 10 구성종목 정보를 가져옵니다."""
    from datetime import datetime

    import yfinance as yf

    logger = get_app_logger()
    # 시스템 표준 티커는 ASX: 접두사를 달고 다니므로 외부 조회 직전에 벗긴다.
    symbol = to_yahoo_symbol(ticker) if is_australian else normalize_ticker(ticker)
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

            # 거래소 접미사가 붙은 원본 심볼(NAB.AX)이 상장 시장을 알려준다.
            # 호주 상장이면 시스템 표준 표기(ASX:NAB)로 저장한다.
            asx_ticker = from_yahoo_symbol(t_code)
            items.append(
                {
                    "ticker": asx_ticker or t_clean,
                    "name": name_val,
                    "weight": weight_val,
                    "yahoo_symbol": t_code,
                }
            )

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
        # 개별 소스 실패는 debug. 모든 소스 실패 시 호출부에서 "holdings 수집 실패(건너뜀)" 로 한 번 경고한다.
        logger.debug(f"[yfinance] {symbol} 수집 실패: {exc}")
        return None


def _refresh_overseas_etf_meta_cache(
    ticker_type: str,
    ticker: str,
    name: str,
    country_code: str,
) -> None:
    """호주/미국 등 해외 ETF 메타와 holdings 캐시를 수집 및 저장한다."""
    from datetime import datetime

    logger = get_app_logger()
    ticker_type_norm = str(ticker_type or "").strip().lower()
    ticker_norm = str(ticker or "").strip().upper()
    name_norm = str(name or "").strip() or ticker_norm
    country_norm = str(country_code or "").strip().lower()

    holdings_info = None

    # 호주 ETF는 발행사 공식 소스(BetaShares CSV → Vanguard API)를 먼저 쓰고
    # 둘 다 없을 때만 yfinance 로 폴백한다. 공식 소스가 구성종목 국가까지 알려준다.
    if country_norm == "au":
        holdings_info = fetch_betashares_holdings(ticker_norm)
        if not holdings_info:
            holdings_info = fetch_vanguard_au_holdings(ticker_norm)
        if not holdings_info:
            holdings_info = fetch_yfinance_holdings(ticker_norm, is_australian=True)
    else:
        holdings_info = fetch_yfinance_holdings(ticker_norm, is_australian=False)

    # 구성종목 수집이 실패해도 배당·상장일·순자산 등 나머지 메타는 저장한다.
    # (여기서 return 하면 캐시 문서 자체가 만들어지지 않아 화면이 통째로 빈다.)
    holdings_cache = None
    if holdings_info:
        holdings_cache = {
            "source": holdings_info["source"],
            "updated_at": holdings_info["fetched_at"],
            "reference_date": holdings_info["as_of_date"],
            "holdings_count": holdings_info["holdings_count"],
            "items": holdings_info["holdings"],
        }
    else:
        logger.warning(f"[{ticker_type_norm.upper()}/{ticker_norm}] 해외 ETF holdings 수집 실패 (메타만 저장)")

    meta_cache = {
        "source": "overseas_etf_meta",
        "updated_at": datetime.now().isoformat(),
        "listed_date": None,
        "dividend_yield_ttm": None,
        "expense_ratio": None,
        "total_net_assets": None,
        "issue_name": name_norm,
    }

    # yfinance를 통해 추가적인 ETF 메타 정보 보완
    try:
        import yfinance as yf

        symbol = to_yahoo_symbol(ticker_norm) if country_norm == "au" else ticker_norm
        t_data = yf.Ticker(symbol)
        t_info = getattr(t_data, "info", {})
        if t_info:
            # 배당/보수는 미국 경로와 같은 정규화 함수를 쓴다(둘 다 '퍼센트' 단위로 저장 — 화면 표기 규칙 동일).
            meta_cache["dividend_yield_ttm"] = _yf_dividend_yield_pct(t_info)
            meta_cache["expense_ratio"] = _yf_expense_ratio_pct(t_info)
            meta_cache["total_net_assets"] = t_info.get("totalAssets") or t_info.get("marketCap")
            epoch = t_info.get("firstTradeDateEpochUtc") or t_info.get("fundInceptionDate")
            if isinstance(epoch, (int, float)) and not isinstance(epoch, bool) and epoch > 0:
                meta_cache["listed_date"] = datetime.fromtimestamp(float(epoch), tz=timezone.utc).strftime("%Y-%m-%d")
            elif t_info.get("startDate"):
                meta_cache["listed_date"] = _format_iso_date(t_info.get("startDate"))
            if not meta_cache.get("market"):
                meta_cache["market"] = str(t_info.get("exchange") or "").strip().upper() or None
            if not meta_cache.get("industry"):
                meta_cache["industry"] = str(t_info.get("category") or "").strip() or None
    except Exception as e:
        logger.debug(f"yfinance 추가 메타데이터 수집 건너뜀: {e}")

    # yfinance 는 호주 ETF 의 운용보수를 제공하지 않는다. Vanguard 상품이면 공식 API 로 채운다.
    if country_norm == "au" and meta_cache.get("expense_ratio") is None:
        vanguard_expense_ratio = fetch_vanguard_au_expense_ratio_pct(ticker_norm)
        if vanguard_expense_ratio is not None:
            meta_cache["expense_ratio"] = vanguard_expense_ratio

    refresh_stock_cache(
        ticker_type_norm,
        ticker_norm,
        country_code=country_norm,
        name=name_norm,
        meta_cache=meta_cache,
        holdings_cache=holdings_cache,
    )
    if holdings_info:
        logger.info(
            f"[{ticker_type_norm.upper()}/{ticker_norm}] 해외 ETF 구성종목 캐시 갱신 완료 ({holdings_info['holdings_count']}개 종목)"
        )
    else:
        logger.info(f"[{ticker_type_norm.upper()}/{ticker_norm}] 해외 ETF 메타 캐시 갱신 완료 (구성종목 없음)")


def _load_ticker_entries(type_norm: str) -> tuple[str, list[dict[str, Any]]] | None:
    """종목타입의 국가코드와 (삭제 포함) 전체 종목 목록을 로드한다. 없으면 None."""
    logger = get_app_logger()
    try:
        settings = get_ticker_type_settings(type_norm)
        country_code = str(settings.get("country_code") or "").strip().lower()
    except Exception as e:
        logger.error(f"종목타입 설정을 로드할 수 없습니다 ({type_norm}): {e}")
        return None

    stock_data = get_all_etfs_including_deleted(type_norm)
    if not stock_data:
        logger.warning(f"'{type_norm}' 종목타입의 종목 데이터가 비어있습니다.")
        return None
    return country_code, list(stock_data)


def _update_reference_meta_for_type(
    type_norm: str, progress_callback: Callable[[int, int, str], None] | None = None
) -> list[tuple[str, str]]:
    """배치 B(식별·상세 메타) — 이름/상장일/마켓/업종 + ETF 상세 캐시(holdings/배당/ETFBase).

    반환: 이 풀의 실패 목록 [(ticker, "단계: 사유")] — 상위에서 모아 슬랙으로 보고한다.
    """
    logger = get_app_logger()
    loaded = _load_ticker_entries(type_norm)
    if loaded is None:
        return []
    country_code, ticker_entries = loaded
    total_count = len(ticker_entries)
    logger.info(f"[{type_norm.upper()}] 식별·상세 메타 업데이트 시작 (총 {total_count}개 종목)")
    if progress_callback:
        progress_callback(0, total_count, "데이터 준비 중...")

    # [KOR] 전체 종목(일반주/ETF) 맵 구성하여 루프 내 호출 최소화
    naver_etf_map: dict[str, str] = {}
    if country_code == "kor":
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
            logger.info(f"[{type_norm.upper()}] 기존 메타 캐시 문서 {len(existing_meta_cache_map)}건 로드 (TTL 판정용)")
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

    def _process_one_stock(stock: dict[str, Any]) -> tuple[str, str, dict[str, Any] | None, list[str]]:
        """단일 종목 처리 (워커 스레드). 반환: (ticker, name, update_doc|None, 실패목록).

        실패 목록 항목은 "단계: 사유" 형식 — 배치 끝에 모아 슬랙으로 보고한다.
        공유 맵(naver_*_map, existing_meta_cache_map)은 루프 전에 빌드 완료된 읽기 전용이다.
        """
        ticker = str(stock.get("ticker") or "")
        failures: list[str] = []
        try:
            update_single_stock_metadata(
                stock,
                country_code,
                naver_etf_map,
                type_norm,
                naver_us_stock_map=naver_us_stock_map,
            )
        except Exception as e:
            failures.append(f"식별 메타: {str(e)[:80]}")
            return ticker, str(stock.get("name") or "-"), None, failures

        name = str(stock.get("name") or "-")
        # 저장 필드 = 식별 필드만(가격지표는 배치 A 담당)
        update_doc: dict[str, Any] = {"ticker": ticker}
        for f in (
            "name",
            "listing_date",
            "market",
            "is_etf",
            "etf_category",
            "dividend_yield_ttm",
            "market_cap",
            "sector",
            "industry",
        ):
            if f in stock:
                update_doc[f] = stock[f]

        # ETF 상세 캐시 갱신(holdings/배당 등).
        # ETF 가 아닌 국내 개별주(예: KOSDAQ 다우데이타)는 ETF 상세 API 가 404 이므로 호출 자체를 건너뛴다.
        if country_code == "kor" and stock.get("is_etf"):
            try:
                existing_doc = existing_meta_cache_map.get(ticker.strip().upper())
                _refresh_korean_etf_meta_cache(type_norm, ticker, name, existing_cache_doc=existing_doc)
            except Exception as e:
                logger.warning(f"[{type_norm.upper()}/{ticker}] ETF 상세 캐시 갱신 건너뜀: {e}")
                failures.append(f"ETF 상세: {str(e)[:80]}")
        elif country_code == "au":
            try:
                _refresh_overseas_etf_meta_cache(type_norm, ticker, name, country_code)
            except Exception as e:
                logger.warning(f"[{type_norm.upper()}/{ticker}] 호주 ETF 상세 캐시 갱신 실패: {e}")
                failures.append(f"호주 상세: {str(e)[:80]}")
        elif country_code == "us":
            # 네이버에 없어도(미국 ETF 등) yfinance 로 배당·순자산을 보완하도록 항상 호출.
            naver_entry = naver_us_stock_map.get(ticker.strip().upper(), {})
            try:
                _refresh_us_stock_meta_cache(type_norm, ticker, name, naver_entry, is_etf=bool(stock.get("is_etf")))
            except Exception as e:
                logger.warning(f"[{type_norm.upper()}/{ticker}] 미국 메타 캐시 갱신 건너뜀: {e}")
                failures.append(f"미국 메타: {str(e)[:80]}")

        return ticker, name, update_doc, failures

    # 종목 처리 병렬 실행 — 종목당 작업이 HTTP 대기 위주라 워커 수만큼 단축된다.
    # DB 저장·진행 로그·progress_callback 은 메인 스레드에서만 수행한다.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    valid_entries = [s for s in ticker_entries if s.get("ticker")]
    updates_for_db: list[dict[str, Any]] = []
    pool_failures: list[tuple[str, str]] = []  # (ticker, "단계: 사유")
    done_count = 0

    def _handle_result(ticker: str, name: str, update_doc: dict[str, Any] | None, failures: list[str]) -> None:
        nonlocal done_count
        done_count += 1
        for failure in failures:
            pool_failures.append((ticker, failure))
        if update_doc is None:
            logger.error(f"[{type_norm.upper()}/{ticker}] 식별 메타 업데이트 실패: {failures}")
        else:
            logger.info(f"  -> 식별 메타 획득: {done_count}/{total_count} - {name}({ticker})")
            updates_for_db.append(update_doc)
        if len(updates_for_db) >= 100:
            try:
                modified = bulk_update_stocks(type_norm, updates_for_db)
                logger.info(f"[{type_norm.upper()}] 식별 메타 중간 저장 ({done_count}/{total_count}, {modified}건)")
                updates_for_db.clear()
            except Exception as e:
                logger.error(f"[{type_norm.upper()}] 중간 저장 실패: {e}")
                pool_failures.append(("(일괄)", f"중간 저장: {str(e)[:80]}"))
        if progress_callback:
            progress_callback(done_count, total_count, f"{name}({ticker})")

    if META_FETCH_WORKERS <= 1:
        for stock in valid_entries:
            _handle_result(*_process_one_stock(stock))
    else:
        with ThreadPoolExecutor(max_workers=META_FETCH_WORKERS) as executor:
            futures = [executor.submit(_process_one_stock, stock) for stock in valid_entries]
            for future in as_completed(futures):
                _handle_result(*future.result())

    try:
        if updates_for_db:
            modified = bulk_update_stocks(type_norm, updates_for_db)
            logger.info(f"[{type_norm.upper()}] 식별 메타 최종 저장 완료 ({modified}건)")
    except Exception as e:
        logger.error(f"'{type_norm}' 식별 메타 최종 저장 실패: {e}")
        pool_failures.append(("(일괄)", f"최종 저장: {str(e)[:80]}"))

    return pool_failures


def _update_price_metrics_for_type(type_norm: str, progress_callback: Callable[[int, int, str], None] | None = None):
    """배치 A(가격 지표) — OHLCV 1회 로드로 거래량·기간수익률만 갱신."""
    logger = get_app_logger()
    loaded = _load_ticker_entries(type_norm)
    if loaded is None:
        return
    country_code, ticker_entries = loaded
    total_count = len(ticker_entries)
    logger.info(f"[{type_norm.upper()}] 가격 지표 업데이트 시작 (총 {total_count}개 종목)")
    if progress_callback:
        progress_callback(0, total_count, "데이터 준비 중...")

    updates_for_db: list[dict[str, Any]] = []
    for idx, stock in enumerate(ticker_entries, start=1):
        ticker = stock.get("ticker")
        if not ticker:
            continue
        try:
            df = fetch_ohlcv(ticker, country=country_code, months_back=60, ticker_type=type_norm)
            price_metrics = compute_price_metrics(df)
            if not price_metrics:
                continue
            updates_for_db.append({"ticker": ticker, **price_metrics})

            if len(updates_for_db) >= 100:
                try:
                    modified = bulk_update_stocks(type_norm, updates_for_db)
                    logger.info(f"[{type_norm.upper()}] 가격 지표 중간 저장 ({idx}/{total_count}, {modified}건)")
                    updates_for_db.clear()
                except Exception as e:
                    logger.error(f"[{type_norm.upper()}] 중간 저장 실패: {e}")

            if progress_callback:
                progress_callback(idx, total_count, str(ticker))
        except Exception as e:
            logger.error(f"[{type_norm.upper()}/{ticker}] 가격 지표 업데이트 실패: {e}")

    try:
        if updates_for_db:
            modified = bulk_update_stocks(type_norm, updates_for_db)
            logger.info(f"[{type_norm.upper()}] 가격 지표 최종 저장 완료 ({modified}건)")
    except Exception as e:
        logger.error(f"'{type_norm}' 가격 지표 최종 저장 실패: {e}")


def _resolve_ticker_types_to_update(ticker_type: str | None) -> list[str] | None:
    """대상 종목타입 목록을 확정한다. 잘못된 타입이면 None(호출부에서 중단)."""
    logger = get_app_logger()
    available_ticker_types = list_available_ticker_types()
    if ticker_type:
        type_norm = ticker_type.strip().lower()
        if type_norm not in available_ticker_types:
            logger.error(f"대상 종목타입 '{ticker_type}'를 찾을 수 없습니다.")
            return None
        return [type_norm]
    return available_ticker_types.copy()


def update_stock_reference_metadata(ticker_type: str | None = None):
    """배치 B — 식별·상세 메타(이름/상장일/마켓/업종 + ETF holdings·배당) + KIS 마스터 캐시."""
    logger = get_app_logger()
    _reset_batch_caches()  # 풀 간 공유 네이버 이름 맵을 1회만 빌드

    targets = _resolve_ticker_types_to_update(ticker_type)
    if targets is None:
        return

    if not ticker_type:
        try:
            logger.info("KIS 국내 ETF 마스터 캐시 갱신을 시작합니다.")
            refreshed_count = refresh_kis_domestic_etf_master_cache()
            logger.info("KIS 국내 ETF 마스터 캐시 갱신 완료: %d건", refreshed_count)
        except Exception as exc:
            logger.error("KIS 국내 ETF 마스터 캐시 갱신 실패: %s", exc)

    logger.info(f"[배치 B] 식별·상세 메타 대상 종목타입: {targets}")
    failures_by_pool: dict[str, list[tuple[str, str]]] = {}
    for type_norm in targets:
        pool_failures = _update_reference_meta_for_type(type_norm)
        if pool_failures:
            failures_by_pool[type_norm] = pool_failures
    logger.info("[배치 B] 식별·상세 메타 업데이트 완료.")

    # 시총 순위 — 종목별 갱신이 meta_cache 를 통째로 덮으므로 반드시 그 뒤에 적는다.
    try:
        from utils.market_cap_rank import update_market_cap_ranks

        update_market_cap_ranks(targets)
    except Exception as exc:
        logger.error("[배치 B] 시총 순위 기록 실패: %s", exc)

    # 부분 실패 슬랙 보고 — 어떤 풀/종목/단계에서 실패했는지 명시한다.
    # (배치 자체는 계속 진행되므로 run_batch 의 exit!=0 실패 알림으로는 잡히지 않는 케이스)
    if failures_by_pool:
        total_failures = sum(len(v) for v in failures_by_pool.values())
        lines = [f"⚠️ [배치 B] 종목 메타 업데이트 부분 실패: {total_failures}건"]
        shown = 0
        for pool, pool_failures in failures_by_pool.items():
            for ticker, reason in pool_failures:
                if shown >= 15:
                    break
                lines.append(f"• [{pool.upper()}/{ticker}] {reason}")
                shown += 1
            if shown >= 15:
                break
        if total_failures > shown:
            lines.append(f"… 외 {total_failures - shown}건 — 상세는 logs/cron/reference_meta_updater.log")
        try:
            from utils.notification import send_slack_message_v2

            send_slack_message_v2("\n".join(lines))
        except Exception as exc:
            logger.error(f"[배치 B] 부분 실패 슬랙 보고 실패: {exc}")


def update_stock_price_metrics(ticker_type: str | None = None):
    """배치 A — 가격 지표(거래량·기간수익률)만 갱신. OHLCV 캐시만 사용."""
    logger = get_app_logger()
    targets = _resolve_ticker_types_to_update(ticker_type)
    if targets is None:
        return
    logger.info(f"[배치 A] 가격 지표 대상 종목타입: {targets}")
    for type_norm in targets:
        _update_price_metrics_for_type(type_norm)
    logger.info("[배치 A] 가격 지표 업데이트 완료.")


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

            # 웹(종목 추가 검증) 경로라 락으로 직렬화한다 — 배치 워커는 이 함수를 쓰지 않는다.
            from utils.yfinance_guard import yfinance_lock

            with yfinance_lock():
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
            _refresh_overseas_etf_meta_cache(
                type_norm, ticker_norm, str(stock.get("name") or ticker_norm), country_code
            )
        except Exception as meta_cache_error:
            logger.error(f"[{type_norm.upper()}/{ticker_norm}] 호주 ETF 상세 캐시 갱신 실패: {meta_cache_error}")
    elif country_code == "us":
        # 네이버에 없어도(미국 ETF 등) yfinance 로 배당·순자산을 보완하도록 항상 호출.
        naver_entry = naver_us_stock_map.get(ticker_norm, {})
        try:
            _refresh_us_stock_meta_cache(
                type_norm,
                ticker_norm,
                str(stock.get("name") or ticker_norm),
                naver_entry,
                is_etf=bool(stock.get("is_etf")),
            )
        except Exception as meta_cache_error:
            logger.error(f"[{type_norm.upper()}/{ticker_norm}] 미국 메타 캐시 갱신 실패: {meta_cache_error}")

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
        "sector",
        "industry",
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
    elif country_code == "au":
        yfinance_ticker = to_yahoo_symbol(ticker)
    else:
        yfinance_ticker = ticker

    listing_date_str = None

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

        # 4. 업종 — 국내 개별주는 네이버에서 받는다(한국어 원본, 번역 불필요).
        # yfinance 는 국내 종목 셋 중 하나에 분류가 없어서 그만큼 업종 상한에서 빠졌다.
        # 값이 이미 있으면 건너뛴다 — 최초 1회만 무겁고 이후엔 신규 종목만 조회한다.
        # 빈 문자열은 '미설정'이라 그대로 둔다(임의 값으로 메우면 업종 상한이 엉뚱하게 묶인다).
        if not stock.get("is_etf") and not stock.get("industry"):
            from services.naver_industry_service import fetch_industry

            industry = fetch_industry(ticker)
            if industry:
                stock["industry"] = industry
                if account_norm:
                    logger.debug(f"[{account_norm.upper()}/{ticker}] 업종 획득: {industry}")

    elif country_code in ("us", "au"):
        # 호주 풀은 ETF 전용. 미국은 종목 자체 속성(yfinance quoteType)으로 판정한다 — 아래 .info 블록에서 설정.
        # (어느 풀에 넣었는지로 추정하면 같은 종목이 풀마다 다른 값을 갖게 되므로 쓰지 않는다.)
        if country_code == "au":
            stock["is_etf"] = True

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

            # 미국: 종목 자체 속성으로 ETF 판정(quoteType == "ETF"). 이름 보완도 같은 .info 로 처리.
            need_name = not stock.get("name") or stock.get("name") == ticker
            if country_code == "us" or need_name:
                try:
                    info = t.info
                    if country_code == "us":
                        is_etf = str(info.get("quoteType") or "").strip().upper() == "ETF"
                        if not is_etf:
                            # Yahoo 가 신생 ETF 를 EQUITY 로 잘못 분류하는 경우가 있다
                            # (NYSX 'Global X NYSE 100 ETF', 2026-03 상장에서 확인).
                            # 이름이 'ETF' 로 끝나면 이름을 믿는다 — ETN 은 'ETN' 으로 끝나 안 걸린다.
                            yahoo_name = str(
                                info.get("longName") or info.get("shortName") or stock.get("name") or ""
                            ).strip()
                            if yahoo_name.upper().endswith("ETF"):
                                is_etf = True
                                logger.info(
                                    f"[{account_norm.upper()}/{ticker}] quoteType={info.get('quoteType')} 이지만 "
                                    f"종목명({yahoo_name})이 ETF 로 끝나 ETF 로 판정합니다."
                                )
                        stock["is_etf"] = is_etf
                    if need_name:
                        fetched_name = info.get("longName") or info.get("shortName")
                        if fetched_name:
                            stock["name"] = fetched_name
                            if account_norm:
                                logger.debug(f"[{account_norm.upper()}/{ticker}] 종목명 업데이트: {fetched_name}")
                except Exception as e:
                    logger.warning(f"[{account_norm.upper()}/{ticker}] yfinance 정보 조회 실패: {e}")

            # 상장일이 이미 있으면 history 호출 자체를 생략
            if not stock.get("listing_date"):
                try:
                    hist = t.history(period="max", auto_adjust=True)
                    if hist is not None and not hist.empty:
                        first_date = hist.index.min()
                        listing_date_str = first_date.strftime("%Y-%m-%d")
                        if account_norm:
                            logger.debug(f"[{account_norm.upper()}/{ticker}] yfinance 상장일 획득: {listing_date_str}")
                except Exception as e:
                    logger.warning(f"[{account_norm.upper()}/{ticker}] yfinance history 조회 실패: {e}")
        except Exception as e:
            logger.warning(f"[{account_norm.upper()}/{ticker}] yfinance 메타데이터 조회 조회 실패: {e}")

    if not listing_date_str:
        listing_date_str = stock.get("listing_date")

    if listing_date_str:
        stock["listing_date"] = listing_date_str

    # 가격 파생 지표(거래량·기간수익률)는 이 함수에서 계산하지 않는다.
    # 호출부가 OHLCV 를 1회 로드해 compute_price_metrics() 로 계산·병합한다(가격지표 배치와 공유).
    stock.pop("1_month_avg_volume", None)
    stock.pop("1_week_avg_turnover", None)
    stock.pop("1_month_avg_turnover", None)
