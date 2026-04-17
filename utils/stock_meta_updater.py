"""계좌 종목 메타데이터를 업데이트합니다."""

import time
import xml.etree.ElementTree as ET
from typing import Any

import pandas as pd
import requests
import yfinance as yf

from services.etf_holdings_service import fetch_korean_etf_holdings_from_naver
from services.etf_meta_service import fetch_korean_etf_info_from_naver
from services.stock_cache_service import refresh_stock_cache
from utils.data_loader import fetch_ohlcv, fetch_pykrx_market, fetch_pykrx_name
from utils.kis_market import refresh_kis_domestic_etf_master_cache
from config import (
    NAVER_ETF_CATEGORY_CONFIG,
    NAVER_ETF_CATEGORY_HEADERS,
    NAVER_ETF_DOMESTIC_URL,
    NAVER_ETF_THEMES_URL,
    NAVER_STOCK_MARKET_VALUE_HEADERS,
    NAVER_STOCK_MARKET_VALUE_URL,
)
from utils.logger import get_app_logger
from utils.settings_loader import get_ticker_type_settings, list_available_ticker_types


def fetch_naver_kor_stock_map() -> dict[str, dict[str, str]]:
    """
    네이버 marketValue API를 사용하여 모든 한국 종목(KOSPI, KOSDAQ)의 {티커: {이름, 마켓}} 맵을 생성한다.
    ETN은 제외한다.
    """
    logger = get_app_logger()
    stock_map = {}

    for market in ["KOSPI", "KOSDAQ"]:
        page = 1
        page_size = 100
        while True:
            try:
                url = NAVER_STOCK_MARKET_VALUE_URL.format(market=market)
                resp = requests.get(
                    url,
                    params={"page": page, "pageSize": page_size},
                    headers=NAVER_STOCK_MARKET_VALUE_HEADERS,
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.warning(f"[Naver 벌크] {market} {page}페이지 조회 실패: {exc}")
                break

            stocks = data.get("stocks") or []
            if not stocks:
                break

            for s in stocks:
                ticker = str(s.get("itemCode") or "").strip().upper()
                name = str(s.get("stockName") or "").strip()
                stock_end_type = str(s.get("stockEndType") or "").lower()

                if not ticker or not name:
                    continue
                
                # ETN 제외 (stockEndType이 'etn'인 경우 제외)
                if stock_end_type == "etn":
                    continue

                stock_map[ticker] = {"name": name, "market": market}

            if len(stocks) < page_size:
                break
            page += 1
            time.sleep(0.05)

    logger.info(f"[Naver 벌크] 총 {len(stock_map)}개 한국 종목 정보 맵 생성 완료")
    return stock_map


def _build_naver_category_map() -> dict[str, dict[str, str]]:
    """
    네이버 ETF 카테고리(투자국가/섹터/지수) 테마별 티커 목록을 조회해
    {ticker: {"category_country": str, "category_sector": str, "category_index": str}} 형태의
    역인덱스 맵을 구성한다.

    - 대분류 0201 (투자국가), 0401 (섹터), 0501 (지수)만 대상으로 한다.
    - 동일 대분류 내 여러 중분류에 해당될 경우 첫 번째 값만 보존한다.
    """

    logger = get_app_logger()
    ticker_categories: dict[str, dict[str, str]] = {}

    try:
        themes_response = requests.get(
            NAVER_ETF_THEMES_URL,
            headers=NAVER_ETF_CATEGORY_HEADERS,
            timeout=10,
        )
        themes_response.raise_for_status()
        themes_payload = themes_response.json()
    except Exception as exc:
        logger.warning(f"[Naver 카테고리] 테마 목록 조회 실패: {exc}")
        return ticker_categories

    # API 응답은 리스트 형태임
    large_categories: list[dict[str, Any]] = []
    if isinstance(themes_payload, list):
        large_categories = themes_payload
    elif isinstance(themes_payload, dict):
        # 만약 dict 구조일 경우를 대비한 fallback (upperCategories 등)
        large_categories = list(themes_payload.get("upperCategories") or themes_payload.get("largeCategories") or [])

    # 모든 대분류 코드 목록 (0101~0803 전체)
    all_codes = {c["code"] for c in NAVER_ETF_CATEGORY_CONFIG}
    # 대표 분류(best) 결정 시 사용할 대분류 코드 세트 (use가 True인 것만)
    use_codes = {c["code"] for c in NAVER_ETF_CATEGORY_CONFIG if c.get("use")}

    # 티커별로 (최대 활성 코드, 중분류 이름, 대분류별 맵) 저장
    ticker_cat_info: dict[str, dict[str, Any]] = {}

    for large in large_categories:
        large_code = str(large.get("largeCategoryCode") or large.get("upperCategoryCode") or "").strip()
        if large_code not in all_codes:
            continue

        middle_categories = large.get("middleCategories") or []
        for middle in middle_categories:
            middle_code = str(middle.get("code") or middle.get("middleCategoryCode") or "").strip()
            middle_name = str(middle.get("name") or middle.get("middleCategoryName") or "").strip()
            if not middle_code or not middle_name:
                continue

            # 중분류별 ETF 목록 조회 (페이지네이션)
            idx = 0
            page_size = 100
            while True:
                try:
                    resp = requests.get(
                        NAVER_ETF_DOMESTIC_URL,
                        params={
                            "listingType": "aumDesc",
                            "size": page_size,
                            "index": idx,
                            "middleCategoryCode": middle_code,
                        },
                        headers=NAVER_ETF_CATEGORY_HEADERS,
                        timeout=10,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as exc:
                    logger.warning(
                        f"[Naver 카테고리] 중분류 조회 실패(large={large_code}, mid={middle_code}): {exc}"
                    )
                    break

                items = []
                if isinstance(data, dict):
                    items = list(data.get("etfs") or data.get("items") or [])

                if not items:
                    break

                for item in items:
                    ticker_raw = (
                        item.get("itemCode")
                        or item.get("symbolCode")
                        or item.get("ticker")
                    )
                    ticker_norm = str(ticker_raw or "").strip().upper()
                    if not ticker_norm:
                        continue

                    info = ticker_cat_info.setdefault(ticker_norm, {
                        "best_code": "",
                        "best_name": "",
                        "details": {cat["code"]: "" for cat in NAVER_ETF_CATEGORY_CONFIG}
                    })

                    # 1. 상세 맵 업데이트 (대분류별 중분류명 보존)
                    # 값이 여러 개인 경우 가장 처음 발견된 값만 사용
                    if not info["details"].get(large_code):
                        info["details"][large_code] = middle_name

                    # 2. 대표 분류(best) 업데이트 (use가 True인 코드 중 코드가 클수록 우선순위 높음)
                    if large_code in use_codes:
                        if large_code > info["best_code"]:
                            info["best_code"] = large_code
                            info["best_name"] = middle_name

                if len(items) < page_size:
                    break
                idx += page_size
                time.sleep(0.05)  # 과도한 호출 방지

    # 최종 결과 구성
    final_map = {}
    for ticker, info in ticker_cat_info.items():
        final_map[ticker] = {
            "best": info["best_name"],
            "details": info["details"]
        }

    logger.info(
        f"[Naver 카테고리] 역인덱스 구성 완료: 티커 {len(final_map)}개 "
        f"(상세 분류 필드 생성 및 우선순위 단일화 병행)"
    )
    return final_map

    logger.info(
        f"[Naver 카테고리] 역인덱스 구성 완료: 티커 {len(final_map)}개 "
        f"(전체 분류 필드 생성 및 우선순위 단일화 병행)"
    )
    return final_map


def fetch_naver_etf_names_map() -> dict[str, str]:
    """
    네이버 ETF API를 호출하여 전체 ETF 종목의 {코드: 이름} 맵을 반환합니다.
    """
    from config import NAVER_FINANCE_ETF_API_URL, NAVER_FINANCE_HEADERS

    url = NAVER_FINANCE_ETF_API_URL
    names_map = {}

    try:
        response = requests.get(url, headers=NAVER_FINANCE_HEADERS, timeout=5)
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
        response = requests.get(url, timeout=5)
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


def _refresh_korean_etf_meta_cache(ticker_type: str, ticker: str, name: str, category_data: dict[str, Any] | None = None) -> None:
    """한국 ETF 메타/구성종목 캐시를 네이버 기준으로 함께 갱신한다."""
    ticker_type_norm = str(ticker_type or "").strip().lower()
    ticker_norm = str(ticker or "").strip().upper()
    name_norm = str(name or "").strip() or ticker_norm
    if not ticker_type_norm or not ticker_norm:
        raise ValueError("ticker_type과 ticker가 필요합니다.")

    etf_info = fetch_korean_etf_info_from_naver(ticker_norm)
    holdings_info = fetch_korean_etf_holdings_from_naver(ticker_norm)
    meta_cache = {
        "source": str(etf_info.get("source") or "naver_etf_meta"),
        "updated_at": str(etf_info.get("fetched_at") or ""),
        "reference_date": etf_info.get("reference_date"),
        "listed_date": etf_info.get("listed_date"),
        "dividend_yield_ttm": etf_info.get("dividend_yield_ttm"),
        "dividend_per_share_ttm": etf_info.get("dividend_per_share_ttm"),
        "recent_ex_dividend_at": etf_info.get("recent_ex_dividend_at"),
        "expense_ratio": etf_info.get("expense_ratio"),
        "total_net_assets": etf_info.get("total_net_assets"),
        "issue_name": etf_info.get("issue_name"),
        "base_index": etf_info.get("base_index"),
    }
    # 추가 카테고리 정보가 있으면 업데이트
    if category_data:
        for k, v in category_data.items():
            if k.startswith("cat_"):
                meta_cache[k] = v

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
        meta_cache=meta_cache,
        holdings_cache=holdings_cache,
    )


def update_ticker_type_metadata(
    ticker_type: str, progress_callback: Callable[[int, int, str], None] | None = None
):
    """지정된 종목타입의 모든 종목 메타데이터를 업데이트합니다."""
    logger = get_app_logger()
    type_norm = (ticker_type or "").strip().lower()

    try:
        settings = get_ticker_type_settings(type_norm)
        country_code = str(settings.get("country_code") or "").strip().lower()
        type_source = str(settings.get("type_source") or "").strip()
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
    naver_etf_map = {}
    naver_kor_stock_map = {}
    if country_code == "kor":
        logger.info("네이버 API에서 전체 종목 정보들을 수집합니다...")
        naver_etf_map = fetch_naver_etf_names_map()
        naver_kor_stock_map = fetch_naver_kor_stock_map()

    # type_source == "Naver" 인 종목풀만 카테고리(투자국가/섹터/지수) 역인덱스 맵 구성
    naver_category_map: dict[str, dict[str, str]] = {}
    if type_source.lower() == "naver":
        logger.info(f"[{type_norm.upper()}] 네이버 ETF 카테고리 맵을 구성합니다...")
        naver_category_map = _build_naver_category_map()

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
                naver_category_map=naver_category_map,
                naver_kor_stock_map=naver_kor_stock_map,
            )

            name = stock.get("name") or "-"
            logger.info(f"  -> 메타데이터 획득 중: {idx}/{total_count} - {name}({ticker})")

            # 저장할 필드들을 딕셔너리로 구성
            update_doc = {"ticker": ticker}

            # 메타데이터 업데이트 시 갱신되는 주요 필드 지정
            fields_to_update = [
                "name",
                "listing_date",
                "market",
                "1_week_avg_volume",
                "volume",
                "1_week_earn_rate",
                "2_week_earn_rate",
                "1_month_earn_rate",
                "3_month_earn_rate",
                "6_month_earn_rate",
                "12_month_earn_rate",
                "etf_category",
            ]
            # 개별 분류 컬럼들을 업데이트 필드에 추가
            for cat in NAVER_ETF_CATEGORY_CONFIG:
                fields_to_update.append(f"cat_{cat['code']}")

            for f in fields_to_update:
                if f in stock:
                    update_doc[f] = stock[f]

            # 한국 종목인 경우에만 상세 캐시(배당률 등) 갱신 시도
            if country_code == "kor":
                try:
                    # 카테고리 정보만 추출하여 전달
                    cat_info = {f"cat_{c['code']}": stock.get(f"cat_{c['code']}") for c in NAVER_ETF_CATEGORY_CONFIG if f"cat_{c['code']}" in stock}
                    _refresh_korean_etf_meta_cache(type_norm, str(ticker), str(name), category_data=cat_info)
                except Exception as e:
                    logger.warning(f"[{type_norm.upper()}/{ticker}] ETF 상세 캐시 갱신 건너뜀: {e}")

            updates_for_db.append(update_doc)

            # 중간 저장 (20개 단위)
            if len(updates_for_db) >= 20:
                try:
                    modified = bulk_update_stocks(type_norm, updates_for_db)
                    logger.info(f"[{type_norm.upper()}] 중간 저장 완료 ({idx}/{total_count}, {modified}건)")
                    updates_for_db.clear()
                except Exception as e:
                    logger.error(f"[{type_norm.upper()}] 중간 저장 실패: {e}")

            if progress_callback:
                progress_callback(idx, total_count, f"{name}({ticker})")
            updated_count += 1
            time.sleep(0.1)  # 속도 조절

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


def _fetch_naver_stock_name_scraping(ticker: str) -> str | None:
    """
    네이버 금융 페이지 크롤링을 통해 종목명을 가져옵니다.
    API나 pykrx에서 조회되지 않는 신규 상장 ETF 등을 위한 폴백입니다.
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        url = f"https://finance.naver.com/item/main.naver?code={ticker}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=5)
        resp.raise_for_status()

        # EUC-KR 디코딩
        try:
            html = resp.content.decode("euc-kr")
        except Exception:
            html = resp.text

        soup = BeautifulSoup(html, "html.parser")
        # <div class="wrap_company"><h2><a href="#">종목명</a></h2>...</div>
        name_tag = soup.select_one(".wrap_company h2 a")
        if name_tag:
            return name_tag.text.strip()

    except Exception as e:
        get_app_logger().debug(f"네이버 금융 크롤링 실패 ({ticker}): {e}")

    return None


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

            # 2. 이름 못 찾으면 Naver Map 시도 (비효율적이지만 정확도 높음)
            if not result["name"]:
                try:
                    naver_map = fetch_naver_etf_names_map()
                    if ticker in naver_map:
                        result["name"] = naver_map[ticker]
                except Exception:
                    pass

            # 3. 그래도 없으면 크롤링 폴백 시도 (0111J0 등 API 누락 대비)
            if not result["name"]:
                try:
                    scraped_name = _fetch_naver_stock_name_scraping(ticker)
                    if scraped_name:
                        result["name"] = scraped_name
                except Exception:
                    pass

            # 4. 상장일 조회
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
        type_source = str(settings.get("type_source") or "").strip()
    except Exception as exc:
        raise RuntimeError(f"[{type_norm.upper()}/{ticker_norm}] 종목타입 설정 로드 실패: {exc}") from exc

    naver_etf_map = {}
    if country_code == "kor":
        naver_etf_map = fetch_naver_etf_names_map()

    naver_category_map: dict[str, dict[str, str]] = {}
    if type_source.lower() == "naver":
        naver_category_map = _build_naver_category_map()

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

    # [KOR] 전체 종목 정보 맵 확보 (일개 종목 업데이트 시에도 정합성 위해 벌크 활용)
    naver_kor_stock_map = {}
    if country_code == "kor":
        naver_kor_stock_map = fetch_naver_kor_stock_map()

    update_single_stock_metadata(
        stock,
        country_code,
        naver_etf_map,
        type_norm,
        naver_category_map=naver_category_map,
        naver_kor_stock_map=naver_kor_stock_map,
    )

    if country_code == "kor":
        try:
            _refresh_korean_etf_meta_cache(type_norm, ticker_norm, str(stock.get("name") or ticker_norm))
        except Exception as meta_cache_error:
            logger.error(f"[{type_norm.upper()}/{ticker_norm}] ETF 메타 캐시 갱신 실패: {meta_cache_error}")

    update_doc = {"ticker": ticker_norm}
    fields_to_update = [
        "name",
        "listing_date",
        "market",
        "1_week_avg_volume",
        "volume",
        "1_week_earn_rate",
        "2_week_earn_rate",
        "1_month_earn_rate",
        "3_month_earn_rate",
        "6_month_earn_rate",
        "12_month_earn_rate",
        "etf_category",
    ]
    # 개별 분류 컬럼들을 업데이트 필드에 추가
    for cat in NAVER_ETF_CATEGORY_CONFIG:
        fields_to_update.append(f"cat_{cat['code']}")

    for f in fields_to_update:
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
    naver_category_map: dict[str, Any] | None = None,
    naver_kor_stock_map: dict[str, dict[str, str]] | None = None,
):
    """단일 종목의 메타데이터를 업데이트합니다."""
    logger = get_app_logger()
    ticker = stock.get("ticker")
    if not ticker:
        return

    # Naver 카테고리 맵이 제공된 경우 (type_source == "Naver" 종목풀)
    if naver_category_map:
        ticker_norm = str(ticker).strip().upper()
        entry = naver_category_map.get(ticker_norm, {})
        stock["etf_category"] = entry.get("best", "")
        # 개별 대분류 필드들 채우기
        details = entry.get("details", {})
        for code, name in details.items():
            stock[f"cat_{code}"] = name

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
        elif not stock.get("name") or stock.get("name") == ticker:
            try:
                fetched_name = fetch_pykrx_name(ticker)
                if fetched_name:
                    stock["name"] = fetched_name
                    if account_norm:
                        logger.info(f"[{account_norm.upper()}/{ticker}] pykrx에서 종목명 획득: {fetched_name}")
            except Exception as e:
                logger.warning(f"[{account_norm.upper()}/{ticker}] pykrx 종목명 조회 실패: {e}")

        # 3. 마켓(KOSPI/KOSDAQ) 조회
        if not stock.get("market"):
            try:
                market = fetch_pykrx_market(ticker)
                if market:
                    stock["market"] = market
                    if account_norm:
                        logger.info(f"[{account_norm.upper()}/{ticker}] pykrx에서 마켓 정보 획득: {market}")
            except Exception as e:
                logger.warning(f"[{account_norm.upper()}/{ticker}] pykrx 마켓 정보 조회 실패: {e}")

    elif country_code in ("us", "au"):
        try:
            t = yf.Ticker(yfinance_ticker)

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

            hist = t.history(period="max")
            if not hist.empty:
                first_date = hist.index.min()
                listing_date_str = first_date.strftime("%Y-%m-%d")
                if account_norm:
                    logger.debug(f"[{account_norm.upper()}/{ticker}] yfinance에서 상장일 획득: {listing_date_str}")
        except Exception as e:
            logger.warning(f"[{account_norm.upper()}/{ticker}] yfinance 메타데이터 조회 조회 실패: {e}")

    if not listing_date_str:
        listing_date_str = stock.get("listing_date")

    if listing_date_str:
        stock["listing_date"] = listing_date_str

    # 먼저 캐시된 데이터(MongoDB)가 있는지 확인
    from utils.cache_utils import load_cached_frame

    # [Improvement] yfinance를 매번 호출하기보다는, 이미 수집된(fetch_ohlcv로) 캐시 데이터를 우선 사용
    # 캐시가 있으면 그것으로 수익률 계산 -> 훨씬 빠르고 정합성 높음 (특히 KOR 종목)
    try:
        cached_df = load_cached_frame(account_norm, ticker)
        if cached_df is not None and not cached_df.empty:
            data = cached_df
            # 캐시 데이터는 이미 'Close', 'Volume' 등이 포함되어 있음 (한글 컬럼일 수도 있음 - data_loader 참고)
            # data_loader.fetch_ohlcv 결과는 영문 컬럼 (Open, High, Low, Close, Volume, 등락률)
            # load_cached_frame 결과도 동일.
    except Exception:
        pass

    if data is None:
        # 캐시에 없으면 yfinance 시도 (기존 로직)
        data = yf.download(yfinance_ticker, period="2y", progress=False, auto_adjust=True)
        if isinstance(data, pd.DataFrame) and not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
                data = data.loc[:, ~data.columns.duplicated()]
            if not data.index.is_unique:
                data = data[~data.index.duplicated(keep="last")]
        else:
            data = None

    stock.pop("1_month_avg_volume", None)
    stock.pop("1_week_avg_turnover", None)
    stock.pop("1_month_avg_turnover", None)

    if data is not None and not data.empty and len(data) >= 1:
        if "Volume" in data.columns:
            last_week = data.tail(5)
            avg_volume = last_week["Volume"].mean()
            if pd.notna(avg_volume):
                stock["1_week_avg_volume"] = int(avg_volume)

            non_empty_vols = data["Volume"].dropna()
            if not non_empty_vols.empty:
                stock["volume"] = int(non_empty_vols.iloc[-1])

        def calc_rate_safe(df, days_lookback):
            if len(df) < days_lookback + 1:
                return None
            if "Close" not in df.columns:
                return None
            subset = df.tail(days_lookback + 1)
            if len(subset) < 2:
                return None
            start_price = subset.iloc[0]["Close"]
            end_price = subset.iloc[-1]["Close"]
            if pd.notna(start_price) and pd.notna(end_price) and start_price > 0:
                return round(((end_price - start_price) / start_price) * 100, 4)
            return None

        stock["1_week_earn_rate"] = calc_rate_safe(data, 5)
        stock["2_week_earn_rate"] = calc_rate_safe(data, 10)
        stock["1_month_earn_rate"] = calc_rate_safe(data, 21)
        stock["3_month_earn_rate"] = calc_rate_safe(data, 63)
        stock["6_month_earn_rate"] = calc_rate_safe(data, 126)
        stock["12_month_earn_rate"] = calc_rate_safe(data, 252)

        ordered_stock = {}
        for k in ["ticker", "name", "note", "listing_date"]:
            if k in stock:
                ordered_stock[k] = stock[k]

        if "1_week_avg_volume" in stock:
            ordered_stock["1_week_avg_volume"] = stock["1_week_avg_volume"]

        rate_keys = [
            "volume",
        "1_week_earn_rate",
            "2_week_earn_rate",
            "1_month_earn_rate",
            "3_month_earn_rate",
            "6_month_earn_rate",
            "12_month_earn_rate",
        ]
        for k in rate_keys:
            if k in stock:
                ordered_stock[k] = stock[k]

        known_keys = set(["ticker", "name", "note", "listing_date", "1_week_avg_volume"] + rate_keys)
        for k, v in stock.items():
            if k not in known_keys:
                ordered_stock[k] = v

        stock.clear()
        stock.update(ordered_stock)
