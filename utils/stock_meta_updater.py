"""
종목 파일(zsettings/<account>/stocks.json)에 메타데이터(상장일, 주간 평균 거래량/거래대금)를 업데이트합니다.
"""

import time
import xml.etree.ElementTree as ET
from typing import Any

import pandas as pd
import requests
import yfinance as yf

from utils.data_loader import fetch_pykrx_name
from utils.logger import get_app_logger
from utils.settings_loader import get_account_settings, list_available_accounts


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


def update_account_metadata(account_id: str, progress_callback: Callable[[int, int, str], None] | None = None):
    """지정된 계정의 모든 종목에 대한 메타데이터를 업데이트합니다."""
    logger = get_app_logger()
    account_norm = (account_id or "").strip().lower()

    try:
        settings = get_account_settings(account_norm)
        country_code = settings.get("country_code", "kor").lower()
    except Exception as e:
        logger.error(f"계정 설정을 로드할 수 없습니다: {e}")
        return

    # 삭제된 종목 포함하여 모든 종목 로드
    stock_data = get_all_etfs_including_deleted(account_norm)
    if not stock_data:
        logger.warning(f"'{account_norm}' 계정의 종목 데이터가 비어있습니다.")
        return

    updated_count = 0
    ticker_entries: list[dict[str, Any]] = []

    for stock in stock_data:
        ticker_entries.append(stock)

    total_count = len(ticker_entries)
    logger.info(f"[{account_norm.upper()}] 메타데이터 업데이트 시작 (총 {total_count}개 종목)")

    if progress_callback:
        progress_callback(0, total_count, "데이터 준비 중...")

    # [KOR] 전체 ETF 목록(이름 포함)을 한 번에 조회하여 맵 구성
    naver_etf_map = {}
    if country_code == "kor":
        logger.info("네이버 ETF API에서 전체 종목명 목록을 가져옵니다...")
        naver_etf_map = fetch_naver_etf_names_map()
        logger.info(f"  -> {len(naver_etf_map)}개 ETF 정보 획득")

    # 업데이트 사항을 모아두기 위한 리스트
    updates_for_db = []

    for idx, stock in enumerate(ticker_entries, start=1):
        ticker = stock.get("ticker")
        if not ticker:
            continue

        try:
            update_single_stock_metadata(stock, country_code, naver_etf_map, account_norm)

            name = stock.get("name") or "-"
            logger.info(f"  -> 메타데이터 획득 중: {idx}/{total_count} - {name}({ticker})")

            # 저장할 필드들을 딕셔너리로 구성
            update_doc = {"ticker": ticker}

            # 메타데이터 업데이트 시 갱신되는 주요 필드 지정
            fields_to_update = [
                "name",
                "listing_date",
                "1_week_avg_volume",
                "1_week_earn_rate",
                "1_month_earn_rate",
                "3_month_earn_rate",
                "6_month_earn_rate",
                "12_month_earn_rate",
            ]
            for f in fields_to_update:
                if f in stock:
                    update_doc[f] = stock[f]

            updates_for_db.append(update_doc)

            if progress_callback:
                progress_callback(idx, total_count, f"{name}({ticker})")
            updated_count += 1
            time.sleep(0.2)  # API 호출 속도 조절

        except Exception as e:
            logger.error(f"[{account_norm.upper()}/{ticker}] 메타데이터 업데이트 실패: {e}")

    # 메타데이터가 갱신된 필드만 추출하여 bulk_update 수행 (save_etfs를 쓰면 is_deleted가 리셋되는 버그 방지)
    try:
        if updates_for_db:
            modified = bulk_update_stocks(account_norm, updates_for_db)
            logger.info(f"[{account_norm.upper()}] {modified}개 메타데이터 변경사항 저장 완료")
    except Exception as e:
        logger.error(f"'{account_id}' 설정 저장 실패: {e}")


def update_stock_metadata(account_id: str | None = None):
    """
    모든 계정 또는 특정 계정의 종목 메타데이터를 업데이트합니다.
    account_id가 None이면 모든 계정(kor/us/usa)을 업데이트합니다.
    """
    logger = get_app_logger()

    accounts_to_update = []

    if account_id:
        norm_id = account_id.strip().lower()
        if norm_id in list_available_accounts():
            accounts_to_update.append(norm_id)
        else:
            logger.error(f"계정 ID '{account_id}'를 찾을 수 없습니다.")
            return
    else:
        all_accounts = list_available_accounts()
        for account in all_accounts:
            try:
                settings = get_account_settings(account)
                c_code = settings.get("country_code", "").lower()
                if c_code in ["kor", "us", "usa"]:
                    accounts_to_update.append(account)
            except Exception:
                pass

    logger.info(f"메타데이터 업데이트 대상 계정: {accounts_to_update}")

    for account in accounts_to_update:
        update_account_metadata(account)

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
        if country_norm in ("kor", "kr"):
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

        elif country_norm in ("us", "au", "usa"):
            yf_ticker = ticker
            if country_norm == "au" and not ticker.endswith(".AX"):
                yf_ticker = f"{ticker}.AX"

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


def update_single_stock_metadata(
    stock: dict[str, Any], country_code: str, naver_etf_map: dict[str, str], account_norm: str = ""
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
        elif not stock.get("name"):
            try:
                fetched_name = fetch_pykrx_name(ticker)
                if fetched_name:
                    stock["name"] = fetched_name
                    if account_norm:
                        logger.info(f"[{account_norm.upper()}/{ticker}] pykrx에서 종목명 획득: {fetched_name}")
            except Exception as e:
                logger.warning(f"[{account_norm.upper()}/{ticker}] pykrx 종목명 조회 실패: {e}")

    elif country_code in ("us", "au"):
        try:
            t = yf.Ticker(yfinance_ticker)

            if not stock.get("name"):
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
            "1_week_earn_rate",
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
