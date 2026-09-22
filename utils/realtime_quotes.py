"""실시간 시세 스냅샷 유틸 — 네이버·토스·호주 quoteapi·iNAV.

`utils/data_loader.py` 에서 분리(이동만, 로직 불변). 기존 임포트 경로는
data_loader 가 re-export 로 유지한다. TTL 캐시는 모듈 전역이라 프로세스 내 공유된다.
"""

import re
from collections.abc import Sequence
from datetime import datetime, timedelta
from math import isfinite
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    requests = None  # type: ignore
    BeautifulSoup = None  # type: ignore

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None  # type: ignore

from config import (
    AU_QUOTEAPI_APP_ID,
    AU_QUOTEAPI_HEADERS,
    AU_QUOTEAPI_URL,
    CACHE_TTL_COMPUTE,
    CACHE_TTL_LIVE,
    NAVER_FINANCE_ETF_API_URL,
    NAVER_FINANCE_HEADERS,
    TOSS_INVEST_API_BASE_URL,
    TOSS_INVEST_HEADERS,
)
from utils.asx_ticker import strip_asx_prefix, to_yahoo_symbol
from utils.logger import get_app_logger

logger = get_app_logger()


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_naver_realtime_price(ticker: str) -> float | None:
    """
    네이버 금융 웹 스크레이핑을 통해 종목의 실시간 현재가를 조회합니다.
    주의: 이 방법은 웹페이지 구조 변경에 취약하며, 비공식적인 방법입니다.
    """
    if not requests or not BeautifulSoup:
        return None

    try:
        url = f"https://finance.naver.com/item/sise.naver?code={ticker}"
        # 네이버의 차단을 피하기 위해 브라우저처럼 보이는 User-Agent를 설정합니다.
        headers = {  # noqa: F841
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생

        soup = BeautifulSoup(response.text, "html.parser")
        # 현재가를 담고 있는 HTML 요소를 id를 통해 찾습니다.
        price_element = soup.select_one("#_nowVal")

        if price_element:
            price_str = price_element.get_text().replace(",", "")
            return float(price_str)
    except Exception as e:
        logger.warning("%s의 실시간 가격 조회 중 오류 발생: %s", ticker, e)
    return None


# ─── ETF iNAV 글로벌 캐시 ───
# etfItemList.nhn은 전체 ETF 목록을 반환하므로 글로벌로 캐시하고 공유한다.
_ETF_INAV_GLOBAL_CACHE: dict[str, Any] = {}
_ETF_INAV_GLOBAL_TTL_SECONDS = CACHE_TTL_LIVE


def _fetch_etf_inav_all() -> dict[str, dict[str, float]]:
    """네이버 ETF API에서 전체 ETF iNAV 데이터를 가져온다. 글로벌 캐시를 사용한다."""

    now = datetime.now()
    expires_at = _ETF_INAV_GLOBAL_CACHE.get("expires_at")
    if isinstance(expires_at, datetime) and now < expires_at:
        return _ETF_INAV_GLOBAL_CACHE.get("data", {})

    if not requests:
        logger.debug("requests 라이브러리가 없어 네이버 iNAV 조회를 건너뜁니다.")
        return _ETF_INAV_GLOBAL_CACHE.get("data", {})

    try:
        response = requests.get(NAVER_FINANCE_ETF_API_URL, headers=NAVER_FINANCE_HEADERS, timeout=5)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.warning("네이버 ETF iNAV 조회 실패: %s", exc)
        # 실패 시 stale 캐시 재사용
        return _ETF_INAV_GLOBAL_CACHE.get("data", {})

    items = payload.get("result", {}).get("etfItemList")
    if not isinstance(items, list):
        return _ETF_INAV_GLOBAL_CACHE.get("data", {})

    snapshot: dict[str, dict[str, float]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue

        code = str(item.get("itemcode") or "").strip().upper()
        if not code:
            continue

        nav_raw = item.get("nav")
        price_raw = item.get("nowVal")
        change_rate_raw = item.get("changeRate")
        open_raw = item.get("openVal")
        high_raw = item.get("highVal")
        low_raw = item.get("lowVal")
        vol_raw = item.get("quant")
        name_raw = item.get("itemname")
        return_3m_raw = item.get("threeMonthEarnRate")

        try:
            nav_value = float(str(nav_raw).replace(",", ""))
            price_value = float(str(price_raw).replace(",", ""))
        except (TypeError, ValueError):
            continue

        if nav_value <= 0:
            deviation = None
        else:
            deviation = ((price_value / nav_value) - 1.0) * 100.0

        entry: dict[str, Any] = {
            "nav": nav_value,
            "nowVal": price_value,
            "deviation": deviation,
        }

        try:
            entry["changeRate"] = float(str(change_rate_raw).replace(",", ""))
        except (TypeError, ValueError):
            pass

        if name_raw:
            entry["itemname"] = str(name_raw).strip()

        try:
            entry["threeMonthEarnRate"] = float(str(return_3m_raw).replace(",", ""))
        except (TypeError, ValueError):
            pass

        try:
            if open_raw:
                entry["open"] = float(str(open_raw).replace(",", ""))
            if high_raw:
                entry["high"] = float(str(high_raw).replace(",", ""))
            if low_raw:
                entry["low"] = float(str(low_raw).replace(",", ""))
            if vol_raw:
                entry["volume"] = float(str(vol_raw).replace(",", ""))
        except (TypeError, ValueError):
            pass

        snapshot[code] = entry

    _ETF_INAV_GLOBAL_CACHE["data"] = snapshot
    _ETF_INAV_GLOBAL_CACHE["expires_at"] = datetime.now() + timedelta(seconds=_ETF_INAV_GLOBAL_TTL_SECONDS)
    return snapshot


def fetch_naver_etf_inav_snapshot(tickers: Sequence[str]) -> dict[str, dict[str, float]]:
    """네이버 API에서 한국 ETF의 실시간 NAV 정보를 조회합니다. 글로벌 캐시를 사용합니다."""

    normalized_codes = {str(t).strip().upper() for t in tickers if str(t or "").strip()}
    if not normalized_codes:
        return {}

    all_data = _fetch_etf_inav_all()
    return {code: all_data[code] for code in normalized_codes if code in all_data}


# 해외 ETF NAV 캐시 — 종목별로 조회하므로 티커 단위로 담는다(한국은 전체 목록 1회 조회).
# (모듈 상단에서 datetime.time 을 import 하고 있어 시간 비교는 datetime 으로 한다.)
_OVERSEAS_NAV_CACHE: dict[str, tuple[datetime, dict[str, float]]] = {}
_OVERSEAS_NAV_TTL_SECONDS = CACHE_TTL_COMPUTE


def fetch_overseas_etf_nav_snapshot(ticker: str, country_code: str) -> dict[str, float]:
    """해외(미국·호주) ETF 의 NAV·괴리율을 yfinance ``navPrice`` 로 조회한다.

    괴리율 계산식은 한국과 동일하다: ``(현재가 / NAV - 1) × 100``.
    한국 네이버 iNAV 와 달리 실시간 추정치(iNAV)가 아니라 **직전 공시 NAV** 라 장중에는 지연이 있다.
    값이 없으면 빈 dict(화면은 '-').
    """
    code = str(ticker or "").strip().upper()
    cc = str(country_code or "").strip().lower()
    if not code or cc not in ("us", "au"):
        return {}

    cache_key = f"{cc}:{code}"
    now = datetime.now()
    cached = _OVERSEAS_NAV_CACHE.get(cache_key)
    if cached is not None and (now - cached[0]).total_seconds() < _OVERSEAS_NAV_TTL_SECONDS:
        return dict(cached[1])

    symbol = to_yahoo_symbol(code) if cc == "au" else code
    try:
        import yfinance as yf

        from utils.yfinance_guard import yfinance_lock

        # 동시 호출 시 남의 티커 데이터를 받는 것을 막는다(utils/yfinance_guard).
        with yfinance_lock():
            info = getattr(yf.Ticker(symbol), "info", {}) or {}
    except Exception as exc:
        logger.debug("해외 ETF NAV 조회 실패 (%s): %s", symbol, exc)
        return {}

    def _positive_float(value: Any) -> float | None:
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return float(value)
        return None

    nav_value = _positive_float(info.get("navPrice"))
    price_value = _positive_float(info.get("regularMarketPrice")) or _positive_float(info.get("previousClose"))
    if nav_value is None:
        return {}

    entry: dict[str, float] = {"nav": nav_value}
    if price_value is not None:
        entry["price"] = price_value
        entry["deviation"] = ((price_value / nav_value) - 1.0) * 100.0

    _OVERSEAS_NAV_CACHE[cache_key] = (now, dict(entry))
    return entry


def _parse_comma_number(value: str | None) -> float | None:
    """쉼표가 포함된 숫자 문자열을 float로 변환한다."""
    if value is None:
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _parse_naver_signed_change_rate(item: dict[str, Any]) -> float | None:
    """네이버 등락률 필드와 등락 방향 코드를 함께 해석한다."""
    change_rate = _parse_comma_number(item.get("fluctuationsRatio"))
    if change_rate is None:
        return None

    compare_code = (item.get("compareToPreviousPrice") or {}).get("code", "")
    if compare_code == "5" and change_rate > 0:
        return -change_rate
    return change_rate


def _get_naver_over_market_price_info(item: dict[str, Any]) -> tuple[dict[str, Any], str] | None:
    """네이버 국내주식 응답에서 **열려 있는** 시간외 가격 정보와 그 구간을 반환한다.

    장전(``PRE_MARKET``)·장후(``AFTER_MARKET``) 모두 대상이다. 닫힌 뒤에도 `overPrice`
    값은 응답에 남아 있으므로 ``overMarketStatus == "OPEN"`` 일 때만 쓴다 — 시간외가
    끝나면 정규장 종가(`closePrice`)로 돌아온다.
    """
    over_market_info = item.get("overMarketPriceInfo")
    if not isinstance(over_market_info, dict):
        return None

    trading_session_type = str(over_market_info.get("tradingSessionType") or "").strip().upper()
    over_market_status = str(over_market_info.get("overMarketStatus") or "").strip().upper()
    if trading_session_type not in ("PRE_MARKET", "AFTER_MARKET") or over_market_status != "OPEN":
        return None

    over_price = _parse_comma_number(over_market_info.get("overPrice"))
    if over_price is None:
        return None

    return over_market_info, trading_session_type


def fetch_naver_stock_realtime_snapshot(tickers: Sequence[str]) -> dict[str, dict[str, Any]]:
    """stock.naver.com 폴링 API에서 한국 개별 종목의 실시간 가격 정보를 조회합니다."""

    normalized_codes = [str(t).strip().upper() for t in tickers if str(t or "").strip()]
    if not normalized_codes:
        return {}

    if not requests:
        logger.debug("requests 라이브러리가 없어 네이버 주식 조회를 건너뜁니다.")
        return {}

    naver_stock_polling_url = "https://stock.naver.com/api/polling/domestic/stock"
    naver_stock_polling_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Referer": "https://stock.naver.com/",
        "Accept": "application/json, text/plain, */*",
    }

    def _fetch_chunk(chunk: list[str]) -> dict[str, dict[str, Any]]:
        item_codes = ",".join(chunk)
        url = f"{naver_stock_polling_url}?itemCodes={item_codes}"

        try:
            response = requests.get(url, headers=naver_stock_polling_headers, timeout=5)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.warning("네이버 주식 실시간 조회 실패: %s", exc)
            return {}

        result: dict[str, dict[str, float]] = {}
        datas = data.get("datas") or []
        for item in datas:
            code = str(item.get("itemCode") or "").strip().upper()
            if not code:
                continue

            # `closePrice` 는 시간외가 열려 있어도 **정규장 종가**를 유지한다(실측 확인:
            # SK하이닉스 closePrice 1,879,000 = 일봉 종가, 같은 응답 overPrice 1,878,000).
            regular_close = _parse_comma_number(item.get("closePrice"))
            over_market = _get_naver_over_market_price_info(item)
            price_source = over_market[0] if over_market else item
            price_field = "overPrice" if over_market else "closePrice"
            price_value = _parse_comma_number(price_source.get(price_field))
            if price_value is None:
                continue

            entry: dict[str, Any] = {"nowVal": price_value}
            # 정규장 종가 — 확정 일봉 저장(`fetch_naver_daily_ohlcv_snapshot`)이 쓴다.
            # 표시·판정용 `nowVal` 은 시간외가 닫히면 이 값으로 돌아온다(위 분기).
            if regular_close is not None:
                entry["regularClose"] = regular_close
            # 일봉 스냅샷(fetch_naver_daily_ohlcv_snapshot)이 날짜 정합 검증에 쓴다.
            local_traded_at = str(item.get("localTradedAt") or "").strip()
            if local_traded_at:
                entry["localTradedAt"] = local_traded_at
            if over_market is not None:
                entry["session"] = over_market[1]
                if over_market[1] == "PRE_MARKET":
                    entry["is_pre_market"] = True

            change_rate = _parse_naver_signed_change_rate(price_source)
            if change_rate is not None:
                entry["changeRate"] = change_rate

            open_val = _parse_comma_number(price_source.get("openPrice"))
            if open_val is not None:
                entry["open"] = open_val
            high_val = _parse_comma_number(price_source.get("highPrice"))
            if high_val is not None:
                entry["high"] = high_val
            low_val = _parse_comma_number(price_source.get("lowPrice"))
            if low_val is not None:
                entry["low"] = low_val
            vol_val = _parse_comma_number(price_source.get("accumulatedTradingVolume"))
            if vol_val is not None:
                entry["volume"] = vol_val

            # 확정 일봉 저장용 — 시간외 구간에도 정규장 OHLCV 를 그대로 보존한다.
            for key, field in (
                ("regularOpen", "openPrice"),
                ("regularHigh", "highPrice"),
                ("regularLow", "lowPrice"),
                ("regularVolume", "accumulatedTradingVolume"),
            ):
                parsed = _parse_comma_number(item.get(field))
                if parsed is not None:
                    entry[key] = parsed

            result[code] = entry

        return result

    # 청크 단위로 호출 (URL 길이 제한 대비)
    snapshot: dict[str, dict[str, Any]] = {}
    chunk_size = 50
    for i in range(0, len(normalized_codes), chunk_size):
        chunk = normalized_codes[i : i + chunk_size]
        snapshot.update(_fetch_chunk(chunk))

    return snapshot


def fetch_naver_daily_ohlcv_snapshot(tickers: Sequence[str], target_day: pd.Timestamp) -> dict[str, dict[str, float]]:
    """한국 종목들의 **확정 당일 일봉(OHLCV)** 을 폴링 API 로 일괄 조회한다.

    가격 캐시 증분 갱신용 — 종목당 pykrx 호출 대신 50종목 단위 배치 호출로
    `target_day`(마감된 최신 거래일)의 일봉 행을 만든다. 값의 날짜(localTradedAt)가
    target_day 와 다르거나(거래정지·이월 표시), 장 전 예상가 상태(is_pre_market)거나,
    OHLC 중 하나라도 없는 종목은 **제외**한다 — 잘못된 일봉을 저장하느니 빼고
    종목별 pykrx 경로에 맡기는 쪽이 안전하다.

    쓰는 값은 **정규장 OHLCV**(`regular*`)다. 표시용 `nowVal`·`open`·`high` 등은 장후
    시간외가 열려 있으면 시간외 값으로 바뀌므로, 그대로 저장하면 일봉이 오염된다.
    """
    target = pd.Timestamp(target_day).normalize()
    snapshot = fetch_naver_stock_realtime_snapshot(tickers)
    result: dict[str, dict[str, float]] = {}
    for code, entry in snapshot.items():
        if entry.get("is_pre_market"):
            continue
        traded_at = str(entry.get("localTradedAt") or "")[:10]
        traded_ts = pd.to_datetime(traded_at, errors="coerce")
        if traded_ts is pd.NaT or pd.Timestamp(traded_ts).normalize() != target:
            continue
        open_val = entry.get("regularOpen")
        high_val = entry.get("regularHigh")
        low_val = entry.get("regularLow")
        close_val = entry.get("regularClose")
        volume_val = entry.get("regularVolume")
        if any(v is None for v in (open_val, high_val, low_val, close_val, volume_val)):
            continue
        if min(float(open_val), float(high_val), float(low_val), float(close_val)) <= 0:
            continue
        result[code] = {
            "Open": float(open_val),
            "High": float(high_val),
            "Low": float(low_val),
            "Close": float(close_val),
            "Volume": float(volume_val),
        }
    return result


def fetch_naver_worldstock_snapshot(reuters_codes: Sequence[str]) -> dict[str, dict[str, float | str]]:
    """네이버 worldstock 폴링 API에서 해외 종목의 지연 시세를 조회합니다."""

    normalized_codes = [str(code).strip().upper() for code in reuters_codes if str(code or "").strip()]
    if not normalized_codes:
        return {}

    if not requests:
        logger.debug("requests 라이브러리가 없어 네이버 worldstock 조회를 건너뜁니다.")
        return {}

    naver_worldstock_polling_url = "https://stock.naver.com/api/polling/worldstock/stock"
    naver_worldstock_polling_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Referer": "https://stock.naver.com/",
        "Accept": "application/json, text/plain, */*",
    }

    def _fetch_chunk(chunk: list[str]) -> dict[str, dict[str, float | str]]:
        codes = ",".join(chunk)
        url = f"{naver_worldstock_polling_url}?reutersCodes={codes}"
        try:
            response = requests.get(url, headers=naver_worldstock_polling_headers, timeout=5)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.warning("네이버 worldstock 조회 실패: %s", exc)
            return {}

        result: dict[str, dict[str, float | str]] = {}
        datas = data.get("datas") or []
        for item in datas:
            code = str(item.get("reutersCode") or "").strip().upper()
            if not code:
                continue

            price_value = _parse_comma_number(item.get("closePrice"))
            if price_value is None:
                continue

            # worldstock 은 정규장 종가만 주는 지연 시세다 — 시간외 값이 따로 없다.
            entry: dict[str, float | str] = {"nowVal": price_value}

            prev_close = _parse_comma_number(item.get("compareToPreviousClosePrice"))
            compare_code = str(((item.get("compareToPreviousPrice") or {}).get("code")) or "").strip()
            if prev_close is not None:
                signed_diff = -prev_close if compare_code == "5" and prev_close > 0 else prev_close
                entry["prevClose"] = price_value - signed_diff

            change_rate = _parse_comma_number(item.get("fluctuationsRatio"))
            if change_rate is not None:
                if compare_code == "5" and change_rate > 0:
                    change_rate = -change_rate
                entry["changeRate"] = change_rate

            open_val = _parse_comma_number(item.get("openPrice"))
            if open_val is not None:
                entry["open"] = open_val
            high_val = _parse_comma_number(item.get("highPrice"))
            if high_val is not None:
                entry["high"] = high_val
            low_val = _parse_comma_number(item.get("lowPrice"))
            if low_val is not None:
                entry["low"] = low_val
            vol_val = _parse_comma_number(item.get("accumulatedTradingVolume"))
            if vol_val is not None:
                entry["volume"] = vol_val

            currency_code = str(((item.get("currencyType") or {}).get("code")) or "").strip().upper()
            if currency_code:
                entry["currency"] = currency_code

            result[code] = entry

        return result

    snapshot: dict[str, dict[str, float | str]] = {}
    chunk_size = 50
    for i in range(0, len(normalized_codes), chunk_size):
        chunk = normalized_codes[i : i + chunk_size]
        snapshot.update(_fetch_chunk(chunk))

    return snapshot


def fetch_au_quoteapi_snapshot(tickers: Sequence[str]) -> dict[str, dict[str, float]]:
    """호주 MarketIndex QuoteAPI에서 ETF의 실시간 가격 정보를 조회합니다.

    Args:
        tickers: 조회할 호주 ETF 티커 리스트 (예: ["ACDC", "MNRS"])

    Returns:
        티커별 가격 정보 딕셔너리
        {
            "ACDC": {"nowVal": 155.81, "changeRate": -0.44, "open": ..., "high": ..., "low": ..., "volume": ...},
            ...
        }
    """
    if not requests:
        logger.debug("requests 라이브러리가 없어 호주 QuoteAPI 조회를 건너뜁니다.")
        return {}

    normalized_tickers = [str(t).strip().upper() for t in tickers if str(t or "").strip()]
    if not normalized_tickers:
        return {}

    import concurrent.futures

    # 병렬 처리를 위한 내부 함수
    def _fetch_single_quote(ticker: str) -> tuple[str, dict[str, float] | None]:
        try:
            # 호주 ETF 티커 형식: ticker.asx (소문자). 시스템 표준 ASX: 접두사는 벗겨서 보낸다.
            url = f"{AU_QUOTEAPI_URL}/{strip_asx_prefix(ticker).lower()}.asx"
            # params = {"appID": AU_QUOTEAPI_APP_ID} # URL에 포함되지 않는 경우도 있음

            # API 호출 (타임아웃 단축)
            response = requests.get(url, params={"appID": AU_QUOTEAPI_APP_ID}, headers=AU_QUOTEAPI_HEADERS, timeout=3)
            response.raise_for_status()

            data = response.json()
            quote = data.get("quote", {})

            if not quote:
                return ticker, None

            price = quote.get("price")
            if price is None or price <= 0:
                return ticker, None

            entry: dict[str, float] = {
                "nowVal": float(price),
            }

            prev_close = quote.get("prevClose")
            if prev_close is not None:
                try:
                    entry["prevClose"] = float(prev_close)
                except (TypeError, ValueError):
                    pass

            pct_change = quote.get("pctChange")
            if pct_change is not None:
                try:
                    entry["pctChange"] = float(pct_change)
                    entry["changeRate"] = float(pct_change)
                except (TypeError, ValueError):
                    pass
            elif entry.get("prevClose"):
                try:
                    entry["changeRate"] = ((entry["nowVal"] / entry["prevClose"]) - 1.0) * 100.0
                except Exception:
                    pass

            # OHLCV 데이터
            for field in ["open", "high", "low", "volume"]:
                val = quote.get(field)
                if val:
                    try:
                        entry[field] = float(val)
                    except (TypeError, ValueError):
                        pass

            return ticker, entry
        except Exception:
            return ticker, None

    snapshot: dict[str, dict[str, float]] = {}

    # ThreadPoolExecutor를 사용하여 병렬 요청
    # 최대 10개 스레드로 제한
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(_fetch_single_quote, ticker): ticker for ticker in normalized_tickers}
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker_res, entry_res = future.result()
            if entry_res:
                snapshot[ticker_res] = entry_res

    if snapshot:
        global _AU_QUOTEAPI_SNAPSHOT_CACHE, _AU_QUOTEAPI_SNAPSHOT_FETCHED_AT
        _AU_QUOTEAPI_SNAPSHOT_CACHE = snapshot
        _AU_QUOTEAPI_SNAPSHOT_FETCHED_AT = pd.Timestamp.now()
        logger.info(f"[AU] QuoteAPI에서 {len(snapshot)}개 종목의 실시간 가격을 조회했습니다.")
    else:
        _AU_QUOTEAPI_SNAPSHOT_CACHE = {}
        _AU_QUOTEAPI_SNAPSHOT_FETCHED_AT = None

    return snapshot


_AU_QUOTEAPI_SNAPSHOT_CACHE: dict[str, dict[str, float]] = {}
_AU_QUOTEAPI_SNAPSHOT_FETCHED_AT: pd.Timestamp | None = None


def prime_au_etf_realtime_snapshot(tickers: Sequence[str]) -> None:
    """Fetch and cache real-time price snapshot for given Australian ETF tickers."""

    global _AU_QUOTEAPI_SNAPSHOT_CACHE, _AU_QUOTEAPI_SNAPSHOT_FETCHED_AT

    try:
        from services.price_service import get_realtime_snapshot

        snapshot = get_realtime_snapshot("au", tickers)
    except Exception as exc:
        logger.warning("호주 ETF 실시간 스냅샷 조회 실패: %s", exc)
        return

    if snapshot:
        _AU_QUOTEAPI_SNAPSHOT_CACHE = snapshot
        _AU_QUOTEAPI_SNAPSHOT_FETCHED_AT = pd.Timestamp.now()
    else:
        _AU_QUOTEAPI_SNAPSHOT_CACHE = {}
        _AU_QUOTEAPI_SNAPSHOT_FETCHED_AT = None


def get_cached_au_etf_snapshot_entry(ticker: str) -> dict[str, float] | None:
    """Return cached price snapshot entry for the given Australian ETF ticker."""

    key = str(ticker or "").strip().upper()
    if not key:
        return None
    return _AU_QUOTEAPI_SNAPSHOT_CACHE.get(key)


# ────────────────────────────────────────────
# 토스증권 API — 미국 주식 실시간 가격
# ────────────────────────────────────────────

# symbol → productCode 영구 매핑 캐시 (프로세스 수명 동안 유효)
_TOSS_SYMBOL_CODE_CACHE: dict[str, str] = {}


def _resolve_toss_product_codes(symbols: Sequence[str]) -> dict[str, str]:
    """미국 티커 심볼을 토스 productCode로 변환한다.

    캐시에 없는 심볼만 검색 API를 호출하고, 결과를 영구 캐시에 저장한다.
    Returns:
        {symbol: productCode} 매핑 (매핑 실패 심볼은 제외)
    """
    import concurrent.futures

    from utils.symbol_resolution_blacklist import get_active_blacklist, mark_failed

    result: dict[str, str] = {}
    uncached: list[str] = []

    blacklist = get_active_blacklist()

    for sym in symbols:
        cached_code = _TOSS_SYMBOL_CODE_CACHE.get(sym)
        if cached_code:
            result[sym] = cached_code
        elif sym in blacklist:
            # 24시간 내 실패한 심볼은 재시도하지 않음
            continue
        else:
            uncached.append(sym)

    if not uncached:
        return result

    if not requests:
        logger.debug("requests 라이브러리가 없어 토스 심볼 검색을 건너뜁니다.")
        return result

    search_url = f"{TOSS_INVEST_API_BASE_URL}/api/v2/search/stocks"

    def _search_one(sym: str) -> tuple[str, str | None]:
        """심볼 1개를 검색해 (심볼, productCode) 를 반환한다. 실패 시 코드는 None."""
        try:
            # 토스는 BRK-A 등 하이픈(-)이 들어간 경우 BRK.A로 검색해야 결과가 나옵니다.
            search_query = sym.replace("-", ".")
            resp = requests.post(
                search_url,
                headers=TOSS_INVEST_HEADERS,
                json={"query": search_query},
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
            stocks = (data.get("result") or {}).get("stocks") or []

            # 한국 종목(stockCode가 'A' + 숫자 6자리)만 제외한다 — 단순 'A' 시작 판정은
            # AMEX 상장 종목(AMX… 코드, 예: DRAM)까지 한국 종목으로 오인해 버린다.
            us_stocks = [s for s in stocks if not re.fullmatch(r"A\d{6}", str(s.get("stockCode") or ""))]

            # matchType이 EXACT인 첫 번째 미국 종목 사용
            for stock in us_stocks:
                if stock.get("matchType") == "EXACT":
                    return sym, stock.get("stockCode")

            # EXACT 없으면 stockName이 심볼(원래 심볼 또는 검색 심볼)과 동일한 첫 번째 미국 종목
            for stock in us_stocks:
                name = str(stock.get("stockName") or "").strip().upper()
                if name == sym or name == search_query:
                    return sym, stock.get("stockCode")

            logger.warning("토스 심볼 매핑 실패: %s (미국 주식 검색 결과 없음)", sym)
            mark_failed(sym, source="토스", reason="미국 주식 검색 결과 없음")
            return sym, None
        except Exception as exc:
            logger.warning("토스 심볼 검색 API 실패: %s error=%s", sym, exc)
            return sym, None

    # 심볼당 1회 요청이라 순차로 돌면 수백 개일 때 수십 초가 걸린다(응답 자체는 ~80ms).
    # 실시간 시세 병렬 조회와 동일하게 10 스레드로 제한해 동시에 요청한다.
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for sym, product_code in executor.map(_search_one, uncached):
            if product_code:
                _TOSS_SYMBOL_CODE_CACHE[sym] = product_code
                result[sym] = product_code

    return result


def resolve_toss_us_product_codes(symbols: Sequence[str]) -> dict[str, str]:
    """미국 티커를 토스 캔들 조회용 상품 코드로 변환한다."""
    normalized_symbols = [str(symbol).strip().upper() for symbol in symbols if str(symbol or "").strip()]
    return _resolve_toss_product_codes(normalized_symbols)


def _toss_us_price_entry(item: dict[str, Any]) -> dict[str, Any]:
    """**지금 열려 있는 세션**의 가격을 고른다. 애프터 값을 정규 종가로 저장하지 않는다.

    세션은 응답의 거래 시각이 아니라 **현재 시각**으로 판정한다. `tradeDateTime` 은
    종목별 마지막 체결 시각이라, 그 기준으로 고르면 애프터가 끝난 뒤에도 마지막 체결이
    애프터 시각이어서 애프터 가격에 눌러앉는다 — 세션이 끝나면 정규장 종가로 돌아와야 한다.
    """
    from utils.market_session import AFTERMARKET, CLOSED, PREMARKET, market_session

    stamp = pd.Timestamp(item.get("tradeDateTime"))
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("토스 미국 시세의 tradeDateTime(시간대 포함)이 필요합니다.")
    local_stamp = stamp.tz_convert(ZoneInfo("America/New_York"))
    session = market_session("us")["session"]
    # 세션별 필드 — 2026-09-21/22 실측으로 확정했다.
    #   애프터(16:00~20:00 ET): `afterMarketClose`. 애프터가 끝나면 이 값은 0.0 으로 리셋된다.
    #   데이장(20:00~03:50 ET): `close` 가 데이장 현재가를 준다(HOOD 121.82→121.89 로 움직임).
    #                           프리장·정규장도 `close` 가 그 세션의 현재가다.
    #   마감(03:50~04:00 ET·주말·휴장): `close` 는 **데이장 마지막 가격에 머문다**. 그걸 쓰면
    #                           금요일 봉이 금요일 밤 데이장 가격으로 덮이므로, 직전 정규장
    #                           종가인 `base` 로 되돌린다(애프터·데이장 모두 123.30 로 일치).
    if session == AFTERMARKET:
        price_field = "afterMarketClose"
    elif session == CLOSED:
        price_field = "base"
    else:
        price_field = "close"
    price = _safe_float(item.get(price_field))
    if price is None or not isfinite(price) or price <= 0:
        raise ValueError(f"토스 미국 {session} 시세의 {price_field}가 유효하지 않습니다.")

    entry: dict[str, Any] = {
        "nowVal": price,
        "localTradedAt": local_stamp.isoformat(),
        "tradeDateTime": stamp.isoformat(),
        "session": session,
        "priceSource": f"toss_invest.{price_field}",
        "is_pre_market": session == PREMARKET,
    }
    # 일간 등락률은 애프터장에서도 전일 정규장 기준가 대비로 유지한다.
    # 마감 구간은 가격 자체가 `base` 라 여기서 계산하면 항상 0% 가 된다 — 그때는 실시간으로
    # 덮지 않고 비워 두어, 화면이 확정 종가 시리즈로 계산한 일간(%) 을 그대로 쓰게 한다.
    base = _safe_float(item.get("base"))
    if session != CLOSED and base is not None and isfinite(base) and base > 0:
        entry["prevClose"] = base
        entry["changeRate"] = (price / base - 1.0) * 100.0
    # 거래량·거래대금은 API의 누적값을 전달한다. 세션별 값으로 임의 분리하지 않는다.
    for key, field in (("tradeValue", "value"), ("tradeVolume", "volume"), ("volume", "volume")):
        parsed = _safe_float(item.get(field))
        if parsed is not None and isfinite(parsed) and parsed > 0:
            entry[key] = parsed
    return entry


def fetch_toss_us_stock_snapshot(tickers: Sequence[str]) -> dict[str, dict[str, Any]]:
    """토스증권 API에서 미국 주식의 실시간 가격 정보를 조회합니다.

    Args:
        tickers: 미국 주식 티커 리스트 (예: ["TSLA", "AAPL", "NVDA"])

    Returns:
        티커별 가격 정보 딕셔너리
        {
            "TSLA": {"nowVal": 397.27, "changeRate": 1.36, "prevClose": 391.95},
            ...
        }
    """
    normalized_symbols = [str(t).strip().upper() for t in tickers if str(t or "").strip()]
    if not normalized_symbols:
        return {}

    if not requests:
        logger.debug("requests 라이브러리가 없어 토스 가격 조회를 건너뜁니다.")
        return {}

    # 1단계: symbol → productCode 매핑
    symbol_to_code = _resolve_toss_product_codes(normalized_symbols)
    if not symbol_to_code:
        return {}

    # code → symbol 역매핑
    code_to_symbol = {code: sym for sym, code in symbol_to_code.items()}

    # 2단계: productCode로 벌크 가격 조회 (50개씩 청크)
    price_url = f"{TOSS_INVEST_API_BASE_URL}/api/v3/stock-prices/details"
    all_codes = list(symbol_to_code.values())
    snapshot: dict[str, dict[str, Any]] = {}

    chunk_size = 50
    for i in range(0, len(all_codes), chunk_size):
        chunk = all_codes[i : i + chunk_size]
        codes_param = ",".join(chunk)

        try:
            resp = requests.get(
                price_url,
                params={"productCodes": codes_param},
                headers=TOSS_INVEST_HEADERS,
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("토스 가격 API 실패: %s", exc)
            continue

        items = data if isinstance(data, list) else (data.get("result") or [])
        for item in items:
            if not isinstance(item, dict):
                continue
            code = item.get("code")
            sym = code_to_symbol.get(code)
            if not sym:
                continue

            try:
                snapshot[sym] = _toss_us_price_entry(item)
            except (ValueError, TypeError) as exc:
                logger.warning("토스 미국 시세 제외 (%s): %s", sym, exc)

    if snapshot:
        logger.info("[US] 토스증권 API에서 %d개 종목의 실시간 가격을 조회했습니다.", len(snapshot))

    return snapshot


# 토스 국내 스냅샷 TTL — 앱 표준 실시간 주기(60초)에 맞춘다.
# 신고가 화면과 종목풀 순위가 같은 종목을 잇달아 조회하므로 프로세스 전역으로 공유한다.
_TOSS_KR_SNAPSHOT_TTL_SECONDS = CACHE_TTL_LIVE
_TOSS_KR_SNAPSHOT_CACHE: dict[str, tuple[datetime, dict[str, float]]] = {}


def fetch_toss_kr_stock_snapshot(tickers: Sequence[str]) -> dict[str, dict[str, float]]:
    """토스증권 API에서 한국 주식의 실시간 시세를 조회한다.

    같은 응답에 **당일 누적 거래대금(`value`)·거래량(`volume`)** 도 들어 있어 함께 담는다.
    가격 캐시의 거래대금은 `종가 × 거래량` 추정이고 그나마 마감된 날 것뿐이라,
    장중 거래대금을 보여주려면 이 값이 필요하다. 50개씩 묶어 부르므로 358종목이 8회다.
    """
    normalized_tickers = [str(ticker).strip().upper() for ticker in tickers if str(ticker or "").strip()]
    if not normalized_tickers or not requests:
        return {}

    now = datetime.now()
    snapshot: dict[str, dict[str, float]] = {}
    expired: list[str] = []
    for ticker in normalized_tickers:
        cached = _TOSS_KR_SNAPSHOT_CACHE.get(ticker)
        if cached and (now - cached[0]).total_seconds() < _TOSS_KR_SNAPSHOT_TTL_SECONDS:
            snapshot[ticker] = cached[1]
        else:
            expired.append(ticker)
    if not expired:
        return snapshot

    code_to_ticker = {
        (ticker if ticker.startswith("A") else f"A{ticker}"): ticker.removeprefix("A") for ticker in expired
    }
    price_url = f"{TOSS_INVEST_API_BASE_URL}/api/v3/stock-prices/details"

    all_codes = list(code_to_ticker)
    for start in range(0, len(all_codes), 50):
        codes = all_codes[start : start + 50]
        try:
            resp = requests.get(
                price_url,
                params={"productCodes": ",".join(codes)},
                headers=TOSS_INVEST_HEADERS,
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("토스 국내주식 가격 API 실패: %s", exc)
            continue

        items = data if isinstance(data, list) else (data.get("result") or [])
        for item in items:
            if not isinstance(item, dict):
                continue
            ticker = code_to_ticker.get(str(item.get("code") or ""))
            close_val = _safe_float(item.get("close"))
            if not ticker or close_val is None or close_val <= 0:
                continue
            base_val = _safe_float(item.get("base"))
            entry: dict[str, float] = {"nowVal": close_val}
            if base_val is not None and base_val > 0:
                entry["prevClose"] = base_val
                entry["changeRate"] = ((close_val - base_val) / base_val) * 100.0
            for key, field in (("tradeValue", "value"), ("tradeVolume", "volume")):
                parsed = _safe_float(item.get(field))
                if parsed is not None and parsed > 0:
                    entry[key] = parsed
            snapshot[ticker] = entry
            _TOSS_KR_SNAPSHOT_CACHE[ticker] = (now, entry)

    return snapshot


_NAVER_ETF_SNAPSHOT_CACHE: dict[str, dict[str, float]] = {}
_NAVER_ETF_SNAPSHOT_FETCHED_AT: pd.Timestamp | None = None


def prime_naver_etf_realtime_snapshot(tickers: Sequence[str]) -> None:
    """Fetch and cache real-time NAV/price snapshot for given Korean ETF tickers."""

    global _NAVER_ETF_SNAPSHOT_CACHE, _NAVER_ETF_SNAPSHOT_FETCHED_AT

    try:
        from services.price_service import get_realtime_snapshot

        snapshot = get_realtime_snapshot("kor", tickers)
    except Exception as exc:  # pragma: no cover - 외부 요청 방어
        logger.warning("네이버 ETF 실시간 스냅샷 조회 실패: %s", exc)
        return

    if snapshot:
        _NAVER_ETF_SNAPSHOT_CACHE = snapshot
        _NAVER_ETF_SNAPSHOT_FETCHED_AT = pd.Timestamp.now()
    else:
        _NAVER_ETF_SNAPSHOT_CACHE = {}
        _NAVER_ETF_SNAPSHOT_FETCHED_AT = None


def get_cached_naver_etf_snapshot_entry(ticker: str) -> dict[str, float] | None:
    """Return cached NAV snapshot entry for the given Korean ETF ticker."""

    key = str(ticker or "").strip().upper()
    if not key:
        return None
    return _NAVER_ETF_SNAPSHOT_CACHE.get(key)
