import logging
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

# Logger setup
logger = logging.getLogger(__name__)

# 기존 fetch 함수들을 data_loader에서 가져올 준비
# 실제 함수 위치나 구조에 따라 나중에 수정
from config import MARKET_SCHEDULES
from utils.data_loader import (
    fetch_au_quoteapi_snapshot,
    fetch_naver_etf_inav_snapshot,
    fetch_naver_stock_realtime_snapshot,
    get_latest_trading_day,
)

# 글로벌 캐시 저장소
# 구조: {'KOR': {'expires_at': datetime, 'data': {...}}, 'AU': {...}}
_REALTIME_CACHE: dict[str, dict[str, Any]] = {}


def get_realtime_snapshot(country_code: str, tickers: Sequence[str]) -> dict[str, dict[str, float]]:
    """
    국가별 실시간 가격 스냅샷을 가져옵니다.
    장중에는 짧은 TTL, 장 마감 후에는 긴 TTL을 적용하여 불필요한 API 호출을 방지합니다.
    """
    country = str(country_code or "").strip().lower()
    if not country or not tickers:
        return {}

    normalized_tickers = [str(t).strip().upper() for t in tickers if str(t or "").strip()]
    if not normalized_tickers:
        return {}

    now = datetime.now()

    # 1. 캐시 히트 확인
    cache_entry = _REALTIME_CACHE.get(country)
    if cache_entry and cache_entry.get("expires_at") and now < cache_entry["expires_at"]:
        cached_data = cache_entry.get("data", {})
        # 요청한 티커가 모두 캐시에 있는지 확인 (부분 히트면 전체 재조회 혹은 히트된 것만 반환 정책 필요, 여기서는 전체 캐시 유효성만 확인)
        # 단순화를 위해 전체 캐시 유효기간 내면 그대로 반환 (tickers 필터링 적용)
        return {k: v for k, v in cached_data.items() if k in set(normalized_tickers)}

    # 2. 캐시 미스 또는 만료: 새로 조회
    new_data: dict[str, dict[str, float]] = {}
    try:
        if country == "kor":
            # ETF 1차 조회
            new_data = fetch_naver_etf_inav_snapshot(normalized_tickers)
            # 누락된 종목 주식/ETN 2차 조회
            missed = [t for t in normalized_tickers if t not in new_data]
            if missed:
                stock_data = fetch_naver_stock_realtime_snapshot(missed)
                new_data.update(stock_data)
        elif country == "au":
            new_data = fetch_au_quoteapi_snapshot(normalized_tickers)
        else:
            logger.warning(f"지원하지 않는 실시간 가격 국가 코드: {country}")
            return {}

    except Exception as e:
        logger.warning(f"[{country}] 실시간 가격 조회(오케스트레이터) 중 오류: {e}")
        # 오류 시 기존 만료된 캐시라도 있으면 반환 (Stale-while-revalidate 유사 로직)
        if cache_entry and "data" in cache_entry:
            logger.info(f"[{country}] 최신 조회 실패로 기존 만료 캐시 데이터 반환")
            return {k: v for k, v in cache_entry["data"].items() if k in set(normalized_tickers)}
        return {}

    # 3. TTL 계산 및 캐시 갱신
    ttl_seconds = _calculate_ttl(country)
    expires_at = now + timedelta(seconds=ttl_seconds)

    # 기존 캐시가 있다면 업데이트, 없다면 새로 생성 (기존 데이터 유지 목적)
    if country not in _REALTIME_CACHE:
        _REALTIME_CACHE[country] = {"data": {}}

    _REALTIME_CACHE[country]["data"].update(new_data)
    _REALTIME_CACHE[country]["expires_at"] = expires_at

    logger.info(
        f"[CACHE] {country.upper()} 실시간 가격 갱신 완료 (TTL: {ttl_seconds}s, 유효: {expires_at.strftime('%H:%M:%S')})"
    )

    return {k: v for k, v in _REALTIME_CACHE[country]["data"].items() if k in set(normalized_tickers)}


def _calculate_ttl(country: str) -> int:
    """
    현재 시간이 장중인지 장 마감인지 판단하여 TTL을 초 단위로 반환합니다.
    - 장중: 30초 ~ 60초 (자주 갱신)
    - 장 마감 후 / 비거래일: 1시간 (갱신 불필요)
    """
    import pytz

    schedule = MARKET_SCHEDULES.get(country)
    if not schedule:
        return 60  # 기본 1분

    try:
        tz = pytz.timezone(schedule.get("timezone", "UTC"))
        now_local = datetime.now(tz)

        # 오늘이 거래일인지 확인 (최근 거래일이 오늘인지)
        # 달력 모듈을 사용하여 정확히 알 수 있지만, 시간만으로 단순화 + 최신 거래일 비교
        latest_day = get_latest_trading_day(country)
        is_trading_day = latest_day.date() == now_local.date()

        if not is_trading_day:
            return 3600  # 비거래일은 1시간

        # 장중 시간 확인 (10분 여유)
        open_time = schedule["open"]
        close_time = schedule["close"]

        # 장 시작 직전 ~ 장 마감 + 10분 까지는 실시간(짧은 TTL)
        # Python 시간 비교 편의를 위해..
        open_dt = datetime.combine(now_local.date(), open_time, tzinfo=tz)
        close_dt = datetime.combine(now_local.date(), close_time, tzinfo=tz)

        # 여유 시간 추가
        realtime_start = open_dt - timedelta(minutes=10)
        realtime_end = close_dt + timedelta(minutes=20)

        is_market_open = realtime_start <= now_local <= realtime_end

        if is_market_open:
            return 30 if country == "kor" else 60
        else:
            return 3600  # 장 마감 후

    except Exception as e:
        logger.warning(f"TTL 계산 오류 ({country}): {e}")
        return 60


# 앱에서 사용하는 환율 조회도 여기로 통합
def get_exchange_rates() -> dict[str, Any]:
    """
    USD/KRW, AUD/KRW 환율을 가져옵니다. 1시간 TTL.
    """
    now = datetime.now()
    cache_entry = _REALTIME_CACHE.get("EXCHANGE_RATES")

    if cache_entry and cache_entry.get("expires_at") and now < cache_entry["expires_at"]:
        return cache_entry.get("data", {})

    import yfinance as yf

    rates = {
        "USD": {"rate": 0.0, "change_pct": 0.0},
        "AUD": {"rate": 0.0, "change_pct": 0.0},
        "updated_at": now,
    }

    # USD
    try:
        usd_ticker = yf.Ticker("KRW=X")
        curr_usd = float(usd_ticker.fast_info.last_price)
        prev_usd = float(usd_ticker.fast_info.previous_close)
        rates["USD"]["rate"] = curr_usd
        if prev_usd > 0:
            rates["USD"]["change_pct"] = ((curr_usd - prev_usd) / prev_usd) * 100
    except Exception as e:
        logger.warning(f"USD 환율 조회 오류: {e}")

    # AUD
    try:
        aud_ticker = yf.Ticker("AUDKRW=X")
        curr_aud = float(aud_ticker.fast_info.last_price)
        prev_aud = float(aud_ticker.fast_info.previous_close)
        rates["AUD"]["rate"] = curr_aud
        if prev_aud > 0:
            rates["AUD"]["change_pct"] = ((curr_aud - prev_aud) / prev_aud) * 100
    except Exception as e:
        logger.warning(f"AUD 환율 조회 오류: {e}")

    # 캐시 갱신 (항상 1시간 TTL)
    if "EXCHANGE_RATES" not in _REALTIME_CACHE:
        _REALTIME_CACHE["EXCHANGE_RATES"] = {}

    _REALTIME_CACHE["EXCHANGE_RATES"]["data"] = rates
    _REALTIME_CACHE["EXCHANGE_RATES"]["expires_at"] = now + timedelta(hours=1)

    return rates
