from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import datetime, timedelta
from math import isfinite
from typing import Any
from zoneinfo import ZoneInfo

from config import CACHE_TTL_COMPUTE, CACHE_TTL_LIVE, MARKET_SCHEDULES
from utils.data_loader import (
    fetch_au_quoteapi_snapshot,
    fetch_naver_etf_inav_snapshot,
    fetch_naver_stock_realtime_snapshot,
    fetch_naver_worldstock_snapshot,
    fetch_toss_us_stock_snapshot,
    get_latest_trading_day,
)
from utils.data_loader import (
    get_exchange_rate_series as load_exchange_rate_series,
)
from utils.logger import get_app_logger
from utils.yfinance_guard import yfinance_lock

logger = get_app_logger()

# ────────────────────────────────────────────
# 종목별 캐시 (ticker-level cache)
# ────────────────────────────────────────────
# key: "{country}:{ticker}" → {"data": {...}, "fetched_at": dt, "expires_at": dt, "source": str}
_TICKER_PRICE_CACHE: dict[str, dict[str, Any]] = {}

_KOR_ACTIVE_TTL_SECONDS = CACHE_TTL_LIVE
_AU_ACTIVE_TTL_SECONDS = CACHE_TTL_LIVE
_US_ACTIVE_TTL_SECONDS = CACHE_TTL_LIVE
_WORLDSTOCK_TTL_SECONDS = CACHE_TTL_COMPUTE
_YAHOO_SYMBOL_TTL_SECONDS = CACHE_TTL_COMPUTE
_IDLE_TTL_SECONDS = CACHE_TTL_LIVE
_FX_TTL_SECONDS = CACHE_TTL_COMPUTE
# 환율은 유효기간 내 값을 DB로 공유해 별도 배치 프로세스의 중복 조회를 줄인다.
_FX_FETCH_MAX_ATTEMPTS = 3
_FX_RETRY_BASE_SECONDS = 3.0

# 하위 호환: 기존 쿼리 단위 캐시 (환율 전용으로 유지)
_FX_CACHE: dict[str, dict[str, Any]] = {}


def get_realtime_snapshot(country_code: str, tickers: Sequence[str]) -> dict[str, dict[str, float]]:
    """국가별 실시간 스냅샷을 반환한다. 종목별 캐시를 사용한다."""

    country = _normalize_country_code(country_code)
    normalized_tickers = _normalize_tickers(tickers)
    if not normalized_tickers:
        return {}

    now = datetime.now()
    ttl_seconds = _get_realtime_ttl_seconds(country)

    # 캐시 히트/만료 분리
    cached_result: dict[str, dict[str, float]] = {}
    expired_tickers: list[str] = []

    for ticker in normalized_tickers:
        cache_key = f"{country}:{ticker}"
        entry = _TICKER_PRICE_CACHE.get(cache_key)
        # expires_at 대신 fetched_at + 현재 TTL로 판단한다.
        # 장 전에 idle TTL(3600s)로 캐시된 항목도 개장 후 active TTL(30s) 경과 즉시 만료된다.
        fetched_at = entry.get("fetched_at") if entry else None
        is_alive = (
            entry is not None and isinstance(fetched_at, datetime) and (now - fetched_at).total_seconds() < ttl_seconds
        )
        if is_alive:
            cached_result[ticker] = entry["data"]  # type: ignore[index]
        else:
            expired_tickers.append(ticker)

    if not expired_tickers:
        return cached_result

    # 만료 종목만 벌크 조회
    try:
        fetched_data, source = _fetch_realtime_snapshot(country, expired_tickers)
    except Exception as exc:
        # 조회 실패 시 stale 캐시 재사용
        stale_result = _reuse_stale_ticker_cache(country, expired_tickers, exc)
        return {**cached_result, **stale_result}

    # 종목별 캐시 갱신
    fetched_at = datetime.now()
    expires_at = fetched_at + timedelta(seconds=ttl_seconds)
    for ticker, data in fetched_data.items():
        _TICKER_PRICE_CACHE[f"{country}:{ticker}"] = {
            "data": data,
            "fetched_at": fetched_at,
            "expires_at": expires_at,
            "source": source,
            "is_stale": False,
        }

    # 조회 요청했지만 API 응답에 없는 종목은 stale 캐시 재사용
    missing_tickers = [t for t in expired_tickers if t not in fetched_data]
    for ticker in missing_tickers:
        cache_key = f"{country}:{ticker}"
        entry = _TICKER_PRICE_CACHE.get(cache_key)
        if entry and "data" in entry:
            cached_result[ticker] = entry["data"]

    # 캐시 히트 + 새로 조회한 데이터 병합
    fetched_filtered = {t: fetched_data[t] for t in expired_tickers if t in fetched_data}
    return {**cached_result, **fetched_filtered}


def get_worldstock_snapshot(reuters_codes: Sequence[str]) -> dict[str, dict[str, float | str]]:
    """Reuters code 기반 해외 종목 지연 시세를 반환한다."""
    from utils.symbol_resolution_blacklist import get_active_blacklist

    normalized_codes = _normalize_tickers(reuters_codes)
    if not normalized_codes:
        return {}

    blacklist = get_active_blacklist()
    now = datetime.now()
    cached_result: dict[str, dict[str, float | str]] = {}
    expired_codes: list[str] = []

    for code in normalized_codes:
        if code in blacklist:
            continue
        cache_key = f"worldstock:{code}"
        entry = _TICKER_PRICE_CACHE.get(cache_key)
        fetched_at = entry.get("fetched_at") if entry else None
        is_alive = (
            entry is not None
            and isinstance(fetched_at, datetime)
            and (now - fetched_at).total_seconds() < _WORLDSTOCK_TTL_SECONDS
        )
        if is_alive:
            cached_result[code] = entry["data"]  # type: ignore[index]
        else:
            expired_codes.append(code)

    if not expired_codes:
        return cached_result

    try:
        fetched_data = fetch_naver_worldstock_snapshot(expired_codes)
    except Exception as exc:
        logger.warning("네이버 worldstock 조회 실패로 stale 캐시를 재사용합니다. error=%s", exc)
        stale_result: dict[str, dict[str, float | str]] = {}
        for code in expired_codes:
            entry = _TICKER_PRICE_CACHE.get(f"worldstock:{code}")
            if entry and "data" in entry:
                entry["is_stale"] = True
                stale_result[code] = entry["data"]
        return {**cached_result, **stale_result}

    fetched_at = datetime.now()
    expires_at = fetched_at + timedelta(seconds=_WORLDSTOCK_TTL_SECONDS)
    for code, data in fetched_data.items():
        _TICKER_PRICE_CACHE[f"worldstock:{code}"] = {
            "data": data,
            "fetched_at": fetched_at,
            "expires_at": expires_at,
            "source": "naver_worldstock",
            "is_stale": False,
        }

    fetched_filtered = {code: fetched_data[code] for code in expired_codes if code in fetched_data}

    # 일괄 조회 후 응답에 빠진 코드는 영구 누락 가능성 → 블랙리스트에 마킹
    from utils.symbol_resolution_blacklist import mark_failed

    for code in expired_codes:
        if code not in fetched_data:
            mark_failed(code, source="네이버", reason="Worldstock: 응답에 누락된 코드")

    return {**cached_result, **fetched_filtered}


def get_yahoo_symbol_snapshot(symbols: Sequence[str]) -> dict[str, dict[str, float]]:
    """Yahoo 심볼 기반 지연 시세를 반환한다."""

    normalized_symbols = _normalize_tickers(symbols)
    if not normalized_symbols:
        return {}

    now = datetime.now()
    cached_result: dict[str, dict[str, float]] = {}
    expired_symbols: list[str] = []

    for symbol in normalized_symbols:
        cache_key = f"yahoo:{symbol}"
        entry = _TICKER_PRICE_CACHE.get(cache_key)
        fetched_at = entry.get("fetched_at") if entry else None
        is_alive = (
            entry is not None
            and isinstance(fetched_at, datetime)
            and (now - fetched_at).total_seconds() < _YAHOO_SYMBOL_TTL_SECONDS
        )
        if is_alive:
            cached_result[symbol] = entry["data"]  # type: ignore[index]
        else:
            expired_symbols.append(symbol)

    if not expired_symbols:
        return cached_result

    fetched_data = _fetch_yahoo_symbol_snapshot(expired_symbols)

    fetched_at = datetime.now()
    expires_at = fetched_at + timedelta(seconds=_YAHOO_SYMBOL_TTL_SECONDS)
    for symbol, data in fetched_data.items():
        _TICKER_PRICE_CACHE[f"yahoo:{symbol}"] = {
            "data": data,
            "fetched_at": fetched_at,
            "expires_at": expires_at,
            "source": "yahoo_download",
            "is_stale": False,
        }

    fetched_filtered = {symbol: fetched_data[symbol] for symbol in expired_symbols if symbol in fetched_data}
    return {**cached_result, **fetched_filtered}


def get_exchange_rates(currencies: Sequence[str] | None = None) -> dict[str, Any]:
    """원화 환율을 반환한다 — ``currencies`` 를 주면 **그 통화만** 조회한다.

    소스는 야후 하나다. 현재가와 전일 종가가 같은 소스라 변동률 기준이 어긋나지 않는다.

    캐시는 **통화 단위**다. 예전에는 8개를 한 덩어리(`fx:major`)로 받아 캐시했는데,
    필요한 통화가 하나뿐인 화면도 8개 값을 전부 기다려야 했다(실측 3.7초). 통화별로
    나누면 이미 받아 둔 통화는 그대로 쓰고 모자란 것만 조회한다.

    ``currencies=None`` 은 지원 통화 전체다 — 무엇이 필요한지 모르는 호출부의 기본값이다.
    """
    wanted = (
        list(_FX_SYMBOL_BY_CODE)
        if currencies is None
        else [c for c in dict.fromkeys(str(c).strip().upper() for c in currencies) if c in _FX_SYMBOL_BY_CODE]
    )
    if not wanted:
        return {}

    now = datetime.now()
    rates: dict[str, Any] = {}
    missing: list[str] = []
    for currency in wanted:
        entry = _FX_CACHE.get(f"fx:{currency}")
        if _is_cache_alive(entry, now):
            rates[currency] = dict(entry["data"])
        else:
            missing.append(currency)

    if not missing:
        return rates

    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("환율 공유 캐시 DB 연결 실패")
    collection = db["fx_quote_cache"]
    for doc in collection.find({"_id": {"$in": missing}, "expires_at": {"$gt": datetime.utcnow()}}):
        currency = doc["_id"]
        rates[currency] = dict(doc["data"])
        _FX_CACHE[f"fx:{currency}"] = {key: doc[key] for key in ("data", "fetched_at", "expires_at")}
        # DB 시각은 UTC, 기존 메모리 캐시 시각은 로컬이다.
        remaining = max(0, (doc["expires_at"] - datetime.utcnow()).total_seconds())
        _FX_CACHE[f"fx:{currency}"]["expires_at"] = datetime.now() + timedelta(seconds=remaining)
        _FX_CACHE[f"fx:{currency}"]["fetched_at"] = datetime.now() - (datetime.utcnow() - doc["fetched_at"])
        missing.remove(currency)
    if not missing:
        return rates

    try:
        fetched = _fetch_exchange_rates(missing)
    except Exception as exc:
        for currency in missing:
            stale = _reuse_stale_fx_cache(f"fx:{currency}", exc)
            if stale is not None:
                rates[currency] = dict(stale)
        if len(rates) == len(wanted):
            return rates
        raise

    for currency in missing:
        value = fetched.get(currency)
        if not isinstance(value, dict):
            continue
        rates[currency] = value
        saved_at = datetime.utcnow()
        collection.update_one(
            {"_id": currency},
            {
                "$set": {
                    "data": dict(value),
                    "fetched_at": saved_at,
                    "expires_at": saved_at + timedelta(seconds=_FX_TTL_SECONDS),
                }
            },
            upsert=True,
        )
        _FX_CACHE[f"fx:{currency}"] = {
            "data": dict(value),
            "fetched_at": now,
            "expires_at": now + timedelta(seconds=_FX_TTL_SECONDS),
            "is_stale": False,
        }
    rates["updated_at"] = fetched.get("updated_at", now)
    return rates


def get_exchange_rate_series(
    start_date: str | Any,
    end_date: str | Any,
    symbol: str = "KRW=X",
    *,
    allow_partial: bool = False,
) -> Any:
    """환율 시계열을 반환한다."""

    return load_exchange_rate_series(
        start_date=start_date,
        end_date=end_date,
        symbol=symbol,
        allow_partial=allow_partial,
    )


def clear_price_service_cache() -> None:
    """가격 서비스 메모리 캐시를 초기화한다."""

    _TICKER_PRICE_CACHE.clear()
    _FX_CACHE.clear()


def get_realtime_cache_meta(cache_key: str) -> dict[str, Any] | None:
    """캐시 메타데이터를 반환한다. 하위 호환용."""

    # 종목별 캐시에서 조회 시도
    entry = _TICKER_PRICE_CACHE.get(str(cache_key or "").strip())
    if not entry:
        # 환율 캐시에서 조회
        entry = _FX_CACHE.get(str(cache_key or "").strip())
    if not entry:
        return None
    return {
        "fetched_at": entry.get("fetched_at"),
        "expires_at": entry.get("expires_at"),
        "is_stale": entry.get("is_stale", False),
        "source": entry.get("source"),
        "size": 1 if "data" in entry else 0,
    }


def get_realtime_snapshot_meta(country_code: str, tickers: Sequence[str]) -> dict[str, Any] | None:
    """국가별 실시간 스냅샷 캐시 메타데이터를 반환한다."""

    country = _normalize_country_code(country_code)
    normalized_tickers = _normalize_tickers(tickers)
    if not normalized_tickers:
        return None

    # 종목별 캐시에서 가장 최근 fetched_at을 찾는다
    latest_fetched_at: datetime | None = None
    latest_expires_at: datetime | None = None
    latest_source: str | None = None
    is_any_stale = False
    count = 0

    for ticker in normalized_tickers:
        entry = _TICKER_PRICE_CACHE.get(f"{country}:{ticker}")
        if not entry:
            continue
        count += 1
        fetched_at = entry.get("fetched_at")
        if isinstance(fetched_at, datetime):
            if latest_fetched_at is None or fetched_at > latest_fetched_at:
                latest_fetched_at = fetched_at
                latest_expires_at = entry.get("expires_at")
                latest_source = entry.get("source")
        if entry.get("is_stale"):
            is_any_stale = True

    if count == 0:
        return None

    return {
        "fetched_at": latest_fetched_at,
        "expires_at": latest_expires_at,
        "is_stale": is_any_stale,
        "source": latest_source,
        "size": count,
    }


# ────────────────────────────────────────────
# 내부 함수
# ────────────────────────────────────────────


def _normalize_country_code(country_code: str) -> str:
    country = str(country_code or "").strip().lower()
    if not country:
        raise ValueError("country_code는 필수입니다.")
    if country not in MARKET_SCHEDULES:
        raise ValueError(f"지원하지 않는 country_code입니다: {country}")
    return country


def _normalize_tickers(tickers: Sequence[str]) -> tuple[str, ...]:
    normalized = sorted({str(ticker or "").strip().upper() for ticker in tickers if str(ticker or "").strip()})
    return tuple(normalized)


def _is_cache_alive(cache_entry: dict[str, Any] | None, now: datetime) -> bool:
    if not cache_entry:
        return False
    expires_at = cache_entry.get("expires_at")
    if not isinstance(expires_at, datetime):
        return False
    return now < expires_at


def _fetch_realtime_snapshot(country: str, tickers: Sequence[str]) -> tuple[dict[str, dict[str, float]], str]:
    if country == "kor":
        etf_snapshot = fetch_naver_etf_inav_snapshot(tickers)
        missing_tickers = [ticker for ticker in tickers if ticker not in etf_snapshot]
        if not missing_tickers:
            return etf_snapshot, "naver_etf"

        stock_snapshot = fetch_naver_stock_realtime_snapshot(missing_tickers)
        merged_snapshot = dict(etf_snapshot)
        merged_snapshot.update(stock_snapshot)
        return merged_snapshot, "mixed"

    if country == "au":
        return fetch_au_quoteapi_snapshot(tickers), "au_quoteapi"

    if country == "us":
        return fetch_toss_us_stock_snapshot(tickers), "toss_invest"

    raise ValueError(f"지원하지 않는 country_code입니다: {country}")


def _reuse_stale_ticker_cache(
    country: str,
    tickers: Sequence[str],
    exc: Exception,
) -> dict[str, dict[str, float]]:
    """조회 실패 시 만료된 종목별 캐시를 재사용한다."""

    logger.warning("실시간 가격 조회 실패로 stale 캐시를 재사용합니다. country=%s error=%s", country, exc)
    result: dict[str, dict[str, float]] = {}
    for ticker in tickers:
        cache_key = f"{country}:{ticker}"
        entry = _TICKER_PRICE_CACHE.get(cache_key)
        if entry and "data" in entry:
            entry["is_stale"] = True
            result[ticker] = entry["data"]
    return result


def _reuse_stale_fx_cache(cache_key: str, exc: Exception) -> dict[str, Any] | None:
    cached_entry = _FX_CACHE.get(cache_key)
    if not cached_entry or "data" not in cached_entry:
        return None

    cached_entry["is_stale"] = True
    logger.warning("환율 조회 실패로 stale 캐시를 재사용합니다. key=%s error=%s", cache_key, exc)
    return dict(cached_entry["data"])


def _get_realtime_ttl_seconds(country: str) -> int:
    return 60


def _is_market_active(country: str) -> bool:
    schedule = MARKET_SCHEDULES.get(country)
    if not schedule:
        raise ValueError(f"시장 스케줄이 없습니다: {country}")

    timezone_name = str(schedule.get("timezone") or "").strip()
    if not timezone_name:
        raise ValueError(f"시장 타임존이 없습니다: {country}")

    now_local = datetime.now(ZoneInfo(timezone_name))
    latest_trading_day = get_latest_trading_day(country).normalize()
    if latest_trading_day.date() != now_local.date():
        return False

    open_time = schedule.get("open")
    close_time = schedule.get("close")
    open_offset_minutes = int(schedule.get("open_offset_minutes") or 0)
    close_offset_minutes = int(schedule.get("close_offset_minutes") or 0)
    if open_time is None or close_time is None:
        raise ValueError(f"시장 시간이 없습니다: {country}")

    market_open_dt = datetime.combine(now_local.date(), open_time, tzinfo=now_local.tzinfo) - timedelta(
        minutes=open_offset_minutes
    )
    market_close_dt = datetime.combine(now_local.date(), close_time, tzinfo=now_local.tzinfo) + timedelta(
        minutes=close_offset_minutes
    )
    return market_open_dt <= now_local <= market_close_dt


# 지원 통화 → 야후 심볼. 조회 대상은 호출부가 고른다 — 8개를 전부 받는 데 실측 3.7초가
# 들었고(통화당 ~0.39초 순차), 국내 구성종목뿐인 ETF 상세는 그 결과를 하나도 쓰지 않았다.
_FX_SYMBOL_BY_CODE: dict[str, str] = {
    "USD": "KRW=X",
    "AUD": "AUDKRW=X",
    "JPY": "JPYKRW=X",
    "CNY": "CNYKRW=X",
    "TWD": "TWDKRW=X",
    "HKD": "HKDKRW=X",
    "GBP": "GBPKRW=X",
    "EUR": "EURKRW=X",
}


def _fetch_exchange_rates(currencies: Sequence[str] | None = None) -> dict[str, Any]:
    import yfinance as yf

    if currencies is None:
        mapping = dict(_FX_SYMBOL_BY_CODE)
    else:
        mapping = {c: _FX_SYMBOL_BY_CODE[c] for c in currencies if c in _FX_SYMBOL_BY_CODE}
    if not mapping:
        return {"updated_at": datetime.now()}

    def _put(currency: str, current_rate: float, previous_close: float) -> None:
        change_pct = ((current_rate - previous_close) / previous_close * 100.0) if previous_close > 0 else 0.0
        rates[currency] = {"rate": current_rate, "change_pct": change_pct}

    rates: dict[str, Any] = {}
    # 통화별 quote 요약(fast_info)으로 (현재가, 전일 종가)를 받는다. 예전엔 일봉 일괄
    # 조회(_fetch_bulk_fx_closes)로 요청 수를 줄였는데, 야후 일봉의 직전 봉 종가가 실제
    # 공식 전일 종가보다 하루 이상 뒤처져 변동률이 크게 부풀었다(관측: USD +1.65% vs
    # 실제 +0.27%, 2026-09-16). 레이트리밋은 아래 재시도 + TTL 캐시로 감당한다.
    pending = dict(mapping)
    errors: dict[str, str] = {}
    for attempt in range(1, _FX_FETCH_MAX_ATTEMPTS + 1):
        failed: dict[str, str] = {}
        for currency, symbol in pending.items():
            try:
                with yfinance_lock():
                    ticker = yf.Ticker(symbol)
                    _put(currency, float(ticker.fast_info.last_price), float(ticker.fast_info.previous_close))
            except Exception as exc:
                errors[currency] = f"{type(exc).__name__}: {str(exc)[:240]}"
                if attempt >= _FX_FETCH_MAX_ATTEMPTS:
                    logger.warning("%s 환율 조회 실패: %s", currency, exc)
                failed[currency] = symbol

        # 남은 목록을 먼저 갱신한다 — 성공해서 break 할 때도 pending 이 비어야
        # 아래 최종 검사에 걸리지 않는다.
        pending = failed
        if not pending:
            break
        if attempt < _FX_FETCH_MAX_ATTEMPTS:
            logger.warning(
                "환율 일시 오류 재시도 (%d/%d): %s",
                attempt,
                _FX_FETCH_MAX_ATTEMPTS,
                ", ".join(sorted(pending)),
            )
            # 레이트리밋은 초 단위로 풀린다 — 짧게 끊지 말고 넉넉히 기다린다.
            time.sleep(_FX_RETRY_BASE_SECONDS * attempt)

    if pending:
        joined = ", ".join(sorted(pending))
        raise RuntimeError(
            f"Yahoo 환율 조회 실패 ({_FX_FETCH_MAX_ATTEMPTS}회 시도): {joined}. "
            + " | ".join(f"{currency}({mapping[currency]}): {errors[currency]}" for currency in sorted(pending))
        )

    rates["updated_at"] = datetime.now()
    return rates


def _fetch_yahoo_symbol_snapshot(symbols: Sequence[str]) -> dict[str, dict[str, float]]:
    import pandas as pd
    import yfinance as yf

    from utils.symbol_resolution_blacklist import get_active_blacklist, mark_failed

    normalized_symbols = [str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip()]
    if not normalized_symbols:
        return {}

    blacklist = get_active_blacklist()
    normalized_symbols = [s for s in normalized_symbols if s not in blacklist]
    if not normalized_symbols:
        return {}

    result: dict[str, dict[str, float]] = {}
    if "NQ=F" in normalized_symbols:
        # 연속 선물 일봉은 월물 전환·사후 갱신으로 quote의 전일 기준과 달라질 수 있다.
        # 같은 응답의 가격·기준가·변동률만 사용하며 일봉이나 시간봉으로 대체하지 않는다.
        with yfinance_lock():
            metadata = yf.Ticker("NQ=F").get_history_metadata()
        values = {}
        for field in ("regularMarketPrice", "previousClose", "regularMarketChangePercent"):
            value = metadata.get(field)
            if value is None or not isfinite(float(value)):
                raise RuntimeError(f"나스닥선물 시세 응답에 유효한 {field} 값이 없습니다.")
            values[field] = float(value)
        if values["regularMarketPrice"] <= 0 or values["previousClose"] <= 0:
            raise RuntimeError("나스닥선물 현재가·전일 기준가는 양수여야 합니다.")
        result["NQ=F"] = {
            "nowVal": values["regularMarketPrice"],
            "prevClose": values["previousClose"],
            "changeRate": round(values["regularMarketChangePercent"], 2),
        }
        normalized_symbols = [symbol for symbol in normalized_symbols if symbol != "NQ=F"]
        if not normalized_symbols:
            return result
    downloaded = yf.download(
        normalized_symbols,
        period="7d",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if downloaded is None or downloaded.empty:
        for symbol in normalized_symbols:
            mark_failed(symbol, source="야후", reason="yfinance: 일괄 조회 결과 비어있음")
        return result

    def _extract_symbol_frame(symbol: str) -> pd.DataFrame | None:
        if isinstance(downloaded.columns, pd.MultiIndex):
            if symbol not in downloaded.columns.get_level_values(0):
                return None
            frame = downloaded[symbol].copy()
        else:
            if len(normalized_symbols) != 1:
                return None
            frame = downloaded.copy()
        if frame.empty or "Close" not in frame.columns:
            return None
        return frame

    for symbol in normalized_symbols:
        frame = _extract_symbol_frame(symbol)
        if frame is None:
            mark_failed(symbol, source="야후", reason="yfinance: 가격 데이터 없음")
            continue

        close_series = pd.to_numeric(frame["Close"], errors="coerce").dropna()
        if close_series.empty:
            continue

        now_val = float(close_series.iloc[-1])
        prev_close = None
        if len(close_series) >= 2:
            prev_close = float(close_series.iloc[-2])

        # 환율의 기존 quote 기준가 처리는 유지한다.
        if symbol.endswith("=X"):
            try:
                with yfinance_lock():
                    official_prev = float(yf.Ticker(symbol).fast_info.previous_close)
                if official_prev > 0:
                    prev_close = official_prev
            except Exception as exc:
                logger.warning("공식 전일 종가 조회 실패(%s) — 일봉 직전 종가 사용: %s", symbol, exc)

        change_rate = None
        if prev_close not in (None, 0):
            change_rate = round(((now_val - prev_close) / prev_close) * 100.0, 2)

        result[symbol] = {
            "nowVal": now_val,
            "prevClose": prev_close,
            "changeRate": change_rate,
        }

    return result


__all__ = [
    "clear_price_service_cache",
    "get_exchange_rate_series",
    "get_exchange_rates",
    "get_realtime_cache_meta",
    "get_realtime_snapshot",
    "get_realtime_snapshot_meta",
    "get_worldstock_snapshot",
    "get_yahoo_symbol_snapshot",
]
