"""24H 실시간 주식 및 선물 시세 서비스."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import requests

from config import HYPERLIQUID_DEX, HYPERLIQUID_INFO_URL, HYPERLIQUID_SYMBOLS, MARKET_SCHEDULES
from services.price_service import get_exchange_rates
from utils.market_trend_service import _fetch_naver_kor_index_close

logger = logging.getLogger(__name__)


def _is_regular_session_open(country: str) -> bool:
    """해당 국가 정규장이 진행 중인지 — 정확한 현지 시간창 + 평일.

    (get_latest_trading_day 기반 판정은 진행 중 세션을 '미완료 거래일'로 보고 놓치므로 쓰지 않는다.
    평일 공휴일엔 시세 소스의 현재가가 직전 종가와 같아 평일만 보면 충분하다.)
    """
    schedule = MARKET_SCHEDULES.get(country) or {}
    tz_name = schedule.get("timezone")
    open_t = schedule.get("open")
    close_t = schedule.get("close")
    if not tz_name or open_t is None or close_t is None:
        return False
    now_local = datetime.now(ZoneInfo(tz_name))
    if now_local.weekday() >= 5:
        return False
    return open_t <= now_local.time() <= close_t


def _fetch_dex_ctxs(*, max_attempts: int = 3) -> dict[str, dict[str, Any]]:
    """Hyperliquid `metaAndAssetCtxs` 를 호출해 {심볼: ctx} 맵을 반환한다 (심볼=dex 접두사 제거)."""
    payload = {"type": "metaAndAssetCtxs", "dex": HYPERLIQUID_DEX}
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(HYPERLIQUID_INFO_URL, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            if attempt < max_attempts:
                logger.warning("Hyperliquid 조회 재시도 (%d/%d): %s", attempt, max_attempts, exc)
                time.sleep(0.6 * attempt)
                continue
            raise RuntimeError(f"Hyperliquid 시세 조회에 실패했습니다: {exc}") from exc

        if not (isinstance(data, list) and len(data) == 2):
            raise RuntimeError("Hyperliquid 응답 형식이 올바르지 않습니다.")
        universe = (data[0] or {}).get("universe") or []
        ctxs = data[1] or []
        result: dict[str, dict[str, Any]] = {}
        for u, ctx in zip(universe, ctxs):
            name = str(u.get("name") or "").split(":")[-1].strip().upper()
            if name:
                result[name] = ctx
        return result
    raise RuntimeError("Hyperliquid 시세 조회에 실패했습니다.")


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None





import threading

# 24H 캔들 데이터 메모리 캐시 및 갱신 동기화용 변수 (OHLC 구조화)
_HYPERLIQUID_CANDLE_CACHE: dict[str, list[dict[str, float]]] = {}
_CACHE_LAST_UPDATED: float = 0.0
_CACHE_LOCK = threading.Lock()
_CACHE_UPDATING = False

def _update_candle_caches_sync(usd_krw: float | None) -> None:
    """모든 심볼의 24H OHLC 캔들 데이터를 동기적으로 갱신한다."""
    global _CACHE_LAST_UPDATED
    if usd_krw is None:
        usd_krw = 1400.0

    hl_temp: dict[str, list[dict[str, float]]] = {}
    start_time = int((time.time() - 24 * 3600) * 1000)

    for spec in HYPERLIQUID_SYMBOLS:
        symbol = str(spec["symbol"]).upper()
        # 1. Hyperliquid 캔들 (Spot 토큰이므로 xyz: 접두사 필수)
        hl_symbol = f"xyz:{symbol}"
        hl_candles = []
        try:
            url = "https://api.hyperliquid.xyz/info"
            payload = {
                "type": "candleSnapshot",
                "req": {
                    "coin": hl_symbol,
                    "interval": "30m",
                    "startTime": start_time
                }
            }
            resp = requests.post(url, json=payload, timeout=5)
            data = resp.json()
            if isinstance(data, list):
                raw_candles = []
                for c in data:
                    o = _to_float(c.get("o"))
                    h = _to_float(c.get("h"))
                    low = _to_float(c.get("l"))
                    close_val = _to_float(c.get("c"))
                    if None not in (o, h, low, close_val):
                        if spec.get("type") == "stock" and spec.get("country") == "kor":
                            o *= usd_krw
                            h *= usd_krw
                            low *= usd_krw
                            close_val *= usd_krw
                        raw_candles.append({"o": o, "h": h, "l": low, "c": close_val})
                hl_candles = raw_candles[-48:]
        except Exception as exc:
            logger.warning("Hyperliquid 캔들 조회 실패 (%s): %s", hl_symbol, exc)

        if hl_candles:
            hl_temp[symbol] = hl_candles

    with _CACHE_LOCK:
        _HYPERLIQUID_CANDLE_CACHE.update(hl_temp)
        _CACHE_LAST_UPDATED = time.time()

def _trigger_candle_cache_update(usd_krw: float | None) -> None:
    """비동기 스레드를 띄워 백그라운드에서 캐시를 업데이트한다."""
    global _CACHE_UPDATING
    with _CACHE_LOCK:
        if _CACHE_UPDATING:
            return
        _CACHE_UPDATING = True

    def run():
        global _CACHE_UPDATING
        try:
            _update_candle_caches_sync(usd_krw)
        except Exception as exc:
            logger.error("비동기 캔들 캐시 갱신 중 예외 발생: %s", exc)
        finally:
            with _CACHE_LOCK:
                _CACHE_UPDATING = False

    t = threading.Thread(target=run, daemon=True)
    t.start()

def load_live_24h_quotes() -> dict[str, Any]:
    """설정된 심볼들의 24H 실시간 시세 + 실제가 대비 차이를 반환한다."""
    ctx_map = _fetch_dex_ctxs()

    # 환율(USD→KRW) — 한국 개별주 환산용
    try:
        rates = get_exchange_rates()
        usd_krw = _to_float((rates.get("USD") or {}).get("rate"))
    except Exception as exc:
        logger.warning("Hyperliquid 환율 조회 실패: %s", exc)
        usd_krw = None

    # 캐시 만료 여부 검사 및 비동기 업데이트 트리거 (만료 주기 5분)
    now = time.time()
    need_sync = False
    with _CACHE_LOCK:
        cache_age = now - _CACHE_LAST_UPDATED
        is_empty = not _HYPERLIQUID_CANDLE_CACHE

    if is_empty:
        # 최초 1회는 화면 렌더링을 위해 동기적으로 조회
        need_sync = True
    elif cache_age > 300:
        # 그 이후로는 백그라운드 비동기로 갱신하여 대기 딜레이 유발 방지
        _trigger_candle_cache_update(usd_krw)

    if need_sync:
        _update_candle_caches_sync(usd_krw)

    # 종목별 '장중/시간외' 표기 + 정규장 종가 기준에 쓸 각 시장의 정규장 개장 여부.
    # (정규장 종가/변동률은 한국주=네이버 일봉, 미국주/지수=yfinance 일봉으로 세션 인지 산출)
    us_market_open = _is_regular_session_open("us")
    kor_market_open = _is_regular_session_open("kor")

    quotes: list[dict[str, Any]] = []
    for spec in HYPERLIQUID_SYMBOLS:
        symbol = str(spec["symbol"]).upper()
        kind = spec.get("type", "stock")
        ctx = ctx_map.get(symbol) or {}
        mark = _to_float(ctx.get("markPx"))
        prev = _to_float(ctx.get("prevDayPx"))
        change_24h = ((mark / prev - 1.0) * 100.0) if (mark and prev) else None

        if kind == "index":
            currency = "POINT"
            country = "us"
            hyper_price = mark
            actual_price, actual_change_pct = _fetch_regular_close(str(spec.get("yahoo_symbol") or ""), us_market_open)
        elif spec["country"] == "kor":
            currency = "KRW"
            country = "kor"
            hyper_price = (mark * usd_krw) if (mark is not None and usd_krw) else None
            # US 와 동일: 네이버 정규장 일봉(세션 인지, 원화). 장중=전일 종가 기준(당일 변화),
            # 마감 후=당일 종가 기준. naver nowVal(실시간가) 기준의 '프리미엄만' 보이던 문제 해결.
            actual_price, actual_change_pct = _fetch_kr_regular_close(spec["actual_ticker"], kor_market_open)
        else:
            currency = "USD"
            country = "us"
            hyper_price = mark
            # 토스 base 는 마감 후에도 '어제 종가'라 시간외 변동만 떼어내지 못한다.
            # yfinance 정규장 일봉(세션 인지)으로 '직전 완료 정규장 종가'를 일관되게 쓴다.
            actual_price, actual_change_pct = _fetch_regular_close(str(spec.get("actual_ticker") or ""), us_market_open)

        diff_pct = (
            (hyper_price / actual_price - 1.0) * 100.0
            if (hyper_price is not None and actual_price and actual_price > 0)
            else None
        )

        # 종목 자기 시장의 정규장 개장 여부 (헤드라인 '장중/시간외' 표기용)
        session_open = kor_market_open if country == "kor" else us_market_open

        # 1. Hyperliquid 캔들 데이터 매핑
        hl_candles = _HYPERLIQUID_CANDLE_CACHE.get(symbol) or []

        quotes.append(
            {
                "symbol": symbol,
                "name": spec["name"],
                "type": kind,
                "country": country,
                "currency": currency,
                "hyper_price": hyper_price,
                "change_24h_pct": change_24h,
                "actual_price": actual_price,
                "actual_change_pct": actual_change_pct,
                "diff_pct": diff_pct,
                "session_open": session_open,
                "candles": hl_candles,
            }
        )

    return {"quotes": quotes, "usd_krw": usd_krw}


_REGULAR_CLOSE_CACHE: dict[tuple[str, bool], tuple[tuple[float | None, float | None], float]] = {}
_REGULAR_CLOSE_TTL = 60.0  # 정규장 종가는 하루 1회만 바뀌므로 짧은 TTL 로 yfinance 호출을 줄인다.


def _regular_close_from_series(closes, session_open: bool, tz_name: str) -> tuple[float | None, float | None]:
    """일봉 종가 시리즈에서 '직전 완료 정규장 종가 + 그 변동률'을 세션 인지로 뽑는다.

    장중이고 마지막 봉이 '오늘'(현지 기준)이면 형성 중이므로 제외하고 직전(어제) 종가를 앵커로 한다.
    → 장중엔 전일 종가 기준(당일 변화), 마감 후엔 당일 종가 기준(시간외 변화).
    """
    if closes is None or closes.empty:
        return None, None
    anchor = -1
    if session_open and len(closes) >= 2:
        try:
            if closes.index[-1].date() == datetime.now(ZoneInfo(tz_name)).date():
                anchor = -2
        except Exception:
            pass
    close = float(closes.iloc[anchor])
    change = None
    if len(closes) >= abs(anchor) + 1:
        prev = float(closes.iloc[anchor - 1])
        if prev:
            change = (close / prev - 1.0) * 100.0
    return close, change


def _fetch_regular_close(yahoo_symbol: str, session_open: bool) -> tuple[float | None, float | None]:
    """US 종목/지수의 '직전 완료 정규장 종가'와 그 정규장 변동률 (yfinance 정규장 일봉, 세션 인지).

    토스 base/naver nowVal 은 실시간/시간외가라 '직전 완료 종가'를 안정적으로 못 주므로 일봉을 쓴다.
    """
    if not yahoo_symbol:
        return None, None
    key = (f"us:{yahoo_symbol}", session_open)
    now = time.time()
    cached = _REGULAR_CLOSE_CACHE.get(key)
    if cached and now - cached[1] < _REGULAR_CLOSE_TTL:
        return cached[0]
    result: tuple[float | None, float | None] = (None, None)
    try:
        import yfinance as yf

        hist = yf.Ticker(yahoo_symbol).history(period="7d", interval="1d")
        closes = hist["Close"].dropna() if hist is not None and "Close" in hist else None
        result = _regular_close_from_series(closes, session_open, "America/New_York")
    except Exception as exc:
        logger.warning("Hyperliquid 정규장 종가 조회 실패 (%s): %s", yahoo_symbol, exc)
        result = cached[0] if cached else (None, None)
    _REGULAR_CLOSE_CACHE[key] = (result, now)
    return result


def _fetch_kr_regular_close(ticker: str, session_open: bool) -> tuple[float | None, float | None]:
    """한국 종목의 '직전 완료 정규장 종가'와 변동률 (네이버 일봉, 세션 인지). US 와 동일 의미.

    naver nowVal 은 장중에 실시간 정규장가라 '전일 종가 기준 당일 변화'를 못 준다.
    그래서 일봉으로 장중엔 전일 종가, 마감 후엔 당일 종가를 앵커로 쓴다.
    """
    if not ticker:
        return None, None
    key = (f"kr:{ticker}", session_open)
    now = time.time()
    cached = _REGULAR_CLOSE_CACHE.get(key)
    if cached and now - cached[1] < _REGULAR_CLOSE_TTL:
        return cached[0]
    result: tuple[float | None, float | None] = (None, None)
    try:
        closes = _fetch_naver_kor_index_close(ticker, 10)
        result = _regular_close_from_series(closes, session_open, "Asia/Seoul")
    except Exception as exc:
        logger.warning("Hyperliquid 한국 정규장 종가 조회 실패 (%s): %s", ticker, exc)
        result = cached[0] if cached else (None, None)
    _REGULAR_CLOSE_CACHE[key] = (result, now)
    return result


