"""24H 실시간 주식 및 선물 시세 서비스."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import requests

from config import HYPERLIQUID_DEX, HYPERLIQUID_INFO_URL, HYPERLIQUID_SYMBOLS, MARKET_SCHEDULES
from services.price_service import get_exchange_rates, get_realtime_snapshot

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

    # 실제 시장가 — 개별주는 국가별 스냅샷 일괄 조회
    kor_tickers = [
        s["actual_ticker"] for s in HYPERLIQUID_SYMBOLS if s.get("type") == "stock" and s["country"] == "kor"
    ]
    us_tickers = [s["actual_ticker"] for s in HYPERLIQUID_SYMBOLS if s.get("type") == "stock" and s["country"] == "us"]
    kor_snap = _safe_snapshot("kor", kor_tickers)
    us_snap = _safe_snapshot("us", us_tickers)
    # 종목별 '장중/시간외' 표기에 쓸 각 시장의 정규장 개장 여부.
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
            actual_price = _fetch_index_value(spec, us_market_open)
        elif spec["country"] == "kor":
            currency = "KRW"
            country = "kor"
            hyper_price = (mark * usd_krw) if (mark is not None and usd_krw) else None
            # naver nowVal 은 시간외 오염이 없어 폐장 시 = 정규장 종가, 장중 = 실시간 정규장가.
            actual_price = _to_float((kor_snap.get(spec["actual_ticker"]) or {}).get("nowVal"))
        else:
            currency = "USD"
            country = "us"
            hyper_price = mark
            # 토스 'close'(nowVal)는 프리/애프터가를 포함하므로, 정규장 기준은 항상
            # 'base'(prevClose=기준가=직전 정규장 종가)를 쓴다. → '정규장 종가 대비'가
            # 장중엔 당일 상승분, 폐장 후엔 시간외 변동을 일관되게 보여준다.
            actual_price = _to_float((us_snap.get(spec["actual_ticker"]) or {}).get("prevClose"))

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
                "diff_pct": diff_pct,
                "session_open": session_open,
                "candles": hl_candles,
            }
        )

    return {"quotes": quotes, "usd_krw": usd_krw}


def _fetch_index_value(spec: dict[str, Any], session_open: bool) -> float | None:
    """지수의 '직전 완료 정규장 종가'. (24시간 토큰의 '정규장 종가 대비' 기준값)

    intraday 1분봉은 휴장 중 며칠 stale 할 수 있어 정규장 일봉 종가를 쓴다.
    장중에는 마지막 일봉이 '형성 중인 오늘' 값이므로, 개별주(prevClose)와 일관되게
    당일 봉을 제외하고 직전(완료된) 종가를 기준으로 삼아 당일 상승분이 보이게 한다.
    """
    symbol = spec.get("yahoo_symbol")
    if not symbol:
        return None
    try:
        import yfinance as yf

        hist = yf.Ticker(symbol).history(period="7d", interval="1d")
        if hist is None or "Close" not in hist:
            return None
        closes = hist["Close"].dropna()
        if closes.empty:
            return None
        # 장중이고 마지막 봉이 오늘(ET)이면 형성 중 → 직전 완료 종가 사용.
        if session_open and len(closes) >= 2:
            try:
                last_date = closes.index[-1].date()
                today_et = datetime.now(ZoneInfo("America/New_York")).date()
                if last_date == today_et:
                    return float(closes.iloc[-2])
            except Exception:
                pass
        return float(closes.iloc[-1])
    except Exception as exc:
        logger.warning("Hyperliquid 지수 정규장 종가 조회 실패 (%s): %s", spec.get("symbol"), exc)
        return None


def _safe_snapshot(country: str, tickers: list[str]) -> dict[str, dict[str, Any]]:
    if not tickers:
        return {}
    try:
        return get_realtime_snapshot(country, tickers)
    except Exception as exc:
        logger.warning("Hyperliquid 실제가(%s) 조회 실패: %s", country, exc)
        return {}
