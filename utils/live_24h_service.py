"""24H 실시간 주식 및 선물 시세 서비스."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from config import HYPERLIQUID_DEX, HYPERLIQUID_INFO_URL, HYPERLIQUID_SYMBOLS
from services.price_service import (
    get_confirmed_regular_close,
    get_exchange_rates,
    get_realtime_snapshot,
    quote_daily_change,
)
from utils.data_loader import resolve_toss_us_product_codes
from utils.market_session import CLOSED, REGULAR, market_session

logger = logging.getLogger(__name__)


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

_TOSS_STOCK_SPECS = (
    {"symbol": "SKHY_TOSS", "name": "SK하이닉스", "tickers": ("SKHY",)},
    {"symbol": "MU_TOSS", "name": "마이크론", "tickers": ("MU",)},
)

_KR_TOSS_STOCK_SPECS = (
    {"symbol": "SKHX_KR_TOSS", "name": "SK하이닉스", "ticker": "000660"},
    {"symbol": "SMSN_KR_TOSS", "name": "삼성전자", "ticker": "005930"},
)

# 지표 카드(나스닥 100 선물 · 달러 환율 · VIX) — 시세·캔들 모두 야후.
# 상단 헤더 환율(get_exchange_rates, KRW=X)과 같은 소스라 값이 어긋나지 않는다.
# 환율은 사실상 실시간이고, 선물(CME)·VIX(CBOE)는 야후가 10~15분 지연 시세를 준다
# — 화면이 그 카드에 '15분 지연' 배지를 붙인다.
_YAHOO_INDICATOR_SYMBOLS = (
    ("NQ_FUT", "NQ=F"),
    ("USDKRW", "KRW=X"),
    ("VIX", "^VIX"),
)


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
            payload = {"type": "candleSnapshot", "req": {"coin": hl_symbol, "interval": "15m", "startTime": start_time}}
            resp = requests.post(url, json=payload, timeout=5)
            data = resp.json()
            if isinstance(data, list):
                raw_candles = []
                for c in data:
                    timestamp = _to_float(c.get("t"))
                    o = _to_float(c.get("o"))
                    h = _to_float(c.get("h"))
                    low = _to_float(c.get("l"))
                    close_val = _to_float(c.get("c"))
                    if None not in (timestamp, o, h, low, close_val):
                        if spec.get("type") == "stock" and spec.get("country") == "kor":
                            o *= usd_krw
                            h *= usd_krw
                            low *= usd_krw
                            close_val *= usd_krw
                        raw_candles.append({"t": int(timestamp), "o": o, "h": h, "l": low, "c": close_val})
                hl_candles = raw_candles[-96:]
        except Exception as exc:
            logger.warning("Hyperliquid 캔들 조회 실패 (%s): %s", hl_symbol, exc)

        if hl_candles:
            hl_temp[symbol] = hl_candles

    from services.toss_market_service import fetch_toss_candles, fetch_toss_stock_candles

    toss_stock_tickers = [ticker for spec in _TOSS_STOCK_SPECS for ticker in spec["tickers"]]
    toss_product_codes = resolve_toss_us_product_codes(toss_stock_tickers)
    for spec in _TOSS_STOCK_SPECS:
        resolved_ticker = next((ticker for ticker in spec["tickers"] if ticker in toss_product_codes), None)
        if resolved_ticker is None:
            continue
        toss_code = toss_product_codes[resolved_ticker]
        try:
            hl_temp[spec["symbol"]] = fetch_toss_candles(toss_code, interval="min:15", count=96)
        except Exception as exc:
            logger.warning("토스 미국주식 캔들 조회 실패 (%s/%s): %s", resolved_ticker, toss_code, exc)

    for spec in _KR_TOSS_STOCK_SPECS:
        toss_code = f"A{spec['ticker']}"
        try:
            hl_temp[spec["symbol"]] = fetch_toss_stock_candles(
                toss_code,
                securities_type="kr-s",
                interval="min:15",
                count=96,
            )
        except Exception as exc:
            logger.warning("토스 국내주식 캔들 조회 실패 (%s): %s", toss_code, exc)

    # 지표 15분봉 (최근 24시간 = 96개) — 나스닥 100 선물 · 달러 환율 · VIX 전부 야후.
    # 지표 카드의 현재가·기준가와 소스를 맞춘다(상단 헤더 환율도 같은 야후 기준).
    import yfinance as yf

    from utils.yfinance_guard import yfinance_lock

    for cache_key, yf_symbol in _YAHOO_INDICATOR_SYMBOLS:
        try:
            # 동시 호출 시 남의 티커 데이터를 받는 것을 막는다(utils/yfinance_guard).
            with yfinance_lock():
                bars = yf.Ticker(yf_symbol).history(period="2d", interval="15m")
            candles = []
            for timestamp, row in bars.iterrows():
                o, h, low, c = (_to_float(row.get(k)) for k in ("Open", "High", "Low", "Close"))
                if None not in (o, h, low, c):
                    candles.append(
                        {
                            "t": int(timestamp.timestamp() * 1000),
                            "o": o,
                            "h": h,
                            "l": low,
                            "c": c,
                        }
                    )
            if candles:
                hl_temp[cache_key] = candles[-96:]
        except Exception as exc:
            logger.warning("야후 지표 캔들 조회 실패 (%s): %s", yf_symbol, exc)

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

    # 카드의 세션 표시는 공통 시장 시간표를 따른다.
    us_price_data_session = market_session("us")["session"]
    kr_price_data_session = market_session("kor")["session"]
    us_market_open = us_price_data_session == REGULAR
    kor_market_open = kr_price_data_session == REGULAR

    quotes: list[dict[str, Any]] = []

    # ── 야후 지표 카드 (나스닥 100 선물 · 달러 환율 · VIX) — 실패 시 카드만 생략 ──
    # 환율은 상단 헤더와 같은 함수(get_exchange_rates)를 써서 값·캐시가 헤더와 일치한다.
    try:
        from services.price_service import get_yahoo_symbol_snapshot

        snapshot = get_yahoo_symbol_snapshot([code for _, code in _YAHOO_INDICATOR_SYMBOLS if code != "KRW=X"])
        # 환율 기준가(전일 종가)는 rate·change_pct 에서 되구한다 — 둘 다 같은 야후 조회다.
        usd_change = _to_float((rates.get("USD") or {}).get("change_pct")) if usd_krw is not None else None
        usd_base = usd_krw / (1.0 + usd_change / 100.0) if usd_krw is not None and usd_change is not None else None
        indicator_prices = {
            "NQ_FUT": (
                _to_float((snapshot.get("NQ=F") or {}).get("nowVal")),
                _to_float((snapshot.get("NQ=F") or {}).get("prevClose")),
            ),
            "USDKRW": (usd_krw, usd_base),
            "VIX": (
                _to_float((snapshot.get("^VIX") or {}).get("nowVal")),
                _to_float((snapshot.get("^VIX") or {}).get("prevClose")),
            ),
        }
        for symbol, name, currency in (
            ("NQ_FUT", "나스닥 100 선물", "POINT"),
            ("USDKRW", "달러 환율", "FX"),
            ("VIX", "VIX", "POINT"),
        ):
            latest, base = indicator_prices[symbol]
            candles = _HYPERLIQUID_CANDLE_CACHE.get(symbol) or []
            change_24h = None
            if len(candles) >= 2 and candles[0].get("c"):
                change_24h = (candles[-1]["c"] / candles[0]["c"] - 1.0) * 100.0
            quotes.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "type": "yahoo",
                    "country": "us",
                    "currency": currency,
                    "hyper_price": latest,
                    "change_24h_pct": change_24h,
                    "actual_price": base,
                    "actual_change_pct": None,
                    "diff_pct": ((latest / base - 1.0) * 100.0) if (latest and base) else None,
                    "session_open": True,
                    "candles": candles,
                }
            )
    except Exception as exc:
        logger.warning("야후 지표 카드 구성 실패: %s", exc)

    toss_stock_tickers = [ticker for spec in _TOSS_STOCK_SPECS for ticker in spec["tickers"]]
    try:
        toss_stock_snapshot = get_realtime_snapshot("us", toss_stock_tickers)
    except Exception as exc:
        logger.warning("토스 미국주식 카드 구성 실패: %s", exc)
        toss_stock_snapshot = {}

    for spec in _TOSS_STOCK_SPECS:
        resolved_ticker = next((ticker for ticker in spec["tickers"] if ticker in toss_stock_snapshot), None)
        if resolved_ticker is None:
            continue
        info = toss_stock_snapshot[resolved_ticker]
        latest = _to_float(info.get("nowVal"))
        base, change_pct = quote_daily_change("us", resolved_ticker, info)
        candles = _HYPERLIQUID_CANDLE_CACHE.get(spec["symbol"]) or []
        change_24h = None
        if len(candles) >= 2 and candles[0].get("c"):
            change_24h = (candles[-1]["c"] / candles[0]["c"] - 1.0) * 100.0
        quotes.append(
            {
                "symbol": spec["symbol"],
                "name": spec["name"],
                "type": "toss",
                "country": "us",
                "currency": "USD",
                "hyper_price": latest,
                "change_24h_pct": change_24h,
                "actual_price": base,
                "actual_change_pct": None,
                "diff_pct": change_pct,
                "session_open": us_market_open,
                "price_data_open": us_price_data_session != CLOSED,
                "price_data_session": us_price_data_session,
                "candles": candles,
                "source_ticker": resolved_ticker,
            }
        )

    kr_toss_tickers = [spec["ticker"] for spec in _KR_TOSS_STOCK_SPECS]
    kr_toss_stock_snapshot = get_realtime_snapshot("kor", kr_toss_tickers)
    for spec in _KR_TOSS_STOCK_SPECS:
        ticker = spec["ticker"]
        info = kr_toss_stock_snapshot.get(ticker) or {}
        latest = _to_float(info.get("nowVal"))
        previous_close, change_pct = quote_daily_change("kor", ticker, info)
        candles = _HYPERLIQUID_CANDLE_CACHE.get(spec["symbol"]) or []
        change_24h = None
        if len(candles) >= 2 and candles[0].get("c"):
            change_24h = (candles[-1]["c"] / candles[0]["c"] - 1.0) * 100.0
        quotes.append(
            {
                "symbol": spec["symbol"],
                "name": spec["name"],
                "type": "toss",
                "country": "kor",
                "currency": "KRW",
                "hyper_price": latest,
                "change_24h_pct": change_24h,
                "actual_price": previous_close,
                "actual_change_pct": None,
                "diff_pct": change_pct,
                "session_open": kor_market_open,
                "price_data_open": kr_price_data_session != CLOSED,
                "price_data_session": kr_price_data_session,
                "candles": candles,
                "source_ticker": ticker,
            }
        )

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
            actual_price, actual_change_pct = get_confirmed_regular_close("us", str(spec.get("yahoo_symbol") or ""))
        elif spec["country"] == "kor":
            currency = "KRW"
            country = "kor"
            hyper_price = (mark * usd_krw) if (mark is not None and usd_krw) else None
            actual_price, actual_change_pct = get_confirmed_regular_close("kor", spec["actual_ticker"])
        else:
            currency = "USD"
            country = "us"
            hyper_price = mark
            actual_price, actual_change_pct = get_confirmed_regular_close("us", str(spec.get("actual_ticker") or ""))

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
