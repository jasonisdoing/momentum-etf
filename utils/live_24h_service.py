"""24H 실시간 주식 및 선물 시세 서비스."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from config import HYPERLIQUID_DEX, HYPERLIQUID_INFO_URL, HYPERLIQUID_SYMBOLS
from services.price_service import get_exchange_rates, get_realtime_snapshot

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


def _fetch_binance_ticker(symbol: str) -> dict[str, Any] | None:
    """바이낸스 선물 24시간 ticker 정보를 조회한다."""
    url = f"https://fapi.binance.com/fapi/v1/ticker/24hr?symbol={symbol}"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Binance ticker 조회 실패 (%s): %s", symbol, exc)
        return None


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

    # 실제 시장가 — 개별주는 국가별 스냅샷 일괄 조회
    kor_tickers = [
        s["actual_ticker"] for s in HYPERLIQUID_SYMBOLS if s.get("type") == "stock" and s["country"] == "kor"
    ]
    us_tickers = [s["actual_ticker"] for s in HYPERLIQUID_SYMBOLS if s.get("type") == "stock" and s["country"] == "us"]
    kor_snap = _safe_snapshot("kor", kor_tickers)
    us_snap = _safe_snapshot("us", us_tickers)

    quotes: list[dict[str, Any]] = []
    for spec in HYPERLIQUID_SYMBOLS:
        symbol = str(spec["symbol"]).upper()
        kind = spec.get("type", "stock")
        ctx = ctx_map.get(symbol) or {}
        mark = _to_float(ctx.get("markPx"))
        prev = _to_float(ctx.get("prevDayPx"))
        change_24h = ((mark / prev - 1.0) * 100.0) if (mark and prev) else None

        if kind == "index":
            # 지수: 포인트 그대로(통화 없음), 실제 지수값과 비교.
            currency = "POINT"
            country = "kor" if spec.get("naver_symbol") else "us"
            hyper_price = mark
            actual_price = _fetch_index_value(spec)
        elif spec["country"] == "kor":
            currency = "KRW"
            country = "kor"
            hyper_price = (mark * usd_krw) if (mark is not None and usd_krw) else None
            actual_price = _to_float((kor_snap.get(spec["actual_ticker"]) or {}).get("nowVal"))
        else:
            currency = "USD"
            country = "us"
            hyper_price = mark
            actual_price = _to_float((us_snap.get(spec["actual_ticker"]) or {}).get("nowVal"))

        diff_pct = (
            (hyper_price / actual_price - 1.0) * 100.0
            if (hyper_price is not None and actual_price and actual_price > 0)
            else None
        )

        # 바이낸스 선물 시세 추가 정보 (SAMSUNGUSDT, SKHYNIXUSDT 등)
        binance_data = None
        binance_symbol = spec.get("binance_symbol")
        if binance_symbol:
            ticker = _fetch_binance_ticker(binance_symbol)
            if ticker:
                b_price_usd = _to_float(ticker.get("lastPrice"))
                b_change = _to_float(ticker.get("priceChangePercent"))

                if b_price_usd is not None:
                    # 바이낸스 시세 승수 보정 (예: S&P500의 SPYUSDT의 경우 10.0배 보정)
                    b_multiplier = spec.get("binance_multiplier", 1.0)
                    b_price_usd = b_price_usd * b_multiplier

                    # 국내 종목인 경우 환율을 적용하여 원화로 환산
                    if spec.get("country") == "kor":
                        b_price_converted = b_price_usd * usd_krw if usd_krw else None
                    else:
                        b_price_converted = b_price_usd

                    b_diff_pct = (
                        (b_price_converted / actual_price - 1.0) * 100.0
                        if (b_price_converted is not None and actual_price and actual_price > 0)
                        else None
                    )

                    binance_data = {
                        "symbol": binance_symbol,
                        "price": b_price_converted,
                        "change_24h_pct": b_change,
                        "diff_pct": b_diff_pct,
                    }

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
                "binance": binance_data,
            }
        )

    return {"quotes": quotes, "usd_krw": usd_krw}


def _fetch_index_value(spec: dict[str, Any]) -> float | None:
    """지수의 실제 현재값. 한국 지수는 네이버, 그 외는 야후(intraday 보강) 로 조회."""
    try:
        if spec.get("naver_symbol"):
            from utils.market_trend_service import _fetch_naver_kor_index_close

            series = _fetch_naver_kor_index_close(spec["naver_symbol"], count=3)
            if series is not None and not series.empty:
                return float(series.iloc[-1])
            return None
        from utils.market_trend_service import _fetch_yf_intraday_last_close

        intraday = _fetch_yf_intraday_last_close(spec["yahoo_symbol"])
        return float(intraday[1]) if intraday else None
    except Exception as exc:
        logger.warning("Hyperliquid 지수 실제값 조회 실패 (%s): %s", spec.get("symbol"), exc)
        return None


def _safe_snapshot(country: str, tickers: list[str]) -> dict[str, dict[str, Any]]:
    if not tickers:
        return {}
    try:
        return get_realtime_snapshot(country, tickers)
    except Exception as exc:
        logger.warning("Hyperliquid 실제가(%s) 조회 실패: %s", country, exc)
        return {}
