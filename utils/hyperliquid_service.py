"""Hyperliquid 24시간 토큰화 주식 시세 (/hyperliquid 화면).

빌더 DEX `xyz` 의 perp(SMSN/SKHX/MU 등) 24시간 가격을 가져와, 한국 종목은 환율로 KRW 환산,
미국 종목은 USD 그대로 표시한다. 각 종목의 실제 시장가(네이버 KRX / 토스 US)와 비교해
"24시간 가격이 실제 대비 얼마나 벌어졌나"(프리미엄/디스카운트) 도 함께 계산한다.
"""

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


def load_hyperliquid_quotes() -> dict[str, Any]:
    """설정된 심볼들의 Hyperliquid 24h 시세 + 실제가 대비 차이를 반환한다."""
    ctx_map = _fetch_dex_ctxs()

    # 환율(USD→KRW) — 한국 개별주 환산용
    try:
        rates = get_exchange_rates()
        usd_krw = _to_float((rates.get("USD") or {}).get("rate"))
    except Exception as exc:
        logger.warning("Hyperliquid 환율 조회 실패: %s", exc)
        usd_krw = None

    # 실제 시장가 — 개별주는 국가별 스냅샷 일괄 조회
    kor_tickers = [s["actual_ticker"] for s in HYPERLIQUID_SYMBOLS if s.get("type") == "stock" and s["country"] == "kor"]
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
