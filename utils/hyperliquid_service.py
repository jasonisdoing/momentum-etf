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

    # 환율(USD→KRW) — 한국 종목 환산용
    try:
        rates = get_exchange_rates()
        usd_krw = _to_float((rates.get("USD") or {}).get("rate"))
    except Exception as exc:
        logger.warning("Hyperliquid 환율 조회 실패: %s", exc)
        usd_krw = None

    # 실제 시장가 — 국가별 일괄 조회
    kor_tickers = [s["actual_ticker"] for s in HYPERLIQUID_SYMBOLS if s["country"] == "kor"]
    us_tickers = [s["actual_ticker"] for s in HYPERLIQUID_SYMBOLS if s["country"] == "us"]
    kor_snap = _safe_snapshot("kor", kor_tickers)
    us_snap = _safe_snapshot("us", us_tickers)

    quotes: list[dict[str, Any]] = []
    for spec in HYPERLIQUID_SYMBOLS:
        symbol = str(spec["symbol"]).upper()
        ctx = ctx_map.get(symbol) or {}
        mark_usd = _to_float(ctx.get("markPx"))
        prev_usd = _to_float(ctx.get("prevDayPx"))
        change_24h = ((mark_usd / prev_usd - 1.0) * 100.0) if (mark_usd and prev_usd) else None

        country = spec["country"]
        currency = "KRW" if country == "kor" else "USD"

        # 표시 가격: 한국=USD×환율→KRW, 미국=USD 그대로
        if country == "kor":
            hyper_price = (mark_usd * usd_krw) if (mark_usd is not None and usd_krw) else None
            actual = (kor_snap.get(spec["actual_ticker"]) or {}).get("nowVal")
        else:
            hyper_price = mark_usd
            actual = (us_snap.get(spec["actual_ticker"]) or {}).get("nowVal")
        actual_price = _to_float(actual)

        diff_pct = (
            (hyper_price / actual_price - 1.0) * 100.0
            if (hyper_price is not None and actual_price and actual_price > 0)
            else None
        )

        quotes.append(
            {
                "symbol": symbol,
                "name": spec["name"],
                "country": country,
                "currency": currency,
                "hyper_price": hyper_price,
                "change_24h_pct": change_24h,
                "actual_price": actual_price,
                "diff_pct": diff_pct,
            }
        )

    return {"quotes": quotes, "usd_krw": usd_krw}


def _safe_snapshot(country: str, tickers: list[str]) -> dict[str, dict[str, Any]]:
    if not tickers:
        return {}
    try:
        return get_realtime_snapshot(country, tickers)
    except Exception as exc:
        logger.warning("Hyperliquid 실제가(%s) 조회 실패: %s", country, exc)
        return {}
