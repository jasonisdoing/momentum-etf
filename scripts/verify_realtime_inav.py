"""실시간 추정 iNAV 검증 스크립트 (PoC).

대상 ETF 의 구성종목에 대해 Yahoo Finance 실시간/프리·애프터마켓 가격을 가져와
'포트폴리오 변동' 과 '추정 iNAV' 를 etfnow.co.kr 방식으로 계산해본다.

사용 예:
    python scripts/verify_realtime_inav.py --ticker 0015B0 --ticker-type kor_us

목표:
    https://etfnow.co.kr/etf/0015B0 의 '포트폴리오 변동' / '추정 iNAV' 와 우리 계산이
    일치(또는 ±0.1%p 이내)하는지 확인한다.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.price_service import get_exchange_rates  # noqa: E402
from services.stock_cache_service import get_stock_cache_meta  # noqa: E402


def _fetch_yahoo_quotes(symbols: list[str]) -> dict[str, dict]:
    """Yahoo Finance v7 quote API 를 yfinance 인증 세션으로 호출한다."""
    from yfinance.data import YfData

    if not symbols:
        return {}

    data = YfData()
    url = "https://query1.finance.yahoo.com/v7/finance/quote"
    # 한 번에 너무 많이 보내면 잘리므로 50개 단위로 분할
    out: dict[str, dict] = {}
    for i in range(0, len(symbols), 50):
        chunk = symbols[i : i + 50]
        params = {"symbols": ",".join(chunk)}
        resp = data.get(url, params=params)
        if resp.status_code != 200:
            print(f"[WARN] Yahoo {resp.status_code}: {resp.text[:120]}")
            continue
        for q in resp.json().get("quoteResponse", {}).get("result", []):
            out[q["symbol"]] = q
    return out


def _resolve_live_change_pct(quote: dict) -> tuple[float | None, str]:
    """quote 에서 '오늘 공식 iNAV 이후 발생한 변동률(%)' 을 산출한다.

    Yahoo 가 직접 제공하는 pre/regular/post change% 를 시장 상태에 맞게 골라준다.
    공식 iNAV 가 마지막 정규장 종가를 반영한다는 가정 하에:
      - PRE  → preMarketChangePercent  (전일 종가 → 현재 프리마켓)
      - REGULAR → regularMarketChangePercent (전일 종가 → 현재가)
      - POST → regularMarketChangePercent + postMarketChangePercent compound
      - CLOSED 이고 postMarketPrice 있음 → 위와 동일
      - CLOSED 이고 후장 없음 → 0% (반영할 신규 데이터 없음)
    """
    state = str(quote.get("marketState") or "").upper()
    pre_pct = quote.get("preMarketChangePercent")
    reg_pct = quote.get("regularMarketChangePercent")
    post_pct = quote.get("postMarketChangePercent")

    def _f(v):
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    pre_pct = _f(pre_pct)
    reg_pct = _f(reg_pct)
    post_pct = _f(post_pct)

    if state == "PRE" and pre_pct is not None:
        return pre_pct, "PRE"
    if state == "REGULAR" and reg_pct is not None:
        return reg_pct, "REGULAR"
    if state == "POST":
        # 정규장 + 애프터마켓 누적 변동
        components = [v for v in (reg_pct, post_pct) if v is not None]
        if not components:
            return None, state
        compound = 1.0
        for v in components:
            compound *= 1.0 + v / 100.0
        return (compound - 1.0) * 100.0, "POST"
    if state == "CLOSED":
        if post_pct is not None:
            return post_pct, "CLOSED+POST"
        return 0.0, "CLOSED"
    return None, state or "UNKNOWN"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="0015B0")
    parser.add_argument("--ticker-type", default="kor_us")
    args = parser.parse_args()

    ticker_norm = args.ticker.strip().upper()
    type_norm = args.ticker_type.strip().lower()

    # 1) holdings_cache + 공식 iNAV 조회
    doc = get_stock_cache_meta(type_norm, ticker_norm)
    if not doc:
        print(f"[FAIL] {type_norm}/{ticker_norm} 메타 캐시가 없습니다.")
        return 1

    meta = doc.get("meta_cache") or {}
    holdings_cache = doc.get("holdings_cache") or {}
    items = holdings_cache.get("items") or []
    nav = float(meta.get("nav") or 0.0)
    if not items or nav <= 0:
        print(f"[FAIL] holdings 또는 nav 누락 (holdings={len(items)}, nav={nav})")
        return 2

    # 2) Yahoo 심볼 수집
    symbols: list[str] = []
    for it in items:
        sym = (it.get("yahoo_symbol") or it.get("ticker") or "").strip().upper()
        if sym:
            symbols.append(sym)

    print(f"[INFO] holdings={len(items)}, fetching {len(symbols)} symbols from Yahoo...")
    t0 = time.time()
    quotes = _fetch_yahoo_quotes(symbols)
    print(f"[INFO] fetched in {time.time() - t0:.2f}s — got {len(quotes)} quotes")

    # 3) 환율
    rates = get_exchange_rates()
    usd = rates.get("USD") or {}
    fx_change_pct = float(usd.get("change_pct") or 0.0)
    fx_rate = float(usd.get("rate") or 0.0)

    # 4) 종목별 변동률 + 가중 평균 (USD 기준)
    print()
    print(f"{'symbol':<8} {'weight':>7} {'state':<12} {'change%':>9}  comment")
    print("-" * 70)

    total_weight = 0.0
    weighted_sum = 0.0
    matched = 0
    for it in items:
        sym = (it.get("yahoo_symbol") or it.get("ticker") or "").strip().upper()
        weight = float(it.get("weight") or 0.0)
        q = quotes.get(sym)
        if not q:
            print(f"{sym:<8} {weight:>6.2f}% {'NO_DATA':<12}")
            total_weight += weight
            continue
        change_pct, src = _resolve_live_change_pct(q)
        comment = ""
        if change_pct is None:
            comment = f"reg={q.get('regularMarketChangePercent')} post={q.get('postMarketChangePercent')} pre={q.get('preMarketChangePercent')}"
            change_pct = 0.0
        else:
            matched += 1

        weighted_sum += weight * change_pct
        total_weight += weight
        print(f"{sym:<8} {weight:>6.2f}% {src:<12} {change_pct:>+8.2f}%  {comment}")

    # 가중치 합이 100 미만이면 나머지는 현금(0% 변동) 가정
    denom = max(total_weight, 100.0)
    portfolio_change_usd_pct = weighted_sum / denom

    print()
    print(f"[합계] 비중 합        : {total_weight:.2f}%")
    print(f"[합계] 매칭 종목       : {matched} / {len(items)}")
    print(f"[합계] USD 가중평균    : {portfolio_change_usd_pct:+.4f}%")
    print(f"[환율] USD/KRW 변동    : {fx_change_pct:+.4f}% (rate={fx_rate:.2f})")

    # 5) compound: KRW 기준 포트폴리오 변동
    portfolio_change_krw_pct = ((1 + portfolio_change_usd_pct / 100.0) * (1 + fx_change_pct / 100.0) - 1.0) * 100.0
    print(f"[합산] KRW 포트폴리오 변동: {portfolio_change_krw_pct:+.4f}%")

    # 6) 추정 iNAV
    estimated_inav = nav * (1.0 + portfolio_change_krw_pct / 100.0)
    delta = estimated_inav - nav
    print()
    print(f"[iNAV] 공식 iNAV       : {nav:>12,.2f} 원")
    print(f"[iNAV] 추정 iNAV       : {estimated_inav:>12,.2f} 원  ({delta:+,.2f} 원, {portfolio_change_krw_pct:+.2f}%)")

    now_kst = datetime.now(ZoneInfo("Asia/Seoul"))
    print()
    print(f"[시각] {now_kst.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"[참고] etfnow.co.kr/etf/{ticker_norm} 의 '포트폴리오 변동' / '추정 iNAV' 와 비교해주세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
