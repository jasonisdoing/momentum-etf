"""S&P500, NASDAQ100 구성종목을 Wikipedia에서 가져오고 yfinance로 시가총액을 보강해 data/ 폴더에 JSON으로 저장한다.

사용법:
    python scripts/fetch_index_constituents.py
"""

from __future__ import annotations

import io
import json
import sys
import time
from datetime import date
from pathlib import Path

import requests
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).parent.parent / "data"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}
_YFINANCE_BATCH_SIZE = 50
_YFINANCE_BATCH_DELAY = 1.0  # 초


def _read_html(url: str) -> list[pd.DataFrame]:
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))


def _fetch_sp500() -> list[dict]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = _read_html(url)
    df = tables[0]
    result = []
    for _, row in df.iterrows():
        ticker = str(row.get("Symbol") or "").strip().upper().replace(".", "-")
        name = str(row.get("Security") or "").strip()
        sector = str(row.get("GICS Sector") or "").strip()
        sub_industry = str(row.get("GICS Sub-Industry") or "").strip()
        if not ticker:
            continue
        result.append({"ticker": ticker, "name": name, "sector": sector, "industry": sub_industry})
    return result


def _fetch_ndx100() -> list[dict]:
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = _read_html(url)
    df = None
    for table in tables:
        cols = [str(c).strip() for c in table.columns]
        if any("ticker" in c.lower() or "symbol" in c.lower() for c in cols):
            df = table
            break
    if df is None:
        raise RuntimeError("NASDAQ-100 구성종목 테이블을 찾지 못했습니다.")

    cols = {str(c).strip(): c for c in df.columns}
    ticker_col = next((cols[c] for c in cols if "ticker" in c.lower() or "symbol" in c.lower()), None)
    name_col = next((cols[c] for c in cols if "company" in c.lower() or "name" in c.lower()), None)
    sector_col = next((cols[c] for c in cols if "sector" in c.lower()), None)
    industry_col = next((cols[c] for c in cols if "industry" in c.lower() and "sector" not in c.lower()), None)

    result = []
    for _, row in df.iterrows():
        ticker = str(row.get(ticker_col) or "").strip().upper().replace(".", "-") if ticker_col else ""
        name = str(row.get(name_col) or "").strip() if name_col else ""
        sector = str(row.get(sector_col) or "").strip() if sector_col else ""
        industry = str(row.get(industry_col) or "").strip() if industry_col else ""
        if not ticker or ticker in ("NAN", "TICKER", "SYMBOL"):
            continue
        result.append({"ticker": ticker, "name": name, "sector": sector, "industry": industry})
    return result


def _fetch_stock_meta(tickers: list[str]) -> dict[str, dict[str, Any]]:
    """yfinance fast_info 로 시가총액(USD)과 거래량을 배치 조회한다."""
    result: dict[str, dict[str, Any]] = {}
    batches = [tickers[i:i + _YFINANCE_BATCH_SIZE] for i in range(0, len(tickers), _YFINANCE_BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches):
        print(f"  데이터 조회 중... {batch_idx * _YFINANCE_BATCH_SIZE + len(batch)}/{len(tickers)}", end="\r")
        # yfinance 심볼은 "-" 지원
        yf_symbols = [t.replace("-", "-") for t in batch]
        try:
            tickers_obj = yf.Tickers(" ".join(yf_symbols))
            for ticker, yf_sym in zip(batch, yf_symbols):
                try:
                    info = tickers_obj.tickers[yf_sym].fast_info
                    cap = info.market_cap
                    vol = info.last_volume
                    result[ticker] = {
                        "market_cap": int(cap) if cap and cap > 0 else None,
                        "volume": int(vol) if vol and vol > 0 else None,
                    }
                except Exception:
                    result[ticker] = {"market_cap": None, "volume": None}
        except Exception as exc:
            print(f"\n  배치 조회 실패: {exc}")
            for t in batch:
                result[t] = {"market_cap": None, "volume": None}

        if batch_idx < len(batches) - 1:
            time.sleep(_YFINANCE_BATCH_DELAY)

    print()
    return result


def _save(filename: str, tickers: list[dict], source_url: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": date.today().isoformat(),
        "source": source_url,
        "count": len(tickers),
        "tickers": tickers,
    }
    path = DATA_DIR / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장 완료: {path} ({len(tickers)}개)")


def _enrich_constituents(items: list[dict]) -> list[dict]:
    ticker_list = [item["ticker"] for item in items]
    meta_map = _fetch_stock_meta(ticker_list)
    for item in items:
        meta = meta_map.get(item["ticker"], {})
        item["market_cap"] = meta.get("market_cap")
        item["volume"] = meta.get("volume")
    return items


def main() -> None:
    print("S&P500 구성종목 조회 중...")
    try:
        sp500 = _fetch_sp500()
        print(f"  Wikipedia에서 {len(sp500)}개 종목 확인. 시가총액 조회 시작...")
        sp500 = _enrich_constituents(sp500)
        _save("sp500_tickers.json", sp500, "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    except Exception as exc:
        print(f"S&P500 조회 실패: {exc}", file=sys.stderr)

    print("NASDAQ100 구성종목 조회 중...")
    try:
        ndx100 = _fetch_ndx100()
        print(f"  Wikipedia에서 {len(ndx100)}개 종목 확인. 시가총액 조회 시작...")
        ndx100 = _enrich_constituents(ndx100)
        _save("ndx100_tickers.json", ndx100, "https://en.wikipedia.org/wiki/Nasdaq-100")
    except Exception as exc:
        print(f"NASDAQ100 조회 실패: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
