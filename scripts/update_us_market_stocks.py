"""미국 개별주 인덱스 구성종목을 갱신하고 시가총액·3개월 수익률을 저장한다.

사용법:
    python scripts/update_us_market_stocks.py
"""

from __future__ import annotations

import io
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests
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


def _fetch_sp500() -> list[dict[str, Any]]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = _read_html(url)
    df = tables[0]
    result: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        ticker = str(row.get("Symbol") or "").strip().upper().replace(".", "-")
        name = str(row.get("Security") or "").strip()
        sector = str(row.get("GICS Sector") or "").strip()
        sub_industry = str(row.get("GICS Sub-Industry") or "").strip()
        if not ticker:
            continue
        result.append({"ticker": ticker, "name": name, "sector": sector, "industry": sub_industry})
    return result


def _fetch_ndx100() -> list[dict[str, Any]]:
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

    result: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        ticker = str(row.get(ticker_col) or "").strip().upper().replace(".", "-") if ticker_col else ""
        name = str(row.get(name_col) or "").strip() if name_col else ""
        sector = str(row.get(sector_col) or "").strip() if sector_col else ""
        industry = str(row.get(industry_col) or "").strip() if industry_col else ""
        if not ticker or ticker in ("NAN", "TICKER", "SYMBOL"):
            continue
        result.append({"ticker": ticker, "name": name, "sector": sector, "industry": industry})
    return result


def _normalize_yfinance_symbol(ticker: str) -> str:
    return str(ticker or "").strip().upper()


def _extract_close_frame(downloaded: pd.DataFrame, yf_symbols: list[str]) -> pd.DataFrame:
    if downloaded.empty:
        return pd.DataFrame()

    if isinstance(downloaded.columns, pd.MultiIndex):
        if "Close" not in downloaded.columns.get_level_values(0):
            return pd.DataFrame()
        close = downloaded["Close"]
    else:
        if "Close" not in downloaded.columns:
            return pd.DataFrame()
        close = downloaded[["Close"]]
        if len(yf_symbols) == 1:
            close = close.rename(columns={"Close": yf_symbols[0]})

    if isinstance(close, pd.Series):
        close = close.to_frame()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    return close.sort_index()


def _calculate_three_month_return(series: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    clean = clean[clean > 0]
    if clean.empty:
        return {
            "return_3m_base_date": None,
            "return_3m_base_price": None,
            "return_3m_latest_price": None,
            "return_3m_pct": None,
        }

    latest_date = clean.index.max()
    latest_price = float(clean.loc[latest_date])
    target_date = pd.Timestamp(date.today()).normalize() - pd.DateOffset(months=3)
    base_candidates = clean[clean.index >= target_date]
    if base_candidates.empty:
        base_date = clean.index.min()
    else:
        base_date = base_candidates.index.min()

    base_price = float(clean.loc[base_date])
    return_pct = ((latest_price / base_price) - 1.0) * 100.0 if base_price > 0 else None
    return {
        "return_3m_base_date": pd.Timestamp(base_date).date().isoformat(),
        "return_3m_base_price": round(base_price, 6),
        "return_3m_latest_price": round(latest_price, 6),
        "return_3m_pct": round(return_pct, 4) if return_pct is not None else None,
    }


def _fetch_stock_meta(tickers: list[str]) -> dict[str, dict[str, Any]]:
    """yfinance fast_info 와 시계열로 시가총액·거래량·3개월 수익률을 배치 조회한다."""
    result: dict[str, dict[str, Any]] = {}
    batches = [tickers[i:i + _YFINANCE_BATCH_SIZE] for i in range(0, len(tickers), _YFINANCE_BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches):
        print(f"  데이터 조회 중... {batch_idx * _YFINANCE_BATCH_SIZE + len(batch)}/{len(tickers)}", end="\r")
        yf_symbols = [_normalize_yfinance_symbol(ticker) for ticker in batch]
        symbol_to_ticker = dict(zip(yf_symbols, batch, strict=True))

        try:
            tickers_obj = yf.Tickers(" ".join(yf_symbols))
            for ticker, yf_sym in zip(batch, yf_symbols, strict=True):
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
            print(f"\n  fast_info 배치 조회 실패: {exc}")
            for ticker in batch:
                result[ticker] = {"market_cap": None, "volume": None}

        try:
            history = yf.download(
                yf_symbols,
                period="4mo",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            close_frame = _extract_close_frame(history, yf_symbols)
            for yf_sym, ticker in symbol_to_ticker.items():
                if close_frame.empty or yf_sym not in close_frame.columns:
                    return_data = _calculate_three_month_return(pd.Series(dtype=float))
                else:
                    return_data = _calculate_three_month_return(close_frame[yf_sym])
                result.setdefault(ticker, {}).update(return_data)
        except Exception as exc:
            print(f"\n  3개월 수익률 배치 조회 실패: {exc}")
            for ticker in batch:
                result.setdefault(ticker, {}).update(_calculate_three_month_return(pd.Series(dtype=float)))

        if batch_idx < len(batches) - 1:
            time.sleep(_YFINANCE_BATCH_DELAY)

    print()
    return result


def _save(filename: str, tickers: list[dict[str, Any]], source_url: str) -> None:
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


def _enrich_constituents(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ticker_list = [str(item["ticker"]) for item in items]
    meta_map = _fetch_stock_meta(ticker_list)
    for item in items:
        meta = meta_map.get(str(item["ticker"]), {})
        item["market_cap"] = meta.get("market_cap")
        item["volume"] = meta.get("volume")
        item["return_3m_base_date"] = meta.get("return_3m_base_date")
        item["return_3m_base_price"] = meta.get("return_3m_base_price")
        item["return_3m_latest_price"] = meta.get("return_3m_latest_price")
        item["return_3m_pct"] = meta.get("return_3m_pct")
    return items


def main() -> None:
    print("S&P500 구성종목 조회 중...")
    try:
        sp500 = _fetch_sp500()
        print(f"  Wikipedia에서 {len(sp500)}개 종목 확인. 시가총액/3개월 수익률 조회 시작...")
        sp500 = _enrich_constituents(sp500)
        _save("sp500_tickers.json", sp500, "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    except Exception as exc:
        print(f"S&P500 조회 실패: {exc}", file=sys.stderr)

    print("NASDAQ100 구성종목 조회 중...")
    try:
        ndx100 = _fetch_ndx100()
        print(f"  Wikipedia에서 {len(ndx100)}개 종목 확인. 시가총액/3개월 수익률 조회 시작...")
        ndx100 = _enrich_constituents(ndx100)
        _save("ndx100_tickers.json", ndx100, "https://en.wikipedia.org/wiki/Nasdaq-100")
    except Exception as exc:
        print(f"NASDAQ100 조회 실패: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
