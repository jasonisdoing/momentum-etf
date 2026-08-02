"""호주 S&P/ASX 200 구성종목을 갱신하고 시가총액·기간 수익률을 저장한다.

사용법:
    python scripts/update_aus_market_stocks.py
"""

from __future__ import annotations

import io
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.index_constituents_loader import save_index_constituents  # noqa: E402

_SOURCE_URL = "https://en.wikipedia.org/wiki/S%26P/ASX_200"
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


def _normalize_asx_ticker(value: Any) -> str:
    ticker = str(value or "").strip().upper()
    if not ticker or ticker == "NAN":
        return ""
    if ticker.startswith("ASX:"):
        ticker = ticker[4:]
    if ticker.endswith(".AX"):
        ticker = ticker[:-3]
    return ticker.replace(".", "-")


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> Any | None:
    for col in df.columns:
        name = str(col).strip().lower()
        if any(candidate in name for candidate in candidates):
            return col
    return None


def _fetch_asx200() -> list[dict[str, Any]]:
    tables = _read_html(_SOURCE_URL)
    target = None
    for table in tables:
        ticker_col = _find_column(table, ("code", "ticker", "symbol"))
        name_col = _find_column(table, ("company", "security", "name"))
        if ticker_col is not None and name_col is not None and len(table) >= 100:
            target = table
            break
    if target is None:
        raise RuntimeError("S&P/ASX 200 구성종목 테이블을 찾지 못했습니다.")

    ticker_col = _find_column(target, ("code", "ticker", "symbol"))
    name_col = _find_column(target, ("company", "security", "name"))
    sector_col = _find_column(target, ("sector", "industry", "gics"))

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, row in target.iterrows():
        ticker = _normalize_asx_ticker(row.get(ticker_col))
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        name = str(row.get(name_col) or "").strip()
        sector = str(row.get(sector_col) or "").strip() if sector_col is not None else ""
        rows.append(
            {
                "ticker": f"ASX:{ticker}",
                "name": name or ticker,
                "sector": sector,
                "industry": sector,
            }
        )
    return rows[:200]


def _to_yfinance_symbol(ticker: str) -> str:
    raw = str(ticker or "").strip().upper()
    if raw.startswith("ASX:"):
        raw = raw[4:]
    if raw.endswith(".AX"):
        return raw
    return f"{raw}.AX"


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


def _calculate_period_return(clean: pd.Series, latest_price: float, months: int) -> dict[str, Any]:
    target_date = pd.Timestamp(date.today()).normalize() - pd.DateOffset(months=months)
    base_candidates = clean[clean.index >= target_date]
    base_date = clean.index.min() if base_candidates.empty else base_candidates.index.min()
    base_price = float(clean.loc[base_date])
    return_pct = ((latest_price / base_price) - 1.0) * 100.0 if base_price > 0 else None
    prefix = f"return_{months}m"
    return {
        f"{prefix}_base_date": pd.Timestamp(base_date).date().isoformat(),
        f"{prefix}_base_price": round(base_price, 6),
        f"{prefix}_latest_price": round(latest_price, 6),
        f"{prefix}_pct": round(return_pct, 4) if return_pct is not None else None,
    }


def _calculate_mdd(clean: pd.Series, months: int) -> dict[str, Any]:
    target_date = pd.Timestamp(date.today()).normalize() - pd.DateOffset(months=months)
    period = clean[clean.index >= target_date]
    if period.empty:
        period = clean
    running_peak = period.cummax()
    drawdown = (period / running_peak - 1.0) * 100.0
    mdd = float(drawdown.min()) if not drawdown.empty else None
    return {f"mdd_{months}m_pct": round(mdd, 4) if mdd is not None else None}


def _calculate_return_metrics(series: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    clean = clean[clean > 0]
    if clean.empty:
        return {
            "return_1m_base_date": None,
            "return_1m_base_price": None,
            "return_1m_latest_price": None,
            "return_1m_pct": None,
            "return_3m_base_date": None,
            "return_3m_base_price": None,
            "return_3m_latest_price": None,
            "return_3m_pct": None,
            "return_12m_base_date": None,
            "return_12m_base_price": None,
            "return_12m_latest_price": None,
            "return_12m_pct": None,
            "mdd_12m_pct": None,
            "current_price": None,
            "change_pct": None,
        }
    latest_date = clean.index.max()
    latest_price = float(clean.loc[latest_date])
    previous_prices = clean[clean.index < latest_date]
    previous_price = float(previous_prices.iloc[-1]) if not previous_prices.empty else None
    change_pct = ((latest_price / previous_price) - 1.0) * 100.0 if previous_price and previous_price > 0 else None
    return {
        **_calculate_period_return(clean, latest_price, 1),
        **_calculate_period_return(clean, latest_price, 3),
        **_calculate_period_return(clean, latest_price, 12),
        **_calculate_mdd(clean, 12),
        "current_price": round(latest_price, 6),
        "change_pct": round(change_pct, 4) if change_pct is not None else None,
    }


def _fetch_stock_meta(tickers: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    batches = [tickers[i:i + _YFINANCE_BATCH_SIZE] for i in range(0, len(tickers), _YFINANCE_BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches):
        print(f"  데이터 조회 중... {batch_idx * _YFINANCE_BATCH_SIZE + len(batch)}/{len(tickers)}", end="\r")
        yf_symbols = [_to_yfinance_symbol(ticker) for ticker in batch]
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
                period="13mo",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            close_frame = _extract_close_frame(history, yf_symbols)
            for yf_sym, ticker in symbol_to_ticker.items():
                return_data = (
                    _calculate_return_metrics(close_frame[yf_sym])
                    if not close_frame.empty and yf_sym in close_frame.columns
                    else _calculate_return_metrics(pd.Series(dtype=float))
                )
                result.setdefault(ticker, {}).update(return_data)
        except Exception as exc:
            print(f"\n  기간 수익률 배치 조회 실패: {exc}")
            for ticker in batch:
                result.setdefault(ticker, {}).update(_calculate_return_metrics(pd.Series(dtype=float)))

        if batch_idx < len(batches) - 1:
            time.sleep(_YFINANCE_BATCH_DELAY)

    print()
    return result


def _save(items: list[dict[str, Any]]) -> None:
    save_index_constituents(
        "ASX200", items, {"updated_at": date.today().isoformat(), "source": _SOURCE_URL}
    )
    print(f"저장 완료: ASX200 ({len(items)}개)")


def main() -> None:
    print("S&P/ASX 200 구성종목 조회 중...")
    try:
        items = _fetch_asx200()
        print(f"  Wikipedia에서 {len(items)}개 종목 확인. 시가총액/기간 수익률 조회 시작...")
        meta_map = _fetch_stock_meta([str(item["ticker"]) for item in items])
        for item in items:
            item.update(meta_map.get(str(item["ticker"]), {}))
        items.sort(key=lambda row: (-(row.get("market_cap") or 0), str(row.get("ticker") or "")))
        _save(items)
    except Exception as exc:
        print(f"S&P/ASX 200 조회 실패: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
