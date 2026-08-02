"""미국 개별주 인덱스 구성종목을 갱신하고 시가총액·기간 수익률을 저장한다.

출처
----
- S&P500: 위키피디아 구성종목 표
- NASDAQ100: 나스닥 공식 API (지수 산출 주체가 직접 제공)
- 섹터·업종: yfinance

구성종목 수가 기대 범위를 벗어나면 저장하지 않고 실패로 끝난다(종료 코드 1).
원본 구조가 바뀌었을 때 조용히 낡은 데이터를 쓰는 상황을 막기 위한 것이다.

섹터·업종은 yfinance 에서 받는다. 종목별 개별 호출이라 비싸서, **이미 저장된
값이 있으면 다시 조회하지 않는다**(분류는 거의 바뀌지 않는다). 전체를 다시 받으려면
`--refresh-classification` 을 준다.

사용법:
    python scripts/update_us_market_stocks.py
    python scripts/update_us_market_stocks.py --refresh-classification
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.index_constituents_loader import (  # noqa: E402
    load_index_constituents,
    load_index_meta,
    save_index_constituents,
)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}
_YFINANCE_BATCH_SIZE = 50
_YFINANCE_BATCH_DELAY = 1.0  # 초
_CLASSIFICATION_WORKERS = 8  # 섹터·업종 개별 조회 동시 실행 수

_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_NDX100_API_URL = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
_API_HEADERS = {**_HEADERS, "Accept": "application/json, */*"}

# 구성종목 수가 이 범위를 벗어나면 원본이 깨진 것으로 보고 저장하지 않는다.
# (위키 문서가 옮겨졌을 때처럼 조용히 실패하는 것을 막기 위한 장치)
_EXPECTED_COUNTS = {"S&P500": (490, 510), "NASDAQ100": (95, 110)}


def _check_count(label: str, items: list[dict[str, Any]]) -> None:
    low, high = _EXPECTED_COUNTS[label]
    if not low <= len(items) <= high:
        raise RuntimeError(
            f"{label} 구성종목 수가 비정상입니다: {len(items)}개 (기대 {low}~{high}). "
            "원본 페이지·API 구조가 바뀌었을 수 있어 저장하지 않습니다."
        )


def _load_saved_classification(index: str) -> dict[str, dict[str, str]]:
    """이미 저장된 값에서 티커별 섹터·업종을 읽는다. 없으면 빈 dict.

    `classification_source` 가 yfinance 인 경우만 재사용한다. 예전에 위키 표에서
    받은 값이 남아 있으면 체계가 다른 값을 그대로 물려받게 되기 때문이다
    (ARM 이 섹터 `Semiconductors` · 업종 `Technology` 로 뒤집혀 남아 있던 사례).
    """
    try:
        meta = load_index_meta(index)
        if meta.get("classification_source") != "yfinance":
            return {}
        stored = load_index_constituents(index)
    except Exception:
        return {}
    saved: dict[str, dict[str, str]] = {}
    for item in stored:
        ticker = str(item.get("ticker") or "").strip().upper()
        industry = str(item.get("industry") or "").strip()
        if ticker and industry:  # 업종이 비어 있으면 실패했던 것이므로 다시 받는다
            saved[ticker] = {"sector": str(item.get("sector") or "").strip(), "industry": industry}
    return saved


def _fetch_classification(tickers: list[str]) -> dict[str, dict[str, str]]:
    """yfinance 에서 섹터·업종을 받는다 (종목별 호출이라 병렬로 돈다)."""
    def one(ticker: str) -> tuple[str, dict[str, str]]:
        try:
            info = yf.Ticker(_normalize_yfinance_symbol(ticker)).get_info() or {}
            return ticker, {
                "sector": str(info.get("sector") or "").strip(),
                "industry": str(info.get("industry") or "").strip(),
            }
        except Exception:
            return ticker, {"sector": "", "industry": ""}

    with ThreadPoolExecutor(max_workers=_CLASSIFICATION_WORKERS) as pool:
        return dict(pool.map(one, tickers))


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

        if not ticker:
            continue
        result.append({"ticker": ticker, "name": name})
    return result


def _fetch_ndx100() -> list[dict[str, Any]]:
    """나스닥 공식 API 에서 나스닥100 구성종목을 받는다.

    위키피디아를 쓰다가 옮겼다 — 문서가 `List_of_NASDAQ-100_companies` 로 분리되면서
    한 달 넘게 조용히 실패했다. 지수를 산출하는 나스닥이 직접 주는 JSON 이라
    문서 편집·이동에 영향받지 않는다.
    """
    resp = requests.get(_NDX100_API_URL, headers=_API_HEADERS, timeout=30)
    resp.raise_for_status()
    payload = resp.json().get("data") or {}
    rows = ((payload.get("data") or {}).get("rows")) or []

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        ticker = str(row.get("symbol") or "").strip().upper().replace(".", "-")
        if not ticker or ticker in seen:
            continue
        name = str(row.get("companyName") or "").strip()
        # "Apple Inc. Common Stock" 처럼 증권 종류가 붙어 있어 떼어낸다.
        for suffix in (" Common Stock", " Common Shares", " Ordinary Shares", " Class A", " Class C"):
            if name.endswith(suffix):
                name = name[: -len(suffix)].strip()
        seen.add(ticker)
        result.append({"ticker": ticker, "name": name})
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
        }

    latest_date = clean.index.max()
    latest_price = float(clean.loc[latest_date])
    return {
        **_calculate_period_return(clean, latest_price, 1),
        **_calculate_period_return(clean, latest_price, 3),
        **_calculate_period_return(clean, latest_price, 12),
        **_calculate_mdd(clean, 12),
    }


def _fetch_stock_meta(tickers: list[str]) -> dict[str, dict[str, Any]]:
    """yfinance fast_info 와 시계열로 시가총액·거래량·기간 수익률을 배치 조회한다."""
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
                period="13mo",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            close_frame = _extract_close_frame(history, yf_symbols)
            for yf_sym, ticker in symbol_to_ticker.items():
                if close_frame.empty or yf_sym not in close_frame.columns:
                    return_data = _calculate_return_metrics(pd.Series(dtype=float))
                else:
                    return_data = _calculate_return_metrics(close_frame[yf_sym])
                result.setdefault(ticker, {}).update(return_data)
        except Exception as exc:
            print(f"\n  기간 수익률 배치 조회 실패: {exc}")
            for ticker in batch:
                result.setdefault(ticker, {}).update(_calculate_return_metrics(pd.Series(dtype=float)))

        if batch_idx < len(batches) - 1:
            time.sleep(_YFINANCE_BATCH_DELAY)

    print()
    return result


def _save(index: str, tickers: list[dict[str, Any]], source_url: str) -> None:
    save_index_constituents(
        index,
        tickers,
        {
            "updated_at": date.today().isoformat(),
            "source": source_url,
            # 구성종목 목록은 위 주소에서, 섹터·업종은 yfinance 에서 받는다.
            "classification_source": "yfinance",
        },
    )
    print(f"저장 완료: {index} ({len(tickers)}개)")


def _enrich_constituents(
    items: list[dict[str, Any]], index: str, refresh_classification: bool
) -> list[dict[str, Any]]:
    ticker_list = [str(item["ticker"]) for item in items]

    # 섹터·업종 — 저장된 값이 있으면 재사용한다. 종목별 호출이라 전체를 매번 받으면
    # 배치가 몇 분씩 길어지는데, 분류는 거의 바뀌지 않아 그럴 이유가 없다.
    saved = {} if refresh_classification else _load_saved_classification(index)
    todo = [t for t in ticker_list if t not in saved]
    if todo:
        print(f"  섹터·업종 조회 {len(todo)}종목 (재사용 {len(saved)}종목)...")
        saved.update(_fetch_classification(todo))
    else:
        print(f"  섹터·업종 전부 재사용 ({len(saved)}종목)")

    meta_map = _fetch_stock_meta(ticker_list)
    for item in items:
        classification = saved.get(str(item["ticker"]), {})
        meta = meta_map.get(str(item["ticker"]), {})
        item["sector"] = classification.get("sector") or ""
        item["industry"] = classification.get("industry") or ""
        item["market_cap"] = meta.get("market_cap")
        item["volume"] = meta.get("volume")
        for months in (1, 3, 12):
            item[f"return_{months}m_base_date"] = meta.get(f"return_{months}m_base_date")
            item[f"return_{months}m_base_price"] = meta.get(f"return_{months}m_base_price")
            item[f"return_{months}m_latest_price"] = meta.get(f"return_{months}m_latest_price")
            item[f"return_{months}m_pct"] = meta.get(f"return_{months}m_pct")
        item["mdd_12m_pct"] = meta.get("mdd_12m_pct")

    missing = [item["ticker"] for item in items if not item["industry"]]
    if missing:
        print(f"  업종 조회 실패 {len(missing)}종목: {', '.join(missing[:10])}"
              f"{' 외' if len(missing) > 10 else ''}")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="미국 지수 구성종목 갱신")
    parser.add_argument(
        "--refresh-classification",
        action="store_true",
        help="저장된 섹터·업종을 무시하고 전부 다시 조회한다",
    )
    args = parser.parse_args()

    failed: list[str] = []

    for label, index, fetch, source in (
        ("S&P500", "SP500", _fetch_sp500, _SP500_URL),
        ("NASDAQ100", "NDX100", _fetch_ndx100, _NDX100_API_URL),
    ):
        print(f"{label} 구성종목 조회 중...")
        try:
            items = fetch()
            _check_count(label, items)
            print(f"  {len(items)}개 종목 확인. 시가총액/기간 수익률 조회 시작...")
            items = _enrich_constituents(items, index, args.refresh_classification)
            _save(index, items, source)
        except Exception as exc:
            print(f"{label} 조회 실패: {exc}", file=sys.stderr)
            failed.append(label)

    if failed:
        # 종료 코드를 남겨야 cron 래퍼가 실패로 보고 슬랙 알림을 보낸다.
        print(f"실패한 인덱스: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
