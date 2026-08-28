"""미국 ETF 마켓 목록 — /us-market-etf 화면의 데이터.

한국 ETF 마켓(`kis_market.py` + `market_service.py`)과 같은 구조다.
- 유니버스: KIS 미국 3개 거래소(NAS/NYS/AMS) 종목 마스터에서 증권종류=ETF 만.
  전체 약 6천 개 중 **20일 평균 거래대금 상위 N 개**(`US_ETF_MARKET_TOP_COUNT`)만 담는다 —
  나머지 대부분은 거래가 거의 없는 초소형이라 목록만 무거워진다.
- 가격·수익률: yfinance 일봉으로 배치 때 계산해 함께 저장한다. 한국과 달리 실시간
  스냅샷이 없으므로 화면 값의 기준 시각 = 배치 시각(`updated_at`)이다.
- 저장: 한국과 같은 컬렉션 ``etf_market_master``, master_id ``us_etf_market``.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any
from zipfile import ZipFile

import pandas as pd
import requests

from config import KIS_US_MASTER_URLS, US_ETF_MARKET_TOP_COUNT
from utils.db_manager import get_db_connection
from utils.logger import get_app_logger
from utils.normalization import to_iso_string

logger = get_app_logger()

_MASTER_ID = "us_etf_market"
_COLLECTION_NAME = "etf_market_master"

# 탭 구분 .cod 파일의 필드 위치 (cp949).
_F_TICKER = 4
_F_NAME_KR = 6
_F_NAME_EN = 7
_F_SECURITY_TYPE = 8  # 2=주식, 3=ETF
_F_CURRENCY = 9

_ETF_TYPE = "3"
_DOLLAR_VOLUME_DAYS = 20  # 거래대금 순위의 평균 일수
_YF_CHUNK = 300


def _load_us_etf_master() -> list[dict[str, str]]:
    """3개 거래소 마스터에서 USD ETF 를 모은다. 같은 티커는 먼저 만난 거래소를 유지."""
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for exchange, url in KIS_US_MASTER_URLS.items():
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with ZipFile(BytesIO(response.content)) as zf:
            names = zf.namelist()
            if len(names) != 1:
                raise RuntimeError(f"{exchange} 마스터 zip 파일 구성이 예상과 다릅니다: {names}")
            text = zf.read(names[0]).decode("cp949", errors="replace")
        count = 0
        for line in text.splitlines():
            fields = line.split("\t")
            if len(fields) <= _F_CURRENCY:
                continue
            if fields[_F_SECURITY_TYPE].strip() != _ETF_TYPE or fields[_F_CURRENCY].strip() != "USD":
                continue
            ticker = fields[_F_TICKER].strip().upper()
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            name_kr = fields[_F_NAME_KR].strip()
            name_en = fields[_F_NAME_EN].strip()
            rows.append(
                {
                    "ticker": ticker,
                    # 화면 표기는 한글명 우선 — KIS 가 번역해 둔 종목만 있고, 없으면 영문명.
                    "name": name_kr or name_en,
                    "exchange": exchange,
                }
            )
            count += 1
        logger.info("[미국 ETF] %s 마스터: ETF %d건", exchange, count)
    if not rows:
        raise RuntimeError("KIS 미국 마스터에서 ETF 를 한 건도 찾지 못했습니다.")
    return rows


def _download_daily(tickers: list[str], period: str) -> dict[str, pd.DataFrame]:
    """yfinance 일봉을 청크로 받아 티커별 프레임(Close·Volume)으로 돌려준다."""
    import yfinance as yf

    from utils.yfinance_guard import yfinance_lock

    result: dict[str, pd.DataFrame] = {}
    for start in range(0, len(tickers), _YF_CHUNK):
        chunk = tickers[start : start + _YF_CHUNK]
        with yfinance_lock():
            downloaded = yf.download(
                chunk,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        if downloaded is None or downloaded.empty:
            continue
        for ticker in chunk:
            if isinstance(downloaded.columns, pd.MultiIndex):
                if ticker not in downloaded.columns.get_level_values(0):
                    continue
                frame = downloaded[ticker]
            elif len(chunk) == 1:
                frame = downloaded
            else:
                continue
            closes = pd.to_numeric(frame.get("Close"), errors="coerce")
            if closes is None or closes.dropna().empty:
                continue
            result[ticker] = frame
        logger.info("[미국 ETF] 일봉 조회 %d/%d (수신 %d)", min(start + _YF_CHUNK, len(tickers)), len(tickers), len(result))
    return result


def _avg_dollar_volume(frame: pd.DataFrame) -> float | None:
    closes = pd.to_numeric(frame["Close"], errors="coerce")
    volumes = pd.to_numeric(frame.get("Volume"), errors="coerce")
    if volumes is None:
        return None
    dollar = (closes * volumes).dropna().tail(_DOLLAR_VOLUME_DAYS)
    if dollar.empty:
        return None
    return float(dollar.mean())


def refresh_us_etf_market_cache() -> int:
    """마스터 + yfinance 로 미국 ETF 목록 캐시를 다시 만든다. 반환: 저장 건수."""
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")

    master_rows = _load_us_etf_master()
    by_ticker = {row["ticker"]: row for row in master_rows}

    # 1차: 최근 한 달 일봉으로 거래대금 순위를 매겨 상위 N 을 고른다.
    frames = _download_daily(list(by_ticker), period="1mo")
    ranked = sorted(
        ((ticker, _avg_dollar_volume(frame)) for ticker, frame in frames.items()),
        key=lambda item: item[1] or 0.0,
        reverse=True,
    )
    top = [(t, dv) for t, dv in ranked if dv is not None][:US_ETF_MARKET_TOP_COUNT]
    logger.info("[미국 ETF] 유니버스 %d → 시세 수신 %d → 상위 %d", len(by_ticker), len(frames), len(top))

    # 2차: 상위 N 만 4개월 일봉을 받아 기준종가(1/2/3개월 전)와 일간 변동을 계산한다.
    top_tickers = [t for t, _ in top]
    dollar_volume_by = dict(top)
    frames_long = _download_daily(top_tickers, period="4mo")

    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    bases = {n: today - pd.DateOffset(months=n) for n in (1, 2, 3)}

    def _return_pct(now_val: float, base_close: float | None) -> float | None:
        if base_close in (None, 0) or pd.isna(base_close):
            return None
        return round((now_val / float(base_close) - 1.0) * 100.0, 4)

    rows: list[dict[str, Any]] = []
    for ticker in top_tickers:
        frame = frames_long.get(ticker)
        if frame is None:
            continue
        closes = pd.to_numeric(frame["Close"], errors="coerce").dropna()
        if closes.empty:
            continue
        now_val = float(closes.iloc[-1])
        prev_close = float(closes.iloc[-2]) if len(closes) >= 2 else None
        volumes = pd.to_numeric(frame.get("Volume"), errors="coerce").dropna()
        base_closes = {n: closes.asof(base) for n, base in bases.items()}
        rows.append(
            {
                "ticker": ticker,
                "name": by_ticker[ticker]["name"],
                "exchange": by_ticker[ticker]["exchange"],
                "current_price": round(now_val, 4),
                "daily_change_pct": _return_pct(now_val, prev_close),
                "return_1m_pct": _return_pct(now_val, base_closes[1]),
                "return_2m_pct": _return_pct(now_val, base_closes[2]),
                "return_3m_pct": _return_pct(now_val, base_closes[3]),
                "prev_volume": int(volumes.iloc[-1]) if not volumes.empty else 0,
                # 20일 평균 거래대금(백만$) — 시가총액 대신 규모 지표로 쓴다.
                "dollar_volume_musd": round(dollar_volume_by[ticker] / 1e6, 1),
            }
        )

    rows.sort(key=lambda row: row["dollar_volume_musd"], reverse=True)
    db[_COLLECTION_NAME].update_one(
        {"master_id": _MASTER_ID},
        {"$set": {"rows": rows, "updated_at": pd.Timestamp.utcnow().to_pydatetime()}},
        upsert=True,
    )
    logger.info("[미국 ETF] 캐시 저장 %d건", len(rows))
    return len(rows)


def us_etf_name_of(ticker: str) -> str | None:
    """마켓 캐시에서 미국 ETF 이름을 찾는다 — 종목 추가 검증이 yfinance 호출 전에 쓴다.

    캐시는 거래대금 상위 N 개만 담으므로 없으면 None(호출부가 외부 조회로 넘어간다).
    """
    ticker_norm = str(ticker or "").strip().upper()
    db = get_db_connection()
    if not ticker_norm or db is None:
        return None
    doc = db[_COLLECTION_NAME].find_one({"master_id": _MASTER_ID}, {"rows.ticker": 1, "rows.name": 1}) or {}
    for row in doc.get("rows") or []:
        if str(row.get("ticker") or "").strip().upper() == ticker_norm:
            name = str(row.get("name") or "").strip()
            return name or None
    return None


def load_us_etf_market_data() -> dict[str, Any]:
    """화면용 목록 — 배치가 저장한 값에 종목풀·보유 표시만 붙인다."""
    from utils.market_service import load_ticker_pool_map
    from utils.portfolio_io import load_all_holding_tickers

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")
    doc = db[_COLLECTION_NAME].find_one({"master_id": _MASTER_ID}) or {}
    rows = doc.get("rows") or []
    if not rows:
        raise RuntimeError("미국 ETF 마켓 캐시가 없습니다. update_us_market_etfs 를 먼저 실행하세요.")

    ticker_pool_map = load_ticker_pool_map()
    held_tickers = load_all_holding_tickers()

    result_rows = [
        {
            **row,
            "ticker_pools": ", ".join(ticker_pool_map.get(row["ticker"], [])),
            "is_held": row["ticker"] in held_tickers,
            "listed_at": "",  # 미국 마스터에는 상장일이 없다
            "nav": None,
            "deviation": None,
            "market_cap": row.get("dollar_volume_musd"),
        }
        for row in rows
    ]
    return {
        "updated_at": to_iso_string(doc.get("updated_at")),
        "rows": result_rows,
    }
