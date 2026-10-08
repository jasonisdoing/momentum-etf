"""미국 ETF 마켓 목록 — /us-market-etf 화면의 데이터.

한국 ETF 마켓(`kis_market.py` + `market_service.py`)과 같은 구조다.
- 유니버스: KIS 미국 3개 거래소(NAS/NYS/AMS) 종목 마스터에서 증권종류=ETF 만.
  전체 약 6천 개 중 **20일 평균 거래대금 상위 N 개**(`US_ETF_MARKET_TOP_COUNT`)만 담는다 —
  나머지 대부분은 거래가 거의 없는 초소형이라 목록만 무거워진다.
- 가격·수익률: 한국 ETF 마켓과 같은 방식이다. 배치는 yfinance 일봉으로 **기간별 기준종가**만
  저장하고, 화면은 열 때의 **실시간 시세**(`get_realtime_snapshot`, 순위 화면과 같은 소스)로
  현재가·일간(%)을 채우고 기간 수익률을 계산한다. 예전에는 배치 시각의 종가로 계산한 값을
  저장해, 프리장에 순위 화면은 하락인데 이 화면은 전 거래일 상승이 그대로 보였다.
- 저장: 한국과 같은 컬렉션 ``etf_market_master``, master_id ``us_etf_market``.
"""

from __future__ import annotations

import re
from io import BytesIO
from typing import Any
from zipfile import ZipFile

import pandas as pd
import requests

from config import CACHE_TTL_SLOW, KIS_US_MASTER_URLS, US_ETF_MARKET_TOP_COUNT
from utils.db_manager import get_db_connection
from utils.kis_market import BASE_CLOSE_OFFSETS
from utils.logger import get_app_logger
from utils.normalization import normalize_exchange_symbol, to_iso_string
from utils.ttl_cache import TtlCache

logger = get_app_logger()

_MASTER_ID = "us_etf_market"
_COLLECTION_NAME = "etf_market_master"

# 탭 구분 .cod 파일의 필드 위치 (cp949).
_F_TICKER = 4
_F_NAME_KR = 6
_F_NAME_EN = 7  # 운용사·법적 구조까지 담긴 긴 이름 — PTP 판정에 쓴다
_F_SECURITY_TYPE = 8  # 2=주식, 3=ETF
_F_CURRENCY = 9

_ETF_TYPE = "3"
_MASTER_CACHE = TtlCache(CACHE_TTL_SLOW, name="us_security_master", max_entries=1)
# 기간 수익률의 기준일 — 한국 ETF 마켓과 같은 기간에 3달·1~3년을 더한다. 한국은 3달을 네이버
# 실시간 스냅샷이 직접 주지만, 미국 시세에는 그 값이 없어 기준종가로 계산한다.
_BASE_CLOSE_OFFSETS: tuple[tuple[str, pd.DateOffset], ...] = (
    *BASE_CLOSE_OFFSETS,
    ("3m", pd.DateOffset(months=3)),
    ("1y", pd.DateOffset(years=1)),
    ("2y", pd.DateOffset(years=2)),
    ("3y", pd.DateOffset(years=3)),
)
# 기준일 중 가장 먼 3년 전 종가까지 덮는 일봉 기간(yfinance 가 허용하는 값 중 다음 단계).
_LONG_DOWNLOAD_PERIOD = "5y"
_DOLLAR_VOLUME_DAYS = 20  # 거래대금 순위의 평균 일수
_YF_CHUNK = 300

# PTP(Publicly Traded Partnership) 판정 — 국내 투자자는 **매도 대금 총액의 10%** 가
# 원천징수돼 사실상 거래 대상이 아니다. 그래서 화면에서 기본으로 뺀다.
#
# 판정은 마스터의 **긴 이름**(`_F_NAME_EN`)으로 한다 — 거기에 법적 구조가 드러난다:
#   BWET → "AMPLIFY COMMODITY TRUST BREAKWAVE TANKER SHIPPING ETF"
#   USO  → "UNITED STATES OIL FUND LP UNITS"
#   UCO  → "PROSHARES TRUST II ULTA BLOMBERG CRUD OIL"
# 화면에 쓰는 짧은 이름에는 이 정보가 없어서 이름 키워드 필터로는 못 걸렀다.
#
# `LP` 는 **펀드 이름의 LP** 만 본다 — 그냥 걸면 운용사명("SPROTT ASSET MANAGEMENT LP")이
# 걸려 PTP 가 아닌 PHYS·PSLV·CEF 까지 빠진다.
#
# 마스터의 긴 이름이 종목마다 들쭉날쭉해 못 잡는 것이 있다(BOIL·ZSL·GLL·UVXY·VIXY —
# 형제 종목엔 있는 "TRUST II" 가 빠져 있다). 그건 아래 목록에 티커로 적어 보완한다.
_PTP_NAME_PATTERN = re.compile(
    r"(?:FUND|FD|FDS)\s+LP|LP\s+UNIT|PARTNERSHIP|"
    r"COMMODITY TRUST|COMM TR|COMMODTY|COMMODITY POOL|PROSHARES TRUST II",
    re.I,
)

# 긴 이름으로 못 잡는 PTP — 확인되는 대로 티커를 적는다.
_PTP_EXTRA_TICKERS: frozenset[str] = frozenset()


def _is_ptp(ticker: str, legal_name: str) -> bool:
    """이 ETF 가 PTP 인지. 긴 이름의 법적 구조 + 보완 티커 목록으로 본다."""
    return ticker in _PTP_EXTRA_TICKERS or bool(_PTP_NAME_PATTERN.search(legal_name or ""))


def _fetch_us_security_master() -> list[dict[str, str]]:
    """3개 거래소의 USD 주식·ETF 명단을 공통으로 수집한다."""
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
            security_type = fields[_F_SECURITY_TYPE].strip()
            if security_type not in {"2", _ETF_TYPE} or fields[_F_CURRENCY].strip() != "USD":
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
                    "security_type": security_type,
                    # 법적 구조가 드러나는 긴 이름 — PTP 판정용(화면에는 안 쓴다).
                    "legal_name": name_en,
                }
            )
            count += 1
        logger.info("[미국 종목] %s 마스터: 주식·ETF %d건", exchange, count)
    if not rows:
        raise RuntimeError("KIS 미국 마스터에서 주식·ETF를 한 건도 찾지 못했습니다.")
    return rows


def load_us_security_master() -> list[dict[str, str]]:
    """미국 상장 여부를 확인할 공통 명단을 반환한다."""
    return [dict(row) for row in _MASTER_CACHE.get_or_compute("master", _fetch_us_security_master)]


def _load_us_etf_master() -> list[dict[str, str]]:
    return [row for row in load_us_security_master() if row["security_type"] == _ETF_TYPE]


def load_us_security_exchange_map() -> dict[str, str]:
    """공식 미국 상장 명단을 시세 심볼 → 거래소 맵으로 반환한다."""
    return {normalize_exchange_symbol(row["ticker"], suffix=""): row["exchange"] for row in load_us_security_master()}


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
        logger.info(
            "[미국 ETF] 일봉 조회 %d/%d (수신 %d)", min(start + _YF_CHUNK, len(tickers)), len(tickers), len(result)
        )
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

    # 2차: 상위 N 만 5년 일봉을 받아 기준종가(1주~3년 전)와 일간 변동을 계산한다.
    top_tickers = [t for t, _ in top]
    dollar_volume_by = dict(top)
    frames_long = _download_daily(top_tickers, period=_LONG_DOWNLOAD_PERIOD)

    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    base_dates = {suffix: today - offset for suffix, offset in _BASE_CLOSE_OFFSETS}

    rows: list[dict[str, Any]] = []
    for ticker in top_tickers:
        frame = frames_long.get(ticker)
        if frame is None:
            continue
        closes = pd.to_numeric(frame["Close"], errors="coerce").dropna()
        if closes.empty:
            continue
        volumes = pd.to_numeric(frame.get("Volume"), errors="coerce").dropna()
        rows.append(
            {
                "ticker": ticker,
                "name": by_ticker[ticker]["name"],
                "exchange": by_ticker[ticker]["exchange"],
                # 기준일이 휴장이면 직전 거래일 종가(asof). 현재가·수익률은 화면이 실시간으로 계산한다.
                **{
                    f"base_close_{suffix}": _positive_or_none(closes.asof(base_date))
                    for suffix, base_date in base_dates.items()
                },
                "prev_volume": int(volumes.iloc[-1]) if not volumes.empty else 0,
                # 20일 평균 거래대금(백만$) — 시가총액 대신 규모 지표로 쓴다.
                "dollar_volume_musd": round(dollar_volume_by[ticker] / 1e6, 1),
                # PTP 여부 — 화면이 기본으로 제외한다(국내 매도 시 총액 10% 원천징수).
                "is_ptp": _is_ptp(ticker, by_ticker[ticker].get("legal_name", "")),
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


def _positive_or_none(value: Any) -> float | None:
    return float(value) if value is not None and pd.notna(value) and float(value) > 0 else None


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
    """화면용 목록 — 배치가 저장한 기준종가에 **실시간 시세**와 종목풀·보유 표시를 붙인다."""
    from services.price_service import get_realtime_snapshot
    from utils.market_service import load_ticker_pool_map, load_ticker_pool_type_map, return_pct_from_base
    from utils.portfolio_io import load_all_holding_tickers

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")
    doc = db[_COLLECTION_NAME].find_one({"master_id": _MASTER_ID}) or {}
    rows = doc.get("rows") or []
    if not rows:
        raise RuntimeError("미국 ETF 마켓 캐시가 없습니다. update_us_market_etfs 를 먼저 실행하세요.")

    ticker_pool_map = load_ticker_pool_map()
    ticker_pool_type_map = load_ticker_pool_type_map()
    held_tickers = load_all_holding_tickers()
    # 현재가·일간(%)은 순위 화면과 **같은 실시간 소스**다 — 세션(프리·정규·애프터·데이장)
    # 가격 선택과 기준가 규칙이 한 곳(`utils.realtime_quotes`)에서 정해진다.
    snapshot = get_realtime_snapshot("us", [row["ticker"] for row in rows])

    def _live_fields(row: dict[str, Any]) -> dict[str, Any]:
        snap = snapshot.get(row["ticker"]) or {}
        now_val = snap.get("nowVal")
        return {
            "current_price": now_val,
            "daily_change_pct": snap.get("changeRate"),
            **{
                f"return_{suffix}_pct": return_pct_from_base(now_val, row.get(f"base_close_{suffix}"))
                for suffix, _ in _BASE_CLOSE_OFFSETS
            },
        }

    result_rows = [
        {
            # 기준종가는 화면에 내보내지 않는다 — 수익률로만 쓴다.
            **{key: value for key, value in row.items() if not key.startswith("base_close_")},
            **_live_fields(row),
            "ticker_pools": ", ".join(ticker_pool_map.get(row["ticker"], [])),
            # 종목풀 추가 사전 필터(공용 pool-add)가 풀 id 로 판별한다 — 이름은 표시용.
            "ticker_pool_types": ticker_pool_type_map.get(row["ticker"], []),
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
