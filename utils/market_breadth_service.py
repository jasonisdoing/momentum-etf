"""시장 폭(market breadth) — 일별 상승/하락 종목수 적립과 ADR(등락비율) 산출.

지수는 시가총액 큰 몇 종목에 끌려가지만 ADR 은 '얼마나 많은 종목이 함께 오르는가'를 본다.
지수가 오르는데 ADR 이 빠지면 상승 종목이 좁아지는 국면이라 모멘텀 전략에 불리하다.

    ADR = 기간 내 상승종목수 합 ÷ 하락종목수 합 × 100

대상 종목은 시장마다 다르다. 미국은 지수 구성종목 전체(S&P 500 / 나스닥 100)를 쓰고,
한국은 구성종목 목록을 얻을 수 없어 시가총액 상위 N(코스피 200 / 코스닥 150)으로 근사한다.
이 선정은 배치가 소유한다 — 화면을 열어야 채워지던 캐시에 기대면 조용히 빈 값이 된다.

**남기는 것은 하루치 상승/하락 종목수뿐이다.** ADR 은 20일 누적이라 읽을 때 계산한다.
ADR 값만 남기면 창에서 빠지는 20일 전 카운트가 없어 매번 20일치 가격을 다시 받아야 한다.
가격은 어디에도 저장하지 않는다 — 종목풀 가격 캐시는 다른 배치가 소유한다.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import yfinance as yf

from config import (
    MARKET_ADR_NEUTRAL,
    MARKET_ADR_OVERHEATED,
    MARKET_ADR_OVERSOLD,
    MARKET_ADR_WINDOW_DAYS,
)
from utils.db_manager import get_db_connection
from utils.index_constituents_loader import load_index_constituents
from utils.kor_stock_market_service import _fetch_market_value_page, _parse_float
from utils.naver_chart import fetch_naver_daily_ohlc
from utils.yfinance_guard import yfinance_lock

logger = logging.getLogger(__name__)

COLLECTION_NAME = "market_breadth_daily"

# 시장별 대상 종목과 가격 소스.
#
#   source="naver"       — 시가총액 상위 N 종목. 한국은 지수 구성종목 목록을 얻을 수 없어
#                          상위 N 으로 근사한다(`universe_size`).
#   source="constituents" — `index_constituents` 에 저장된 **실제 구성종목 전체**.
#                          `/us-market-stock` 배치가 유지하므로 지수 정의와 정확히 같다.
#
# `index_symbol` 은 그 시장의 거래일 달력으로 쓰는 지수다(별도 거래일 달력이 없다).
MARKETS: dict[str, dict[str, Any]] = {
    "KOSPI": {
        "name": "코스피",
        "source": "naver",
        "universe_size": 200,
        "index_symbol": "KOSPI",
    },
    "KOSDAQ": {
        "name": "코스닥",
        "source": "naver",
        "universe_size": 150,
        "index_symbol": "KOSDAQ",
    },
    "SP500": {
        "name": "S&P 500",
        "source": "constituents",
        "index_key": "SP500",
        "index_symbol": "^GSPC",
    },
    "NDX100": {
        "name": "나스닥 100",
        "source": "constituents",
        "index_key": "NDX100",
        "index_symbol": "^NDX",
    },
}

# 그날 판정에 쓰인 종목이 대상의 이 비율에 못 미치면 기록하지 않는다.
# 표본이 적은 날은 상승/하락 비가 크게 흔들려 ADR 을 통째로 왜곡한다.
#
# 시장 4개는 지수 구성종목이라 그날 전부 존재한다 — 표본이 모자라면 상장 문제가 아니라
# **조회가 실패한 것**이므로 그날은 버린다.
_MIN_COUNTED_RATIO = 0.9

# 종목풀은 기준이 다르다. 과거로 갈수록 아직 상장 전인 종목이 많아 미달이 정상이고,
# 그날 존재하던 종목들의 폭이 그 시점의 진짜 ADR 이다. 지금 구성의 90% 를 요구하면
# 멀쩡한 과거를 통째로 버린다(kor_kr 은 1882일 중 115일만 남았다).
# 50% 로 두면 실제로 세는 종목이 가장 적은 풀도 41종목이라 비율이 흔들리지 않는다.
_POOL_MIN_COUNTED_RATIO = 0.5

# 지수 티커 → 시장. 화면(`/market-trend`)의 지수 행과 ADR 을 잇는다.
MARKET_BY_INDEX_TICKER = {"^KS11": "KOSPI", "^KQ11": "KOSDAQ", "^GSPC": "SP500", "^NDX": "NDX100"}

# 일봉을 받아 둘 기간. ADR 창(20일)보다 넉넉해야 첫 실행부터 값이 나온다.
_HISTORY_DAYS = 400
_PAGE_SIZE = 100
# yfinance 일괄 다운로드 묶음 크기. 500종목을 한 번에 요청하면 자주 실패한다.
_YF_BATCH_SIZE = 50


def _require_db():
    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패했습니다.")
    return db


def _index_trading_days(spec: dict[str, Any], count: int = _HISTORY_DAYS) -> list[str]:
    """시장의 거래일 목록(오름차순). 거래일 달력이 따로 없어 지수 일봉을 달력으로 쓴다.

    한국 지수는 네이버(yfinance 가 하루 지연되는 이슈 회피), 미국 지수는 yfinance 다.
    """
    symbol = str(spec["index_symbol"])
    if not symbol.startswith("^"):
        ohlc = fetch_naver_daily_ohlc(symbol, count=count)
        if ohlc is None or ohlc.empty:
            raise RuntimeError(f"{symbol} 지수 일봉을 조회하지 못해 거래일을 정할 수 없습니다.")
        return [str(index)[:10] for index in ohlc.index]

    with yfinance_lock():
        frame = yf.download(symbol, period="2y", interval="1d", progress=False, auto_adjust=True)
    if frame is None or frame.empty:
        raise RuntimeError(f"{symbol} 지수 일봉을 조회하지 못해 거래일을 정할 수 없습니다.")
    days = [str(pd.Timestamp(index).date()) for index in frame.index]
    return days[-count:]


def _fetch_kor_universe_with_moves(market: str, size: int) -> tuple[list[str], dict[str, float]]:
    """시가총액 상위 ``size`` 종목과 그 종목의 당일 등락률.

    네이버 시세표가 대상 목록과 등락률을 함께 주므로, 종목별 일봉을 받을 필요가 없다.
    페이지 2~4회 호출로 하루치 집계가 끝난다.
    ETF·ETN 과 일반 주식이 아닌 항목은 제외한다 (`/kor-market-stock` 화면과 같은 기준).
    """
    tickers: list[str] = []
    ratios: dict[str, float] = {}
    page = 1
    while len(tickers) < size:
        payload = _fetch_market_value_page(market, page=page, page_size=_PAGE_SIZE)
        stocks = payload.get("stocks") or []
        if not stocks:
            break
        for item in stocks:
            if len(tickers) >= size:
                break
            name = str(item.get("stockName") or "")
            if str(item.get("stockEndType", "")).lower() != "stock":
                continue
            if any(keyword in name.upper() for keyword in ("ETF", "ETN")):
                continue
            ticker = str(item.get("itemCode") or "").strip()
            if not ticker:
                continue
            tickers.append(ticker)
            ratio = _parse_float(item.get("fluctuationsRatio"))
            if ratio is not None:
                ratios[ticker] = ratio
        page += 1

    if len(tickers) < size:
        # 수를 못 채웠다는 사실을 남긴다 — 조용히 적은 표본으로 계산하면 ADR 이 흔들린다.
        logger.warning("[market_breadth] %s 대상 %d종목 요청, %d종목만 확보", market, size, len(tickers))
    return tickers, ratios


def resolve_market_universe(market: str, spec: dict[str, Any]) -> list[str]:
    """시장의 대상 종목. 미국은 실제 구성종목 전체, 한국은 시가총액 상위 N 이다."""
    if spec["source"] == "constituents":
        items = load_index_constituents(str(spec["index_key"]))
        return [str(item["ticker"]).strip().upper() for item in items if item.get("ticker")]
    return _fetch_kor_universe_with_moves(market, int(spec["universe_size"]))[0]


def _download_us_closes(tickers: list[str], period: str) -> pd.DataFrame:
    """미국 종목 종가를 묶음으로 받아 (날짜 × 티커) 표로 만든다. 저장하지 않는다."""
    columns: dict[str, pd.Series] = {}
    for start in range(0, len(tickers), _YF_BATCH_SIZE):
        batch = tickers[start : start + _YF_BATCH_SIZE]
        try:
            with yfinance_lock():
                downloaded = yf.download(
                    batch,
                    period=period,
                    interval="1d",
                    group_by="column",
                    progress=False,
                    auto_adjust=True,
                    threads=True,
                )
        except Exception:
            logger.exception("[market_breadth] yfinance 묶음 조회 실패 (%s…)", batch[0])
            continue
        if downloaded is None or downloaded.empty:
            continue

        if isinstance(downloaded.columns, pd.MultiIndex):
            if "Close" not in downloaded.columns.get_level_values(0):
                continue
            close_frame = downloaded["Close"]
        else:
            if "Close" not in downloaded.columns:
                continue
            # 종목이 하나뿐이면 컬럼이 평평하게 온다.
            close_frame = downloaded[["Close"]].rename(columns={"Close": batch[0]})

        close_frame.index = pd.to_datetime(close_frame.index).tz_localize(None).normalize()
        for ticker in batch:
            if ticker in close_frame.columns:
                series = close_frame[ticker].dropna()
                if not series.empty:
                    columns[ticker] = series

    return pd.DataFrame(columns).sort_index() if columns else pd.DataFrame()


def _count_daily_moves(frame: pd.DataFrame) -> pd.DataFrame:
    """일자별 상승/하락/보합 종목수를 센다.

    전일 종가가 없는 종목(상장 전·결측)은 그날의 분모에서 빠진다 — 값을 지어내지 않는다.
    """
    change = frame.pct_change(fill_method=None)
    counted = change.notna()
    return pd.DataFrame(
        {
            "advance": (change > 0).sum(axis=1),
            "decline": (change < 0).sum(axis=1),
            "unchanged": ((change == 0) & counted).sum(axis=1),
            "counted": counted.sum(axis=1),
        }
    )


def _counts_from_ratios(ratios: dict[str, float]) -> dict[str, int]:
    """등락률 맵에서 상승/하락/보합 수를 센다. 등락률이 없는 종목은 세지 않는다."""
    values = list(ratios.values())
    return {
        "advance": sum(1 for value in values if value > 0),
        "decline": sum(1 for value in values if value < 0),
        "unchanged": sum(1 for value in values if value == 0),
        "counted": len(values),
    }


def _daily_counts_light(market: str, spec: dict[str, Any], target_date: str) -> tuple[list[str], dict[str, int]] | None:
    """평상시 경로 — 하루치 등락만 받아 센다. 가격은 저장하지 않는다.

    미국은 마지막 봉의 날짜가 기준일과 다르면 None 을 돌려준다. 다른 날 값을
    그날 것으로 기록하면 20일 누적이 통째로 어긋난다.
    """
    if spec["source"] == "naver":
        tickers, ratios = _fetch_kor_universe_with_moves(market, int(spec["universe_size"]))
        return tickers, _counts_from_ratios(ratios)

    tickers = resolve_market_universe(market, spec)
    # 직전 거래일 대비 등락을 봐야 하므로 최소 이틀, 휴장을 감안해 5일치를 받는다.
    closes = _download_us_closes(tickers, period="5d")
    if closes.empty:
        return None
    last_date = str(closes.index[-1])[:10]
    if last_date != target_date:
        logger.warning(
            "[market_breadth] %s 최신 봉(%s)이 기준일(%s)과 달라 기록하지 않습니다.",
            market,
            last_date,
            target_date,
        )
        return None
    counts = _count_daily_moves(closes).iloc[-1]
    return tickers, {key: int(counts[key]) for key in ("advance", "decline", "unchanged", "counted")}


def _daily_counts_backfill(
    market: str, spec: dict[str, Any], missing_dates: set[str]
) -> tuple[list[str], dict[str, dict[str, int]]]:
    """구멍을 메우는 경로 — 일봉을 받아 빠진 날들을 다시 센다.

    첫 실행이거나 배치를 거른 날이 있을 때만 탄다. 받은 일봉은 메모리에서만 쓰고
    가격 캐시에 남기지 않는다 — 남기면 그 캐시를 소유한 배치와 어긋난다.
    """
    tickers = resolve_market_universe(market, spec)

    if spec["source"] == "constituents":
        closes = _download_us_closes(tickers, period="2y")
    else:
        series_map: dict[str, pd.Series] = {}
        for ticker in tickers:
            ohlc = fetch_naver_daily_ohlc(ticker, count=_HISTORY_DAYS)
            if ohlc is None or ohlc.empty or "Close" not in ohlc.columns:
                logger.warning("[market_breadth] %s 일봉 조회 실패 — 대상에서 제외", ticker)
                continue
            series_map[ticker] = ohlc["Close"]
        closes = pd.DataFrame(series_map).sort_index() if series_map else pd.DataFrame()

    if closes.empty:
        raise RuntimeError(f"{market} 대상 종목의 일봉을 하나도 확보하지 못했습니다.")

    counts = _count_daily_moves(closes)
    by_date: dict[str, dict[str, int]] = {}
    for index, row in zip(counts.index, counts.itertuples(index=False)):
        date = str(index)[:10]
        if date not in missing_dates:
            continue
        by_date[date] = {
            "advance": int(row.advance),
            "decline": int(row.decline),
            "unchanged": int(row.unchanged),
            "counted": int(row.counted),
        }
    return tickers, by_date


# 종목풀 ADR — 시장 지수 대신 **그 풀의 종목들** 로 폭을 잰다.
#
# 지수 ADR 은 매매하는 종목과 구성이 다르다(us_stock 은 S&P100+나스닥100 조합이라
# SP500 과도 NDX100 과도 다르다). 풀 자신의 폭을 보는 편이 그 전략에 더 맞다.
#
# 시장 4개와 달리 **외부 조회가 없다** — 가격 캐시가 이미 매일 갱신되므로 그것만 센다.
# 그래서 활성 풀 전부를 매일 돌려도 비용이 거의 없고, 골라 둔 풀만 계산하지 않는다
# (레짐을 바꾼 순간 과거가 비어 백테스트가 그 구간을 못 쓰기 때문).
POOL_MARKET_PREFIX = "pool:"

# 종목풀 레짐을 고른 상태를 나타내는 예약 티커. 지수와 같은 `{ticker, name}` 자리에
# 들어가 저장 스키마를 바꾸지 않는다 — 어느 풀인지는 설정을 읽는 그 풀 자신이다.
SELF_POOL_REGIME_TICKER = "pool:self"


def pool_market_key(pool: str) -> str:
    """종목풀 ADR 의 시장 키 — 시장 4개와 같은 컬렉션에 섞여 들어간다."""
    return f"{POOL_MARKET_PREFIX}{str(pool or '').strip().lower()}"


def pool_of_market_key(market: str) -> str | None:
    """시장 키가 종목풀이면 그 풀 id, 지수면 None."""
    text = str(market or "")
    return text[len(POOL_MARKET_PREFIX) :] if text.startswith(POOL_MARKET_PREFIX) else None


def _pool_daily_counts(pool: str, since: str | None = None) -> tuple[int, dict[str, dict[str, int]]]:
    """가격 캐시에서 그 풀의 일별 상승/하락 종목수를 센다.

    전일 종가 대비로 판정한다 — 시장 4개(등락률·일봉)와 같은 기준이다.
    보합(0%)은 어느 쪽에도 넣지 않는다. `counted` 는 그날 판정에 쓰인 종목 수다.

    ``since`` 를 주면 **그날 이후만** 센다(그날 자신은 최신일 갱신 대상이라 포함).
    전일 대비를 내야 하므로 계산에는 그 하루 앞까지 넣는다.
    """
    from utils.cache_utils import load_cached_frames_bulk_from_ticker_types
    from utils.stock_list_io import get_etfs

    tickers = [str(row["ticker"]).strip() for row in get_etfs(pool) if row.get("ticker")]
    if not tickers:
        return 0, {}
    frames = load_cached_frames_bulk_from_ticker_types([pool], tickers)

    closes: dict[str, pd.Series] = {}
    for ticker, frame in frames.items():
        if frame is None or frame.empty or "Close" not in frame:
            continue
        series = pd.to_numeric(frame["Close"], errors="coerce")
        series = series[series > 0].dropna()
        if len(series) >= 2:
            closes[ticker] = series
    if not closes:
        return len(tickers), {}

    close_df = pd.DataFrame(closes).sort_index()
    if since:
        # 이미 저장된 구간은 다시 세지 않는다. 전일 대비를 내야 해서 한 칸 앞부터 자른다.
        stamps = list(close_df.index)
        keep_from = next((i for i, day in enumerate(stamps) if str(pd.Timestamp(day).date()) >= since), None)
        if keep_from is None:
            return len(tickers), {}
        close_df = close_df.iloc[max(keep_from - 1, 0) :]
    change = close_df.pct_change()
    counts: dict[str, dict[str, int]] = {}
    for day, row in change.iterrows():
        valid = row.dropna()
        if valid.empty:
            continue
        counts[str(pd.Timestamp(day).date())] = {
            "advance": int((valid > 0).sum()),
            "decline": int((valid < 0).sum()),
            "counted": int(len(valid)),
        }
    return len(tickers), counts


def refresh_pool_breadth(pools: list[str] | None = None, *, full: bool = False) -> dict[str, Any]:
    """활성 종목풀의 상승/하락 종목수를 채운다 — 가격 캐시만 읽는다(외부 조회 없음).

    시장 4개(`refresh_market_breadth`)와 같은 컬렉션·스키마를 쓰고, 시장 키만
    `pool:<풀id>` 다. 그래서 읽기(`load_adr_series`)·판정·화면은 손댈 것이 없다.

    **과거는 소급 계산한다.** 가격 캐시에 이력이 있으니 처음부터 백테스트에 쓸 수 있다.
    다만 지금 풀 구성으로 과거를 재구성하는 것이라 생존 편향이 들어간다 — 중간에 빠진
    종목은 없는 셈이 되고, 새로 들어온 종목은 편입 전 구간까지 계산에 들어간다.
    (표본 가드가 이를 일부 걸러낸다 — 대상의 90% 가 안 차는 날은 기록하지 않는다.)

    평상시에는 **마지막 저장일 이후만** 센다. ``full=True`` 면 전 이력을 다시 훑는다
    (풀 종목이 크게 바뀌어 과거까지 다시 세고 싶을 때만 쓴다 — 이미 있는 날짜는
    그래도 덮지 않으므로, 정말 다시 만들려면 그 풀의 문서를 먼저 지워야 한다).
    """
    from utils.settings_loader import list_available_ticker_types

    db = _require_db()
    collection = db[COLLECTION_NAME]
    collection.create_index([("market", 1), ("date", 1)], unique=True, name="market_date_unique")

    targets = pools if pools is not None else list_available_ticker_types()
    summary: dict[str, Any] = {"pools": {}}
    for pool in targets:
        market = pool_market_key(pool)
        # 이미 채운 구간은 건너뛴다 — 첫 실행만 전 이력을 훑고 그 뒤로는 하루치씩이다.
        # 마지막 저장일 자신은 장중 재실행으로 값이 바뀔 수 있어 다시 센다.
        last_doc = None if full else collection.find_one({"market": market}, {"_id": 0, "date": 1}, sort=[("date", -1)])
        since = str(last_doc["date"]) if last_doc else None
        try:
            universe_size, counts_by_date = _pool_daily_counts(pool, since)
        except Exception as exc:
            logger.warning("[pool_breadth] %s 집계 실패: %s", pool, exc)
            summary["pools"][pool] = {"error": str(exc)}
            continue
        if not counts_by_date:
            summary["pools"][pool] = {"skipped": True, "reason": "가격 캐시에 종가가 없습니다."}
            continue

        # 표본 가드 — 종목풀은 시장 4개보다 느슨하다(위 상수 주석 참고).
        min_counted = universe_size * _POOL_MIN_COUNTED_RATIO
        target_date = max(counts_by_date)
        inserted = updated = skipped = 0
        for date, counts in sorted(counts_by_date.items()):
            if counts["counted"] < min_counted:
                skipped += 1
                continue
            doc = {"market": market, "date": date, **counts, "universe_size": universe_size}
            if date == target_date:
                # 최신일은 장중 재실행으로 값이 바뀔 수 있어 갱신한다.
                collection.update_one({"market": market, "date": date}, {"$set": doc}, upsert=True)
                updated += 1
                continue
            result = collection.update_one({"market": market, "date": date}, {"$setOnInsert": doc}, upsert=True)
            if result.upserted_id is not None:
                inserted += 1
        summary["pools"][pool] = {
            "market": market,
            "universe_size": universe_size,
            "inserted": inserted,
            "updated": updated,
            "skipped": skipped,
            "latest_date": target_date,
        }
    return summary


def refresh_market_breadth() -> dict[str, Any]:
    """시장별로 빠진 날의 상승/하락 종목수를 채운다.

    평상시에는 **그날 등락만** 받아 한 줄을 더한다(한국은 시세표 2~4회, 미국은 묶음 조회).
    배치를 거른 날이 있으면 그 구간만 일봉을 받아 되메운다 — 구멍이 생기면 20일 누적이
    조용히 달라지기 때문이다. 어느 경로든 가격은 저장하지 않는다.

    과거 일자는 **이미 있으면 덮지 않는다** — 그날의 대상 종목은 그날 기준이어야 하는데,
    지금 대상으로 과거를 다시 계산해 덮으면 값이 매번 조금씩 달라진다. 가장 최근 하루는
    장중에 여러 번 돌 수 있어 갱신한다.
    """
    db = _require_db()
    collection = db[COLLECTION_NAME]
    collection.create_index([("market", 1), ("date", 1)], unique=True, name="market_date_unique")

    summary: dict[str, Any] = {"markets": {}}
    for market, spec in MARKETS.items():
        trading_days = _index_trading_days(spec)
        target_date = trading_days[-1]
        last_doc = collection.find_one({"market": market}, {"_id": 0, "date": 1}, sort=[("date", -1)])
        last_stored = str(last_doc["date"]) if last_doc else None

        missing = [day for day in trading_days if last_stored is None or day > last_stored]
        # 최신일은 장중 재실행으로 값이 바뀔 수 있어 항상 다시 센다.
        gap_dates = {day for day in missing if day != target_date}

        if gap_dates:
            logger.info("[market_breadth] %s 빠진 %d일을 일봉으로 되메웁니다.", market, len(gap_dates))
            tickers, counts_by_date = _daily_counts_backfill(market, spec, gap_dates | {target_date})
        else:
            light = _daily_counts_light(market, spec, target_date)
            if light is None:
                # 키 이름을 `skipped` 로 쓰지 않는다 — 아래 정상 형태의 `skipped` 는
                # '표본 부족으로 건너뛴 **날짜 수**'라 뜻이 다르다. 같은 이름을 쓰면
                # 요약을 찍는 쪽이 둘을 구분하지 못해 죽는다(KeyError: 'universe').
                summary["markets"][market] = {
                    "market_skipped": True,
                    "reason": "기준일 불일치",
                    "latest_date": target_date,
                }
                continue
            tickers, counts = light
            counts_by_date = {target_date: counts}

        min_counted = len(tickers) * _MIN_COUNTED_RATIO
        inserted = updated = skipped = 0
        for date, counts in sorted(counts_by_date.items()):
            if counts["counted"] < min_counted:
                # 상장 전이라 표본이 적은 초기 구간, 또는 일부 종목만 최신인 날.
                skipped += 1
                continue
            doc = {"market": market, "date": date, **counts, "universe_size": len(tickers)}
            if date == target_date:
                collection.update_one({"market": market, "date": date}, {"$set": doc}, upsert=True)
                updated += 1
                continue
            result = collection.update_one({"market": market, "date": date}, {"$setOnInsert": doc}, upsert=True)
            if result.upserted_id is not None:
                inserted += 1

        summary["markets"][market] = {
            "market_skipped": False,
            "universe": len(tickers),
            "inserted": inserted,
            "updated": updated,
            # 표본 부족으로 건너뛴 **날짜 수** — 시장을 건너뛴 것과 다르다.
            "skipped_days": skipped,
            "latest_date": target_date,
        }
        logger.info(
            "[market_breadth] %s 대상 %d종목 · 신규 %d일 · 갱신 %d일 · 표본부족 %d일 (기준일 %s)",
            market,
            len(tickers),
            inserted,
            updated,
            skipped,
            target_date,
        )

    return summary


def load_adr_series(market: str, limit_days: int | None = None) -> list[dict[str, Any]]:
    """일별 ADR 시계열. 창이 다 차기 전 구간은 ``adr=None`` 으로 둔다(보정하지 않는다)."""
    db = _require_db()
    docs = list(db[COLLECTION_NAME].find({"market": market}, {"_id": 0}).sort("date", 1))
    if not docs:
        return []

    window = MARKET_ADR_WINDOW_DAYS
    points: list[dict[str, Any]] = []
    for position, doc in enumerate(docs):
        adr: float | None = None
        if position >= window - 1:
            chunk = docs[position - window + 1 : position + 1]
            advance_sum = sum(int(item.get("advance") or 0) for item in chunk)
            decline_sum = sum(int(item.get("decline") or 0) for item in chunk)
            # 하락 종목이 하나도 없으면 나눌 수 없다. 큰 값으로 대체하지 않는다.
            if decline_sum > 0:
                adr = advance_sum / decline_sum * 100.0
        points.append(
            {
                "date": doc["date"],
                "adr": adr,
                "advance": int(doc.get("advance") or 0),
                "decline": int(doc.get("decline") or 0),
            }
        )

    return points[-limit_days:] if limit_days else points


def classify_adr(adr: float | None) -> str | None:
    """ADR 값을 4단계로 나눈다.

    100 은 상승 종목수와 하락 종목수가 같은 지점이라 강세/약세의 경계다.
    바깥쪽 두 단계(과매수·과매도)는 되돌림을 경계해야 하는 구간이다.
    """
    if adr is None:
        return None
    if adr >= MARKET_ADR_OVERHEATED:
        return "overbought"
    if adr >= MARKET_ADR_NEUTRAL:
        return "bullish"
    if adr > MARKET_ADR_OVERSOLD:
        return "bearish"
    return "oversold"


# 4단계를 강세/약세 두 덩어리로 묶는다. 과매수는 강세의 연장이고 과매도는 약세의 연장이라,
# 100 을 넘나들지 않았는데 120 을 스쳤다고 '1일차' 로 되돌리면 국면이 끊긴 것처럼 보인다.
_ADR_GROUP = {"overbought": "bullish", "bullish": "bullish", "bearish": "bearish", "oversold": "bearish"}


def adr_group(level: str | None) -> str | None:
    """세부 단계가 속한 큰 국면(강세/약세)."""
    return _ADR_GROUP.get(level or "")


def _count_level_streak(points: list[dict[str, Any]]) -> tuple[str | None, int]:
    """마지막 세부 단계와, 그 단계가 속한 **국면**이 이어진 거래일 수.

    연속일은 강세(강세·과매수)/약세(약세·과매도) 덩어리로 센다. 표시용 단계는 마지막
    값 그대로 돌려줘서 화면이 '강세 3일째 (과매수)' 처럼 괄호로 덧붙일 수 있게 한다.
    값이 끊기면 거기서 멈춘다.
    """
    level: str | None = None
    group: str | None = None
    days = 0
    for point in reversed(points):
        current = classify_adr(point["adr"])
        if current is None:
            break
        if level is None:
            level, group = current, adr_group(current)
            days = 1
            continue
        if adr_group(current) != group:
            break
        days += 1
    return level, days


def load_adr_for_index(yf_ticker: str, limit_days: int | None = None) -> dict[str, Any] | None:
    """지수 티커에 대응하는 ADR. 집계 대상이 아닌 지수(다우·필라델피아 등)는 None."""
    market = MARKET_BY_INDEX_TICKER.get(yf_ticker)
    if not market:
        return None

    points = load_adr_series(market, limit_days)
    if not points:
        return None

    # 대상 종목 수는 적립 당시 값을 그대로 쓴다 — 구성종목이 바뀌면 수도 달라진다.
    latest_doc = (
        _require_db()[COLLECTION_NAME].find_one({"market": market}, {"_id": 0, "universe_size": 1}, sort=[("date", -1)])
        or {}
    )

    latest = next((item for item in reversed(points) if item["adr"] is not None), None)
    level, level_days = _count_level_streak(points)
    return {
        "market": market,
        "market_name": MARKETS[market]["name"],
        "universe_size": int(latest_doc.get("universe_size") or 0),
        "window_days": MARKET_ADR_WINDOW_DAYS,
        "overheated": MARKET_ADR_OVERHEATED,
        # 강세/약세를 가르는 중간선 — 상승 종목수와 하락 종목수가 같아지는 지점이다.
        "neutral": MARKET_ADR_NEUTRAL,
        "oversold": MARKET_ADR_OVERSOLD,
        "latest_adr": latest["adr"] if latest else None,
        # 지금 단계와 그 단계가 이어진 거래일 수 — 그리드의 'ADR' 컬럼이 쓴다.
        "level": level,
        "level_days": level_days,
        "points": points,
    }


__all__ = [
    "COLLECTION_NAME",
    "classify_adr",
    "MARKETS",
    "MARKET_BY_INDEX_TICKER",
    "load_adr_for_index",
    "load_adr_series",
    "refresh_market_breadth",
    "resolve_market_universe",
]
