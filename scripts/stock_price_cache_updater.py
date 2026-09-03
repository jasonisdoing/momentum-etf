#!/usr/bin/env python
"""계좌별 OHLCV 캐시 갱신.

실행 모드
--------
- 기본(증분): 캐시에 없는 구간만 조회한다. 한국 풀은 **최신 거래일 하루만 비는
  종목을 네이버 일봉 스냅샷(50종목 배치 폴링)으로 일괄 적재**해 pykrx 종목별
  호출(+1초 스로틀)을 없앤다. 매시 크론이 이 모드로 돈다.
- ``--full``: 전체 히스토리 강제 재수집 — 수정주가(배당·분할) 재정렬용.
  하루 1회 크론(17:10)이 담당하며, 네이버 스냅샷으로 들어온 당일 행도
  이때 KRX 공식 값으로 덮어써진다.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager, nullcontext

import pandas as pd

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.component_price_service import build_component_price_snapshot, select_component_holdings_for_pricing
from services.portfolio_change_service import compute_and_store_portfolio_change_bundle
from services.stock_cache_service import get_stock_cache_meta
from utils.cache_utils import (
    get_cached_date_range,
    load_cached_frame_with_fallback,
    prune_cache_to_tickers,
    save_cached_frame,
    set_cache_refresh_completed_at,
)
from utils.data_loader import PykrxDataUnavailableError, fetch_ohlcv, repair_recent_trading_day_gaps
from utils.env import load_env_if_present
from utils.logger import get_app_logger
from utils.settings_loader import get_ticker_type_settings, list_available_ticker_types, load_common_settings
from utils.stock_list_io import get_all_etfs_including_deleted, get_etfs

FETCH_RETRY_ATTEMPTS = 3
FETCH_RETRY_DELAY_SECONDS = 2.0
PER_TICKER_TIMEOUT_SECONDS = 90
# KOR_FETCH_TARGET_SECONDS = 0.5 서버에서 freeze 1초로 늘려서 테스트중
KOR_FETCH_TARGET_SECONDS = 1


def _backfill_missing_closes_from_naver(ticker_type: str, tickers: list[str]) -> dict:
    """yfinance 이력의 구멍을 네이버 해외증시로 메운다.

    반환(슬랙 점검 보고용): ``{"filled_tickers": 메운 종목 수,
    "filled_days": {날짜: 종목 수}, "unfilled": {티커: [네이버에도 없어 못 채운 날짜]}}``.

    구멍은 두 형태다 — 둘 다 실제로 관측됐다:
      ① 봉은 있는데 **Close 만 NaN** (2026-08-28: 시가·고가·저가·거래량은 있고 종가만 없음).
      ② 최근 거래일 봉이 이력에서 **통째로 빠짐** (2026-09-01: 재수집이 8/28 행이 아예 없는
        이력을 받아 와, 전날 네이버로 채워둔 금요일 봉까지 사라졌다 — 판정일 데이터가 하루
        뒤로 밀려 모멘텀 선정이 밤 사이 바뀌었다).

    같은 봉을 네이버는 온전히 들고 있고 시가·고가·저가가 yfinance 와 소수점까지 같다.
    ①은 빈 값만 채우고, ②는 그 날짜 행을 새로 만들어 시가·고가·저가·종가를 채운다.
    이미 있는 값은 건드리지 않는다(주 소스는 어디까지나 yfinance).
    거래량은 네이버가 주지 않으므로 빈 값으로 둔다 — 지어내지 않는다.

    ②의 탐지: 최근 30일의 미국 거래일 중 **장 마감이 끝난 날**인데 캐시에 없는 날 —
    캐시 마지막 봉의 앞이든 뒤든 가리지 않는다(야후가 어느 날부터 통째로 안 주는 경우도
    마감된 날이면 채운다). "아직 안 온 오늘 데이터"와의 구분은 마감 완료 여부가 한다.
    네이버로 채운 날의 거래량은 다음 전체 재수집(--full)이 야후 값으로 되채운다.
    """
    from utils.cache_utils import load_cached_frame, save_cached_frame
    from utils.naver_overseas import fetch_daily_ohlc
    from utils.trading_calendar import get_trading_days, is_market_day_completed

    filled = 0
    filled_days: dict[str, int] = {}
    unfilled: dict[str, list[str]] = {}
    calendar_days: list[pd.Timestamp] | None = None  # 종목 공통 — 한 번만 받는다
    for ticker in tickers:
        cached = load_cached_frame(ticker_type, ticker)
        if cached is None or cached.empty or "Close" not in cached.columns:
            continue
        gaps = list(cached.index[cached["Close"].isna()])

        # ② 통째로 빠진 거래일 — 캘린더에는 있고 마감도 끝났는데 캐시 인덱스에 없는 날.
        if calendar_days is None:
            try:
                calendar_days = [
                    day
                    for day in (
                        pd.Timestamp(raw).normalize()
                        for raw in get_trading_days(
                            (pd.Timestamp.now() - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
                            pd.Timestamp.now().strftime("%Y-%m-%d"),
                            "us",
                        )
                    )
                    if is_market_day_completed("us", day)
                ]
            except Exception:
                calendar_days = []
        # 그 종목의 첫 봉 이전은 상장 전이라 데이터가 없는 게 정상 — 탐지 대상이 아니다
        # (신규 상장 ETF 가 "네이버에도 없어 못 채움" 오탐으로 찍혔다).
        absent = [day for day in calendar_days if day >= cached.index[0] and day not in cached.index]
        if not gaps and not absent:
            continue

        # 네이버는 최근분부터 준다 — 가장 오래된 결측까지 덮을 만큼만 요청한다.
        oldest = min([*gaps, *absent])
        span_days = int((pd.Timestamp.now().normalize() - oldest).days) + 5
        naver = fetch_daily_ohlc(ticker, days=max(10, min(span_days, 200)))
        if naver is None or naver.empty:
            continue

        merged = cached.copy()
        touched = 0
        for day in gaps:
            if day not in naver.index:
                unfilled.setdefault(ticker, []).append(str(day.date()))
                continue
            for column in ("Open", "High", "Low", "Close"):
                if column in merged.columns and pd.isna(merged.at[day, column]):
                    merged.at[day, column] = float(naver.at[day, column])
            touched += 1
            filled_days[str(day.date())] = filled_days.get(str(day.date()), 0) + 1
        for day in absent:
            if day not in naver.index:
                unfilled.setdefault(ticker, []).append(str(day.date()))
                continue
            for column in ("Open", "High", "Low", "Close"):
                if column in merged.columns:
                    merged.at[day, column] = float(naver.at[day, column])
            touched += 1
            filled_days[str(day.date())] = filled_days.get(str(day.date()), 0) + 1
        if touched:
            merged = merged.sort_index()
            save_cached_frame(ticker_type, ticker, merged)
            filled += 1
    return {"filled_tickers": filled, "filled_days": filled_days, "unfilled": unfilled}


def _backfill_missing_volumes_from_toss(
    ticker_type: str,
    tickers: list[str],
    *,
    lookback_days: int = 45,
) -> dict:
    """최근 미국 일봉의 빈 거래량을 토스 일봉으로 채운다.

    네이버가 복구한 OHLC 행은 종가가 이미 있어 다음 배치에서 가격 누락으로 다시 잡히지
    않는다. 그래서 완료된 미국 거래일 중 ``Close``는 있고 ``Volume``만 비어 있는 행을
    별도로 찾는다. 기존 OHLC는 건드리지 않고 거래량만 채운다.
    """
    from services.toss_market_service import fetch_toss_us_daily_ohlcv
    from utils.cache_utils import load_cached_frame, save_cached_frame
    from utils.realtime_quotes import resolve_toss_us_product_codes
    from utils.trading_calendar import get_trading_days, is_market_day_completed

    logger = get_app_logger()
    now = pd.Timestamp.now().normalize()
    start = now - pd.Timedelta(days=max(1, int(lookback_days)))
    completed_days = {
        pd.Timestamp(day).normalize()
        for day in get_trading_days(start.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d"), "us")
        if is_market_day_completed("us", pd.Timestamp(day).normalize())
    }
    if not completed_days:
        return {"filled_tickers": 0, "filled_days": {}, "unfilled": {}}

    cached_by_ticker: dict[str, pd.DataFrame] = {}
    missing_by_ticker: dict[str, list[pd.Timestamp]] = {}
    for ticker in tickers:
        cached = load_cached_frame(ticker_type, ticker)
        if cached is None or cached.empty or "Close" not in cached.columns:
            continue
        volume = (
            pd.to_numeric(cached["Volume"], errors="coerce")
            if "Volume" in cached.columns
            else pd.Series(float("nan"), index=cached.index, dtype="float64")
        )
        close = pd.to_numeric(cached["Close"], errors="coerce")
        normalized_index = pd.to_datetime(cached.index).normalize()
        gaps = sorted(
            {
                pd.Timestamp(day).normalize()
                for day, close_value, volume_value in zip(normalized_index, close, volume, strict=True)
                if day in completed_days and pd.notna(close_value) and pd.isna(volume_value)
            }
        )
        if gaps:
            cached_by_ticker[ticker] = cached.copy()
            missing_by_ticker[ticker] = gaps

    if not missing_by_ticker:
        return {"filled_tickers": 0, "filled_days": {}, "unfilled": {}}

    product_codes = resolve_toss_us_product_codes(list(missing_by_ticker))
    filled_tickers = 0
    filled_days: dict[str, int] = {}
    unfilled: dict[str, list[str]] = {}
    for ticker, gaps in missing_by_ticker.items():
        product_code = product_codes.get(ticker)
        if not product_code:
            unfilled[ticker] = [str(day.date()) for day in gaps]
            continue
        oldest = min(gaps)
        count = max(30, min(200, int((now - oldest).days * 2 + 10)))
        try:
            candles = fetch_toss_us_daily_ohlcv(product_code, count=count)
        except Exception:
            logger.exception("[%s] %s 토스 거래량 조회 실패", ticker_type.upper(), ticker)
            unfilled[ticker] = [str(day.date()) for day in gaps]
            continue
        by_date = {pd.Timestamp(str(candle["date"])).normalize(): candle for candle in candles}
        cached = cached_by_ticker[ticker]
        if "Volume" not in cached.columns:
            cached["Volume"] = pd.Series(float("nan"), index=cached.index, dtype="float64")
        normalized_index = pd.to_datetime(cached.index).normalize()
        touched = 0
        for day in gaps:
            candle = by_date.get(day)
            raw_volume = candle.get("volume") if candle else None
            if raw_volume is None or float(raw_volume) <= 0:
                unfilled.setdefault(ticker, []).append(str(day.date()))
                continue
            matching_rows = cached.index[normalized_index == day]
            if len(matching_rows) == 0:
                unfilled.setdefault(ticker, []).append(str(day.date()))
                continue
            cached.at[matching_rows[-1], "Volume"] = float(raw_volume)
            touched += 1
            day_key = str(day.date())
            filled_days[day_key] = filled_days.get(day_key, 0) + 1
        if touched:
            save_cached_frame(ticker_type, ticker, cached.sort_index())
            filled_tickers += 1

    return {"filled_tickers": filled_tickers, "filled_days": filled_days, "unfilled": unfilled}


def _resolve_fetch_workers() -> int:
    """종목 fetch 병렬 워커 수.

    ⚠️ 현재 pykrx/yfinance 가 thread-safe 가 아닐 가능성이 있어 병렬 동작 시 deadlock 발생.
    안전하게 직렬(1) 로 고정한다. 추후 ProcessPoolExecutor 또는 asyncio 기반 재설계 필요.
    """
    return 1


# 풀 전체 NaN 비율이 이 임계값을 초과하는 날짜는 데이터 소스 오류로 간주하고
# 모든 종목 캐시에서 그 날짜 행을 제거한다. 다음 cron 시 자동 재fetch.
SUSPICIOUS_NAN_RATIO_THRESHOLD = 0.5
SUSPICIOUS_LOOKBACK_DAYS = 400


def _determine_start_date() -> str:
    settings = load_common_settings() or {}
    start = settings.get("CACHE_START_DATE")
    if not start:
        raise RuntimeError("CACHE_START_DATE 가 설정되지 않았습니다. config.py 또는 공용 설정을 확인하세요.")
    return str(start)


@contextmanager
def _ticker_refresh_timeout(seconds: int):
    """티커 단위 갱신이 장시간 멈추지 않도록 제한한다."""
    timeout_seconds = int(seconds or 0)
    if timeout_seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _raise_timeout(signum, frame):
        raise TimeoutError(f"티커 처리 제한 시간 {timeout_seconds}초를 초과했습니다.")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(timeout_seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


@contextmanager
def _global_refresh_lock():
    """전체 가격 캐시 갱신이 동시에 실행되지 않도록 파일 잠금을 건다."""
    lock_path = os.path.join("/tmp", "momentum_etf_cache_refresh.lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise RuntimeError("가격 캐시 갱신이 이미 실행 중입니다. 중복 실행을 중단합니다.") from exc

    try:
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode("ascii"))
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _purge_suspicious_dates(
    target_id: str,
    tickers: list[str],
    *,
    lookback_days: int = SUSPICIOUS_LOOKBACK_DAYS,
    nan_threshold: float = SUSPICIOUS_NAN_RATIO_THRESHOLD,
) -> list[pd.Timestamp]:
    """풀 전체 close 의 NaN 비율이 임계값을 초과하는 날짜를 캐시에서 제거한다.

    데이터 소스(yfinance 등)가 특정 날짜에 다수 종목의 데이터를 일시적으로 빠뜨리거나
    합성값을 반환할 때, 그 날짜 행을 모든 종목 캐시에서 제거해 다음 cron 시 재fetch 대상으로 만든다.

    비율의 분모는 **그 날짜에 이미 상장돼 있던 종목**(창 안 첫 봉 이후)만이다. 풀 전체로
    잡으면 신규 상장이 많은 풀에서 상장 전 부재가 NaN 으로 집계돼, 멀쩡히 거래되던 옛
    종목의 진짜 데이터까지 지웠다(kor_us_div_etf 2025-07-28~09-22, 40거래일 실손실).
    """
    logger = get_app_logger()
    if not tickers:
        return []

    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=lookback_days)

    # 1) 모든 티커의 close 시리즈 수집 → 와이드 매트릭스
    close_map: dict[str, pd.Series] = {}
    for ticker in tickers:
        try:
            df = load_cached_frame_with_fallback(target_id, ticker)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        df = df[df.index >= cutoff]
        if df.empty:
            continue
        close_col = next(
            (c for c in ("unadjusted_close", "Close", "close") if c in df.columns),
            None,
        )
        if close_col is None:
            continue
        s = pd.to_numeric(df[close_col], errors="coerce")
        s.index = pd.to_datetime(s.index).normalize()
        s = s[~s.index.duplicated(keep="last")]
        close_map[ticker] = s

    if not close_map:
        return []

    matrix = pd.DataFrame(close_map)
    if matrix.empty:
        return []

    # 각 종목은 자기 첫 봉(창 안 기준)부터만 분모에 들어간다 — 상장 전 부재는 NaN 이 아니다.
    listed = pd.DataFrame(
        {ticker: matrix.index >= series.index.min() for ticker, series in close_map.items()},
        index=matrix.index,
    )
    denominator = listed.sum(axis=1)
    nan_ratio = (matrix.isna() & listed).sum(axis=1) / denominator.where(denominator > 0)
    suspicious = sorted(nan_ratio[nan_ratio > nan_threshold].index.tolist())
    if not suspicious:
        return []

    suspicious_text = ", ".join(pd.Timestamp(d).strftime("%Y-%m-%d") for d in suspicious)
    logger.warning(
        "[%s] 의심 날짜 감지 (NaN 비율 > %.0f%%, 종목 %d개 기준): %s — 캐시에서 제거합니다.",
        target_id.upper(),
        nan_threshold * 100,
        matrix.shape[1],
        suspicious_text,
    )

    # 2) 각 티커 캐시에서 의심 날짜 행 삭제 후 저장
    suspicious_set = {pd.Timestamp(d).normalize() for d in suspicious}
    purged_tickers = 0
    for ticker in tickers:
        try:
            df = load_cached_frame_with_fallback(target_id, ticker)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        before = len(df)
        normalized_index = pd.to_datetime(df.index).normalize()
        keep_mask = ~normalized_index.isin(list(suspicious_set))
        df_purged = df[keep_mask]
        if len(df_purged) >= before:
            continue
        if df_purged.empty:
            logger.warning("[%s] %s 의심 날짜 제거 후 데이터가 비어 저장 생략", target_id.upper(), ticker)
            continue
        try:
            save_cached_frame(target_id, ticker, df_purged)
            purged_tickers += 1
        except Exception as exc:
            logger.warning("[%s] %s 의심 날짜 제거 후 캐시 저장 실패: %s", target_id.upper(), ticker, exc)

    logger.info(
        "[%s] 의심 날짜 정리 완료: %d개 날짜 × 영향 종목 %d개",
        target_id.upper(),
        len(suspicious),
        purged_tickers,
    )
    return suspicious


def _trade_value_fields(df: pd.DataFrame) -> dict[str, float] | None:
    """거래대금 배수와, 장중 배수를 다시 계산할 때 쓸 재료.

    거래대금은 `종가 × 거래량` 이고 분모는 **당일을 포함한** 20일 평균이다
    (`utils.trade_value` 가 단일 소스 — 두 전략 화면이 다른 숫자를 보이면 안 된다).

    `trade_value_sum19` 는 직전 19거래일 합이다. 장중에는 오늘 거래대금을 실시간으로
    받아 `(sum19 + 오늘) / 20` 을 분모로 쓰는데, 순위 화면은 거래량 이력이 없어서
    그 합을 여기서 미리 넘겨준다. 20일이 안 되면 None(미산출 — 값을 지어내지 않는다).
    """
    if "Volume" not in df:
        return None
    close_col = next((column for column in ("unadjusted_close", "Close", "close") if column in df.columns), None)
    if close_col is None:
        return None
    from utils.trade_value import latest_trade_value_fields

    return latest_trade_value_fields(df[close_col], df["Volume"])


def _update_daily_change_pct(target_id: str, tickers: list[str]) -> None:
    """캐시된 종가 시계열로 일간 등락률과 거래대금 배수를 계산해 stock_meta 에 저장한다.

    /system 종목풀 표의 상승수(일간)/상승비율(일간) 과 /pools-rank 의 거래대금 컬럼이
    이 필드를 읽는다. 가격 캐시가 이미 메모리에 적재된 직후라 추가 외부 호출 없이
    공짜로 계산된다 — 순위 화면이 직접 계산하면 거래량 blob 을 받느라 3초가 더 든다.
    """
    logger = get_app_logger()
    if not tickers:
        return
    try:
        from pymongo import UpdateOne

        from utils.db_manager import get_db_connection

        db = get_db_connection()
        if db is None:
            logger.warning("[%s] DB 연결 실패로 일간 등락률 저장 생략", target_id.upper())
            return

        ops: list[UpdateOne] = []
        for ticker in tickers:
            try:
                df = load_cached_frame_with_fallback(target_id, ticker)
            except Exception:
                continue
            if df is None or df.empty:
                continue
            close_col = next(
                (c for c in ("unadjusted_close", "Close", "close") if c in df.columns),
                None,
            )
            if close_col is None:
                continue
            close_series = pd.to_numeric(df[close_col], errors="coerce").dropna()
            close_series = close_series[close_series > 0]
            if len(close_series) < 2:
                continue
            latest = float(close_series.iloc[-1])
            prev = float(close_series.iloc[-2])
            if prev <= 0:
                continue
            change_pct = (latest / prev - 1.0) * 100.0
            fields = {
                "1_day_change_pct": round(change_pct, 4),
                "1_day_change_date": pd.Timestamp(close_series.index[-1]).strftime("%Y-%m-%d"),
            }
            trade_fields = _trade_value_fields(df)
            if trade_fields:
                fields.update(trade_fields)
            ops.append(UpdateOne({"ticker_type": target_id, "ticker": ticker}, {"$set": fields}))
        if ops:
            result = db.stock_meta.bulk_write(ops, ordered=False)
            logger.info(
                "[%s] 일간 등락률·거래대금 배수 저장 완료: %d개 종목 (matched %d)",
                target_id.upper(),
                len(ops),
                result.matched_count,
            )
    except Exception as exc:
        logger.warning("[%s] 일간 등락률 저장 실패: %s", target_id.upper(), exc)


def _update_pool_rank_summary(target_id: str) -> None:
    """종목풀 순위를 계산해 추세(%) 양수 개수를 pool_rank_summary 컬렉션에 저장한다.

    /system 종목풀 표의 매수 후보 수/비율이 이 문서를 읽는다.
    추세 계산은 방금 갱신된 가격 캐시를 사용하므로 외부 호출이 없다.
    """
    logger = get_app_logger()
    try:
        from utils.db_manager import get_db_connection
        from utils.rank_service import load_rank_data

        payload = load_rank_data(ticker_type=target_id)
        rows = payload.get("rows") or []
        total_count = len(rows)
        up_count = 0
        for row in rows:
            try:
                trend = row.get("추세")
                is_below = bool(row.get("is_below_benchmark"))
                is_bm = bool(row.get("is_benchmark"))
                is_excl = bool(row.get("exclude_from_ranking"))
                if trend is not None and float(trend) > 0 and not is_below and not is_bm and not is_excl:
                    up_count += 1
            except (TypeError, ValueError):
                continue

        db = get_db_connection()
        if db is None:
            logger.warning("[%s] 종목풀 요약 저장 생략: DB 연결 실패", target_id.upper())
            return
        db.pool_rank_summary.update_one(
            {"_id": target_id},
            {
                "$set": {
                    "ticker_type": target_id,
                    "score_up_count": up_count,
                    "score_total_count": total_count,
                    "updated_at": pd.Timestamp.utcnow().to_pydatetime(),
                }
            },
            upsert=True,
        )
        logger.info(
            "[%s] 매수 후보 요약 저장 완료: 후보 %d / 전체 %d",
            target_id.upper(),
            up_count,
            total_count,
        )
    except Exception as exc:
        logger.warning("[%s] 종목풀 요약 저장 실패: %s", target_id.upper(), exc)


def _refresh_portfolio_change_cache_for_target(
    target_id: str,
    target_items: list[dict],
    success_tickers: set[str],
) -> None:
    """가격 캐시 갱신 후 ETF 포트폴리오 변동 캐시를 미리 계산한다."""
    logger = get_app_logger()
    target_norm = (target_id or "").strip().lower()
    if not target_norm or not target_items:
        return

    candidates: list[tuple[str, list[dict]]] = []
    snapshot_holdings: list[dict] = []
    for item in target_items:
        ticker = str(item.get("ticker") or "").strip().upper()
        if not ticker or ticker not in success_tickers:
            continue
        try:
            cache_doc = get_stock_cache_meta(target_norm, ticker)
        except Exception as exc:
            logger.warning("[%s] %s 포트폴리오 변동 대상 확인 실패: %s", target_norm.upper(), ticker, exc)
            continue
        holdings = ((cache_doc or {}).get("holdings_cache") or {}).get("items") if isinstance(cache_doc, dict) else None
        if holdings:
            holdings_list = list(holdings)
            candidates.append((ticker, holdings_list))
            snapshot_holdings.extend(select_component_holdings_for_pricing(holdings_list, 100))

    if not candidates:
        logger.info("[%s] 포트폴리오 변동 캐시 갱신 대상이 없습니다.", target_norm.upper())
        return

    succeeded = 0
    failed: list[str] = []
    logger.info(
        "[%s] 포트폴리오 변동 공통 구성종목 가격 스냅샷 생성 시작: %d개 후보",
        target_norm.upper(),
        len(snapshot_holdings),
    )
    component_price_snapshot = build_component_price_snapshot(snapshot_holdings)
    logger.info(
        "[%s] 포트폴리오 변동 공통 구성종목 가격 스냅샷 생성 완료: %d개",
        target_norm.upper(),
        len(component_price_snapshot),
    )

    max_workers = _resolve_fetch_workers()
    logger.info(
        "[%s] 포트폴리오 변동 캐시 갱신 시작: %d개 (병렬 워커 %d)",
        target_norm.upper(),
        len(candidates),
        max_workers,
    )

    def _process_one_bundle(idx: int, ticker: str) -> tuple[bool, str]:
        started_at = time.perf_counter()
        try:
            result = compute_and_store_portfolio_change_bundle(
                ticker,
                target_norm,
                component_price_snapshot=component_price_snapshot,
            )
            elapsed = time.perf_counter() - started_at
            if result:
                logger.info(
                    " -> 포트폴리오 변동 캐시 갱신 완료: %d/%d - %s | 소요 %.1fs",
                    idx,
                    len(candidates),
                    ticker,
                    elapsed,
                )
                return True, ticker
            logger.warning(
                " -> 포트폴리오 변동 캐시 계산 불가: %d/%d - %s | 소요 %.1fs",
                idx,
                len(candidates),
                ticker,
                elapsed,
            )
            return False, ticker
        except Exception as exc:
            elapsed = time.perf_counter() - started_at
            logger.warning(
                " -> 포트폴리오 변동 캐시 갱신 실패: %d/%d - %s: %s | 소요 %.1fs",
                idx,
                len(candidates),
                ticker,
                exc,
                elapsed,
            )
            return False, ticker

    if max_workers <= 1:
        for index, (ticker, _) in enumerate(candidates, 1):
            ok, t = _process_one_bundle(index, ticker)
            if ok:
                succeeded += 1
            else:
                failed.append(t)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_one_bundle, i + 1, t) for i, (t, _) in enumerate(candidates)]
            for future in as_completed(futures):
                try:
                    ok, t = future.result()
                except Exception as exc:
                    logger.warning("포트폴리오 변동 갱신 task 예외: %s", exc)
                    continue
                if ok:
                    succeeded += 1
                else:
                    failed.append(t)

    if failed:
        preview = ", ".join(failed[:10])
        suffix_text = " ..." if len(failed) > 10 else ""
        logger.warning(
            "[%s] 포트폴리오 변동 캐시 일부 실패: %s%s (총 %d개 실패 / %d개 성공)",
            target_norm.upper(),
            preview,
            suffix_text,
            len(failed),
            succeeded,
        )
    else:
        logger.info("[%s] 포트폴리오 변동 캐시 갱신 완료 (%d개).", target_norm.upper(), succeeded)


def refresh_cache_for_target(
    target_id: str,
    start_date: str | None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    *,
    full_refresh: bool = False,
):
    """지정된 계정(target_id)에 대한 가격 데이터 캐시를 새로 고칩니다.

    ``full_refresh=True`` 는 기존 동작(전체 히스토리 강제 재수집 — 수정주가 재정렬)이고,
    기본(False)은 **증분**이다: 캐시에 없는 구간만 조회하며, 한국 풀은 최신 거래일
    하루만 비는 종목을 네이버 일봉 스냅샷으로 일괄 적재해 pykrx 종목별 호출을 없앤다.
    """
    logger = get_app_logger()
    target_norm = (target_id or "").strip().lower()

    try:
        available_types = list_available_ticker_types()
        if target_norm in available_types:
            settings = get_ticker_type_settings(target_norm)
            country_code = settings.get("country_code", "kor").lower()
        else:
            country_code = "kor"
    except Exception:
        logger.warning(f"대상 종목풀 설정을 불러올 수 없어 기본 국가코드(kor)를 사용합니다: {target_norm}")
        country_code = "kor"

    logger.info("[%s] 캐시 갱신 시작 (국가설정: %s, 시작일: %s)", target_norm.upper(), country_code, start_date)

    def _is_today_unavailable_warning(exc: PykrxDataUnavailableError) -> bool:
        """한국장 당일 데이터가 아직 집계되지 않은 정상 상황만 경고로 낮춘다."""
        if str(exc.country or "").strip().lower() != "kor":
            return False
        today = pd.Timestamp.now().normalize()
        return exc.start_dt.normalize() == today and exc.end_dt.normalize() == today

    def _refresh_single_ticker_with_retry(
        *,
        ticker: str,
        country_code: str,
        range_start: str,
        account_id: str,
    ) -> list[pd.Timestamp]:
        """일시적인 원천 응답 공백을 고려해 티커 단위로 재시도한다."""
        last_error: Exception | None = None

        for attempt in range(1, FETCH_RETRY_ATTEMPTS + 1):
            try:
                fetched_df = fetch_ohlcv(
                    ticker,
                    country=country_code,
                    months_back=None,
                    date_range=[range_start, None],
                    update_listing_meta=False,
                    # 증분 모드는 캐시에 없는 구간만 조회한다. 전체 재수집(수정주가
                    # 재정렬)은 하루 1회 --full 실행이 담당한다.
                    force_refresh=full_refresh,
                    ticker_type=account_id,
                )
                if fetched_df is None or fetched_df.empty:
                    raise RuntimeError(f"{ticker} 원천 가격 데이터가 비어 있습니다.")

                unresolved_days = repair_recent_trading_day_gaps(
                    ticker,
                    country_code,
                    ticker_type=account_id,
                    lookback_days=15,
                )

                cached_range = get_cached_date_range(account_id, ticker)
                if cached_range is None:
                    raise RuntimeError(f"{ticker} 캐시 저장 결과를 확인할 수 없습니다.")

                return unresolved_days
            except PykrxDataUnavailableError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt >= FETCH_RETRY_ATTEMPTS:
                    break
                logger.warning(
                    "%s 데이터 조회/저장 재시도 예정 (%d/%d): %s",
                    ticker,
                    attempt,
                    FETCH_RETRY_ATTEMPTS,
                    exc,
                )
                time.sleep(FETCH_RETRY_DELAY_SECONDS)

        if last_error is None:
            raise RuntimeError(f"{ticker} 데이터 갱신 실패 원인을 확인할 수 없습니다.")
        raise last_error

    # 전체 실행 잠금은 main()에서 한 번만 잡는다.
    with nullcontext():
        # 종목 리스트 로드
        try:
            all_etfs_from_file = get_all_etfs_including_deleted(target_norm)
        except Exception:
            all_etfs_from_file = []

        all_map = {
            str(item.get("ticker") or "").strip().upper(): item for item in all_etfs_from_file if item.get("ticker")
        }

        # 종목풀 실행 시 해당 종목풀의 모든 종목 반영
        if target_norm in list_available_ticker_types():
            pass  # get_all_etfs_including_deleted가 이미 수행함

        # 벤치마크 추가
        benchmark_tickers = _collect_benchmark_tickers(target_norm)
        for bench in benchmark_tickers:
            norm = str(bench or "").strip().upper()
            if not norm or norm in all_map:
                continue
            all_map[norm] = {
                "ticker": norm,
                "name": norm,
                "type": "etf",
            }

        if not all_map:
            logger.warning(
                "[%s] 갱신할 종목이 없습니다 (stock_meta/portfolio_master 모두 비어있음).", target_norm.upper()
            )
            return

        target_items = list(all_map.values())
        total_tickers = len(target_items)
        failed_tickers: list[str] = []
        succeeded_count = 0
        if not start_date:
            raise RuntimeError(
                "refresh_cache_for_target 에 start_date 가 전달되지 않았습니다. "
                "_determine_start_date() 결과를 명시적으로 넘기세요."
            )
        range_start = start_date

        max_workers = _resolve_fetch_workers()
        logger.info(
            "[%s] 가격 캐시 갱신 시작: %d개 종목, 병렬 워커 %d",
            target_norm.upper(),
            total_tickers,
            max_workers,
        )

        # US/AUS 풀: yfinance 일괄 prefetch 적용 (종목당 호출 → 풀당 1회 호출).
        # prefetch 실패 시 _fetch_ohlcv_core 가 종목별 호출로 자동 fallback 한다.
        if country_code in ("us", "au"):
            try:
                from utils.data_loader import (
                    prefetch_yfinance_bulk,
                    reset_yf_bulk_prefetch,
                )

                reset_yf_bulk_prefetch()
                start_ts = pd.to_datetime(range_start)
                end_ts = pd.Timestamp.now().normalize()
                pf_tickers = [
                    str(item.get("ticker") or "").strip().upper()
                    for item in target_items
                    if str(item.get("ticker") or "").strip()
                ]
                saved_count = prefetch_yfinance_bulk(pf_tickers, country_code, start_ts, end_ts)
                logger.info(
                    "[%s] yfinance 일괄 prefetch: %d/%d 종목 캐시 적재",
                    target_norm.upper(),
                    saved_count,
                    len(pf_tickers),
                )
            except Exception as exc:
                logger.warning(
                    "[%s] yfinance 일괄 prefetch 건너뜀(종목별 호출 fallback): %s",
                    target_norm.upper(),
                    exc,
                )

        # ── KOR 증분: 최신 거래일 하루만 비는 종목은 네이버 일봉 스냅샷으로 일괄 적재 ──
        # (종목당 pykrx 호출 + 1초 스로틀 → 50종목 배치 폴링 몇 번으로 대체)
        # 스냅샷으로 채워졌거나 이미 최신인 종목은 아래 루프에서 스로틀 없이 캐시 확인만 한다.
        kor_snapshot_done: set[str] = set()
        if country_code == "kor" and not full_refresh:
            try:
                from utils.cache_utils import load_cached_frame
                from utils.data_loader import get_latest_trading_day, get_trading_days
                from utils.realtime_quotes import fetch_naver_daily_ohlcv_snapshot

                latest_day = get_latest_trading_day("kor").normalize()
                recent_days = get_trading_days(
                    (latest_day - pd.Timedelta(days=15)).strftime("%Y-%m-%d"),
                    latest_day.strftime("%Y-%m-%d"),
                    "kor",
                )
                prev_day = recent_days[-2].normalize() if len(recent_days) >= 2 else None
                need_latest: list[str] = []
                for item in target_items:
                    t = str(item.get("ticker") or "").strip().upper()
                    if not t:
                        continue
                    cached_range = get_cached_date_range(target_norm, t)
                    if cached_range is None:
                        continue  # 캐시 없음(신규) → 종목별 전체 수집 경로
                    cache_end = pd.Timestamp(cached_range[1]).normalize()
                    if cache_end >= latest_day:
                        kor_snapshot_done.add(t)  # 이미 최신 — 네트워크·스로틀 불필요
                    elif prev_day is not None and cache_end == prev_day:
                        need_latest.append(t)  # 최신 거래일 하루만 부족 → 스냅샷 대상
                if need_latest:
                    snapshot = fetch_naver_daily_ohlcv_snapshot(need_latest, latest_day)
                    for t, row in snapshot.items():
                        cached_df = load_cached_frame(target_norm, t)
                        if cached_df is None or cached_df.empty:
                            continue
                        addition = pd.DataFrame([row], index=pd.DatetimeIndex([latest_day]))
                        merged = pd.concat(
                            [cached_df[cached_df.index.normalize() != latest_day], addition]
                        ).sort_index()
                        save_cached_frame(target_norm, t, merged)
                        kor_snapshot_done.add(t)
                    logger.info(
                        "[%s] 네이버 일봉 스냅샷 일괄 적재: %d/%d 종목 (기준일 %s, 미적재는 pykrx fallback)",
                        target_norm.upper(),
                        len(snapshot),
                        len(need_latest),
                        latest_day.date(),
                    )
            except Exception as exc:
                logger.warning(
                    "[%s] 네이버 일봉 스냅샷 건너뜀(종목별 pykrx fallback): %s",
                    target_norm.upper(),
                    exc,
                )

        def _process_one(idx: int, etf_item: dict) -> tuple[bool, str, str]:
            """단일 종목 처리. 반환: (성공여부, ticker, log_message).

            KOR 풀(pykrx 사용)은 KRX 가 단위 시간당 호출 빈도로 IP 차단을 거는 듯하다.
            서버처럼 응답이 너무 빠른 환경(종목당 ~0.1s)에서는 30종목 즈음 차단되어 hang
            상태로 빠진다. 종목당 목표 간격(KOR_FETCH_TARGET_SECONDS) 미만으로
            끝났을 때 부족분만큼 동적으로 sleep 한다. 로컬처럼 자연 소요가 충분히 느린
            환경(0.3s 이상)은 영향이 없다. 스냅샷/캐시로 이미 최신인 종목은 pykrx 를
            부르지 않으므로 스로틀도 건너뛴다.
            """
            t = str(etf_item.get("ticker") or "").strip().upper()
            n = etf_item.get("name") or "-"
            started = time.perf_counter()
            try:
                unresolved_days = _refresh_single_ticker_with_retry(
                    ticker=t,
                    country_code=country_code,
                    range_start=range_start,
                    account_id=target_norm,
                )
                elapsed = time.perf_counter() - started

                # KOR 풀에만 동적 sleep — 부족분만큼 채워 호출 빈도를 늦춘다.
                # (스냅샷으로 이미 채워진 종목은 pykrx 호출이 없어 스로틀 제외)
                sleep_secs = 0.0
                if country_code == "kor" and (full_refresh or t not in kor_snapshot_done):
                    sleep_secs = max(0.0, KOR_FETCH_TARGET_SECONDS - elapsed)
                    if sleep_secs > 0:
                        time.sleep(sleep_secs)
                sleep_suffix = f" + {sleep_secs:.1f}s 대기(속도조절)" if sleep_secs > 0 else ""

                if unresolved_days:
                    unresolved_text = ", ".join(day.strftime("%Y-%m-%d") for day in unresolved_days)
                    msg = (
                        f" -> 가격 캐시 갱신 완료: {idx}/{total_tickers} - {n}({t})"
                        f" - 최근 거래일 누락 유지: {unresolved_text}"
                        f" | 소요 {elapsed:.1f}s{sleep_suffix}"
                    )
                    logger.warning(msg)
                else:
                    msg = (
                        f" -> 가격 캐시 갱신 완료: {idx}/{total_tickers} - {n}({t}) | 소요 {elapsed:.1f}s{sleep_suffix}"
                    )
                    logger.info(msg)
                return True, t, msg
            except PykrxDataUnavailableError as e:
                elapsed = time.perf_counter() - started
                if _is_today_unavailable_warning(e):
                    logger.warning("%s 당일 데이터 미집계: %s | 소요 %.1fs", t, e, elapsed)
                else:
                    logger.error("%s 데이터 처리 중 오류 발생: %s | 소요 %.1fs", t, e, elapsed)
                return False, t, ""
            except Exception as e:
                elapsed = time.perf_counter() - started
                logger.error("%s 데이터 처리 중 오류 발생: %s | 소요 %.1fs", t, e, elapsed)
                return False, t, ""

        if max_workers <= 1:
            # 직렬 모드 (기존 호환)
            for i, etf in enumerate(target_items, 1):
                if progress_callback:
                    name = etf.get("name") or "-"
                    ticker = str(etf.get("ticker") or "").strip().upper()
                    progress_callback(i, total_tickers, f"{name}({ticker})")
                ok, t, _ = _process_one(i, etf)
                if ok:
                    succeeded_count += 1
                else:
                    failed_tickers.append(t)
        else:
            # 병렬 모드
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from concurrent.futures import TimeoutError as FuturesTimeoutError

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(_process_one, i + 1, etf): (i + 1, etf) for i, etf in enumerate(target_items)
                }
                completed = 0
                for future in as_completed(future_to_idx):
                    idx, etf = future_to_idx[future]
                    try:
                        ok, t, _ = future.result(timeout=PER_TICKER_TIMEOUT_SECONDS)
                    except FuturesTimeoutError:
                        t = str(etf.get("ticker") or "").strip().upper()
                        logger.error("%s 종목 처리 타임아웃 (%ds 초과)", t, PER_TICKER_TIMEOUT_SECONDS)
                        ok = False
                    except Exception as e:
                        t = str(etf.get("ticker") or "").strip().upper()
                        logger.error("%s 처리 중 예외: %s", t, e)
                        ok = False
                    if ok:
                        succeeded_count += 1
                    else:
                        failed_tickers.append(t)
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total_tickers, t)

        if failed_tickers:
            preview = ", ".join(failed_tickers[:10])
            suffix_text = " ..." if len(failed_tickers) > 10 else ""
            logger.warning(
                "[%s] 일부 종목 캐시 갱신 실패: %s%s (총 %d개 실패 / %d개 성공)",
                target_norm.upper(),
                preview,
                suffix_text,
                len(failed_tickers),
                succeeded_count,
            )
        else:
            logger.info("-> [%s] 캐시 갱신 완료 (%d개 종목).", target_norm.upper(), succeeded_count)

        # 미국 풀: yfinance 이력의 구멍(종가 빈칸·통누락)을 네이버로 메운다.
        backfill_report: dict | None = None
        volume_backfill_report: dict | None = None
        if country_code == "us":
            pool_tickers = [
                str(item.get("ticker") or "").strip().upper()
                for item in target_items
                if str(item.get("ticker") or "").strip()
            ]
            try:
                backfill_report = _backfill_missing_closes_from_naver(target_norm, pool_tickers)
                if backfill_report["filled_tickers"]:
                    logger.info(
                        "[%s] 네이버 해외증시로 야후 누락 보강: %d종목 %s",
                        target_norm.upper(),
                        backfill_report["filled_tickers"],
                        backfill_report["filled_days"],
                    )
            except Exception as exc:
                logger.warning("[%s] 네이버 종가 보강 건너뜀: %s", target_norm.upper(), exc)
            try:
                volume_backfill_report = _backfill_missing_volumes_from_toss(target_norm, pool_tickers)
                if volume_backfill_report["filled_tickers"]:
                    logger.info(
                        "[%s] 토스 일봉으로 거래량 누락 보강: %d종목 %s",
                        target_norm.upper(),
                        volume_backfill_report["filled_tickers"],
                        volume_backfill_report["filled_days"],
                    )
            except Exception as exc:
                logger.warning("[%s] 토스 거래량 보강 건너뜀: %s", target_norm.upper(), exc)

        success_tickers = [
            str(etf.get("ticker") or "").strip().upper()
            for etf in target_items
            if str(etf.get("ticker") or "").strip().upper() not in failed_tickers
        ]

        # 이 배치가 다루지 않는 캐시 문서 제거. 남겨두면 그 시점에 얼어붙은 채로 남아,
        # 나중에 컬렉션을 통째로 읽는 쪽이 최신 종목과 멈춘 종목을 한 표본으로 쓰게 된다.
        # `all_map` 은 풀 종목 + 삭제된 종목 + 벤치마크 + 보유 종목을 모두 담고 있다.
        try:
            orphans = prune_cache_to_tickers(target_norm, all_map.keys())
            if orphans:
                preview = ", ".join(orphans[:10])
                suffix = " ..." if len(orphans) > 10 else ""
                logger.info(
                    "[%s] 갱신 대상 밖 캐시 %d건 제거: %s%s", target_norm.upper(), len(orphans), preview, suffix
                )
        except Exception as exc:
            logger.warning("[%s] 고아 캐시 정리 중 오류: %s", target_norm.upper(), exc)

        # 풀 전체 검증: 데이터 소스 오류로 다수 종목의 close 가 NaN인 날짜 자동 제거
        purged_dates: list = []
        try:
            purged_dates = _purge_suspicious_dates(target_norm, success_tickers)
        except Exception as exc:
            logger.warning("[%s] 의심 날짜 자동 정리 중 오류: %s", target_norm.upper(), exc)

        # /system 종목풀 표용 일간 등락률을 stock_meta 에 저장
        _update_daily_change_pct(target_norm, success_tickers)

        # /system 종목풀 표용 추세(%) 양수 요약을 pool_rank_summary 에 저장
        _update_pool_rank_summary(target_norm)

        # 포트폴리오 변동 캐시는 조회 시 TTL 기준으로 갱신한다.
        set_cache_refresh_completed_at(target_norm, pd.Timestamp.utcnow().to_pydatetime())

        # 슬랙 점검 보고용 — main 이 풀별로 모아 문제가 있을 때만 1건 발송한다.
        return {
            "pool": target_norm,
            # 통보 필터가 그 나라 거래일 달력을 골라야 해서 함께 싣는다.
            "country_code": country_code,
            "failed": list(failed_tickers),
            "backfill": backfill_report,
            "volume_backfill": volume_backfill_report,
            "purged_dates": [str(pd.Timestamp(day).date()) for day in purged_dates],
        }


def _collect_benchmark_tickers(target_id: str) -> list[str]:
    """해당 종목풀 설정에 정의된 벤치마크 티커를 수집합니다.

    벤치마크는 종목풀 구성 종목이 아닐 수 있으므로(예: kor 풀의 226490) 여기서 따로 담아
    가격 캐시에 넣는다. 넣지 않으면 백테스트가 벤치마크를 못 읽는다.

    키 해석은 ``get_pool_benchmark_ticker`` 한 곳에서만 한다 — 예전에 여기서 직접
    ``settings.get("benchmark")`` (소문자)로 읽어 항상 빈 목록이 나오던 버그가 있었다.
    """
    from utils.pool_settings_store import get_pool_benchmark_ticker

    try:
        if target_id not in list_available_ticker_types():
            return []
        ticker = get_pool_benchmark_ticker(get_ticker_type_settings(target_id))
    except Exception:
        get_app_logger().exception("벤치마크 티커 수집 실패: %s", target_id)
        return []
    return [ticker] if ticker else []


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OHLCV 캐시 갱신 스크립트",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--start",
        help="데이터 조회 시작일 (YYYY-MM-DD). 지정하지 않으면 공통 설정",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="전체 히스토리를 강제 재수집한다(수정주가 재정렬 — 하루 1회 크론용). 기본은 증분 갱신.",
    )
    return parser


def refresh_portfolio_change_for_all_targets() -> None:
    """모든 종목풀의 ETF 포트폴리오 변동 캐시를 갱신한다 (가격 캐시는 건드리지 않음)."""
    logger = get_app_logger()
    targets_to_update = list_available_ticker_types()
    if not targets_to_update:
        logger.warning("포트폴리오 변동 캐시 갱신 대상이 없습니다.")
        return

    logger.info("전체 종목풀 포트폴리오 변동 캐시 갱신 시작: targets=%s", targets_to_update)
    for t_id in targets_to_update:
        target_norm = (t_id or "").strip().lower()
        if not target_norm:
            continue
        try:
            target_items = list(get_etfs(target_norm) or [])
        except Exception as exc:
            logger.warning("[%s] 종목 목록 조회 실패: %s", target_norm.upper(), exc)
            continue
        if not target_items:
            continue
        success_tickers = {
            str(item.get("ticker") or "").strip().upper()
            for item in target_items
            if str(item.get("ticker") or "").strip()
        }
        try:
            _refresh_portfolio_change_cache_for_target(target_norm, target_items, success_tickers)
        except Exception as exc:
            logger.warning("[%s] 포트폴리오 변동 캐시 갱신 실패: %s", target_norm.upper(), exc)


def main():
    """CLI 진입점"""
    logger = get_app_logger()
    load_env_if_present()

    parser = _build_parser()
    args = parser.parse_args()

    start_date = args.start or _determine_start_date()
    targets_to_update = list_available_ticker_types()

    if not targets_to_update:
        logger.warning("갱신할 대상이 없습니다.")
        return

    logger.info(
        "전체 종목풀 가격 캐시 갱신 시작: targets=%s, start=%s, mode=%s",
        targets_to_update,
        start_date,
        "full(전체 재수집)" if args.full else "incremental(증분)",
    )

    reports: list[dict] = []
    with _global_refresh_lock():
        for t_id in targets_to_update:
            report = refresh_cache_for_target(t_id, start_date, full_refresh=args.full)
            if isinstance(report, dict):
                reports.append(report)

    _notify_cache_issues(reports, full_refresh=args.full)


def _recent_trading_days(country_code: str, days: int) -> set[str] | None:
    """그 나라의 최근 `days` 거래일(오늘 포함) 날짜 문자열. 달력을 못 읽으면 None."""
    if days <= 0:
        return set()
    from utils.trading_calendar import get_trading_days

    end = pd.Timestamp.now().normalize()
    # 연휴가 길어도 days 개를 채우도록 넉넉히 뒤로 잡는다.
    start = end - pd.Timedelta(days=days * 3 + 14)
    try:
        calendar = get_trading_days(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), country_code)
    except Exception:
        return None  # 달력이 없으면 걸러내지 않는다 — 조용히 통보를 삼키지 않기 위해서다.
    return {str(pd.Timestamp(day).date()) for day in calendar[-days:]}


def _recent_unfilled(unfilled: dict, recent_days: set[str] | None) -> dict:
    """못 채운 결측 중 최근 거래일 것만 남긴다. 기준을 못 구했으면(None) 그대로 둔다."""
    if recent_days is None:
        return dict(unfilled)
    kept: dict[str, list[str]] = {}
    for ticker, dates in unfilled.items():
        recent = [day for day in dates if day in recent_days]
        if recent:
            kept[ticker] = recent
    return kept


def _notify_cache_issues(reports: list[dict], *, full_refresh: bool) -> None:
    """캐시 갱신 중 발견된 문제를 슬랙 1건으로 통보한다. 문제가 없으면 보내지 않는다.

    항목: 수집 실패 / 외부 소스에도 없어 못 채운 결측 / 의심 날짜 자동 제거.
    자동 복구에 성공한 보강은 알리지 않는다(로그에만 남긴다) — 소음이 되는 항목은 여기서 뺀다.

    **날짜 기반 문제는 최근 것만 알린다**(`CACHE_ISSUE_NOTIFY_RECENT_TRADING_DAYS`).
    다음 실행에도 그대로 남거나 전체 재수집 때 다시 생겨, 조건 없이 알리면 같은 내용이
    매일 오기 때문이다. 걸러진 건은 아래 로그에 남는다.
    """
    from config import CACHE_ISSUE_NOTIFY_RECENT_TRADING_DAYS

    logger = get_app_logger()
    lines: list[str] = []
    for report in reports:
        pool = str(report["pool"]).upper()
        pool_lines: list[str] = []
        failed = report.get("failed") or []
        if failed:
            preview = ", ".join(failed[:10]) + (" …" if len(failed) > 10 else "")
            pool_lines.append(f"· 수집 실패 {len(failed)}종목: {preview}")
        recent_days = _recent_trading_days(
            str(report.get("country_code") or ""), int(CACHE_ISSUE_NOTIFY_RECENT_TRADING_DAYS)
        )
        backfill = report.get("backfill") or {}
        all_unfilled = backfill.get("unfilled") or {}
        unfilled = _recent_unfilled(all_unfilled, recent_days)
        if len(all_unfilled) != len(unfilled):
            logger.info(
                "[cache] %s 종가 결측 %d종목 중 %d종목만 통보 (최근 %d거래일 기준, 나머지는 로그만): %s",
                pool,
                len(all_unfilled),
                len(unfilled),
                CACHE_ISSUE_NOTIFY_RECENT_TRADING_DAYS,
                all_unfilled,
            )
        if unfilled:
            preview = ", ".join(f"{t}({', '.join(d)})" for t, d in list(unfilled.items())[:10])
            suffix = " …" if len(unfilled) > 10 else ""
            pool_lines.append(f"· 네이버에도 없어 못 채움 {len(unfilled)}종목: {preview}{suffix}")
        volume_backfill = report.get("volume_backfill") or {}
        all_volume_unfilled = volume_backfill.get("unfilled") or {}
        volume_unfilled = _recent_unfilled(all_volume_unfilled, recent_days)
        if len(all_volume_unfilled) != len(volume_unfilled):
            logger.info(
                "[cache] %s 거래량 결측 %d종목 중 %d종목만 통보 (최근 %d거래일 기준, 나머지는 로그만): %s",
                pool,
                len(all_volume_unfilled),
                len(volume_unfilled),
                CACHE_ISSUE_NOTIFY_RECENT_TRADING_DAYS,
                all_volume_unfilled,
            )
        if volume_unfilled:
            preview = ", ".join(f"{t}({', '.join(d)})" for t, d in list(volume_unfilled.items())[:10])
            suffix = " …" if len(volume_unfilled) > 10 else ""
            pool_lines.append(f"· 토스에도 없어 못 채운 거래량 {len(volume_unfilled)}종목: {preview}{suffix}")
        all_purged = report.get("purged_dates") or []
        purged = list(all_purged) if recent_days is None else [day for day in all_purged if day in recent_days]
        if len(all_purged) != len(purged):
            logger.info(
                "[cache] %s 의심 날짜 제거 %d일 중 %d일만 통보 (최근 %d거래일 기준, 나머지는 로그만): %s",
                pool,
                len(all_purged),
                len(purged),
                CACHE_ISSUE_NOTIFY_RECENT_TRADING_DAYS,
                all_purged,
            )
        if purged:
            pool_lines.append(f"· 의심 날짜(다수 종목 종가 NaN) 제거: {', '.join(purged)}")
        if pool_lines:
            lines.append(f"*{pool}*")
            lines.extend(pool_lines)
    if not lines:
        return
    mode = "전체 재수집" if full_refresh else "증분"
    message = "\n".join([f"⚠️ 가격 캐시 점검 ({mode}) — 문제 발견", *lines])
    try:
        from utils.notification import send_slack_message_v2

        send_slack_message_v2(message)
    except Exception:
        logger.exception("[cache] 점검 슬랙 발송 실패")


if __name__ == "__main__":
    main()
