#!/usr/bin/env python
"""계좌별 OHLCV 캐시를 종목 단위로 incremental 갱신합니다."""

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
        raise RuntimeError(
            "CACHE_START_DATE 가 설정되지 않았습니다. config.py 또는 공용 설정을 확인하세요."
        )
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

    nan_ratio = matrix.isna().sum(axis=1) / float(matrix.shape[1])
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


def _update_daily_change_pct(target_id: str, tickers: list[str]) -> None:
    """캐시된 종가 시계열의 마지막 2개로 일간 등락률을 계산해 stock_meta 에 저장한다.

    /system 종목풀 표의 상승수(일간)/상승비율(일간) 이 이 필드를 읽는다.
    가격 캐시가 이미 메모리에 적재된 직후라 추가 외부 호출 없이 공짜로 계산된다.
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
            ops.append(
                UpdateOne(
                    {"ticker_type": target_id, "ticker": ticker},
                    {
                        "$set": {
                            "1_day_change_pct": round(change_pct, 4),
                            "1_day_change_date": pd.Timestamp(close_series.index[-1]).strftime("%Y-%m-%d"),
                        }
                    },
                )
            )
        if ops:
            result = db.stock_meta.bulk_write(ops, ordered=False)
            logger.info(
                "[%s] 일간 등락률 저장 완료: %d개 종목 (matched %d)",
                target_id.upper(),
                len(ops),
                result.matched_count,
            )
    except Exception as exc:
        logger.warning("[%s] 일간 등락률 저장 실패: %s", target_id.upper(), exc)


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
                    idx, len(candidates), ticker, elapsed,
                )
                return True, ticker
            logger.warning(
                " -> 포트폴리오 변동 캐시 계산 불가: %d/%d - %s | 소요 %.1fs",
                idx, len(candidates), ticker, elapsed,
            )
            return False, ticker
        except Exception as exc:
            elapsed = time.perf_counter() - started_at
            logger.warning(
                " -> 포트폴리오 변동 캐시 갱신 실패: %d/%d - %s: %s | 소요 %.1fs",
                idx, len(candidates), ticker, exc, elapsed,
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
            futures = [
                executor.submit(_process_one_bundle, i + 1, t)
                for i, (t, _) in enumerate(candidates)
            ]
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
):
    """지정된 계정(target_id)에 대한 가격 데이터 캐시를 새로 고칩니다."""
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
                    force_refresh=True,
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

        def _process_one(idx: int, etf_item: dict) -> tuple[bool, str, str]:
            """단일 종목 처리. 반환: (성공여부, ticker, log_message).

            KOR 풀(pykrx 사용)은 KRX 가 단위 시간당 호출 빈도로 IP 차단을 거는 듯하다.
            서버처럼 응답이 너무 빠른 환경(종목당 ~0.1s)에서는 30종목 즈음 차단되어 hang
            상태로 빠진다. 종목당 목표 간격(KOR_FETCH_TARGET_MS, 기본 300ms) 미만으로
            끝났을 때 부족분만큼 동적으로 sleep 한다. 로컬처럼 자연 소요가 충분히 느린
            환경(0.3s 이상)은 영향이 없다.
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
                sleep_secs = 0.0
                if country_code == "kor":
                    target = max(0.0, float(os.environ.get("KOR_FETCH_TARGET_MS") or 300) / 1000.0)
                    sleep_secs = max(0.0, target - elapsed)
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
                        f" -> 가격 캐시 갱신 완료: {idx}/{total_tickers} - {n}({t})"
                        f" | 소요 {elapsed:.1f}s{sleep_suffix}"
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
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(_process_one, i + 1, etf): (i + 1, etf)
                    for i, etf in enumerate(target_items)
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

        success_tickers = [
            str(etf.get("ticker") or "").strip().upper()
            for etf in target_items
            if str(etf.get("ticker") or "").strip().upper() not in failed_tickers
        ]

        # 풀 전체 검증: 데이터 소스 오류로 다수 종목의 close 가 NaN인 날짜 자동 제거
        try:
            _purge_suspicious_dates(target_norm, success_tickers)
        except Exception as exc:
            logger.warning("[%s] 의심 날짜 자동 정리 중 오류: %s", target_norm.upper(), exc)

        # /system 종목풀 표용 일간 등락률을 stock_meta 에 저장
        _update_daily_change_pct(target_norm, success_tickers)

        # 포트폴리오 변동 캐시는 조회 시 TTL 기준으로 갱신한다.
        set_cache_refresh_completed_at(target_norm, pd.Timestamp.utcnow().to_pydatetime())


def _collect_benchmark_tickers(target_id: str) -> list[str]:
    """해당 종목풀 설정에 정의된 벤치마크 티커들을 수집합니다."""
    tickers = set()

    try:
        if target_id not in list_available_ticker_types():
            return []
        settings = get_ticker_type_settings(target_id)

        # 'benchmark' (dict, single) 처리
        single_bm = settings.get("benchmark")
        if single_bm and isinstance(single_bm, dict):
            ticker = str(single_bm.get("ticker") or "").strip().upper()
            if ticker:
                tickers.add(ticker)

        return sorted(tickers)
    except Exception:
        pass

    return sorted(tickers)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OHLCV 캐시 갱신 스크립트",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--start",
        help="데이터 조회 시작일 (YYYY-MM-DD). 지정하지 않으면 공통 설정",
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

    logger.info("전체 종목풀 가격 캐시 갱신 시작: targets=%s, start=%s", targets_to_update, start_date)

    with _global_refresh_lock():
        for t_id in targets_to_update:
            refresh_cache_for_target(t_id, start_date)


if __name__ == "__main__":
    main()
