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
from contextlib import contextmanager

import pandas as pd

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from utils.stock_list_io import get_active_holding_tickers, get_all_etfs_including_deleted

FETCH_RETRY_ATTEMPTS = 3
FETCH_RETRY_DELAY_SECONDS = 2.0
PER_TICKER_TIMEOUT_SECONDS = 90

# 풀 전체 NaN 비율이 이 임계값을 초과하는 날짜는 데이터 소스 오류로 간주하고
# 모든 종목 캐시에서 그 날짜 행을 제거한다. 다음 cron 시 자동 재fetch.
SUSPICIOUS_NAN_RATIO_THRESHOLD = 0.5
SUSPICIOUS_LOOKBACK_DAYS = 400


def _determine_start_date() -> str:
    settings = load_common_settings() or {}
    start = settings.get("CACHE_START_DATE")
    if start:
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
def _target_refresh_lock(target_id: str):
    """동일 대상 캐시 갱신이 동시에 실행되지 않도록 파일 잠금을 건다."""
    target_norm = (target_id or "").strip().lower()
    if not target_norm:
        raise ValueError("잠금을 위한 target_id가 필요합니다.")

    lock_path = os.path.join("/tmp", f"momentum_etf_cache_refresh_{target_norm}.lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise RuntimeError(f"[{target_norm.upper()}] 캐시 갱신이 이미 실행 중입니다. 중복 실행을 중단합니다.") from exc

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

    cutoff = (pd.Timestamp.now().normalize() - pd.Timedelta(days=lookback_days))

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

    with _target_refresh_lock(target_norm):
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
            pass # get_all_etfs_including_deleted가 이미 수행함
            holdings = _collect_portfolio_master_holdings(target_norm)
            for item in holdings:
                ticker = str(item.get("ticker") or "").strip().upper()
                if not ticker or ticker in all_map:
                    continue
                all_map[ticker] = item

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

        for i, etf in enumerate(target_items, 1):
            ticker = str(etf.get("ticker") or "").strip().upper()
            name = etf.get("name") or "-"
            started_at = time.perf_counter()

            if progress_callback:
                progress_callback(i, total_tickers, f"{name}({ticker})")

            logger.info(" -> 가격 캐시 처리 시작: %d/%d - %s(%s)", i, total_tickers, name, ticker)

            try:
                range_start = start_date or "1990-01-01"
                with _ticker_refresh_timeout(PER_TICKER_TIMEOUT_SECONDS):
                    unresolved_days = _refresh_single_ticker_with_retry(
                        ticker=ticker,
                        country_code=country_code,
                        range_start=range_start,
                        account_id=target_norm,
                    )

                if unresolved_days:
                    unresolved_text = ", ".join(day.strftime("%Y-%m-%d") for day in unresolved_days)
                    logger.warning(
                        " -> 가격 캐시 갱신 완료: %d/%d - %s(%s) - 최근 거래일 누락 유지: %s | 소요 %.1fs",
                        i,
                        total_tickers,
                        name,
                        ticker,
                        unresolved_text,
                        time.perf_counter() - started_at,
                    )
                else:
                    logger.info(
                        " -> 가격 캐시 갱신 완료: %d/%d - %s(%s) | 소요 %.1fs",
                        i,
                        total_tickers,
                        name,
                        ticker,
                        time.perf_counter() - started_at,
                    )
                succeeded_count += 1
            except PykrxDataUnavailableError as e:
                failed_tickers.append(ticker)
                if _is_today_unavailable_warning(e):
                    logger.warning(
                        "%s 당일 데이터 미집계: %s | 소요 %.1fs", ticker, e, time.perf_counter() - started_at
                    )
                else:
                    logger.error(
                        "%s 데이터 처리 중 오류 발생: %s | 소요 %.1fs", ticker, e, time.perf_counter() - started_at
                    )
            except Exception as e:
                failed_tickers.append(ticker)
                logger.error(
                    "%s 데이터 처리 중 오류 발생: %s | 소요 %.1fs", ticker, e, time.perf_counter() - started_at
                )

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

        # 풀 전체 검증: 데이터 소스 오류로 다수 종목의 close 가 NaN인 날짜 자동 제거
        try:
            success_tickers = [
                str(etf.get("ticker") or "").strip().upper()
                for etf in target_items
                if str(etf.get("ticker") or "").strip().upper() not in failed_tickers
            ]
            _purge_suspicious_dates(target_norm, success_tickers)
        except Exception as exc:
            logger.warning("[%s] 의심 날짜 자동 정리 중 오류: %s", target_norm.upper(), exc)

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


def _collect_portfolio_master_holdings(target_id: str) -> list[dict[str, str]]:
    """최신 스냅샷 기준 현재 보유 종목을 지정 종목풀의 캐시 갱신 대상에 추가한다."""
    results: list[dict[str, str]] = []
    seen: set[str] = set()

    try:
        active_holdings_by_type = get_active_holding_tickers()
    except Exception:
        return []

    for ticker in sorted(active_holdings_by_type.get(target_id, set())):
        ticker_norm = str(ticker or "").strip().upper()
        if not ticker_norm or ticker_norm in seen:
            continue
        seen.add(ticker_norm)
        results.append(
            {
                "ticker": ticker_norm,
                "name": ticker_norm,
                "type": "etf",
            }
        )
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OHLCV 캐시 갱신 스크립트",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("target", nargs="?", help="Account ID")
    parser.add_argument(
        "--start",
        help="데이터 조회 시작일 (YYYY-MM-DD). 지정하지 않으면 공통 설정",
    )
    return parser


def main():
    """CLI 진입점"""
    logger = get_app_logger()
    load_env_if_present()

    parser = _build_parser()
    args = parser.parse_args()

    target = (args.target or "").strip().lower()
    start_date = args.start or _determine_start_date()

    targets_to_update: list[str] = []
    available_types = list_available_ticker_types()
    
    if not target:
        targets_to_update = available_types
    else:
        if target in available_types:
            targets_to_update = [target]
        else:
            logger.error(f"Target '{target}' is not a valid ticker pool ID.")
            return

    if not targets_to_update:
        logger.warning("갱신할 대상이 없습니다.")
        return

    logger.info("입력 파라미터: targets=%s, start=%s", targets_to_update, start_date)

    for t_id in targets_to_update:
        refresh_cache_for_target(t_id, start_date)


if __name__ == "__main__":
    main()
