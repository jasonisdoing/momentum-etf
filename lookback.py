"""
Walk-Forward Analysis를 통한 최적 룩백 기간 탐색

최근 12개월 동안 롤링 방식으로:
- 여러 룩백 기간(3, 6, 9, 12, 18, 24개월)을 테스트
- 각 룩백 기간으로 파라미터 최적화 후 다음 1개월 성과 측정
- 어떤 룩백 기간이 평균적으로 최적인지 분석

사용법:
    python lookback.py k1
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pandas import Timestamp

# 프로젝트 루트 디렉토리를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ACCOUNT_PARAMETER_SEARCH_CONFIG
from logic.backtest.account_runner import run_account_backtest
from utils.data_loader import get_latest_trading_day, prepare_price_data
from utils.logger import get_app_logger
from utils.report import render_table_eaw
from utils.settings_loader import get_account_settings
from utils.stock_list_io import get_etfs
from utils.cache_utils import save_cached_frame

logger = get_app_logger()

RESULTS_DIR = Path(__file__).resolve().parent / "data" / "results"
# DEFAULT_LOOKBACK_PERIODS = [3, 6, 9, 12, 18, 24]
DEFAULT_LOOKBACK_PERIODS = [3, 6, 9, 12]
DEFAULT_TEST_MONTHS = 12
DEFAULT_TEST_PERIOD_MONTHS = 1


def _calculate_month_offset(base_date: Timestamp, months: int) -> Timestamp:
    """기준일로부터 N개월 전 날짜 계산"""
    year = base_date.year
    month = base_date.month - months

    while month <= 0:
        month += 12
        year -= 1

    while month > 12:
        month -= 12
        year += 1

    try:
        result = Timestamp(year=year, month=month, day=base_date.day)
    except ValueError:
        # 날짜가 유효하지 않으면 (예: 2월 30일) 해당 월의 마지막 날로 설정
        if month == 12:
            next_month = Timestamp(year=year + 1, month=1, day=1)
        else:
            next_month = Timestamp(year=year, month=month + 1, day=1)
        result = next_month - timedelta(days=1)

    return result


def _find_best_params_simple(
    account_id: str,
    lookback_start: Timestamp,
    lookback_end: Timestamp,
    prefetched_data: Dict[str, pd.DataFrame],
) -> Tuple[Optional[Dict[str, Any]], float]:
    """
    간단한 그리드 서치로 최적 파라미터 찾기

    Returns:
        (best_params, best_cagr)
    """
    # 공통 계정별 설정 사용 (없으면 즉시 오류)
    config = ACCOUNT_PARAMETER_SEARCH_CONFIG.get(account_id)
    if config is None:
        raise KeyError(f"계정 {account_id}에 대한 ACCOUNT_PARAMETER_SEARCH_CONFIG 항목이 없습니다.")

    required_keys = [
        "MA_RANGE",
        "PORTFOLIO_TOPN",
        "REPLACE_SCORE_THRESHOLD",
        "OVERBOUGHT_SELL_THRESHOLD",
        "COOLDOWN_DAYS",
    ]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"ACCOUNT_PARAMETER_SEARCH_CONFIG[{account_id}]에 필수 키가 없습니다: {', '.join(missing_keys)}")

    # 탐색 공간 - 공통 설정 직접 사용
    ma_values = list(config["MA_RANGE"])
    topn_values = list(config["PORTFOLIO_TOPN"])
    threshold_values = list(config["REPLACE_SCORE_THRESHOLD"])
    rsi_values = list(config["OVERBOUGHT_SELL_THRESHOLD"])
    cooldown_values = list(config["COOLDOWN_DAYS"])

    # 최적화 지표 선택
    optimization_metric = config.get("OPTIMIZATION_METRIC", "SDR").upper()

    best_params = None
    best_metric_value = float("-inf")

    total_combos = len(ma_values) * len(topn_values) * len(threshold_values) * len(rsi_values) * len(cooldown_values)
    logger.info(
        f"[최적화 시작] 파라미터 조합 수: {total_combos}개 (MA:{len(ma_values)} × TOPN:{len(topn_values)} × RSI:{len(rsi_values)}) | 기준: {optimization_metric}"
    )

    current_combo = 0
    last_log_pct = 0

    # 그리드 서치
    for ma in ma_values:
        for topn in topn_values:
            for threshold in threshold_values:
                for rsi in rsi_values:
                    for cooldown in cooldown_values:
                        current_combo += 1
                        progress_pct = (current_combo / total_combos) * 100

                        # 10%마다 진행률 로그
                        if progress_pct - last_log_pct >= 10.0:
                            logger.info(f"[최적화 진행] {progress_pct:.0f}% ({current_combo}/{total_combos})")
                            last_log_pct = progress_pct

                        params = {
                            "ma_period": int(ma),
                            "portfolio_topn": int(topn),
                            "replace_threshold": float(threshold),
                            "rsi_sell_threshold": int(rsi),
                            "cooldown_days": int(cooldown),
                            "ma_type": "SMA",
                        }

                        try:
                            from logic.entry_point import StrategyRules

                            strategy_rules = StrategyRules.from_values(
                                ma_period=params["ma_period"],
                                portfolio_topn=params["portfolio_topn"],
                                replace_threshold=params["replace_threshold"],
                                ma_type=params.get("ma_type", "SMA"),
                                core_holdings=[],
                            )

                            result = run_account_backtest(
                                account_id=account_id,
                                override_settings={
                                    "start_date": lookback_start,
                                    "end_date": lookback_end,
                                    "strategy_overrides": {
                                        "RSI_SELL_THRESHOLD": params.get("rsi_sell_threshold", 10),
                                        "COOLDOWN_DAYS": params.get("cooldown_days", 1),
                                    },
                                },
                                strategy_override=strategy_rules,
                                prefetched_data=prefetched_data,
                                quiet=True,
                            )

                            if result and result.summary:
                                # CAGR 계산
                                cumulative_return = result.summary.get("period_return", 0.0)
                                start_date = result.summary.get("start_date")
                                end_date = result.summary.get("end_date")

                                # 기간 계산 (연 단위)
                                if start_date and end_date:
                                    days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
                                    years = days / 365.25
                                    if years > 0 and cumulative_return > -100:
                                        cagr = (((1 + cumulative_return / 100) ** (1 / years)) - 1) * 100
                                    else:
                                        cagr = 0.0
                                else:
                                    cagr = 0.0

                                # 최적화 지표 선택
                                sharpe = result.summary.get("sharpe", 0.0)
                                mdd = abs(result.summary.get("mdd", 0.0))
                                sdr = sharpe / mdd if mdd > 0 else 0.0

                                if optimization_metric == "CAGR":
                                    metric_value = cagr
                                elif optimization_metric == "SHARPE":
                                    metric_value = sharpe
                                else:  # SDR
                                    metric_value = sdr

                                logger.debug(f"  MA={ma}, TOPN={topn}, RSI={rsi} → CAGR={cagr:.2f}%, Sharpe={sharpe:.2f}, SDR={sdr:.3f}")
                                if metric_value > best_metric_value:
                                    best_metric_value = metric_value
                                    best_params = params
                                    logger.debug(f"  ✓ 새로운 최적 파라미터 발견! ({optimization_metric}={metric_value:.3f})")
                        except Exception as e:
                            logger.debug(f"백테스트 실패 (MA={ma}, TOPN={topn}, TH={threshold}, RSI={rsi}, CD={cooldown}): {e}")
                            continue

    if best_params:
        logger.debug(
            f"[최적화 완료] 최적 파라미터: MA={best_params['ma_period']}, "
            f"TOPN={best_params['portfolio_topn']}, RSI={best_params.get('rsi_sell_threshold', 10)}, "
            f"최고 {optimization_metric}={best_metric_value:.3f}"
        )
    else:
        logger.warning("[최적화 실패] 유효한 파라미터를 찾지 못했습니다.")

    return best_params, best_metric_value


def _run_single_walk_forward_test(
    account_id: str,
    lookback_months: int,
    test_start_date: Timestamp,
    test_end_date: Timestamp,
    prefetched_data: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """
    단일 Walk-Forward 테스트 실행 (병렬 처리용)

    Args:
        account_id: 계정 ID
        lookback_months: 룩백 기간 (개월)
        test_start_date: 테스트 시작일
        test_end_date: 테스트 종료일
        country_code: 국가 코드

    Returns:
        테스트 결과 딕셔너리
    """
    try:
        # 1. 룩백 기간 설정
        lookback_end_date = test_start_date
        lookback_start_date = _calculate_month_offset(lookback_end_date, lookback_months)

        logger.debug(
            f"[룩백 {lookback_months}개월] 룩백: {lookback_start_date:%Y-%m-%d} ~ {lookback_end_date:%Y-%m-%d}, "
            f"테스트: {test_start_date:%Y-%m-%d} ~ {test_end_date:%Y-%m-%d}"
        )

        # 2. 룩백 기간에서 최적 파라미터 찾기 (간단한 그리드 서치)
        best_params, best_cagr = _find_best_params_simple(
            account_id=account_id,
            lookback_start=lookback_start_date,
            lookback_end=lookback_end_date,
            prefetched_data=prefetched_data,
        )

        if not best_params:
            logger.warning(f"튜닝 실패: 룩백 {lookback_months}개월")
            return {
                "lookback_months": lookback_months,
                "test_start": test_start_date,
                "test_end": test_end_date,
                "error": "튜닝 실패",
            }

        # 디버깅: 찾은 파라미터 출력
        logger.debug(
            f"[룩백 {lookback_months}개월] 최적 파라미터: "
            f"MA={best_params['ma_period']}, TOPN={best_params['portfolio_topn']}, "
            f"RSI={best_params.get('rsi_sell_threshold', 10)}, CAGR={best_cagr:.2f}%"
        )

        # 3. 테스트 기간에서 해당 파라미터로 백테스트
        from logic.entry_point import StrategyRules

        strategy_rules = StrategyRules.from_values(
            ma_period=best_params["ma_period"],
            portfolio_topn=best_params["portfolio_topn"],
            replace_threshold=best_params["replace_threshold"],
            ma_type=best_params.get("ma_type", "SMA"),
            core_holdings=[],
        )

        logger.info(
            f"[테스트 시작] 룩백 {lookback_months}개월, 테스트 {test_start_date:%Y-%m-%d}~{test_end_date:%Y-%m-%d}, "
            f"파라미터: MA={best_params['ma_period']}, RSI={best_params.get('rsi_sell_threshold', 10)}"
        )

        test_result = run_account_backtest(
            account_id=account_id,
            override_settings={
                "start_date": test_start_date,
                "end_date": test_end_date,
                "strategy_overrides": {
                    "RSI_SELL_THRESHOLD": best_params.get("rsi_sell_threshold", 10),
                    "COOLDOWN_DAYS": best_params.get("cooldown_days", 1),
                },
            },
            strategy_override=strategy_rules,
            prefetched_data=prefetched_data,
            quiet=True,
        )

        if test_result and test_result.summary:
            logger.info(f"[테스트 완료] 룩백 {lookback_months}개월, 수익률: {test_result.summary.get('period_return', 0):.2f}%")

        if not test_result or not test_result.summary:
            logger.warning(f"백테스트 실패: 룩백 {lookback_months}개월")
            return {
                "lookback_months": lookback_months,
                "test_start": test_start_date,
                "test_end": test_end_date,
                "error": "백테스트 실패",
            }

        summary = test_result.summary

        # 디버깅: summary 내용 확인
        if not summary:
            logger.warning(f"summary가 비어있음: {lookback_months}개월")
        else:
            logger.debug(f"summary 키: {list(summary.keys())}")

        # 4. 결과 수집
        return {
            "lookback_months": lookback_months,
            "lookback_start": lookback_start_date,
            "lookback_end": lookback_end_date,
            "test_start": test_start_date,
            "test_end": test_end_date,
            "best_params": best_params,
            "total_return": summary.get("period_return", 0.0),
            "cagr": summary.get("cagr", 0.0),
            "sharpe": summary.get("sharpe", 0.0),
            "sharpe_to_mdd": summary.get("sharpe_to_mdd", 0.0),
            "mdd": summary.get("mdd", 0.0),
            "win_rate": summary.get("win_rate", 0.0),
            "error": None,
        }

    except Exception as e:
        logger.error(f"Walk-Forward 테스트 실패 (룩백 {lookback_months}개월): {e}")
        return {
            "lookback_months": lookback_months,
            "test_start": test_start_date,
            "test_end": test_end_date,
            "error": str(e),
        }


def run_walk_forward_analysis(
    account_id: str,
    lookback_periods: List[int] = None,
    test_months: int = DEFAULT_TEST_MONTHS,
    test_period_months: int = DEFAULT_TEST_PERIOD_MONTHS,
    max_workers: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-Forward Analysis 실행

    Args:
        account_id: 계정 ID
        lookback_periods: 테스트할 룩백 기간 리스트 (개월)
        test_months: 테스트할 총 기간 (개월)
        test_period_months: 각 테스트 기간 (개월)
        max_workers: 병렬 처리 워커 수

    Returns:
        (상세 결과 DataFrame, 요약 DataFrame)
    """
    if lookback_periods is None:
        lookback_periods = DEFAULT_LOOKBACK_PERIODS

    # 계정 설정 로드
    account_settings = get_account_settings(account_id)
    country_code = account_settings.get("country_code", "kor")

    # 최신 거래일 가져오기 (백테스트는 확정된 과거 데이터만 사용)
    latest_trading_day = get_latest_trading_day(country_code)
    today = pd.Timestamp.now().normalize()

    # 오늘 날짜가 최신 거래일과 같으면 전일 거래일 사용 (종가 미확정)
    if latest_trading_day >= today:
        from utils.data_loader import get_trading_days

        # 전일 거래일 찾기
        past_30_days = (today - pd.DateOffset(days=30)).strftime("%Y-%m-%d")
        today_str = today.strftime("%Y-%m-%d")
        recent_trading_days = get_trading_days(past_30_days, today_str, country_code)
        if len(recent_trading_days) >= 2:
            latest_date = pd.to_datetime(recent_trading_days[-2]).normalize()
        else:
            latest_date = latest_trading_day - pd.DateOffset(days=1)
    else:
        latest_date = latest_trading_day

    logger.info("=== Walk-Forward Analysis 시작 ===")
    logger.info(f"계정: {account_id} ({country_code})")
    logger.info(f"기준일: {latest_date:%Y-%m-%d} (백테스트용 확정 데이터)")
    logger.info(f"테스트 기간: 최근 {test_months}개월")
    logger.info(f"룩백 기간 후보: {lookback_periods}")
    logger.info(f"각 테스트 기간: {test_period_months}개월")

    # 테스트할 월별 기간 생성
    test_periods = []
    for month_offset in range(test_months, 0, -test_period_months):
        test_end = _calculate_month_offset(latest_date, month_offset - test_period_months)
        test_start = _calculate_month_offset(latest_date, month_offset)
        test_periods.append((test_start, test_end))

    logger.info(f"총 {len(test_periods)}개 테스트 기간 생성")

    # === 데이터 프리패치 ===
    max_lookback_months = max(lookback_periods)
    earliest_date = _calculate_month_offset(latest_date, test_months + max_lookback_months)

    logger.info(f"[프리패치] 데이터 로딩 시작: {earliest_date:%Y-%m-%d} ~ {latest_date:%Y-%m-%d}")

    # ETF 목록 가져오기
    etf_universe = get_etfs(country_code)
    tickers = [str(item.get("ticker")) for item in etf_universe if item.get("ticker")]

    # 데이터 프리패치 (백테스트이므로 실시간 가격 조회 비활성화)
    prefetched_data, missing_tickers = prepare_price_data(
        tickers=tickers,
        country=country_code,
        start_date=earliest_date.strftime("%Y-%m-%d"),
        end_date=latest_date.strftime("%Y-%m-%d"),
        warmup_days=100,
        skip_realtime=True,
    )

    # 캐시에 저장
    for ticker, frame in prefetched_data.items():
        save_cached_frame(country_code, ticker, frame)

    logger.info(f"[프리패치] 완료: {len(prefetched_data)}개 종목 로드, {len(missing_tickers)}개 누락")

    if missing_tickers:
        logger.warning(
            f"[프리패치] 데이터 없는 종목: {', '.join(missing_tickers[:10])}"
            + (f" 외 {len(missing_tickers) - 10}개" if len(missing_tickers) > 10 else "")
        )

    # 진행도 표시
    total_tasks = len(test_periods) * len(lookback_periods)
    completed = 0

    # 병렬 처리를 위한 작업 생성
    tasks = []
    for test_start, test_end in test_periods:
        for lookback_months in lookback_periods:
            tasks.append(
                {
                    "account_id": account_id,
                    "lookback_months": lookback_months,
                    "test_start_date": test_start,
                    "test_end_date": test_end,
                    "prefetched_data": prefetched_data,
                }
            )

    total_tasks = len(tasks)
    logger.info(f"총 {total_tasks}개 작업 실행 (병렬 처리)")

    # 중간 결과 저장 경로 (계정별 폴더)
    account_dir = RESULTS_DIR / account_id
    account_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    summary_file = account_dir / f"lookback_summary_{date_str}.log"
    details_file = account_dir / f"lookback_details_{date_str}.log"

    # 병렬 실행
    results = []
    completed = 0
    last_save_pct = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_single_walk_forward_test, **task): task for task in tasks}

        for future in as_completed(futures):
            completed += 1
            progress_pct = (completed / total_tasks) * 100

            try:
                result = future.result()
                results.append(result)

                if result.get("error"):
                    logger.warning(f"[{completed}/{total_tasks}] ({progress_pct:.1f}%) 실패: {result['error']}")
                else:
                    logger.info(
                        f"[{completed}/{total_tasks}] ({progress_pct:.1f}%) 완료: 룩백 {result['lookback_months']}개월, "
                        f"수익률 {result.get('total_return', 0):.2f}%"
                    )

                # 1%마다 중간 저장
                if progress_pct - last_save_pct >= 1.0 or completed == total_tasks:
                    last_save_pct = progress_pct
                    _save_progress(results, summary_file, details_file, completed, total_tasks)

            except Exception as e:
                logger.error(f"작업 실행 중 오류: {e}")

    # DataFrame 생성
    df = pd.DataFrame(results)

    if df.empty:
        logger.error("결과가 없습니다.")
        return df, pd.DataFrame()

    # 에러 제외
    df_valid = df[df["error"].isna()].copy()

    if df_valid.empty:
        logger.error("유효한 결과가 없습니다.")
        return df, pd.DataFrame()

    # 요약 통계 계산
    summary = (
        df_valid.groupby("lookback_months")
        .agg(
            {
                "total_return": ["mean", "std", "min", "max"],
                "cagr": ["mean"],
                "sharpe": ["mean", "std"],
                "sharpe_to_mdd": ["mean"],
                "mdd": ["mean", "min", "max"],
                "win_rate": ["mean"],
            }
        )
        .round(2)
    )

    # 승률 계산 (양수 수익률 비율)
    win_rates = df_valid.groupby("lookback_months").apply(lambda x: (x["total_return"] > 0).sum() / len(x) * 100).round(1)

    summary[("win_rate_pct", "calc")] = win_rates

    return df_valid, summary


def _save_progress(results: List[Dict], summary_file: Path, details_file: Path, completed: int, total: int) -> None:
    """중간 진행 상황을 요약/상세 2개 파일로 저장"""
    if not results:
        return

    try:
        df_valid = pd.DataFrame([r for r in results if not r.get("error")])
        if df_valid.empty:
            return

        # === 1. 요약 파일 (summary) ===
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append(f"Walk-Forward Analysis 진행 상황 ({completed}/{total})")
        summary_lines.append(f"완료: {completed}/{total} ({(completed / total) * 100:.1f}%)")
        summary_lines.append("=" * 80)
        summary_lines.append("")

        # 룩백 기간별 평균 통계
        lookback_summary = (
            df_valid.groupby("lookback_months")
            .agg(
                {
                    "total_return": ["mean", "count"],
                    "cagr": ["mean"],
                    "sharpe": ["mean"],
                    "sharpe_to_mdd": ["mean"],
                }
            )
            .round(2)
        )

        summary_lines.append("[룩백 기간별 평균]")
        for lookback_months in sorted(lookback_summary.index):
            row = lookback_summary.loc[lookback_months]
            count = int(row[("total_return", "count")])
            avg_return = row[("total_return", "mean")]
            avg_sharpe = row[("sharpe", "mean")]
            avg_sdr = row[("sharpe_to_mdd", "mean")]

            summary_lines.append(
                f"  참조 {lookback_months:2d}개월: 월평균수익률 {avg_return:+6.2f}%, "
                f"Sharpe {avg_sharpe:5.2f}, SDR {avg_sdr:5.3f} ({count}개 완료)"
            )

        summary_file.write_text("\n".join(summary_lines), encoding="utf-8")

        # === 2. 상세 파일 (details) ===
        details_lines = []
        details_lines.append("=" * 120)
        details_lines.append(f"Walk-Forward Analysis 상세 결과 ({completed}/{total})")
        details_lines.append("=" * 120)
        details_lines.append("")

        # 현재 날짜 (상대 기간 계산용)
        from pandas import Timestamp

        now = Timestamp.now().normalize()

        # 테스트 시점별로 그룹화 (테스트 시점 중심)
        for test_period in sorted(df_valid[["test_start", "test_end"]].drop_duplicates().values, key=lambda x: x[0], reverse=True):
            test_start, test_end = test_period

            # numpy.datetime64를 pandas Timestamp로 변환
            test_start = pd.Timestamp(test_start)
            test_end = pd.Timestamp(test_end)

            if pd.notna(test_start) and pd.notna(test_end):
                # 상대 기간 계산
                months_ago = (now.year - test_start.year) * 12 + (now.month - test_start.month)
                relative_period = f"{months_ago}개월 전" if months_ago > 0 else "현재"

                test_start_str = test_start.strftime("%Y-%m-%d")
                test_end_str = test_end.strftime("%Y-%m-%d")

                details_lines.append(f"\n[{relative_period}: {test_start_str} ~ {test_end_str}]")
                details_lines.append("-" * 120)

                # 해당 테스트 시점의 모든 룩백 기간 결과
                df_period = df_valid[(df_valid["test_start"] == test_start) & (df_valid["test_end"] == test_end)]
                df_period = df_period.sort_values("lookback_months")

                for _, row in df_period.iterrows():
                    lookback_months = row.get("lookback_months", 0)
                    total_return = row.get("total_return", 0.0)
                    sharpe = row.get("sharpe", 0.0)
                    mdd = row.get("mdd", 0.0)

                    details_lines.append(
                        f"  참조 {lookback_months:2d}개월: 월수익률 {total_return:+6.2f}% | " f"Sharpe {sharpe:5.2f} | MDD {mdd:6.2f}%"
                    )

        details_file.write_text("\n".join(details_lines), encoding="utf-8")

    except Exception as e:
        logger.warning(f"중간 저장 실패: {e}")


def print_summary_table(summary: pd.DataFrame, account_id: str) -> None:
    """요약 결과를 테이블로 출력 (render_table_eaw 사용)"""

    if summary.empty:
        print("결과가 없습니다.")
        return

    # 최적화 지표 가져오기
    config = ACCOUNT_PARAMETER_SEARCH_CONFIG.get(account_id, {})
    optimization_metric = config.get("OPTIMIZATION_METRIC", "SDR").upper()

    print("\n" + "=" * 120)
    print(f"Walk-Forward Analysis 요약 결과 (최적화 기준: {optimization_metric})")
    print("=" * 120)

    headers = [
        "룩백기간",
        "월평균수익률",
        "수익률StdDev",
        "최소수익률",
        "최대수익률",
        "승률",
        "평균Sharpe",
        "평균SDR",
        "평균MDD",
    ]

    aligns = ["center", "right", "right", "right", "right", "right", "right", "right", "right"]

    rows = []
    for lookback_months in summary.index:
        row_data = summary.loc[lookback_months]

        rows.append(
            [
                f"참조 {lookback_months}개월",
                f"{row_data[('total_return', 'mean')]:+.2f}%",
                f"{row_data[('total_return', 'std')]:.2f}%",
                f"{row_data[('total_return', 'min')]:+.2f}%",
                f"{row_data[('total_return', 'max')]:+.2f}%",
                f"{row_data[('win_rate_pct', 'calc')]:.1f}%",
                f"{row_data[('sharpe', 'mean')]:.2f}",
                f"{row_data[('sharpe_to_mdd', 'mean')]:.3f}",
                f"{row_data[('mdd', 'mean')]:.2f}%",
            ]
        )

    table_lines = render_table_eaw(headers, rows, aligns)
    for line in table_lines:
        print(line)

    # 최적 룩백 기간 찾기 (최적화 지표 기준)
    if optimization_metric == "CAGR":
        best_idx = summary[("cagr", "mean")].idxmax()
    elif optimization_metric == "SHARPE":
        best_idx = summary[("sharpe", "mean")].idxmax()
    else:  # SDR
        best_idx = summary[("sharpe_to_mdd", "mean")].idxmax()

    best_return = summary.loc[best_idx, ("total_return", "mean")]
    best_win_rate = summary.loc[best_idx, ("win_rate_pct", "calc")]
    best_sharpe = summary.loc[best_idx, ("sharpe", "mean")]
    best_sdr = summary.loc[best_idx, ("sharpe_to_mdd", "mean")]
    best_cagr = summary.loc[best_idx, ("cagr", "mean")]

    print("\n" + "=" * 120)
    print(f"✅ 최적 룩백 기간: 참조 {best_idx}개월 ({optimization_metric} 기준)")
    print(
        f"   평균 수익률: {best_return:+.2f}% | 승률: {best_win_rate:.1f}% | Sharpe: {best_sharpe:.2f} | SDR: {best_sdr:.3f} | CAGR: {best_cagr:+.2f}%"
    )
    print("=" * 120 + "\n")


def save_results(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    account_id: str,
    output_dir: Path,
) -> None:
    """결과를 요약/상세 2개 .log 파일로 저장"""

    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    summary_file = output_dir / f"lookback_summary_{date_str}.log"
    details_file = output_dir / f"lookback_details_{date_str}.log"

    # === 1. 요약 파일 ===
    summary_lines = []
    summary_lines.append("=" * 120)
    summary_lines.append(f"Walk-Forward Analysis 요약 결과 - {account_id.upper()}")
    summary_lines.append(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("=" * 120)
    summary_lines.append("")

    # 요약 테이블
    headers = [
        "룩백기간",
        "월평균수익률",
        "수익률StdDev",
        "최소수익률",
        "최대수익률",
        "승률",
        "평균Sharpe",
        "평균SDR",
        "평균MDD",
    ]

    aligns = ["center", "right", "right", "right", "right", "right", "right", "right", "right"]

    rows = []
    for lookback_months in summary.index:
        row_data = summary.loc[lookback_months]

        rows.append(
            [
                f"참조 {lookback_months}개월",
                f"{row_data[('total_return', 'mean')]:+.2f}%",
                f"{row_data[('total_return', 'std')]:.2f}%",
                f"{row_data[('total_return', 'min')]:+.2f}%",
                f"{row_data[('total_return', 'max')]:+.2f}%",
                f"{row_data[('win_rate_pct', 'calc')]:.1f}%",
                f"{row_data[('sharpe', 'mean')]:.2f}",
                f"{row_data[('sharpe_to_mdd', 'mean')]:.3f}",
                f"{row_data[('mdd', 'mean')]:.2f}%",
            ]
        )

    table_lines = render_table_eaw(headers, rows, aligns)
    summary_lines.extend(table_lines)

    # 최적화 지표 가져오기
    config = ACCOUNT_PARAMETER_SEARCH_CONFIG.get(account_id, {})
    optimization_metric = config.get("OPTIMIZATION_METRIC", "SDR").upper()

    # 최적 룩백 기간
    if optimization_metric == "CAGR":
        best_idx = summary[("cagr", "mean")].idxmax()
    elif optimization_metric == "SHARPE":
        best_idx = summary[("sharpe", "mean")].idxmax()
    else:  # SDR
        best_idx = summary[("sharpe_to_mdd", "mean")].idxmax()

    best_return = summary.loc[best_idx, ("total_return", "mean")]
    best_win_rate = summary.loc[best_idx, ("win_rate_pct", "calc")]
    best_sharpe = summary.loc[best_idx, ("sharpe", "mean")]
    best_sdr = summary.loc[best_idx, ("sharpe_to_mdd", "mean")]
    best_cagr = summary.loc[best_idx, ("cagr", "mean")]

    summary_lines.append("")
    summary_lines.append("=" * 120)
    summary_lines.append(f"✅ 최적 룩백 기간: 참조 {best_idx}개월 ({optimization_metric} 기준)")
    summary_lines.append(
        f"   평균 수익률: {best_return:+.2f}% | 승률: {best_win_rate:.1f}% | Sharpe: {best_sharpe:.2f} | SDR: {best_sdr:.3f} | CAGR: {best_cagr:+.2f}%"
    )
    summary_lines.append("=" * 120)

    summary_file.write_text("\n".join(summary_lines), encoding="utf-8")
    logger.info(f"요약 결과 저장: {summary_file}")

    # === 2. 상세 파일 ===
    details_lines = []
    details_lines.append("=" * 120)
    details_lines.append(f"Walk-Forward Analysis 상세 결과 - {account_id.upper()}")
    details_lines.append(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    details_lines.append("=" * 120)
    details_lines.append("")

    # 현재 날짜 (상대 기간 계산용)
    from pandas import Timestamp

    now = Timestamp.now().normalize()

    # 테스트 시점별로 그룹화 (테스트 시점 중심)
    for test_period in sorted(df[["test_start", "test_end"]].drop_duplicates().values, key=lambda x: x[0], reverse=True):
        test_start, test_end = test_period

        # numpy.datetime64를 pandas Timestamp로 변환
        test_start = pd.Timestamp(test_start)
        test_end = pd.Timestamp(test_end)

        if pd.notna(test_start) and pd.notna(test_end):
            # 상대 기간 계산
            months_ago = (now.year - test_start.year) * 12 + (now.month - test_start.month)
            relative_period = f"{months_ago}개월 전" if months_ago > 0 else "현재"

            test_start_str = test_start.strftime("%Y-%m-%d")
            test_end_str = test_end.strftime("%Y-%m-%d")

            details_lines.append(f"\n{'=' * 120}")
            details_lines.append(f"[{relative_period}: {test_start_str} ~ {test_end_str}]")
            details_lines.append("=" * 120)

            # 해당 테스트 시점의 모든 룩백 기간 결과
            df_period = df[(df["test_start"] == test_start) & (df["test_end"] == test_end)]
            df_period = df_period.sort_values("lookback_months")

            for _, row in df_period.iterrows():
                lookback_months = row.get("lookback_months", 0)
                lookback_start = row.get("lookback_start")
                lookback_end = row.get("lookback_end")

                lookback_start_str = lookback_start.strftime("%Y-%m-%d") if pd.notna(lookback_start) else "N/A"
                lookback_end_str = lookback_end.strftime("%Y-%m-%d") if pd.notna(lookback_end) else "N/A"

                details_lines.append(f"\n참조 {lookback_months:2d}개월 (룩백: {lookback_start_str} ~ {lookback_end_str})")

                if row.get("error"):
                    details_lines.append(f"  ❌ 오류: {row['error']}")
                else:
                    best_params = row.get("best_params", {})
                    details_lines.append(
                        f"  최적 파라미터: MA={best_params.get('ma_period', 'N/A')}, "
                        f"TOPN={best_params.get('portfolio_topn', 'N/A')}, "
                        f"RSI={best_params.get('rsi_sell_threshold', 'N/A')}"
                    )
                    details_lines.append(
                        f"  월수익률: {row.get('total_return', 0):+6.2f}% | "
                        f"Sharpe: {row.get('sharpe', 0):5.2f} | "
                        f"SDR: {row.get('sharpe_to_mdd', 0):5.3f}"
                    )
                    details_lines.append(f"  MDD: {row.get('mdd', 0):6.2f}%")

    details_file.write_text("\n".join(details_lines), encoding="utf-8")
    logger.info(f"상세 결과 저장: {details_file}")


def main():
    if len(sys.argv) < 2:
        print("사용법: python lookback.py <account_id>")
        print()
        print("예시:")
        print("  python lookback.py k1")
        sys.exit(1)

    account_id = sys.argv[1].strip().lower()

    # Walk-Forward Analysis 실행
    df, summary = run_walk_forward_analysis(
        account_id=account_id,
        lookback_periods=DEFAULT_LOOKBACK_PERIODS,
        test_months=DEFAULT_TEST_MONTHS,
        test_period_months=DEFAULT_TEST_PERIOD_MONTHS,
        max_workers=None,
    )

    # 결과 출력
    print_summary_table(summary, account_id)

    # 결과 저장
    account_dir = RESULTS_DIR / account_id
    save_results(df, summary, account_id, output_dir=account_dir)


if __name__ == "__main__":
    main()
