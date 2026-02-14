"""계정별 추천 실행 스크립트 (백테스트 기반).

백테스트를 실행하고 마지막 날(오늘)의 결과를 추천으로 사용합니다.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from utils.account_registry import (
    get_account_settings,
    get_benchmark_tickers,
    get_strategy_rules,
    list_available_accounts,
)
from utils.data_loader import MissingPriceDataError, get_latest_trading_day, prepare_price_data
from utils.formatters import format_pct_change, format_price, format_price_deviation
from utils.logger import get_app_logger
from utils.notification import (
    compose_recommendation_slack_message,
    send_recommendation_slack_notification,
)
from utils.recommendation_storage import save_recommendation_payload
from utils.report import render_table_eaw
from utils.settings_loader import load_common_settings
from utils.stock_list_io import get_etfs

RESULTS_DIR = Path(__file__).resolve().parent / "zaccounts"
logger = get_app_logger()


# ---------------------------------------------------------------------------
# RecommendationReport 호환 클래스 (기존 인터페이스 유지)
# ---------------------------------------------------------------------------


class RecommendationReport:
    """백테스트 결과 기반 추천 보고서."""

    def __init__(
        self,
        *,
        account_id: str,
        country_code: str,
        base_date: pd.Timestamp,
        recommendations: list[dict[str, Any]],
        summary_data: dict[str, Any] | None = None,
    ):
        self.account_id = account_id
        self.country_code = country_code
        self.base_date = base_date
        self.recommendations = recommendations
        self.report_date = datetime.now()
        self.summary_data = summary_data


# ---------------------------------------------------------------------------
# 백테스트 결과에서 추천 데이터 추출
# ---------------------------------------------------------------------------


def extract_recommendations_from_backtest(
    result: Any,
    *,
    ticker_meta: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """백테스트 결과에서 마지막 날(오늘) 추천 데이터를 추출합니다."""

    ticker_timeseries = getattr(result, "ticker_timeseries", {})
    result_ticker_meta = getattr(result, "ticker_meta", {})
    end_date = getattr(result, "end_date", None)

    if not ticker_timeseries or end_date is None:
        return []

    # 티커 메타정보 병합 (result에서 온 것 + 전달받은 것)
    merged_meta = {**result_ticker_meta}
    if ticker_meta:
        for k, v in ticker_meta.items():
            if k not in merged_meta:
                merged_meta[k] = v
            else:
                merged_meta[k] = {**merged_meta[k], **v}

    recommendations: list[dict[str, Any]] = []

    for ticker, df in ticker_timeseries.items():
        ticker_key = str(ticker).upper()
        if ticker_key == "CASH" or ticker_key.startswith("_"):
            continue

        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        # 마지막 날 데이터 추출
        if end_date in df.index:
            last_row = df.loc[end_date]
        else:
            last_row = df.iloc[-1]

        # 전날 데이터 (일간 수익률 계산용)
        # 방법 1: end_date 이전 데이터 중 마지막 행
        df_before_end = df[df.index < end_date]
        # 방법 2: 충분히 가까운 날짜 (df에 end_date가 없을 수 있음)
        if df_before_end.empty and len(df) >= 2:
            # end_date가 df의 마지막 날짜와 같다면, 두 번째 마지막 행 사용
            prev_row = df.iloc[-2]
        elif not df_before_end.empty:
            prev_row = df_before_end.iloc[-1]
        else:
            prev_row = None

        # 메타 정보
        meta = merged_meta.get(ticker_key, merged_meta.get(ticker, {}))
        name = meta.get("name", ticker_key)

        # [UPDATE] stock_note가 있으면 이름에 병합 (예: 종목명(노트내용))
        # UI 오버레이 복구용 원본 노트도 stock_note 필드로 저장
        stock_note = meta.get("note")
        if stock_note:
            name = f"{name}({stock_note})"

        # 기본 값 추출
        price = _safe_float(last_row.get("price"))
        shares = _safe_float(last_row.get("shares"), 0)
        avg_cost = _safe_float(last_row.get("avg_cost"))
        score = _safe_float(last_row.get("score"))
        rsi_score = _safe_float(last_row.get("rsi_score"))
        filter_val = _safe_float(last_row.get("filter"))
        decision = str(last_row.get("decision", "")).upper() or "WAIT"
        note = str(last_row.get("note", "") or "")

        # nav_price와 price_deviation 계산 (메타에서 가져오거나 계산)
        nav_price = meta.get("nav_price") or meta.get("nav") or None
        price_deviation = None
        if nav_price and price and price > 0:
            price_deviation = ((price - nav_price) / nav_price) * 100

        # 일간 수익률 계산
        # 주의: 백테스트 엔진에서 데이터 없는 날은 price = avg_cost로 설정됨
        # 실제 가격 변동을 계산하려면 price != avg_cost인 날을 찾아야 함
        daily_pct = 0.0
        if prev_row is not None:
            prev_price = _safe_float(prev_row.get("price"))
            prev_avg_cost = _safe_float(prev_row.get("avg_cost"))

            # prev_price가 avg_cost와 같으면 (데이터 없음 표시), 더 이전 날짜를 찾음
            if prev_price and prev_avg_cost and abs(prev_price - prev_avg_cost) < 0.001:
                # df에서 price != avg_cost인 마지막 행 찾기
                df_before_prev = df[df.index < end_date]
                for idx in reversed(df_before_prev.index):
                    row = df_before_prev.loc[idx]
                    row_price = _safe_float(row.get("price"))
                    row_avg_cost = _safe_float(row.get("avg_cost"))
                    if row_price and row_avg_cost and abs(row_price - row_avg_cost) >= 0.001:
                        prev_price = row_price
                        break

            if prev_price and prev_price > 0 and price:
                daily_pct = ((price / prev_price) - 1.0) * 100.0

        # 평가 수익률 계산
        evaluation_pct = 0.0
        if shares > 0 and avg_cost and avg_cost > 0 and price:
            cost_basis = avg_cost * shares
            pv = price * shares
            evaluation_pct = ((pv - cost_basis) / cost_basis) * 100.0

        # 보유일 계산 (연속 보유 기간)
        holding_days = 0
        if shares > 0:
            holding_days = _calculate_holding_days(df, end_date)

        # streak (filter 값 사용)
        streak = int(filter_val) if filter_val and filter_val > 0 else 0

        # phrase (note 사용)
        phrase = note

        # 상태 결정
        state = _decision_to_state(decision, shares)

        # 수익률 및 드로우다운 계산
        df_up_to_end = df[df.index <= end_date]
        return_1w = return_1m = return_3m = 0.0
        drawdown_from_high = 0.0
        trend_prices = []

        if not df_up_to_end.empty:
            historical_prices = df_up_to_end["price"]
            current_p = _safe_float(historical_prices.iloc[-1])

            def _get_ret(days: int) -> float:
                if len(historical_prices) > days and current_p:
                    prev_p = _safe_float(historical_prices.iloc[-(days + 1)])
                    if prev_p and prev_p > 0:
                        return (current_p / prev_p - 1.0) * 100.0

                # [Fallback] 데이터가 살짝 부족해도 전체 기간이 대략 맞으면(예: 12개월) 가장 오래된 데이터 사용
                # 1년 영업일은 보통 252일 전후이므로, 240일 이상이면 1년치로 간주
                if days == 252 and len(historical_prices) >= 240 and current_p:
                    prev_p = _safe_float(historical_prices.iloc[0])
                    if prev_p and prev_p > 0:
                        return (current_p / prev_p - 1.0) * 100.0

                return 0.0

            return_1w = _get_ret(5)
            # return_2w = _get_ret(10)  # Removed as per request
            return_1m = _get_ret(20)
            return_3m = _get_ret(60)
            return_6m = _get_ret(126)
            return_12m = _get_ret(252)

            # 고점대비 하락폭
            max_p = _safe_float(historical_prices.max())
            if max_p and max_p > 0 and current_p:
                drawdown_from_high = (current_p / max_p - 1.0) * 100.0

            # 추세 데이터 (최근 60일)
            trend_prices = historical_prices.iloc[-60:].tolist()

        recommendations.append(
            {
                "ticker": ticker_key,
                "name": name,
                "stock_note": stock_note,  # UI 오버레이 복구용
                "state": state,
                "decision": decision,
                "price": price,
                "nav_price": nav_price,
                "shares": shares,
                "score": score,
                "rsi_score": rsi_score,
                "streak": streak,
                "daily_pct": daily_pct,
                "evaluation_pct": evaluation_pct,
                "price_deviation": price_deviation,
                "holding_days": holding_days,
                "return_1w": return_1w,
                # "return_2w": return_2w,
                "return_1m": return_1m,
                "return_3m": return_3m,
                "return_6m": return_6m,
                "return_12m": return_12m,
                "drawdown_from_high": drawdown_from_high,
                "trend_prices": trend_prices,
                "phrase": phrase,
                "base_date": end_date,
            }
        )

    # 점수로 정렬 (None은 마지막으로)
    recommendations.sort(key=lambda x: (x.get("score") is None, -(x.get("score") or 0)))

    # 순위 부여 (보유/신규 -> 보유N, 그외 -> 대기N)
    held_count = 0
    wait_count = 0

    for rec in recommendations:
        shares = rec.get("shares", 0) or 0
        decision = str(rec.get("decision", "")).upper()

        # [User Request] 웹 화면 기준: 보유중이거나(shares > 0) 신규 매수(BUY)인 경우 '보유' 그룹
        if shares > 0 or decision == "BUY":
            held_count += 1
            rec["rank"] = f"보유{held_count}"
        else:
            wait_count += 1
            rec["rank"] = f"대기{wait_count}"

    return recommendations


def _safe_float(value: Any, default: float | None = None) -> float | None:
    """값을 float으로 변환. pd.isna 체크 포함."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _calculate_holding_days(df: pd.DataFrame, target_date: pd.Timestamp) -> int:
    """보유일 계산: target_date까지 연속으로 shares > 0인 날짜 수."""
    if df.empty:
        return 0

    # target_date 이전 데이터만 (포함)
    df_up_to_date = df[df.index <= target_date]
    if df_up_to_date.empty:
        return 0

    # 역순으로 보유일 계산
    days = 0
    for idx in reversed(df_up_to_date.index):
        row = df_up_to_date.loc[idx]
        shares = _safe_float(row.get("shares"), 0)
        if shares and shares > 0:
            days += 1
        else:
            break

    return days


def _decision_to_state(decision: str, shares: float) -> str:
    """decision 값을 state로 변환합니다."""
    decision_upper = str(decision).upper()

    # [User Request] 교체매수/매도는 별도 상태로 표시
    if decision_upper in (
        "BUY",
        "SELL",
        "BUY_REPLACE",
        "SELL_REPLACE",
        "SELL_TREND",
        "SELL_RSI",
        "SELL_STOP",
        "CUT_STOPLOSS",
        "SELL_TRAILING",
        "SELL_MOMENTUM",
    ):
        return decision_upper
    elif shares and shares > 0:
        return "HOLD"
    else:
        return "WAIT"


def _enrich_with_nav_data(
    recommendations: list[dict[str, Any]],
    tickers: list[str],
) -> list[dict[str, Any]]:
    """한국 ETF의 경우 네이버 API에서 Nav와 괴리율을 가져와 채웁니다."""
    from utils.data_loader import fetch_naver_etf_inav_snapshot

    try:
        snapshot = fetch_naver_etf_inav_snapshot(tickers)
    except Exception as e:
        logger.warning("네이버 ETF Nav 스냅샷 조회 실패: %s", e)
        return recommendations

    for rec in recommendations:
        ticker = str(rec.get("ticker", "")).upper()
        if ticker in snapshot:
            nav_info = snapshot[ticker]
            rec["nav_price"] = nav_info.get("nav")
            rec["price_deviation"] = nav_info.get("deviation")
            # 실시간 가격으로 덮어쓰기 (옵션)
            if nav_info.get("nowVal"):
                rec["price"] = nav_info.get("nowVal")

    return recommendations


# ---------------------------------------------------------------------------
# 추천 리포트 생성 (백테스트 실행 포함)
# ---------------------------------------------------------------------------


def generate_recommendation_report(
    account_id: str,
    *,
    date_str: str | None = None,
) -> RecommendationReport:
    """백테스트를 실행하고 마지막 날 결과를 추천 보고서로 반환합니다."""

    account_settings = get_account_settings(account_id)
    country_code = (account_settings.get("country_code") or account_id).strip().lower()
    strategy_cfg = account_settings.get("strategy", {}) or {}
    backtest_start_date_str = strategy_cfg.get("BACKTEST_START_DATE", "2025-01-01")

    try:
        start_date = pd.to_datetime(backtest_start_date_str)
    except Exception:
        start_date = pd.Timestamp.now().normalize() - pd.DateOffset(months=12)

    # 종료일 결정
    if date_str:
        end_date = pd.to_datetime(date_str)
    else:
        end_date = get_latest_trading_day(country_code)
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.Timestamp.now().normalize()

    # [전략 기준일 설정] 장중 변동 방지 로직
    from datetime import time as dt_time

    from config import MARKET_SCHEDULES, USE_REALTIME_RECOMMENDATION

    strategy_end_date = end_date

    # 날짜가 강제 지정되지 않았고, 실시간 반영 옵션이 꺼져있는 경우
    if not date_str and not USE_REALTIME_RECOMMENDATION:
        schedule = MARKET_SCHEDULES.get(country_code, {})
        timezone_str = schedule.get("timezone", "Asia/Seoul")
        market_close_time = schedule.get("close", dt_time(15, 30))

        try:
            now = pd.Timestamp.now(tz=timezone_str)
        except Exception:
            # Fallback
            now = pd.Timestamp.now().tz_localize(timezone_str)

        # 오늘 날짜이고, 아직 장 마감 전이라면
        if end_date.date() == now.date() and now.time() < market_close_time:
            # 전략 기준일을 하루 전으로 설정 (전일 종가 기준 고정)
            strategy_end_date = end_date - pd.Timedelta(days=1)
            logger.info(
                f"[장중 고정] 전략 기준일을 전일({strategy_end_date.strftime('%Y-%m-%d')})로 고정합니다. (수익률은 실시간 반영)"
            )

    # 전략 규칙 로드
    strategy_rules = get_strategy_rules(account_id)
    warmup_days = strategy_rules.ma_period

    # 종목 로드 (한 번만)
    etf_universe = get_etfs(account_id)
    universe_tickers = [etf["ticker"] for etf in etf_universe if etf.get("ticker")]
    universe_meta = {etf["ticker"]: etf for etf in etf_universe if etf.get("ticker")}
    benchmark_tickers = get_benchmark_tickers(account_settings)
    tickers = sorted({*(str(t).strip().upper() for t in universe_tickers if t), *benchmark_tickers})

    # 캐시 설정
    common_settings = load_common_settings()
    cache_seed_raw = (common_settings or {}).get("CACHE_START_DATE")
    cache_seed_dt = None
    if cache_seed_raw:
        try:
            cache_seed_dt = pd.to_datetime(cache_seed_raw).normalize()
        except Exception:
            pass

    # 1년 수익률 계산 등을 위한 최소 데이터 확보 (400일)
    min_days_needed = 400
    min_start_date_for_stats = end_date - pd.DateOffset(days=min_days_needed)

    prefetch_start = start_date - pd.DateOffset(days=warmup_days)

    # 통계용 최소 시작일과 비교하여 더 이른 날짜 선택
    if min_start_date_for_stats < prefetch_start:
        prefetch_start = min_start_date_for_stats

    if cache_seed_dt is not None and cache_seed_dt < prefetch_start:
        prefetch_start = cache_seed_dt
    date_range = [prefetch_start.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]

    # 가격 데이터 로드
    prefetched_map, missing = prepare_price_data(
        tickers=tickers,
        country=country_code,
        start_date=date_range[0],
        end_date=date_range[1],
        warmup_days=0,
        account_id=account_id,
    )

    if missing:
        raise MissingPriceDataError(
            country=country_code,
            start_date=date_range[0],
            end_date=date_range[1],
            tickers=missing,
        )

    # 백테스트 실행 (ETF 유니버스 전달하여 중복 로딩 방지)
    from logic.backtest.account import run_account_backtest

    # [Data Slicing] 전략 기준일 이후의 데이터(오늘 시가 등)가 백테스트에 영향을 주지 않도록 잘라냄
    # (엔진이 '다음날 시가'를 참조하여 체결가를 계산하는 로직 때문)
    backtest_data = {ticker: df[df.index <= strategy_end_date] for ticker, df in prefetched_map.items()}

    result = run_account_backtest(
        account_id,
        prefetched_data=backtest_data,
        prefetched_etf_universe=etf_universe,
        quiet=True,  # 백테스트 로그 출력 억제
        override_settings={"end_date": strategy_end_date.strftime("%Y-%m-%d")},
    )

    # 마지막 날 추천 데이터 추출
    recommendations = extract_recommendations_from_backtest(result, ticker_meta=universe_meta)

    # 전체 기간 데이터(prefetched_map)를 이용하여 기간별 수익률(6m, 12m 등) 재계산/보강
    # [수익률 표시] 전략은 과거 기준이라도, 수익률은 현재(end_date) 기준으로 표시
    recommendations = _enrich_with_period_returns(
        recommendations,
        prefetched_map,
        base_date=end_date,
    )

    # 한국 종목의 경우 Nav와 괴리율을 네이버 API에서 가져옴
    if country_code in ("kor", "kr"):
        recommendations = _enrich_with_nav_data(recommendations, universe_tickers)

    return RecommendationReport(
        account_id=account_id,
        country_code=country_code,
        base_date=result.end_date,
        recommendations=recommendations,
        summary_data=result.summary if hasattr(result, "summary") else None,
    )


# ---------------------------------------------------------------------------
# CLI 유틸리티
# ---------------------------------------------------------------------------


def print_run_header(account_id: str, *, date_str: str | None) -> None:
    """추천 실행 헤더를 출력합니다."""
    banner = f"=== {account_id.upper()} 추천 생성 ==="
    logger.info("%s", banner)
    logger.info("기준일: %s", date_str or "auto (latest trading day)")


def print_result_summary(
    items: list[dict[str, Any]],
    account_id: str,
    date_str: str | None = None,
) -> None:
    """추천 결과 요약을 출력합니다."""

    if not items:
        logger.warning("%s에 대한 결과가 없습니다.", account_id.upper())
        return

    state_counts = Counter(item.get("state", "UNKNOWN") for item in items)
    state_summary = ", ".join(f"{state}: {count}" for state, count in sorted(state_counts.items()))

    base_date = items[0].get("base_date") if items else (date_str or "N/A")

    logger.info("=== %s 추천 요약 (기준일: %s) ===", account_id.upper(), base_date)

    if state_summary:
        logger.info("상태 요약: %s", state_summary)
    buy_count = sum(1 for item in items if item.get("state") == "BUY")
    logger.info("매수 추천: %d개, 대기: %d개", buy_count, len(items) - buy_count)
    logger.info("결과가 성공적으로 생성되었습니다. (총 %d개 항목)", len(items))


def dump_recommendation_log(
    report: RecommendationReport,
    *,
    results_dir: Path | str | None = None,
) -> Path:
    """추천 결과를 로그 파일로 저장합니다."""

    account_id = report.account_id
    base_date = report.base_date
    recommendations = report.recommendations
    country_code = report.country_code

    # 기본 디렉토리 설정
    if results_dir is None:
        base_dir = Path(__file__).parent / "zaccounts" / account_id / "results"
    else:
        base_dir = Path(results_dir) / account_id / "results"

    base_dir.mkdir(parents=True, exist_ok=True)

    # 파일명
    date_str = datetime.now().strftime("%Y-%m-%d")
    path = base_dir / f"recommend_{date_str}.log"

    lines: list[str] = []

    # 헤더
    lines.append(f"추천 로그 생성: {pd.Timestamp.now().isoformat(timespec='seconds')}")
    base_date_str = base_date.strftime("%Y-%m-%d") if hasattr(base_date, "strftime") else str(base_date)
    lines.append(f"계정: {account_id.upper()} | 기준일: {base_date_str}")
    lines.append("")

    # 상태 카운트
    state_counts = Counter(item.get("state", "UNKNOWN") for item in recommendations)
    lines.append("=== 상태 요약 ===")
    for state, count in sorted(state_counts.items()):
        lines.append(f"  {state}: {count}개")
    lines.append("")

    # 테이블
    lines.append("=== 추천 목록 ===")
    lines.append("")

    country_lower = (country_code or "").strip().lower()
    nav_mode = country_lower in {"kr", "kor"}
    show_deviation = country_lower in {"kr", "kor"}

    headers = ["#", "티커", "종목명", "상태", "보유일", "일간(%)", "평가(%)", "현재가"]
    # [User Request] 현재가 - 괴리율 - Nav
    if show_deviation:
        headers.append("괴리율")
    if nav_mode:
        headers.append("Nav")

    # [User Request] 1주 - 1달 - 3달 - 6달 - 12달
    headers.extend(["1주(%)", "1달(%)", "3달(%)", "6달(%)", "12달(%)", "고점대비"])
    headers.extend(["점수", "RSI", "지속", "문구"])

    aligns = ["right", "left", "left", "left", "center", "right", "right", "right", "right"]
    if show_deviation:
        aligns.append("right")
    if nav_mode:
        aligns.append("right")
    aligns.extend(["right", "right", "right", "right", "right", "right"])  # Returns(5) & Drawdown
    aligns.extend(["right", "right", "right", "left"])

    rows: list[list[str]] = []
    for item in recommendations:
        rank = item.get("rank", 0)
        ticker = item.get("ticker", "-")
        name = item.get("name", "-")

        state = item.get("state", "-")
        holding_days = item.get("holding_days", 0)
        daily_pct = item.get("daily_pct", 0)
        evaluation_pct = item.get("evaluation_pct", 0)
        price = item.get("price")
        nav_price = item.get("nav_price")
        price_deviation = item.get("price_deviation")
        score = item.get("score", 0)
        rsi_score = item.get("rsi_score", 0)
        streak = item.get("streak", 0)
        phrase = item.get("phrase", "")

        return_1w = item.get("return_1w", 0)
        # return_2w = item.get("return_2w", 0)
        return_1m = item.get("return_1m", 0)
        return_3m = item.get("return_3m", 0)
        return_6m = item.get("return_6m", 0)
        return_12m = item.get("return_12m", 0)
        drawdown_from_high = item.get("drawdown_from_high", 0)

        row = [
            str(rank),
            ticker,
            name,
            state,
            str(holding_days) if holding_days > 0 else "-",
            format_pct_change(daily_pct),
            format_pct_change(evaluation_pct) if evaluation_pct != 0 else "-",
            format_price(price, country_code),
        ]
        if show_deviation:
            row.append(format_price_deviation(price_deviation))
        if nav_mode:
            row.append(format_price(nav_price, country_code))

        row.extend(
            [
                format_pct_change(return_1w),
                # format_pct_change(return_2w),
                format_pct_change(return_1m),
                format_pct_change(return_3m),
                format_pct_change(return_6m),
                format_pct_change(return_12m),
                format_pct_change(drawdown_from_high),
            ]
        )

        row.extend(
            [
                f"{score:.1f}" if isinstance(score, (int, float)) else "-",
                f"{rsi_score:.1f}" if isinstance(rsi_score, (int, float)) else "-",
                f"{streak}일" if streak > 0 else "-",
                phrase,
            ]
        )
        rows.append(row)

    table_lines = render_table_eaw(headers, rows, aligns)
    lines.extend(table_lines)
    lines.append("")

    with path.open("w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")

    return path


# ---------------------------------------------------------------------------
# CLI 메인
# ---------------------------------------------------------------------------


def _available_account_choices() -> list[str]:
    choices = list_available_accounts()
    if not choices:
        raise SystemExit("계정 설정(JSON)이 존재하지 않습니다. zaccounts/account/*.json 파일을 확인하세요.")
    return choices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MomentumEtf 계정 추천 실행기 (백테스트 기반)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("account", choices=_available_account_choices(), help="실행할 계정 ID")
    parser.add_argument(
        "--date",
        help="YYYY-MM-DD 형식의 기준일 (미지정 시 최신 거래일)",
    )
    parser.add_argument(
        "--output",
        help="결과 JSON 저장 경로",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    account_id = args.account.lower()

    try:
        account_settings = get_account_settings(account_id)
        get_strategy_rules(account_id)
    except Exception as exc:
        parser.error(f"계정 설정을 로드하는 중 오류가 발생했습니다: {exc}")

    account_country = str((account_settings or {}).get("country_code", "") or "")

    print_run_header(account_id, date_str=args.date)
    start_time = time.time()

    try:
        report = generate_recommendation_report(account_id=account_id, date_str=args.date)
    except MissingPriceDataError as exc:
        logger.error(str(exc))
        raise SystemExit(1)

    if not report.recommendations:
        logger.warning("%s에 대한 추천 결과가 비어 있습니다.", account_id.upper())
        return

    duration = time.time() - start_time
    items = list(report.recommendations)

    print_result_summary(items, account_id, args.date)

    # MongoDB 저장
    try:
        meta = save_recommendation_payload(
            items,
            account_id=account_id,
            country_code=report.country_code,
            base_date=report.base_date,
            summary=report.summary_data,
        )
        logger.info(
            "✅ %s 추천 결과를 MongoDB에 저장했습니다. document_id=%s",
            account_id.upper(),
            meta.get("document_id") if isinstance(meta, dict) else meta,
        )
    except Exception:
        logger.error(
            "기본 추천 결과 저장에 실패했습니다 (account=%s)",
            account_id,
            exc_info=True,
        )

    # 커스텀 JSON 저장
    if args.output:
        import json

        custom_path = Path(args.output)
        custom_path.parent.mkdir(parents=True, exist_ok=True)
        with custom_path.open("w", encoding="utf-8") as fp:
            json.dump(items, fp, ensure_ascii=False, indent=2, default=str)
        logger.info("📄 커스텀 JSON을 '%s'에 저장했습니다.", custom_path)

    # 로그 파일 저장
    try:
        log_path = dump_recommendation_log(report, results_dir=RESULTS_DIR)
        logger.info("✅ 추천 로그를 '%s'에 저장했습니다.", log_path)
    except Exception:
        logger.error("추천 로그 저장에 실패했습니다 (account=%s)", account_id, exc_info=True)

    # Slack 알림
    slack_payload = compose_recommendation_slack_message(
        account_id,
        report,
        duration=duration,
    )

    target_country = (getattr(report, "country_code", "") or account_country or "").strip().lower()
    notified = send_recommendation_slack_notification(
        account_id,
        slack_payload,
    )

    base_date_str = (
        report.base_date.strftime("%Y-%m-%d") if hasattr(report.base_date, "strftime") else str(report.base_date)
    )
    if notified:
        logger.info(
            "[%s/%s] Slack 알림 전송이 완료되었습니다 (소요 %.1fs)",
            (target_country or account_country or "").upper() or "UNKNOWN",
            base_date_str,
            duration,
        )
    else:
        logger.info(
            "[%s/%s] Slack 알림이 전송되지 않았습니다.",
            (target_country or account_country or "").upper() or "UNKNOWN",
            base_date_str,
        )

    elapsed_total = int(time.time() - start_time)
    hours, remainder = divmod(elapsed_total, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}시간")
    if minutes:
        parts.append(f"{minutes}분")
    if seconds or not parts:
        parts.append(f"{seconds}초")
    logger.info(
        "[%s] 총 소요 시간: %s",
        account_id.upper(),
        " ".join(parts),
    )


def _enrich_with_period_returns(
    recommendations: list[dict[str, Any]],
    prefetched_map: dict[str, pd.DataFrame],
    base_date: pd.Timestamp,
) -> list[dict[str, Any]]:
    """추천 목록에 기간별 수익률을 prefetch된 전체 데이터를 이용하여 다시 계산합니다."""

    # 1주, 1달, 3달, 6달, 12달
    periods = {
        "return_1w": 7,
        "return_1m": 30,
        "return_3m": 90,
        "return_6m": 180,
        "return_12m": 365,
    }

    for rec in recommendations:
        ticker = rec.get("ticker")
        if not ticker:
            continue

        df = prefetched_map.get(ticker)
        if df is None or df.empty:
            continue

        # 현재가 결정
        current_price = rec.get("price")
        if not current_price:
            if base_date in df.index:
                row = df.loc[base_date]
                current_price = _safe_float(row.get("close") or row.get("Close"))
            elif not df.empty:
                row = df.iloc[-1]
                current_price = _safe_float(row.get("close") or row.get("Close"))

        if not current_price:
            continue

        # 과거 가격 조회 및 수익률 갱신
        for key, days in periods.items():
            target_date = base_date - pd.Timedelta(days=days)

            try:
                if target_date < df.index[0]:
                    continue

                idx = df.index.get_indexer([target_date], method="pad")[0]
                if idx >= 0:
                    row = df.iloc[idx]
                    prev_price = _safe_float(row.get("close") or row.get("Close"))

                    if prev_price and prev_price > 0:
                        ret = ((current_price - prev_price) / prev_price) * 100.0
                        rec[key] = ret
            except Exception:
                pass

    return recommendations


if __name__ == "__main__":
    main()
