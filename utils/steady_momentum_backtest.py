"""Steady Momentum 월간 리밸런싱 백테스트.

방식
----
- 판정과 체결을 분리한다: **월말 전 거래일(T−1) 종가까지의 데이터**로 선정을
  계산하고, **월말(T) 종가**에 교체를 체결한다. 종가로 판정한 것을 같은 종가에
  체결하는 동시성 편향(look-ahead)을 없앤 것 — 실제로는 전일 밤 신호 계산 후
  다음날 종가 주문에 해당한다.
- 선정은 꾸준한 모멘텀 점수(연율화 상대기울기 × R²) 순 — 화면 선정과 같은
  ``rank_candidates`` 를 써서 두 화면이 항상 일치한다.
- 슬리피지는 편도(%)로, 리밸런싱에서 실제 매매되는 금액 전체에 부과한다:
  편출 전량 매도 + 편입 1/N 매수 + **유지 종목의 1/N 재조정 매매**(한 달간
  흘러간 비중과 목표 1/N 의 차이)까지 포함 — 완전 리밸런싱 모델과 비용이 일치한다.
- 벤치마크: 종목풀 설정(DB `pool_settings`)에 등록된 벤치마크 티커를 그대로 쓴다.
  미설정이면 fallback 없이 에러다.

한계(화면에 명시): 현재 종목풀 기준이라 상장폐지·풀 이탈 종목이 빠진
생존 편향이 있다.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from leverage.engine.backtest.ma_cross import max_drawdown_pct, sortino
from utils.pool_signal_backtest_service import get_max_backtest_months
from utils.steady_momentum_service import (
    POOL_CONFIGS,
    available_backtest_months,
    benchmark_info,
    load_benchmark_close,
    load_price_frames,
    load_settings,
    load_universe,
    month_last_two_trading_days,
    rank_candidates,
    select_candidates,
    validate_settings,
)

# 미국 풀 참고 지수 — 유사 컨셉 ETF(FMTM)와 같은 구간을 나란히 비교한다 (벤치마크 아님).
US_REFERENCE_TICKER = "FMTM"


def _rebalance_dates(benchmark_close: pd.Series, months: int) -> list[pd.Timestamp]:
    """월말 거래일 목록 — 마지막 항목은 최신 거래일(진행 중인 달 포함)."""
    index = benchmark_close.index
    month_ends = index.to_series().groupby(index.to_period("M")).max().tolist()
    if len(month_ends) < months + 1:
        raise ValueError(f"백테스트 {months}개월에 필요한 데이터가 부족합니다.")
    return month_ends[-(months + 1) :]


def _period_return(close: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    series = pd.to_numeric(close, errors="coerce").dropna()
    try:
        start_price = series.asof(start)
        end_price = series.asof(end)
    except Exception:
        return None
    if pd.isna(start_price) or pd.isna(end_price) or float(start_price) <= 0:
        return None
    return float(end_price) / float(start_price) - 1.0


def run_backtest(
    months: int,
    settings: dict[str, Any] | None = None,
    *,
    include_daily: bool,
) -> dict[str, Any]:
    """월간 리밸런싱 백테스트. 월별 전략 vs 벤치마크 수익률을 반환한다.

    ``include_daily`` 가 참일 때만 일별 행까지 만든다. 일별은 구간마다 종목별
    시계열을 재색인해야 하고 응답도 수천 행이 되므로, 화면에서 일간 탭을 볼 때만
    요청한다. 동작이 달라지는 값이라 기본값을 두지 않는다.
    """
    max_months = get_max_backtest_months()
    if not isinstance(months, int) or not 1 <= months <= max_months:
        raise ValueError(f"'months' 는 1~{max_months} 사이의 정수여야 합니다.")
    if settings is None:
        settings = load_settings()
    settings = validate_settings(settings)

    universe = load_universe(settings["pool"])
    name_by_ticker = {row["ticker"]: row["name"] for row in universe}

    def holding_label(ticker: str) -> str:
        """편입·편출 표시용 `종목명(티커)`. 이름을 모르면 티커만 쓴다."""
        name = name_by_ticker.get(ticker)
        return f"{name}({ticker})" if name else ticker

    frames = load_price_frames(universe)
    benchmark_close = load_benchmark_close(settings["pool"])

    # 실제 한계는 종목풀 데이터가 정한다 — 판정일 여유까지 반영해 여기서 다시 막는다.
    lookback_months = int(settings["lookback_months"])
    pool_max = available_backtest_months(benchmark_close, lookback_months)
    if months > pool_max:
        raise ValueError(
            f"룩백 {lookback_months}개월 기준으로 이 종목풀은 최대 {pool_max}개월까지 "
            f"백테스트할 수 있습니다 (요청 {months}개월)."
        )

    # 유사 컨셉 ETF(FMTM)를 참고 지수로 함께 계산한다.
    from utils.cache_utils import load_cached_frames_bulk_from_all_ticker_types

    reference_close: pd.Series | None = None
    reference_frame = load_cached_frames_bulk_from_all_ticker_types([US_REFERENCE_TICKER]).get(
        US_REFERENCE_TICKER
    )
    if reference_frame is not None and not reference_frame.empty:
        reference_close = pd.to_numeric(reference_frame["Close"], errors="coerce").dropna()
    dates = _rebalance_dates(benchmark_close, months)

    # 판정 시점 = 각 리밸런싱일(체결일)의 직전 거래일. 벤치마크 달력 기준.
    bench_index = benchmark_close.index
    signal_dates: list[pd.Timestamp] = []
    for date in dates[:-1]:
        prior = bench_index[bench_index < date]
        if len(prior) == 0:
            raise ValueError("판정 기준일(직전 거래일)을 구할 수 없습니다 — 데이터가 부족합니다.")
        signal_dates.append(prior[-1])

    # '예정' 행 — 마지막 데이터 날짜가 완결된 월말이면(예: 8/1 시점의 7/31),
    # 그 월말 종가 교체분(다음 달 포트폴리오)을 함께 계산해 표 맨 위에 보여준다.
    # 진행 중인 달(부분 월)에서는 마지막 행이 이미 현재 포트폴리오라 예정 행이 없다.
    pending_signal: pd.Timestamp | None = None
    pair = month_last_two_trading_days(
        POOL_CONFIGS[settings["pool"]]["country"], dates[-1].to_period("M")
    )
    if pair is not None:
        _, signal_calendar = pair
        if pd.Timestamp.now().normalize() > signal_calendar:
            prior = bench_index[bench_index <= signal_calendar]
            if len(prior) > 0:
                pending_signal = prior[-1]

    # 리밸런싱 시점별 후보 (모멘텀은 가격 캐시만 쓰므로 즉시 계산된다)
    candidates_by_date: list[list[dict[str, Any]]] = []
    all_signal_dates = signal_dates + ([pending_signal] if pending_signal is not None else [])
    for signal_date in all_signal_dates:
        candidates_by_date.append(
            select_candidates(universe, frames, settings, benchmark_close, as_of=signal_date)
        )

    slippage = float(settings["slippage_pct"]) / 100.0
    top_n = int(settings["top_n"])

    # 일간 표용 — 보유 구간 안에서 매일의 동일가중 포트폴리오 수익률.
    # 종가는 한 번만 정제해 재사용한다(구간마다 다시 정제하면 느리다).
    clean_closes: dict[str, pd.Series] = {}
    clean_benchmark: pd.Series | None = None
    clean_reference: pd.Series | None = None
    if include_daily:
        clean_closes = {
            ticker: pd.to_numeric(frame["Close"], errors="coerce").dropna()
            for ticker, frame in frames.items()
        }
        clean_benchmark = pd.to_numeric(benchmark_close, errors="coerce").dropna()
        clean_reference = (
            pd.to_numeric(reference_close, errors="coerce").dropna()
            if reference_close is not None
            else None
        )
    daily: list[dict[str, Any]] = []

    monthly: list[dict[str, Any]] = []
    strategy_returns: list[float] = []
    benchmark_returns: list[float] = []
    reference_returns: list[float] = []
    previous_holdings: set[str] = set()
    # 직전 보유 구간의 종목별 성장배수(1+수익률) — 리밸런싱 시점의 드리프트 비중 계산용.
    previous_growth: dict[str, float] = {}

    for position in range(months):
        start = dates[position]
        end = dates[position + 1]
        # 판정은 전 거래일(signal_dates) 기준, 체결·보유 구간은 월말 종가(start→end).
        scored = rank_candidates(candidates_by_date[position])
        holdings = [item["ticker"] for item in scored[: int(settings["top_n"])]]
        holdings_set = set(holdings)
        added_tickers = sorted(holdings_set - previous_holdings)
        removed_tickers = sorted(previous_holdings - holdings_set)
        target_weight = 1.0 / max(len(holdings_set), 1)

        # ── 리밸런싱 매매 금액(포트폴리오 대비 비율) ──
        # 유지 종목도 매월 1/N 로 재조정하므로, 드리프트 비중과 목표의 차이가 전부 매매다.
        if previous_holdings:
            growth = {ticker: previous_growth.get(ticker, 1.0) for ticker in previous_holdings}
            total_growth = sum(growth.values())
            drifted = {ticker: value / total_growth for ticker, value in growth.items()}
            sell_notional = sum(drifted[t] for t in removed_tickers) + sum(
                max(drifted[t] - target_weight, 0.0) for t in holdings_set & previous_holdings
            )
            buy_notional = sum(target_weight for t in added_tickers) + sum(
                max(target_weight - drifted[t], 0.0) for t in holdings_set & previous_holdings
            )
            traded_notional = sell_notional + buy_notional
            turnover_pct = round(len(added_tickers) / top_n * 100.0, 1)
        else:
            traded_notional = 1.0  # 첫 달은 전량 신규 매수 (매수측만)
            turnover_pct = None
        cost = slippage * traded_notional

        period_returns: dict[str, float] = {}
        for ticker in holdings:
            if ticker not in frames:
                continue
            value = _period_return(frames[ticker]["Close"], start, end)
            if value is not None:
                period_returns[ticker] = value
        gross = sum(period_returns.values()) / len(period_returns) if period_returns else None

        strategy_pct = (gross - cost) * 100.0 if gross is not None else None
        benchmark_return = _period_return(benchmark_close, start, end)
        benchmark_pct = benchmark_return * 100.0 if benchmark_return is not None else None
        reference_return = (
            _period_return(reference_close, start, end) if reference_close is not None else None
        )
        reference_pct = reference_return * 100.0 if reference_return is not None else None

        if strategy_pct is not None:
            strategy_returns.append(strategy_pct / 100.0)
        if benchmark_pct is not None:
            benchmark_returns.append(benchmark_pct / 100.0)
        if reference_pct is not None:
            reference_returns.append(reference_pct / 100.0)

        # ── 일간 행 ──
        # 이 구간(start→end)의 보유 종목은 고정이다. start 종가를 1 로 두고 매일의
        # 동일가중 포트폴리오 가치를 구한 뒤 전일 대비 변동률을 낸다. 리밸런싱 비용은
        # 구간 첫날에 한 번 반영한다(월간 계산과 같은 방식).
        window = bench_index[(bench_index > start) & (bench_index <= end)] if include_daily else []
        if len(window) > 0:
            ratios = []
            for ticker in holdings:
                series = clean_closes.get(ticker)
                if series is None:
                    continue
                base = series.asof(start)
                if pd.isna(base) or float(base) <= 0:
                    continue
                ratios.append(series.reindex(window, method="ffill") / float(base))
            portfolio = pd.concat(ratios, axis=1).mean(axis=1) if ratios else None

            def _daily_series(source: pd.Series | None) -> pd.Series | None:
                """구간 시작 종가를 1 로 정규화한 일별 가치."""
                if source is None:
                    return None
                base_value = source.asof(start)
                if pd.isna(base_value) or float(base_value) <= 0:
                    return None
                return source.reindex(window, method="ffill") / float(base_value)

            bench_curve = _daily_series(clean_benchmark)
            ref_curve = _daily_series(clean_reference)

            def _step(curve: pd.Series | None, position_index: int) -> float | None:
                """전일 대비(첫날은 구간 시작 종가 대비) 변동률(%)."""
                if curve is None:
                    return None
                value = curve.iloc[position_index]
                prior = 1.0 if position_index == 0 else curve.iloc[position_index - 1]
                if pd.isna(value) or pd.isna(prior) or float(prior) <= 0:
                    return None
                return (float(value) / float(prior) - 1.0) * 100.0

            for day_index, day in enumerate(window):
                strategy_day = _step(portfolio, day_index)
                if strategy_day is not None and day_index == 0:
                    strategy_day -= cost * 100.0
                daily.append(
                    {
                        "date": day.strftime("%Y-%m-%d"),
                        "strategy_pct": round(strategy_day, 2) if strategy_day is not None else None,
                        "benchmark_pct": (
                            round(value, 2) if (value := _step(bench_curve, day_index)) is not None else None
                        ),
                        "reference_pct": (
                            round(value, 2) if (value := _step(ref_curve, day_index)) is not None else None
                        ),
                    }
                )

        monthly.append(
            {
                "month": end.strftime("%Y-%m"),
                "strategy_pct": round(strategy_pct, 2) if strategy_pct is not None else None,
                "benchmark_pct": round(benchmark_pct, 2) if benchmark_pct is not None else None,
                "reference_pct": round(reference_pct, 2) if reference_pct is not None else None,
                "excess_pp": (
                    round(strategy_pct - benchmark_pct, 2)
                    if strategy_pct is not None and benchmark_pct is not None
                    else None
                ),
                "holdings_count": len(holdings),
                "turnover_pct": turnover_pct,
                # 이 달 시작(직전 월말 종가)에 교체한 종목 — 첫 달은 전량 편입.
                "added": [holding_label(ticker) for ticker in added_tickers],
                "removed": [holding_label(ticker) for ticker in removed_tickers],
            }
        )
        previous_holdings = holdings_set
        # 이번 구간 성장배수 저장 (수익률 없는 종목은 1.0 으로 유지 취급)
        previous_growth = {ticker: 1.0 + period_returns.get(ticker, 0.0) for ticker in holdings_set}

    # 예정 행 — 마지막 월말 종가에 실행될 교체 (수익률은 아직 없음)
    if pending_signal is not None:
        scored = rank_candidates(candidates_by_date[-1])
        pending_holdings = {item["ticker"] for item in scored[: int(settings["top_n"])]}
        monthly.append(
            {
                "month": (dates[-1].to_period("M") + 1).strftime("%Y-%m"),
                "strategy_pct": None,
                "benchmark_pct": None,
                "reference_pct": None,
                "excess_pp": None,
                "holdings_count": len(pending_holdings),
                "turnover_pct": round(len(pending_holdings - previous_holdings) / top_n * 100.0, 1),
                "added": [holding_label(t) for t in sorted(pending_holdings - previous_holdings)],
                "removed": [holding_label(t) for t in sorted(previous_holdings - pending_holdings)],
                "is_pending": True,
            }
        )

    def _summarize(returns: list[float]) -> tuple[float, float | None, float | None, float | None]:
        curve = pd.Series([1.0] + list(pd.Series(returns).add(1.0).cumprod()))
        total = (float(curve.iloc[-1]) - 1.0) * 100.0
        # CAGR — 월별 표본 수 기준 연율화 (12개월 미만이면 연 환산이라 과장될 수 있음)
        sample_months = len(returns)
        cagr = (
            ((1.0 + total / 100.0) ** (12.0 / sample_months) - 1.0) * 100.0
            if sample_months > 0 and total > -100.0
            else None
        )
        # 소르티노는 월별 수익률 기준 연율화 (레버리지 엔진 공용 함수 재사용).
        # 표본 3개 미만이거나 하락 달이 없으면 None → 화면에서 '-'.
        return (
            round(total, 2),
            max_drawdown_pct(curve),
            sortino(pd.Series(returns), periods_per_year=12),
            round(cagr, 1) if cagr is not None else None,
        )

    strategy_total, strategy_mdd, strategy_sortino, strategy_cagr = _summarize(strategy_returns)
    benchmark_total, benchmark_mdd, benchmark_sortino, benchmark_cagr = _summarize(benchmark_returns)
    if reference_close is not None and reference_returns:
        reference_total, reference_mdd, reference_sortino, reference_cagr = _summarize(reference_returns)
    else:
        reference_total = reference_mdd = reference_sortino = reference_cagr = None

    return {
        "start_date": dates[0].strftime("%Y-%m-%d"),
        "end_date": dates[-1].strftime("%Y-%m-%d"),
        "months": months,
        "strategy_total_pct": strategy_total,
        "benchmark_total_pct": benchmark_total,
        "strategy_mdd_pct": strategy_mdd,
        "benchmark_mdd_pct": benchmark_mdd,
        "strategy_sortino": strategy_sortino,
        "benchmark_sortino": benchmark_sortino,
        "strategy_cagr_pct": strategy_cagr,
        "benchmark_cagr_pct": benchmark_cagr,
        "reference_cagr_pct": reference_cagr,
        "benchmark_name": benchmark_info(settings["pool"])["name"],
        "benchmark_ticker": benchmark_info(settings["pool"])["ticker"],
        "reference_name": US_REFERENCE_TICKER if reference_close is not None else None,
        "reference_total_pct": reference_total,
        "reference_mdd_pct": reference_mdd,
        "reference_sortino": reference_sortino,
        # 최신 달이 위로 오게 뒤집는다 (화면 표)
        "monthly": list(reversed(monthly)),
        # 일간 표 — 최신 날짜가 위로 오게 뒤집는다
        "daily": list(reversed(daily)),
    }
