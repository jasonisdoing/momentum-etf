"""Steady Momentum 월간 리밸런싱 백테스트.

방식
----
- 판정과 체결을 분리한다: **월말 전 거래일(T−1) 종가까지의 데이터**로 선정을
  계산하고, **월말(T) 종가**에 교체를 체결한다. 종가로 판정한 것을 같은 종가에
  체결하는 동시성 편향(look-ahead)을 없앤 것 — 실제로는 전일 밤 신호 계산 후
  다음날 종가 주문에 해당한다.
- 선정은 꾸준한 모멘텀 점수(연율화 상대기울기 × R²) 순 — 화면 선정과 같은
  ``rank_candidates`` 를 써서 두 화면이 항상 일치한다.
- 슬롯 고정 비중: 종목당 1/top_n 로 배분하고, 자격 종목이 top_n 보다 적으면
  **빈 슬롯은 현금(수익 0)** 으로 남긴다 — 약세로 후보가 줄면 자동 현금 방어.
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
    available_backtest_months,
    benchmark_info,
    load_benchmark_close,
    load_price_frames,
    load_settings,
    load_universe,
    month_last_two_trading_days,
    pool_info,
    rank_candidates,
    sector_industry_map,
    select_candidates,
    select_top,
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
    stop_loss_exit: bool,
) -> dict[str, Any]:
    """월간 리밸런싱 백테스트. 월별 전략 vs 벤치마크 수익률을 반환한다.

    ``include_daily`` 가 참일 때만 일별 행까지 만든다. 일별은 구간마다 종목별
    시계열을 재색인해야 하고 응답도 수천 행이 되므로, 화면에서 일간 탭을 볼 때만
    요청한다. 동작이 달라지는 값이라 기본값을 두지 않는다.

    ``stop_loss_exit`` (단기이격 손절): 참이면 보유 구간 중 **종가가 단기 이평선
    아래로 처음 내려간 날** 그 종목만 종가 매도하고 월말까지 현금으로 둔다(편도
    슬리피지 1회 추가). 이평선 일수는 진입 필터와 같은 전략 전용 설정이라 새
    파라미터가 없다. 다음 교체일에는 정상 재선정한다.
    """
    max_months = get_max_backtest_months()
    if not isinstance(months, int) or not 1 <= months <= max_months:
        raise ValueError(f"'months' 는 1~{max_months} 사이의 정수여야 합니다.")
    if settings is None:
        settings = load_settings()
    settings = validate_settings(settings)
    pool = str(settings["pool"])

    universe = load_universe(pool)
    name_by_ticker = {row["ticker"]: row["name"] for row in universe}

    def holding_label(ticker: str) -> str:
        """편입·편출 표시용 `종목명(티커)`. 이름을 모르면 티커만 쓴다."""
        name = name_by_ticker.get(ticker)
        return f"{name}({ticker})" if name else ticker

    frames = load_price_frames(universe)
    benchmark_close = load_benchmark_close(pool)

    # 실제 한계는 종목풀 데이터가 정한다 — 판정일 여유까지 반영해 여기서 다시 막는다.
    pool_max = available_backtest_months(benchmark_close, int(settings["long_ma_days"]))
    if months > pool_max:
        raise ValueError(
            f"장기 이평선 기준으로 이 종목풀은 최대 {pool_max}개월까지 "
            f"백테스트할 수 있습니다 (요청 {months}개월)."
        )

    # 참고 지수 — 미국 풀은 유사 컨셉 ETF(FMTM)를 나란히 보여준다
    # (벤치마크가 아니며 선정에 관여하지 않는다).
    from utils.cache_utils import load_cached_frames_bulk_from_all_ticker_types

    reference_close: pd.Series | None = None
    reference_name: str | None = None
    if pool_info(pool)["country"] == "us":
        reference_frame = load_cached_frames_bulk_from_all_ticker_types([US_REFERENCE_TICKER]).get(
            US_REFERENCE_TICKER
        )
        if reference_frame is not None and not reference_frame.empty:
            reference_close = pd.to_numeric(reference_frame["Close"], errors="coerce").dropna()
            reference_name = US_REFERENCE_TICKER
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
        pool_info(pool)["country"], dates[-1].to_period("M")
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
            select_candidates(universe, frames, settings, as_of=signal_date)
        )

    slippage = float(settings["slippage_pct"]) / 100.0
    top_n = int(settings["top_n"])

    # 손절 판정용 단기 이평선 — 진입 필터와 같은 전략 전용 설정/공통 헬퍼.
    from utils.moving_averages import calculate_moving_average

    short_ma_days = int(settings["short_ma_days"])
    _stop_cache: dict[str, tuple[pd.Series, pd.Series]] = {}

    def _close_and_short_ma(ticker: str) -> tuple[pd.Series, pd.Series] | None:
        if ticker in _stop_cache:
            return _stop_cache[ticker]
        frame = frames.get(ticker)
        if frame is None or frame.empty:
            return None
        close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
        short_ma = calculate_moving_average(close, short_ma_days, min_periods=short_ma_days)
        _stop_cache[ticker] = (close, short_ma)
        return _stop_cache[ticker]

    def _stop_day(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Timestamp | None:
        """보유 구간(start, end] 에서 종가 < 단기 이평이 처음 되는 날. 없으면 None."""
        pair = _close_and_short_ma(ticker)
        if pair is None:
            return None
        close, short_ma = pair
        mask = (close.index > start) & (close.index <= end)
        window_close = close[mask]
        window_ma = short_ma[mask]
        below = window_close[(window_ma.notna()) & (window_close < window_ma)]
        return below.index[0] if not below.empty else None
    # 업종 상한 — 선정 화면과 같은 규칙으로 상위 종목을 고른다.
    max_per_industry = int(settings["max_per_industry"])
    industry_by_ticker = {
        ticker: meta["industry"] for ticker, meta in sector_industry_map(pool).items()
    }

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
    # 수익인출(고정원금) 누적 순수익(%) — 월 수익률의 산술 합 (아래 monthly 루프 주석 참고).
    harvest_cum_pct = 0.0
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
        holdings = [
            item["ticker"]
            for item in select_top(scored, top_n, max_per_industry, industry_by_ticker)
        ]
        holdings_set = set(holdings)
        added_tickers = sorted(holdings_set - previous_holdings)
        removed_tickers = sorted(previous_holdings - holdings_set)
        # 슬롯 고정 비중 — 선정이 top_n 보다 적으면 빈 슬롯은 현금(수익 0)으로 남긴다.
        target_weight = 1.0 / top_n

        # ── 리밸런싱 매매 금액(포트폴리오 대비 비율) ──
        # 유지 종목도 매월 1/N 로 재조정하므로, 드리프트 비중과 목표의 차이가 전부 매매다.
        # 현금 슬롯(__CASH__)은 비중 분모에는 들어가지만 매매 비용은 없다.
        if position > 0:
            growth = {ticker: previous_growth.get(ticker, 1.0) for ticker in previous_holdings}
            if "__CASH__" in previous_growth:
                growth["__CASH__"] = previous_growth["__CASH__"]
            total_growth = sum(growth.values()) or 1.0
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
            traded_notional = len(holdings) * target_weight  # 첫 달은 투자분만 신규 매수
            turnover_pct = None
        cost = slippage * traded_notional

        period_returns: dict[str, float] = {}
        stops: dict[str, pd.Timestamp] = {}
        for ticker in holdings:
            if ticker not in frames:
                continue
            exit_date = _stop_day(ticker, start, end) if stop_loss_exit else None
            value = _period_return(frames[ticker]["Close"], start, exit_date or end)
            if value is not None:
                period_returns[ticker] = value
                if exit_date is not None:
                    stops[ticker] = exit_date
        # 슬롯 모델: 보유 종목은 각 1/N, 빈 슬롯은 현금(0%) — 분모는 항상 top_n.
        if not holdings:
            gross: float | None = 0.0  # 전량 현금인 달
        elif period_returns:
            gross = sum(period_returns.values()) / top_n
        else:
            gross = None  # 보유 종목의 가격 데이터가 전혀 없음 — 데이터 문제를 그대로 드러낸다
        # 손절 매도는 구간 중 추가 매매 — 그 종목 비중(1/N)에 편도 슬리피지를 물린다.
        cost += slippage * target_weight * len(stops)

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
                curve = series.reindex(window, method="ffill") / float(base)
                stop = stops.get(ticker)
                if stop is not None:
                    # 손절 이후는 현금 — 매도일 가치로 고정한다.
                    curve = curve.where(curve.index <= stop).ffill()
                ratios.append(curve)
            # 슬롯 모델 — 보유 곡선 합 + 현금 슬롯(가치 1 고정), 분모는 top_n.
            if ratios:
                portfolio = (pd.concat(ratios, axis=1).sum(axis=1) + float(top_n - len(ratios))) / top_n
            elif not holdings:
                portfolio = pd.Series(1.0, index=window)  # 전량 현금인 달 — 변동 없음
            else:
                portfolio = None

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

        # 수익인출(고정원금) 누적 — 매월 원금으로 리셋하고 수익은 잘라내는 운용이라
        # 원금 대비 누적 순수익 = 월 수익률의 **산술 합**이다 (복리 총수익과 비교용).
        # 그 달의 인출/입금 흐름 자체는 전략(%)과 같은 값이라 따로 싣지 않는다.
        if strategy_pct is not None:
            harvest_cum_pct += strategy_pct

        monthly.append(
            {
                "month": end.strftime("%Y-%m"),
                "strategy_pct": round(strategy_pct, 2) if strategy_pct is not None else None,
                "harvest_cum_pct": round(harvest_cum_pct, 2) if strategy_pct is not None else None,
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
                # 이 달 중 단기이격 손절로 먼저 매도한 종목 (매도일 포함).
                "stopped": [
                    f"{holding_label(ticker)} {stop.strftime('%m/%d')}"
                    for ticker, stop in sorted(stops.items(), key=lambda pair: pair[1])
                ],
            }
        )
        # 손절 종목은 이미 팔아 현금이다 — 다음 교체일에 또 매도 비용을 물지 않게
        # 보유에서 빼고, 그 가치는 현금 항목으로 이월해 비중 분모에는 남긴다.
        previous_holdings = holdings_set - set(stops)
        previous_growth = {
            ticker: 1.0 + period_returns.get(ticker, 0.0) for ticker in previous_holdings
        }
        # 현금 = 빈 슬롯(각 성장배수 1.0) + 이 달 손절로 판 슬롯 — 다음 교체일 비중 분모에 남긴다.
        cash_multiplier = float(top_n - len(holdings)) + sum(
            1.0 + period_returns.get(t, 0.0) for t in stops
        )
        if cash_multiplier > 0:
            previous_growth["__CASH__"] = cash_multiplier

    # 예정 행 — 마지막 월말 종가에 실행될 교체 (수익률은 아직 없음)
    if pending_signal is not None:
        scored = rank_candidates(candidates_by_date[-1])
        pending_holdings = {
            item["ticker"]
            for item in select_top(scored, top_n, max_per_industry, industry_by_ticker)
        }
        monthly.append(
            {
                "month": (dates[-1].to_period("M") + 1).strftime("%Y-%m"),
                "strategy_pct": None,
                "harvest_cum_pct": None,
                "benchmark_pct": None,
                "reference_pct": None,
                "excess_pp": None,
                "holdings_count": len(pending_holdings),
                "turnover_pct": round(len(pending_holdings - previous_holdings) / top_n * 100.0, 1),
                "added": [holding_label(t) for t in sorted(pending_holdings - previous_holdings)],
                "removed": [holding_label(t) for t in sorted(previous_holdings - pending_holdings)],
                "stopped": [],
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

    # 수익인출전략 요약 — 총자산(원금+남은 수익) 경로 기준. 자산 = 1 + 월 수익률의
    # 산술 누적이고, 월별 자산 수익률 = 그 달 손익 ÷ 직전 총자산 (자산이 커질수록
    # 같은 손익도 비율로는 작아진다 — 고정원금 운용의 실제 체감 경로).
    harvest_wealth_returns: list[float] = []
    wealth = 1.0
    for monthly_return in strategy_returns:
        harvest_wealth_returns.append(monthly_return / wealth)
        wealth += monthly_return
    harvest_total, harvest_mdd, harvest_sortino, harvest_cagr = _summarize(harvest_wealth_returns)
    if reference_close is not None and reference_returns:
        reference_total, reference_mdd, reference_sortino, reference_cagr = _summarize(reference_returns)
    else:
        reference_total = reference_mdd = reference_sortino = reference_cagr = None

    return {
        "start_date": dates[0].strftime("%Y-%m-%d"),
        "end_date": dates[-1].strftime("%Y-%m-%d"),
        "months": months,
        "stop_loss_exit": bool(stop_loss_exit),
        "strategy_total_pct": strategy_total,
        "benchmark_total_pct": benchmark_total,
        "strategy_mdd_pct": strategy_mdd,
        "benchmark_mdd_pct": benchmark_mdd,
        "strategy_sortino": strategy_sortino,
        "benchmark_sortino": benchmark_sortino,
        "strategy_cagr_pct": strategy_cagr,
        "benchmark_cagr_pct": benchmark_cagr,
        "reference_cagr_pct": reference_cagr,
        "harvest_total_pct": harvest_total,
        "harvest_mdd_pct": harvest_mdd,
        "harvest_sortino": harvest_sortino,
        "harvest_cagr_pct": harvest_cagr,
        "benchmark_name": benchmark_info(pool)["name"],
        "benchmark_ticker": benchmark_info(pool)["ticker"],
        "reference_name": reference_name if reference_close is not None else None,
        "reference_total_pct": reference_total,
        "reference_mdd_pct": reference_mdd,
        "reference_sortino": reference_sortino,
        # 최신 달이 위로 오게 뒤집는다 (화면 표)
        "monthly": list(reversed(monthly)),
        # 일간 표 — 최신 날짜가 위로 오게 뒤집는다
        "daily": list(reversed(daily)),
    }
