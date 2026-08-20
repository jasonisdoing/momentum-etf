"""모멘텀 전략 주간 리밸런싱 백테스트.

방식
----
- **매주 재선정한다.** 판정과 체결을 분리한다: **주 마지막 거래일 종가**까지의
  데이터로 선정을 계산하고, **다음 주 첫 거래일 시가**에 체결한다. 한 주 종가를
  모두 보고 판정한 뒤 주말 동안 검토할 시간을 두는 리듬이며, 종가로 판정한 것을
  같은 종가에 체결하는 동시성 편향(look-ahead)도 없다.
- 선정은 꾸준한 모멘텀 점수(연율화 상대기울기 × R²) 순 — 화면 선정과 같은
  ``rank_candidates`` 를 써서 두 화면이 항상 일치한다.
- 슬롯 고정 비중: 종목당 1/top_n 로 배분하고, 자격 종목이 top_n 보다 적으면
  빈 슬롯은 현금으로 남긴다.
- **주중 매도**(simulate_intraweek_exits): 보유 종목이 보유 자격(장기 이격 > 0 &
  단기 이격 >= 0, hold_eligible_mask 와 동일)을 잃으면 다음 거래일 시가에 판다.
  판 슬롯은 다음 주 교체까지 현금이다 — 주중 재매수는 하지 않는다.
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
from utils.momentum_service import (
    available_backtest_months,
    benchmark_info,
    industry_map,
    load_benchmark_close,
    load_price_frames,
    load_settings,
    load_universe,
    pool_info,
    rank_candidates,
    select_candidates,
    select_top,
    simulate_intraweek_exits,
    validate_settings,
    week_last_trading_day,
    week_rebalance_pair,
)
from utils.pool_settings_store import get_pool_slippage
from utils.pool_signal_backtest_service import get_max_backtest_months
from utils.price_series import positive_prices
from utils.trade_stats import summarize_trades

# 미국 풀 참고 지수 — 유사 컨셉 ETF(FMTM)와 같은 구간을 나란히 비교한다 (벤치마크 아님).
US_REFERENCE_TICKER = "FMTM"


def _rebalance_dates(benchmark_close: pd.Series, months: int) -> list[pd.Timestamp]:
    """주 교체일(각 주의 첫 거래일) 목록 — 최근 ``months`` 개월 구간.

    판정은 그 직전 거래일(= 전주 마지막 거래일) 종가다. 마지막 항목은 최신 거래일이라
    진행 중인 구간의 성과까지 보여준다.
    """
    index = benchmark_close.index
    week_ends = index.to_series().groupby(index.to_period("W")).min().tolist()
    start_bound = index[-1] - pd.DateOffset(months=months)
    dates = [stamp for stamp in week_ends if stamp >= start_bound]
    if index[-1] not in dates:
        dates.append(index[-1])
    if len(dates) < 2:
        raise ValueError(f"백테스트 {months}개월에 필요한 데이터가 부족합니다.")
    return dates


def _open_series(frame: pd.DataFrame) -> pd.Series | None:
    """체결가로 쓰는 시가 시계열. 시가가 없는 데이터면 None.

    0 은 거래정지 칸이라 결측으로 돌린다 — 그대로 두면 체결가 0 에 판 것이 되어
    그 구간이 -100% 로 잡힌다(국내 개별주 캐시에 실제로 들어 있다).
    """
    if frame is None or "Open" not in frame.columns:
        return None
    series = positive_prices(frame["Open"]).dropna()
    return series if not series.empty else None


def _open_return(opens: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    """시가 → 시가 수익률. 해당일 시가가 없으면 None(임의 보정하지 않는다)."""
    try:
        start_price = opens.asof(start)
        end_price = opens.asof(end)
    except Exception:
        return None
    if pd.isna(start_price) or pd.isna(end_price) or float(start_price) <= 0:
        return None
    return float(end_price) / float(start_price) - 1.0


def _period_return(close: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    series = positive_prices(close).dropna()
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
            f"장기 이평선 기준으로 이 종목풀은 최대 {pool_max}개월까지 백테스트할 수 있습니다 (요청 {months}개월)."
        )

    # 참고 지수 — 미국 풀은 유사 컨셉 ETF(FMTM)를 나란히 보여준다
    # (벤치마크가 아니며 선정에 관여하지 않는다).
    from utils.cache_utils import load_cached_frames_bulk_from_all_ticker_types

    reference_close: pd.Series | None = None
    reference_name: str | None = None
    if pool_info(pool)["country"] == "us":
        reference_frame = load_cached_frames_bulk_from_all_ticker_types([US_REFERENCE_TICKER]).get(US_REFERENCE_TICKER)
        if reference_frame is not None and not reference_frame.empty:
            reference_close = pd.to_numeric(reference_frame["Close"], errors="coerce").dropna()
            reference_name = US_REFERENCE_TICKER
    dates = _rebalance_dates(benchmark_close, months)

    # 판정 시점 = 각 교체일(체결일)의 직전 거래일. 벤치마크 달력 기준.
    bench_index = benchmark_close.index
    signal_dates: list[pd.Timestamp] = []
    for date in dates[:-1]:
        prior = bench_index[bench_index < date]
        if len(prior) == 0:
            raise ValueError("판정 기준일(직전 거래일)을 구할 수 없습니다 — 데이터가 부족합니다.")
        signal_dates.append(prior[-1])

    # '예정' 행 — 다음 교체일의 판정일(= 이번 주 마지막 거래일) 종가가 이미 확정됐으면
    # 그 판정으로 뽑힐 종목을 미리 보여준다.
    country = pool_info(pool)["country"]
    pending_signal: pd.Timestamp | None = None
    next_rebalance: pd.Timestamp | None = None
    last_cached = dates[-1]
    next_monday = (last_cached - pd.Timedelta(days=int(last_cached.weekday())) + pd.Timedelta(weeks=1)).normalize()
    for monday in (last_cached - pd.Timedelta(days=int(last_cached.weekday())), next_monday):
        pair = week_rebalance_pair(country, monday)
        if pair is None:
            continue
        rebalance_day, signal_day = pair
        if rebalance_day <= last_cached:
            continue  # 이미 지난 교체일
        next_rebalance = rebalance_day
        if signal_day <= last_cached:
            pending_signal = bench_index[bench_index <= signal_day][-1]
        break

    # 교체 시점별 후보 (모멘텀은 가격 캐시만 쓰므로 즉시 계산된다)
    candidates_by_date: list[list[dict[str, Any]]] = []
    all_signal_dates = signal_dates + ([pending_signal] if pending_signal is not None else [])
    for signal_date in all_signal_dates:
        candidates_by_date.append(select_candidates(universe, frames, settings, as_of=signal_date))

    # 슬리피지는 종목풀 설정을 단일 소스로 쓴다 — 매수·매도 편도값을 각각 적용한다.
    buy_slippage_pct, sell_slippage_pct = get_pool_slippage(pool)
    buy_slippage, sell_slippage = buy_slippage_pct / 100.0, sell_slippage_pct / 100.0
    top_n = int(settings["top_n"])

    # 업종 상한 — 선정 화면과 같은 규칙으로 상위 종목을 고른다.
    max_per_industry = int(settings["max_per_industry"])
    industry_by_ticker = industry_map(pool)

    # 일간 표용 — 보유 구간 안에서 매일의 동일가중 포트폴리오 수익률.
    # 가격은 한 번만 정제해 재사용한다(구간마다 다시 정제하면 느리다).
    clean_closes: dict[str, pd.Series] = {}
    clean_opens: dict[str, pd.Series] = {}
    clean_benchmark: pd.Series | None = None
    clean_reference: pd.Series | None = None
    if include_daily:
        clean_closes = {ticker: positive_prices(frame["Close"]).dropna() for ticker, frame in frames.items()}
        for ticker, frame in frames.items():
            opens = _open_series(frame)
            if opens is not None:
                clean_opens[ticker] = opens
        clean_benchmark = pd.to_numeric(benchmark_close, errors="coerce").dropna()
        clean_reference = (
            pd.to_numeric(reference_close, errors="coerce").dropna() if reference_close is not None else None
        )
    daily: list[dict[str, Any]] = []
    # 날짜 -> 그날의 성장배수. 구간 경계일(교체일)은 두 구간이 함께 쓰므로 곱해서 합친다.
    daily_growth: dict[str, float] = {}

    monthly_by_key: dict[str, list[float]] = {}
    monthly_bench: dict[str, list[float]] = {}
    monthly_ref: dict[str, list[float]] = {}
    monthly_order: list[str] = []
    weekly: list[dict[str, Any]] = []
    # 주간 표용 매매 이벤트 — 체결일 기준. 주 행이 이걸로 그 주의 편입·편출을 만든다.
    trade_events: list[dict[str, Any]] = []
    # 체결 목록 — 편입~편출 한 쌍이 한 행. 아직 안 판 종목은 보유중 행으로 남는다.
    open_positions: dict[str, dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []

    def _open_price(ticker: str, day: pd.Timestamp) -> float | None:
        frame = frames.get(ticker)
        opens = _open_series(frame) if frame is not None else None
        if opens is None:
            return None
        value = opens.asof(day)
        return float(value) if pd.notna(value) else None

    def _open_trade(ticker: str, day: pd.Timestamp) -> None:
        price = _open_price(ticker, day)
        if price:
            open_positions[ticker] = {"entry_date": day, "entry_price": price}

    def _open_position_row(ticker: str, position: dict[str, Any]) -> dict[str, Any]:
        """아직 안 판 종목 — 마지막 종가로 평가한 보유중 행."""
        frame = frames.get(ticker)
        close = (
            pd.to_numeric(frame["Close"], errors="coerce").dropna()
            if frame is not None and not frame.empty and "Close" in frame.columns
            else None
        )
        price = float(close.iloc[-1]) if close is not None and not close.empty else None
        return {
            "ticker": ticker,
            "name": name_by_ticker.get(ticker, ticker),
            "entry_date": position["entry_date"].strftime("%Y-%m-%d"),
            "entry_price": round(position["entry_price"], 4),
            "exit_date": None,
            "exit_price": round(price, 4) if price is not None else None,
            "return_pct": round((price / position["entry_price"] - 1) * 100, 2) if price is not None else None,
            "days": int((dates[-1] - position["entry_date"]).days),
            "reason": "보유중",
        }

    def _close_trade(ticker: str, day: pd.Timestamp, reason: str) -> None:
        position = open_positions.pop(ticker, None)
        price = _open_price(ticker, day)
        if position is None or not price:
            return
        trades.append(
            {
                "ticker": ticker,
                "name": name_by_ticker.get(ticker, ticker),
                "entry_date": position["entry_date"].strftime("%Y-%m-%d"),
                "entry_price": round(position["entry_price"], 4),
                "exit_date": day.strftime("%Y-%m-%d"),
                "exit_price": round(price, 4),
                "return_pct": round((price / position["entry_price"] - 1) * 100, 2),
                "days": int((day - position["entry_date"]).days),
                "reason": reason,
            }
        )

    strategy_returns: list[float] = []
    benchmark_returns: list[float] = []
    reference_returns: list[float] = []
    previous_holdings: set[str] = set()
    # 직전 보유 구간의 종목별 성장배수(1+수익률) — 교체 시점의 드리프트 비중 계산용.
    previous_growth: dict[str, float] = {}

    for position in range(len(dates) - 1):
        start = dates[position]
        end = dates[position + 1]
        # 판정은 교체일 직전 거래일(signal_dates), 체결·보유 구간은 교체일 시가(start→end).
        scored = rank_candidates(candidates_by_date[position])
        holdings = [item["ticker"] for item in select_top(scored, top_n, max_per_industry, industry_by_ticker)]
        holdings_set = set(holdings)
        added_tickers = sorted(holdings_set - previous_holdings)
        removed_tickers = sorted(previous_holdings - holdings_set)
        # 슬롯 고정 비중 — 선정이 top_n 보다 적으면 빈 슬롯은 현금으로 남긴다.
        target_weight = 1.0 / top_n

        # ── 주중 매도 — 자격 상실 종목은 다음 거래일 시가 매도 (선정 화면과 같은 함수).
        # 완결된 주는 마지막 판정일(교체일 직전)을 스캔에서 제외한다 — 그 판정의 체결은
        # 주 교체가 대신하기 때문이다(이중 계산 방지). 마지막 구간은 끝까지 스캔해
        # 마지막 종가 판정분을 '다음 거래일 매도 예정'으로 남긴다.
        scan_days = bench_index[(bench_index >= start) & (bench_index < end)]
        exits: list[dict[str, Any]] = []
        if len(scan_days) >= 2:
            exits = simulate_intraweek_exits(frames, settings, holdings_set, bench_index, start, scan_days[-2])
        exited_tickers = {x["ticker"] for x in exits}
        sell_date_by_ticker = {x["ticker"]: x["sell_date"] for x in exits}
        # 이 구간을 끝까지 들고 가는 종목 — 다음 교체·표시가 이 기준을 쓴다.
        survivors = holdings_set - exited_tickers

        for ticker in added_tickers:
            trade_events.append({"date": start, "action": "add", "ticker": ticker})
            _open_trade(ticker, start)
        for ticker in removed_tickers:
            trade_events.append({"date": start, "action": "remove", "ticker": ticker})
            _close_trade(ticker, start, "주간 교체")
        for exit_info in exits:
            trade_events.append({"date": exit_info["sell_date"], "action": "remove", "ticker": exit_info["ticker"]})
            _close_trade(exit_info["ticker"], exit_info["sell_date"], exit_info.get("reason") or "주중 이탈")

        # ── 교체 매매 금액(포트폴리오 대비 비율) ──
        # 유지 종목도 매주 1/N 로 재조정하므로, 드리프트 비중과 목표의 차이가 전부 매매다.
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
        else:
            sell_notional = 0.0
            buy_notional = len(holdings) * target_weight  # 첫 구간은 투자분만 신규 매수
        # 주중 매도도 매매 금액에 넣는다 (슬롯당 1/N).
        sell_notional += len(exits) * target_weight
        cost = buy_slippage * buy_notional + sell_slippage * sell_notional

        # 보유 구간 — 주중 매도된 종목은 매도 체결일 시가까지만 수익이 발생한다.
        period_returns: dict[str, float] = {}
        for ticker in holdings:
            frame = frames.get(ticker)
            opens = _open_series(frame) if frame is not None else None
            if opens is None:
                continue
            value = _open_return(opens, start, sell_date_by_ticker.get(ticker, end))
            if value is not None:
                period_returns[ticker] = value
        # 슬롯 모델: 보유 종목은 각 1/N, 빈 슬롯은 현금(0%) — 분모는 항상 top_n.
        if not holdings:
            gross: float | None = 0.0  # 전량 현금인 주
        elif period_returns:
            gross = sum(period_returns.values()) / top_n
        else:
            gross = None  # 보유 종목의 가격 데이터가 전혀 없음 — 데이터 문제를 그대로 드러낸다

        strategy_pct = (gross - cost) * 100.0 if gross is not None else None
        benchmark_return = _period_return(benchmark_close, start, end)
        benchmark_pct = benchmark_return * 100.0 if benchmark_return is not None else None
        reference_return = _period_return(reference_close, start, end) if reference_close is not None else None
        reference_pct = reference_return * 100.0 if reference_return is not None else None

        if strategy_pct is not None:
            strategy_returns.append(strategy_pct / 100.0)
        if benchmark_pct is not None:
            benchmark_returns.append(benchmark_pct / 100.0)
        if reference_pct is not None:
            reference_returns.append(reference_pct / 100.0)

        # ── 일간 행 ──
        # 이 구간(start→end)의 보유 종목은 고정이다. 체결이 시가이므로 **start 시가**를 1 로
        # 두고, 마지막 날(또는 주중 매도일)은 그날 **시가**로 끊는다 — 종가로 재면 교체일
        # 당일 수익이 통째로 빠져 일별 곡선이 구간 수익률과 어긋난다.
        # start 는 직전 구간의 end 와 같은 날이라 날짜가 겹친다. 겹치는 날은 곱해서 하나로
        # 합친다(직전 구간의 '전일 종가 → 시가' 와 이번 구간의 '시가 → 종가').
        # 교체 비용은 구간 첫날에 한 번 반영한다(주간 계산과 같은 방식).
        window = bench_index[(bench_index >= start) & (bench_index <= end)] if include_daily else []
        if len(window) > 0:

            def position_curve(ticker: str) -> pd.Series | None:
                """구간 시작 시가를 1 로 둔 종목별 가치. 매도일부터는 매도 시가로 고정한다."""
                closes = clean_closes.get(ticker)
                opens = clean_opens.get(ticker)
                if closes is None or opens is None:
                    return None
                base = opens.asof(start)
                if pd.isna(base) or float(base) <= 0:
                    return None
                curve = closes.reindex(window, method="ffill") / float(base)
                # 이 종목이 나가는 날 — 주중 매도일이 없으면 구간 종료일(교체일)이다.
                stop_day = sell_date_by_ticker.get(ticker, end)
                exit_price = opens.asof(stop_day)
                if pd.isna(exit_price) or float(exit_price) <= 0:
                    return curve
                return curve.where(window < stop_day, float(exit_price) / float(base))

            ratios = [curve for curve in (position_curve(t) for t in holdings) if curve is not None]
            # 슬롯 모델 — 보유 곡선 합 + 현금 슬롯(가치 1 고정), 분모는 top_n.
            if ratios:
                portfolio = (pd.concat(ratios, axis=1).sum(axis=1) + float(top_n - len(ratios))) / top_n
            elif not holdings:
                portfolio = pd.Series(1.0, index=window)  # 전량 현금인 주 — 변동 없음
            else:
                portfolio = None

            if portfolio is not None:
                for day_index, day in enumerate(window):
                    value = portfolio.iloc[day_index]
                    prior = 1.0 if day_index == 0 else portfolio.iloc[day_index - 1]
                    if pd.isna(value) or pd.isna(prior) or float(prior) <= 0:
                        continue
                    growth = float(value) / float(prior)
                    if day_index == 0:
                        growth -= cost  # 교체 매매 비용은 체결일에 한 번
                    key = day.strftime("%Y-%m-%d")
                    daily_growth[key] = daily_growth.get(key, 1.0) * growth

        # ── 월간 집계 — 구간 수익률을 구간 종료일이 속한 달로 복리 합산한다.
        month_key = end.strftime("%Y-%m")
        if month_key not in monthly_by_key:
            monthly_order.append(month_key)
            monthly_by_key[month_key], monthly_bench[month_key], monthly_ref[month_key] = [], [], []
        if strategy_pct is not None:
            monthly_by_key[month_key].append(strategy_pct)
        if benchmark_pct is not None:
            monthly_bench[month_key].append(benchmark_pct)
        if reference_pct is not None:
            monthly_ref[month_key].append(reference_pct)

        previous_holdings = survivors
        previous_growth = {ticker: 1.0 + period_returns.get(ticker, 0.0) for ticker in previous_holdings}
        # 현금 = 빈 슬롯 + 주중 매도 슬롯(근사 1.0) — 다음 교체일 비중 분모에 남긴다.
        cash_multiplier = float(top_n - len(previous_holdings))
        if cash_multiplier > 0:
            previous_growth["__CASH__"] = cash_multiplier

    def _compound(values: list[float]) -> float | None:
        if not values:
            return None
        growth = 1.0
        for value in values:
            growth *= 1.0 + value / 100.0
        return round((growth - 1.0) * 100.0, 2)

    monthly: list[dict[str, Any]] = [
        {
            "month": key,
            "strategy_pct": _compound(monthly_by_key[key]),
            "benchmark_pct": _compound(monthly_bench[key]),
            "reference_pct": _compound(monthly_ref[key]),
        }
        for key in monthly_order
    ]

    # ── 이미 체결된 최신 교체 — 마지막 캐시 거래일이 그 주의 교체일이면 그날 시가에
    # 체결이 끝났다. 구간 루프는 이 날을 구간 끝으로만 보므로 여기서 반영한다.
    current_holdings = previous_holdings
    last_pair = week_rebalance_pair(country, last_cached - pd.Timedelta(days=int(last_cached.weekday())))
    if last_pair is not None and last_pair[0] == last_cached:
        selection = {
            item["ticker"]
            for item in select_top(
                rank_candidates(
                    select_candidates(universe, frames, settings, as_of=bench_index[bench_index < last_cached][-1])
                ),
                top_n,
                max_per_industry,
                industry_by_ticker,
            )
        }
        for ticker in sorted(selection - previous_holdings):
            trade_events.append({"date": last_cached, "action": "add", "ticker": ticker})
            _open_trade(ticker, last_cached)
        for ticker in sorted(previous_holdings - selection):
            trade_events.append({"date": last_cached, "action": "remove", "ticker": ticker})
            _close_trade(ticker, last_cached, "주간 교체")
        current_holdings = selection

    # ── 일간 행 조립 — 구간별로 모은 성장배수를 날짜순 행으로 편다.
    # 벤치마크·참조는 구간과 무관하므로 전체 시계열에서 한 번에 일별 변동률을 만든다
    # (구간마다 정규화하면 경계일이 두 번 세어진다).
    if include_daily and daily_growth:
        prior_days = bench_index[bench_index < dates[0]]
        first = prior_days[-1] if len(prior_days) > 0 else dates[0]
        span_index = bench_index[(bench_index >= first) & (bench_index <= dates[-1])]

        def _daily_changes(source: pd.Series | None) -> dict[str, float]:
            if source is None:
                return {}
            changes = source.reindex(span_index, method="ffill").pct_change()
            return {day.strftime("%Y-%m-%d"): float(v) * 100.0 for day, v in changes.items() if pd.notna(v)}

        bench_changes = _daily_changes(clean_benchmark)
        ref_changes = _daily_changes(clean_reference)
        for key in sorted(daily_growth):
            daily.append(
                {
                    "date": key,
                    "strategy_pct": round((daily_growth[key] - 1.0) * 100.0, 2),
                    "benchmark_pct": round(bench_changes[key], 2) if key in bench_changes else None,
                    "reference_pct": round(ref_changes[key], 2) if key in ref_changes else None,
                }
            )

    # ── 주간 행 — 달력 주(월~일) 단위. 기준일은 그 주 마지막 거래일, 수익률은 그 주의
    # 성과, 편입·편출은 그 주에 체결된 매매다.
    if include_daily and daily:

        def _week_monday(stamp: pd.Timestamp) -> pd.Timestamp:
            return (stamp - pd.Timedelta(days=int(stamp.weekday()))).normalize()

        def _compound_daily(values: list[float | None]) -> float | None:
            usable = [value for value in values if value is not None]
            if not usable:
                return None
            growth = 1.0
            for value in usable:
                growth *= 1.0 + value / 100.0
            return round((growth - 1.0) * 100.0, 2)

        buckets: dict[pd.Timestamp, dict[str, Any]] = {}
        for row in daily:
            stamp = pd.Timestamp(row["date"])
            bucket = buckets.setdefault(
                _week_monday(stamp), {"end": stamp, "strategy": [], "benchmark": [], "reference": []}
            )
            bucket["end"] = max(bucket["end"], stamp)
            bucket["strategy"].append(row["strategy_pct"])
            bucket["benchmark"].append(row["benchmark_pct"])
            bucket["reference"].append(row["reference_pct"])

        events_sorted = sorted(trade_events, key=lambda event: event["date"])
        event_index = 0
        holdings_running = 0
        for key in sorted(buckets):
            bucket = buckets[key]
            added_labels: list[str] = []
            removed_labels: list[str] = []
            while event_index < len(events_sorted) and events_sorted[event_index]["date"] <= bucket["end"]:
                event = events_sorted[event_index]
                if event["action"] == "add":
                    holdings_running += 1
                    added_labels.append(holding_label(event["ticker"]))
                else:
                    holdings_running -= 1
                    removed_labels.append(holding_label(event["ticker"]))
                event_index += 1
            weekly.append(
                {
                    "week_end": bucket["end"].strftime("%Y-%m-%d"),
                    "strategy_pct": _compound_daily(bucket["strategy"]),
                    "benchmark_pct": _compound_daily(bucket["benchmark"]),
                    "reference_pct": _compound_daily(bucket["reference"]),
                    "holdings_count": holdings_running,
                    "turnover_pct": round(len(added_labels) / top_n * 100.0, 1),
                    "added": added_labels,
                    "removed": removed_labels,
                }
            )

    # ── 예정 행 — 다음 주. 매매가 없어도 '보유 N 유지'가 보이도록 항상 붙인다.
    if pending_signal is not None:
        # 다음 교체 판정일 종가가 이미 확정됐다 — 그 선정을 그대로 보여준다.
        pending_holdings = {
            item["ticker"]
            for item in select_top(rank_candidates(candidates_by_date[-1]), top_n, max_per_industry, industry_by_ticker)
        }
        pending_added = sorted(pending_holdings - current_holdings)
        pending_removed = sorted(current_holdings - pending_holdings)
        pending_count = len(pending_holdings)
    else:
        # 판정일이 아직 오지 않았다 — 마지막 종가로 판정한 주중 매도 예정만 반영한다.
        pending_exits = simulate_intraweek_exits(
            frames, settings, current_holdings, bench_index, last_cached, last_cached
        )
        pending_added = []
        pending_removed = sorted(x["ticker"] for x in pending_exits)
        pending_count = len(current_holdings) - len(pending_removed)

    weekly.append(
        {
            "week_end": week_last_trading_day(country, next_rebalance or (last_cached + pd.Timedelta(days=7))),
            "strategy_pct": None,
            "benchmark_pct": None,
            "reference_pct": None,
            "holdings_count": pending_count,
            "turnover_pct": round(len(pending_added) / top_n * 100.0, 1),
            "added": [holding_label(t) for t in pending_added],
            "removed": [holding_label(t) for t in pending_removed],
            "is_pending": True,
        }
    )

    def _summarize(returns: list[float]) -> tuple[float, float | None, float | None, float | None]:
        curve = pd.Series([1.0] + list(pd.Series(returns).add(1.0).cumprod()))
        total = (float(curve.iloc[-1]) - 1.0) * 100.0
        # CAGR — 주별 표본 수 기준 연율화 (1년 미만이면 연 환산이라 과장될 수 있음)
        sample_periods = len(returns)
        cagr = (
            ((1.0 + total / 100.0) ** (52.0 / sample_periods) - 1.0) * 100.0
            if sample_periods > 0 and total > -100.0
            else None
        )
        # 소르티노는 주별 수익률 기준 연율화 (레버리지 엔진 공용 함수 재사용).
        # 표본 3개 미만이거나 하락 주가 없으면 None → 화면에서 '-'.
        return (
            round(total, 2),
            max_drawdown_pct(curve),
            sortino(pd.Series(returns), periods_per_year=52),
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
        "benchmark_name": benchmark_info(pool)["name"],
        "benchmark_ticker": benchmark_info(pool)["ticker"],
        "reference_name": reference_name if reference_close is not None else None,
        "reference_total_pct": reference_total,
        "reference_mdd_pct": reference_mdd,
        "reference_sortino": reference_sortino,
        # 최신 달이 위로 오게 뒤집는다 (화면 표)
        "monthly": list(reversed(monthly)),
        # 주간 표 — 매매 내역(편입·편출·교체율·보유 수)을 담는다. 최신 주가 위.
        "weekly": list(reversed(weekly)),
        # 일간 표 — 최신 날짜가 위로 오게 뒤집는다
        "daily": list(reversed(daily)),
        # 체결 목록 — 보유중(청산 전) 행이 위, 그 아래는 청산일 최신순.
        "trades": [
            _open_position_row(ticker, position)
            for ticker, position in sorted(open_positions.items(), key=lambda item: item[1]["entry_date"], reverse=True)
        ]
        + sorted(trades, key=lambda trade: trade["exit_date"], reverse=True),
        # 거래 수·승률·평균 손익 — 세 전략이 같은 공용 계산을 쓴다(청산분만 센다).
        **summarize_trades(trades),
    }
