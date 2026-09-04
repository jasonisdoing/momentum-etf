"""슬롯 백테스트 — 신고가·모멘텀이 함께 쓰는 **일간 슬롯** 엔진.

두 전략은 판정 내용만 다르고 굴리는 방식이 같다.

    매일 종가로 판정 → **다음 거래일 시가**에 체결
    동시 보유 상한 ``slots``, 균등 배분(정수 주수)
    자리가 차 있으면 더 좋은 후보가 와도 **교체하지 않는다**
    ADR 하한에 걸린 날은 **신규 진입만** 건너뛴다(보유 청산은 그대로 돈다)

교체하지 않는 이유: 2026-08-14 kor(24개월)·us(60개월) 백테스트에서 최저수익/손실만/최장보유
교체 전부가 현행보다 나빴다(kor +1912% vs 교체 시 +142~779%, us +240% vs -76~+95%).
보유 중이라는 것 자체가 청산에 안 걸린 살아있는 추세라, 교체는 청산 규칙을 앞질러 이익
종목을 자르고 슬리피지 왕복 비용만 쌓는다.

전략이 채워 넣는 것은 세 장의 표뿐이다(행 = 거래일, 열 = 종목).

    entry     그날 종가로 **살 자격**을 갖췄는가
    exit      그날 종가로 **팔 신호**인가
    priority  자리가 모자랄 때 큰 값부터 담는다

신고가는 (52주 돌파 + 거래대금 하한 / 이탈 이평선 하회 / 거래대금 배수),
모멘텀은 (장기 이격 > 0 그리고 단기 이격 >= 0 / 둘 중 하나라도 깨짐 / 장기 이격률)이다.

이 엔진이 **유일한 판정자**다. 화면·슬랙·합성은 ``planned_exits``·``planned_entries``·
``open_positions`` 를 읽기만 한다. 바꿀 규칙이 생기면 여기를 고쳐야 백테스트·화면·튜닝이
같이 따라온다 (AGENTS.md 10).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from utils.pool_settings_store import get_pool_slippage
from utils.share_allocation import ShareTarget, allocate_integer_shares, backtest_initial_capital
from utils.trade_stats import summarize_trades


def _drawdown_pct(series: pd.Series) -> float:
    """최대 낙폭(%) — 고점 대비 최저."""
    return float(((series / series.cummax()) - 1).min() * 100)


def _cagr_pct(total_pct: float, months: int) -> float | None:
    """연평균 성장률(%)."""
    if months <= 0:
        return None
    return ((1 + total_pct / 100) ** (12 / months) - 1) * 100


def _sortino(returns: pd.Series) -> float | None:
    """하방 변동성 대비 수익(연율). 하락일이 없으면 나눌 수 없어 None."""
    downside = returns[returns < 0]
    if downside.empty or len(returns) < 2:
        return None
    dd = float((downside**2).mean() ** 0.5)
    if dd <= 0:
        return None
    return float(returns.mean() / dd * (252**0.5))


def adr_entry_gate(pool: str, adr_floor: Any) -> tuple[Callable[[pd.Timestamp], bool], Callable[[pd.Timestamp], Any]]:
    """ADR 진입 게이트와 표시값 조회 함수를 한 쌍으로 만든다.

    게이트는 **신규 진입만** 막는다 — 보유 청산은 그대로 돈다. 하한이 없거나 레짐 시장이
    없는 풀이면 아무것도 막지 않고, 표시값도 None 이다. ADR 이력 이전 날짜는 미적용이다.
    """
    from utils.momentum_service import adr_market_of_pool, load_adr_series

    market = adr_market_of_pool(pool)
    series = load_adr_series(market) if market else pd.Series(dtype=float)

    def adr_at(stamp: pd.Timestamp) -> float | None:
        if series.empty:
            return None
        value = series.asof(pd.Timestamp(stamp))
        return round(float(value), 1) if pd.notna(value) else None

    def entry_blocked(stamp: pd.Timestamp) -> bool:
        if adr_floor is None or series.empty:
            return False
        value = series.asof(pd.Timestamp(stamp))
        return bool(pd.notna(value) and float(value) < float(adr_floor))

    return entry_blocked, adr_at


def run_slot_backtest(
    *,
    pool: str,
    months: int,
    panel: dict[str, pd.DataFrame],
    entry: pd.DataFrame,
    exit_signal: pd.DataFrame,
    priority: pd.DataFrame,
    slots: int,
    adr_floor: Any,
    name_by: dict[str, str],
    industry_by: dict[str, str],
    exit_reason: str,
) -> dict[str, Any]:
    """일간 슬롯 시뮬레이션. 자산곡선·지표·체결 내역·현재 보유를 한 형태로 돌려준다.

    ``entry``/``exit_signal``/``priority`` 는 ``panel["close"]`` 와 같은 인덱스·컬럼이어야
    한다. 응답 형태는 두 전략이 같다 — 화면·합성·튜닝이 같은 키를 읽는다.
    """
    entry_blocked, adr_at = adr_entry_gate(pool, adr_floor)
    close_df, open_df = panel["close"], panel["open"]
    buy_slippage, sell_slippage = get_pool_slippage(pool)

    dates = close_df.index
    span = [d for d in dates if d >= dates[-1] - pd.DateOffset(months=months)]
    if len(span) < 2:
        raise RuntimeError("백테스트할 구간의 가격 데이터가 부족합니다.")

    # 자산은 현금 + 보유 주수로 들고 간다. 포지션 손익을 자산에 곱하면 동시에 들고 있던
    # 종목의 손익이 합산이 아니라 곱으로 쌓여 수익이 부풀려진다.
    #
    # 시작 자본은 통화별 상수(config.BACKTEST_INITIAL_CAPITAL)다. 상대곡선(1.0)으로 두면
    # 주수가 소수로 나오는데, 실제로는 정수 주수만 살 수 있어 운용 현황과 결과가 어긋난다.
    # 곡선은 마지막에 시작 자본으로 나눠 배수로 돌려준다.
    initial_capital = backtest_initial_capital(pool)
    cash = float(initial_capital)
    holdings: dict[str, dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []
    curve: list[float] = []
    last_day = span[-1]

    # 평가 전용 종가 — 그날 값이 없으면 **직전 유효 종가**로 본다.
    # 판정에는 쓰지 않는다. 없는 날을 0 으로 치면 그 종목이 사라진 것처럼 계산돼 곡선이
    # 한 번에 무너진다(us_etf 가 마지막 하루로 -100% 가 됐다).
    valuation_close = close_df.ffill()

    def priority_of(ticker: str, day: pd.Timestamp) -> float:
        """자리가 모자랄 때의 줄 세우기 값 — 모르는 종목은 맨 뒤."""
        score = priority.at[day, ticker]
        return float(score) if pd.notna(score) else 0.0

    def _value_at(day: pd.Timestamp) -> float:
        """그날 종가로 평가한 총자산 — 현금 + 보유 평가액."""
        total = cash
        for ticker, position in holdings.items():
            price = valuation_close.at[day, ticker]
            if pd.notna(price):
                total += position["shares"] * float(price)
        return total

    for index, day in enumerate(span[:-1]):
        nxt = span[index + 1]
        # 자산 평가는 **그날의 판정보다 먼저** 한다. 아래 매매는 전부 `nxt` 시가 체결이라,
        # 판정 뒤에 재면 아직 사지도 않은 주식이 오늘 종가로 평가되고 그 대금은 이미 현금에서
        # 빠져 곡선 첫날이 어긋난다(총수익과 일별 합성이 4%p 갈렸다).
        curve.append(_value_at(day))

        # 1) 청산 판정 (오늘 종가) → 내일 시가 체결
        for ticker in list(holdings):
            position = holdings[ticker]
            price = close_df.at[day, ticker]
            if pd.isna(price):
                continue
            if not bool(exit_signal.at[day, ticker]):
                continue
            exit_price = open_df.at[nxt, ticker]
            if pd.isna(exit_price):
                exit_price = price  # 다음 날 시가가 없으면(거래정지) 오늘 종가로 본다
            ret = (float(exit_price) * (1 - sell_slippage / 100)) / position["entry"] - 1
            cash += position["shares"] * float(exit_price) * (1 - sell_slippage / 100)
            trades.append(
                {
                    "ticker": ticker,
                    "name": name_by.get(ticker, ticker),
                    "industry": industry_by.get(ticker, ""),
                    "entry_date": str(position["date"].date()),
                    # 표시는 **실제 체결 시가**다. 슬리피지는 가격이 아니라 비용이라
                    # 손익률에만 반영한다 — 가격에 섞으면 시장에 없는 호가가 찍힌다.
                    "entry_price": round(position["open"], 2),
                    "exit_date": str(nxt.date()),
                    "exit_price": round(float(exit_price), 2),
                    "return_pct": round(ret * 100, 2),
                    # 보유일 — **들고 있던 거래일 수**(체결일 포함). 시가에 사서 그날 종가까지
                    # 들고 있었으면 1일이다. 예전에는 경과일(-1)로 세서 하루 만에 판 거래가
                    # 「보유일 0」 으로, 오늘 산 종목이 「0일」 로 나왔다.
                    "days": len(close_df.loc[position["date"] : day]),
                    "reason": exit_reason,
                }
            )
            del holdings[ticker]

        # 2) 진입 — 빈 자리만큼, 우선순위가 큰 순 (ADR 게이트에 걸린 날은 건너뜀)
        free = slots - len(holdings)
        if free > 0 and not entry_blocked(day):
            # 배정 기준은 **체결 시점(다음 거래일 시가)의 자산**이다. 청산 대금이 이미
            # 현금에 들어와 있으므로 파는 쪽과 사는 쪽이 같은 시점으로 맞는다.
            fill_value = cash
            for held_ticker, held_position in holdings.items():
                held_price = open_df.at[nxt, held_ticker]
                if pd.isna(held_price):
                    held_price = close_df.at[day, held_ticker]
                if pd.notna(held_price):
                    fill_value += held_position["shares"] * float(held_price)
            row = entry.loc[day]
            picks = [t for t in row[row].index if t not in holdings and not pd.isna(open_df.at[nxt, t])]
            picks.sort(key=lambda ticker: priority_of(ticker, day), reverse=True)
            picks = picks[:free]
            # 주수 배분은 운용 현황과 **같은 함수**를 쓴다 — 규칙이 갈라지면 백테스트가
            # 실제로 못 내는 성과를 내게 된다. 예산은 살 수 있는 현금까지만(팔지 않은
            # 평가익으로는 못 산다). 손익 계산에는 슬리피지를 얹은 값을 쓴다.
            fill_price_by_ticker = {t: float(open_df.at[nxt, t]) * (1 + buy_slippage / 100) for t in picks}
            slot_amount = fill_value / slots if slots else 0.0
            quantities = allocate_integer_shares(
                [
                    ShareTarget(key=ticker, target_amount=slot_amount, price=price)
                    for ticker, price in fill_price_by_ticker.items()
                    if price > 0
                ],
                budget=min(slot_amount * len(picks), cash),
            )
            for ticker in picks:
                shares = quantities.get(ticker, 0)
                if shares <= 0:
                    continue
                fill_price = fill_price_by_ticker[ticker]
                holdings[ticker] = {
                    "open": float(open_df.at[nxt, ticker]),
                    "entry": fill_price,
                    "date": nxt,
                    "shares": shares,
                }
                cash -= shares * fill_price

    # 마지막 날은 판정·체결이 없다(체결할 다음 거래일이 없어서). 다만 그날 종가로
    # **평가**는 해야 곡선이 하루 짧아지지 않는다.
    curve.append(_value_at(last_day))

    # 아직 청산하지 않은 종목 — 성과에는 평가손익으로 이미 반영돼 있지만 체결 내역에는 없다.
    # 슬리브 안에서의 현재 비중도 함께 담는다. 진입할 때 1/slots 였다가 시세대로 흘러간
    # 값이라, 합성 화면이 '지금 이 종목이 몇 % 여야 하는지' 를 이 값으로 잡는다.
    sleeve_value = _value_at(last_day)
    open_positions = []
    for ticker, position in holdings.items():
        price = close_df.at[last_day, ticker]
        if pd.isna(price):
            continue
        value = position["shares"] * float(price)
        open_positions.append(
            {
                "ticker": ticker,
                "name": name_by.get(ticker, ticker),
                "industry": industry_by.get(ticker, ""),
                "entry_date": str(position["date"].date()),
                "entry_price": round(position["open"], 2),
                "price": float(price),
                # 표시용 평가손익 — 아직 안 팔았으니 매도 슬리피지는 빼지 않는다.
                "return_pct": round((float(price) / position["open"] - 1) * 100, 2),
                "days": len(close_df.loc[position["date"] : last_day]),
                # 오늘 편입된 종목은 목록에서 따로 표시한다.
                "is_new": position["date"] == last_day,
                # 이 슬리브 안에서의 비중(%) — 슬리브 전체를 100 으로 본다.
                "sleeve_weight_pct": round(value / sleeve_value * 100, 4) if sleeve_value > 0 else 0.0,
            }
        )
    # 오래 들고 있는 것이 위 — 화면이 보여주는 순서다. 여기서 맞춰 두면 이 목록을 그대로
    # 쓰는 합성 슬리브도 같은 순서가 된다.
    open_positions.sort(key=lambda row: row["entry_date"])

    # ── 마지막 날 판정 — **다음 거래일 시가에 할 일** ─────────────────────────
    # 위 루프는 마지막 날을 판정하지 않는다(체결할 다음 날이 없어서). 그래서 여기서 한 번 더
    # 본다. 이걸 엔진이 안 내주면 화면이 같은 판정을 **다시 구현**하게 되고, 한쪽만 고치는
    # 순간 "화면은 사라는데 백테스트는 안 샀다"가 된다 — 그러면 성과 숫자를 믿을 수 없다.
    planned_exits = [
        ticker
        for ticker in holdings
        if pd.notna(close_df.at[last_day, ticker]) and bool(exit_signal.at[last_day, ticker])
    ]
    planned_entries: list[str] = []
    free = slots - (len(holdings) - len(planned_exits))
    if free > 0 and not entry_blocked(last_day):
        row = entry.loc[last_day]
        picks = [ticker for ticker in row[row].index if ticker not in holdings]
        picks.sort(key=lambda ticker: priority_of(ticker, last_day), reverse=True)
        planned_entries = picks[:free]

    # 곡선은 시작 1.0 배수로 되돌린다 — 시작 자본은 정수 주수를 세기 위한 것이고,
    # 성과 지표(수익률·MDD·벤치마크 대비)는 배수 기준으로 읽는다.
    strategy = pd.Series(curve, index=span) / initial_capital
    # 벤치마크는 **시작일 시가**를 1 로 둔다 — 전략도 그날 시가에 사기 때문이다(공용 함수).
    from utils.benchmark_curve import benchmark_growth
    from utils.new_high_service import benchmark_info

    benchmark = benchmark_growth(pool, strategy.index)
    strategy_total = float((strategy.iloc[-1] - 1) * 100)
    benchmark_total = float((benchmark.iloc[-1] - 1) * 100)

    return {
        "start_date": str(strategy.index[0].date()),
        "end_date": str(strategy.index[-1].date()),
        "months": months,
        "strategy_total_pct": round(strategy_total, 2),
        "strategy_cagr_pct": round(_cagr_pct(strategy_total, months) or 0.0, 2),
        "strategy_mdd_pct": round(_drawdown_pct(strategy), 2),
        "strategy_sortino": _sortino(strategy.pct_change().dropna()),
        "benchmark_total_pct": round(benchmark_total, 2),
        "benchmark_cagr_pct": round(_cagr_pct(benchmark_total, months) or 0.0, 2),
        "benchmark_mdd_pct": round(_drawdown_pct(benchmark), 2),
        "benchmark_sortino": _sortino(benchmark.pct_change().dropna()),
        "benchmark_name": benchmark_info(pool)["name"],
        **summarize_trades(trades),
        "trades": sorted(trades, key=lambda t: t["exit_date"], reverse=True),
        "as_of": str(last_day.date()),
        "open_positions": open_positions,
        # 다음 거래일 시가에 할 일 — 화면은 이걸 읽기만 한다(판정을 다시 하지 않는다).
        "planned_exits": planned_exits,
        "planned_entries": planned_entries,
        # 빈 슬롯·잔여 현금 비중 — 종목 비중과 합쳐 100 이 된다.
        "sleeve_cash_weight_pct": round(cash / sleeve_value * 100, 4) if sleeve_value > 0 else 100.0,
        "exited_today": [t for t in trades if t["exit_date"] == str(last_day.date())],
        # 소수 6자리 — 화면은 2자리로 보여주지만, 연간·월간·주간 표와 튜닝 지표는 이 값을
        # **복리로 합성**한다. 2자리로 잘라 보내면 하루치 오차가 250일 쌓여 합계가 총수익과
        # 어긋난다(12개월 기준 4%p 차이가 났다).
        "daily": [
            {
                "date": str(d.date()),
                "strategy_pct": round((v - 1) * 100, 6),
                "benchmark_pct": round((float(benchmark.loc[d]) - 1) * 100, 6),
                "adr": adr_at(d),
            }
            for d, v in strategy.items()
        ],
    }
