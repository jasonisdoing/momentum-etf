"""포트폴리오 시뮬레이션 핵심 — 가격 프레임을 받아 굴린다. 외부 조회는 하지 않는다.

정한 비중으로 시작해 주기마다 그 비중으로 되돌린다. 리밸런싱 때 허용 밴드를 넘긴
종목만 사고팔며, 그 매매만 비용(슬리피지)을 문다. 그 사이에는 시세대로 흘러가게 둔다.
가격 로딩·실시간 오버레이·벤치마크·화면 페이로드는 호출자(`utils/portfolio_backtest`) 몫이다.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def period_key(day: pd.Timestamp, rebalance: str) -> str | None:
    """그 날짜가 속한 리밸런싱 구간의 키. 'none' 이면 None(되돌리지 않음)."""
    if rebalance == "monthly":
        return day.strftime("%Y-%m")
    if rebalance == "quarterly":
        return f"{day.year}-Q{(day.month - 1) // 3 + 1}"
    if rebalance == "yearly":
        return str(day.year)
    return None


def simulate_portfolio(
    *,
    close_df: pd.DataFrame,
    target_by_ticker: dict[str, float],
    cash_target: float,
    band_pct: float,
    rebalance: str,
    buy_slippage: float,
    sell_slippage: float,
) -> dict[str, Any]:
    """자산 1.0 에서 시작하는 고정 비중 시뮬레이션 — 곡선·현금 비중·체결·최종 보유를 낸다.

    ``close_df`` 는 이미 구간·종목이 잘린 (날짜 × 티커) 종가 프레임이고, 첫 행이 최초
    매수일이다. 비중·슬리피지 비율은 0~1 단위가 아니라 저장 형태 그대로다
    (``target_by_ticker`` 는 0~1, ``band_pct`` 는 %p, 슬리피지는 %).
    """
    tickers = list(target_by_ticker)
    index = close_df.index
    shares: dict[str, float] = {}
    cash = 1.0
    trades: list[dict[str, Any]] = []
    first_prices = {ticker: close_df[ticker].first_valid_index() for ticker in tickers}

    def total_value(prices: pd.Series) -> float:
        """미상장 종목은 주수가 없으며, 보유한 종목만 유효 가격으로 평가한다."""
        return cash + sum(quantity * float(prices[ticker]) for ticker, quantity in shares.items() if quantity)

    def rebalance_to_target(day: pd.Timestamp, reason: str, selected: list[str]) -> None:
        """밴드를 벗어난 종목을 매도한 뒤 가용 현금으로 매수한다."""
        nonlocal cash
        prices = close_df.loc[day]
        total = total_value(prices)
        if total <= 0:
            return
        orders: list[tuple[str, float, float]] = []
        for ticker in selected:
            price = float(prices[ticker])
            if pd.isna(price) or price <= 0:
                continue
            held_value = shares.get(ticker, 0.0) * price
            current_pct = held_value / total * 100.0
            target_pct = target_by_ticker[ticker] * 100.0
            # 모든 종목을 매매 전 같은 평가액으로 판정한다.
            if abs(target_pct - current_pct) < band_pct:
                continue
            diff_value = total * target_by_ticker[ticker] - held_value
            if diff_value != 0:
                orders.append((ticker, diff_value, current_pct))

        executed: list[tuple[str, str, float, float]] = []
        for ticker, diff_value, current_pct in orders:
            if diff_value >= 0:
                continue
            price = float(prices[ticker])
            fill = price * (1 - sell_slippage / 100.0)
            # 비용을 주수로 나누면 목표 이상으로 팔거나 공매도가 생긴다.
            quantity = min(shares[ticker], -diff_value / price)
            shares[ticker] -= quantity
            cash += quantity * fill
            executed.append((ticker, "sell", fill, current_pct))

        # 밴드 안의 종목을 강제로 팔지 않는다. 부족하면 매수 요청액 비율로 나눈다.
        # 설정한 현금 몫은 남겨 두며, 매수 비용도 이 예산 안에서 지불한다.
        requested = sum(value for _, value, _ in orders if value > 0)
        # 아직 살 수 없는 종목의 배정분을 다른 종목 매수에 쓰지 않는다.
        reserved_weight = cash_target + sum(target_by_ticker[t] for t in tickers if pd.isna(prices[t]))
        available = max(cash - total * reserved_weight, 0.0)
        scale = min(1.0, available / requested) if requested > 0 else 0.0
        for ticker, diff_value, current_pct in orders:
            if diff_value <= 0 or scale <= 0:
                continue
            spend = min(diff_value * scale, max(cash - total * reserved_weight, 0.0))
            if spend <= 0:
                continue
            fill = float(prices[ticker]) * (1 + buy_slippage / 100.0)
            shares[ticker] = shares.get(ticker, 0.0) + spend / fill
            cash -= spend
            executed.append((ticker, "buy", fill, current_pct))

        final_total = total_value(prices)
        for ticker, side, fill, current_pct in executed:
            trades.append(
                {
                    "date": str(day.date()),
                    "ticker": ticker,
                    "side": side,
                    "reason": reason,
                    "price": round(fill, 4),
                    "weight_before_pct": round(current_pct, 2),
                    "weight_after_pct": round(shares[ticker] * float(prices[ticker]) / final_total * 100.0, 2),
                }
            )

    rebalance_to_target(index[0], "최초 매수", tickers)
    current_period = period_key(index[0], rebalance)

    curve: list[float] = []
    cash_curve: dict[pd.Timestamp, float] = {}
    for day in index:
        period = period_key(day, rebalance)
        if period is not None and period != current_period:
            rebalance_to_target(day, "리밸런싱", tickers)
            current_period = period
        elif day != index[0]:
            newly_available = [ticker for ticker in tickers if first_prices[ticker] == day]
            if newly_available:
                rebalance_to_target(day, "가격 이력 시작 후 최초 매수", newly_available)
        prices = close_df.loc[day]
        curve.append(total_value(prices))
        cash_curve[day] = cash / curve[-1] * 100.0

    return {
        "curve": pd.Series(curve, index=index),
        "cash_curve": cash_curve,
        "trades": trades,
        "shares": shares,
        "cash": cash,
    }
