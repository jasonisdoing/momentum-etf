"""엔진의 날짜별 편입 상태에 고정 기준금액 운용을 적용하는 합성 재생."""

from __future__ import annotations

import math

import pandas as pd

from core.strategy.mix.capital_policy import capital_trade_quantity


def replay_capital(
    *,
    close: pd.DataFrame,
    opened: pd.DataFrame,
    fx: pd.Series,
    targets: dict[str, dict[str, float]],
    capital_krw: float,
    harvest_pct: float,
    refill_pct: float,
    costs: dict[str, tuple[float, float]],
) -> dict:
    """전일 종가로 수량을 정하고 당일 시가에 체결한다. 현금 부족 시 차입하지 않는다.

    목표는 KRW, 가격·현금은 계좌 통화다. 외부 인출은 재생하지 않고 현금으로 보유한다.
    첫날에는 이전 봉이 없으므로 당일 시가로 초기 배정하며, 이후 회수 판단에는 미래 가격을 쓰지 않는다.
    """
    held = {ticker: 0 for ticker in close.columns}
    cash = capital_krw / float(fx.iloc[0])
    curve = {}
    trades = []
    previous_targets: dict[str, float] = {}
    pending_sells: dict[str, int] = {}
    first_price_days = {ticker: close[ticker].first_valid_index() for ticker in close.columns}
    for i, day in enumerate(close.index):
        date = str(day.date())
        amounts = targets[date]
        decision_prices = close.iloc[i - 1] if i else opened.iloc[0]
        decision_fx = float(fx.iloc[i - 1] if i else fx.iloc[0])
        orders = {}
        for ticker in held:
            amount = amounts.get(ticker, 0.0)
            price = decision_prices[ticker]
            if pd.isna(price):
                # 엔진이 최초 가격일에 편입한 종목만 그날 시가로 초기 배정한다.
                # 이후의 데이터 누락을 현재가로 메우거나 상장 전으로 가격을 역채우지 않는다.
                if day != first_price_days[ticker] or amount <= 0:
                    continue
                price = opened.at[day, ticker]
                if pd.isna(price) or price <= 0:
                    raise ValueError(f"합성 최초 진입 시가가 없습니다: {date} {ticker}")
            trade = capital_trade_quantity(
                held=held[ticker],
                price=float(price),
                target_amount=amount / decision_fx,
                harvest_pct=harvest_pct,
                refill_pct=refill_pct,
                previous_target_amount=previous_targets.get(ticker, 0.0) / decision_fx,
            )
            if trade:
                orders[ticker] = trade
        for ticker, quantity in pending_sells.items():
            # 거래정지 중 확정된 매도는 이후 목표가 달라져도 첫 체결 가능일에 실행한다.
            orders[ticker] = -min(held[ticker], max(quantity, -orders.get(ticker, 0)))
        # 매도 대금으로 당일 매수를 충당하되 가용 현금을 넘기는 체결은 만들지 않는다.
        for ticker, trade in orders.items():
            if trade >= 0:
                continue
            price = opened.at[day, ticker]
            if pd.isna(price) or price <= 0:
                pending_sells[ticker] = -trade
                continue
            cash -= trade * float(price) * (1 - costs[ticker][1])
            held[ticker] += trade
            pending_sells.pop(ticker, None)
            trades.append({"date": date, "ticker": ticker, "side": "sell", "quantity": -trade, "price": float(price)})
        requests = {}
        for ticker, trade in orders.items():
            if trade <= 0:
                continue
            price = opened.at[day, ticker]
            if pd.isna(price) or price <= 0:
                # 시가 없는 날의 신규 진입은 슬롯 엔진처럼 체결하지 않는다.
                continue
            requests[ticker] = trade * float(price) * (1 + costs[ticker][0])
        total = sum(requests.values())
        scale = min(1.0, max(cash, 0.0) / total) if total else 0.0
        for ticker in requests:
            quantity = math.floor(orders[ticker] * scale)
            if not quantity:
                continue
            price = float(opened.at[day, ticker])
            cash -= quantity * price * (1 + costs[ticker][0])
            held[ticker] += quantity
            trades.append({"date": date, "ticker": ticker, "side": "buy", "quantity": quantity, "price": price})
        value = cash
        for ticker, quantity in held.items():
            if quantity:
                price = close.at[day, ticker]
                if pd.isna(price):
                    raise ValueError(f"합성 평가 종가가 없습니다: {date} {ticker}")
                value += quantity * float(price)
        curve[date] = value * float(fx.iloc[i]) / capital_krw
        previous_targets = amounts
    return {"curve": pd.Series(curve), "cash": cash, "holdings": held, "executions": trades}
