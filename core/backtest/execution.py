import numpy as np
import pandas as pd

from .domain import BacktestConfig, MarketData, PortfolioState


def calculate_trade_price(
    i: int, total_days: int, open_vals: np.ndarray, close_vals: np.ndarray, country_code: str, is_buy: bool
) -> float:
    """Calculates trade price (next day open) with slippage."""
    if i + 1 >= total_days:
        return 0.0

    price = open_vals[i + 1]  # Next day open
    if np.isnan(price) or price <= 0:
        return 0.0

    # Slippage
    slippage = 0.0005  # 0.05%
    if country_code == "kor":
        # Krx tax/slippage might be different but let's stick to simple model
        pass

    if is_buy:
        return price * (1 + slippage)
    else:
        return price * (1 - slippage)


def apply_trade(
    state: PortfolioState,
    ticker: str,
    amount: float,
    price: float,
    shares: float,
    decision: str,
    dt: pd.Timestamp,
    commission: float = 0.0,
    data: MarketData | None = None,
    i: int = -1,
) -> None:
    """Updates portfolio state with a executed trade."""
    if shares == 0:
        return

    # Update Cash
    if amount > 0:  # Buy
        state.cash -= amount
    else:  # Sell
        state.cash -= amount  # amount is negative for sell, so cash increases

    # Sync CASH Daily Record if exists for today
    if "CASH" in state.daily_records and state.daily_records["CASH"]:
        cash_rec = state.daily_records["CASH"][-1]
        if cash_rec["date"] == dt:
            cash_rec["shares"] = state.cash
            cash_rec["pv"] = state.cash

    # Update Position
    pos = state.positions.setdefault(
        ticker, {"shares": 0.0, "avg_cost": 0.0, "buy_block_until": -1, "sell_block_until": -1}
    )

    prev_shares = pos["shares"]
    prev_cost = pos["avg_cost"]

    if decision.startswith("BUY"):
        new_shares = prev_shares + shares
        # Weighted Average Cost
        if new_shares > 0:
            total_cost = (prev_shares * prev_cost) + (shares * price)
            pos["avg_cost"] = total_cost / new_shares
        pos["shares"] = new_shares
    else:  # SELL
        pos["shares"] = max(0.0, prev_shares + shares)  # shares is negative
        if pos["shares"] == 0:
            pos["avg_cost"] = 0.0

    # Record Trade
    stats = {}
    if decision.startswith("SELL"):
        stats["profit"] = (-amount) - ((-shares) * prev_cost)
        stats["pct"] = (stats["profit"] / ((-shares) * prev_cost)) * 100 if prev_cost > 0 else 0.0

    trade_record = {
        "date": dt,
        "ticker": ticker,
        "decision": decision,
        "price": price,
        "shares": shares,  # Signed (+ buy, - sell)
        "amount": amount,  # Signed
        "remaining_shares": pos["shares"],
        "profit": stats.get("profit", 0.0),
        "pct": stats.get("pct", 0.0),
    }
    state.trades.append(trade_record)

    # Update (or Create) Daily Record for this ticker
    record = None
    if ticker in state.daily_records and state.daily_records[ticker]:
        last_rec = state.daily_records[ticker][-1]
        if last_rec["date"] == dt:
            record = last_rec

    if record is None and data is not None and i >= 0:
        # Create missing record (e.g. for new Buy of non-held ticker)
        # We use current price/metrics
        price_val = data.close_prices[ticker][i]

        score_val = data.scores[ticker][i] if pd.notna(data.scores[ticker][i]) else None
        rsi_val = data.rsi_scores[ticker][i] if pd.notna(data.rsi_scores[ticker][i]) else None
        ma_val = data.ma_values[ticker][i] if pd.notna(data.ma_values[ticker][i]) else None
        filter_val = int(data.buy_signals[ticker][i])

        record = {
            "date": dt,
            "price": price_val if pd.notna(price_val) else price,
            "shares": pos["shares"],
            "pv": pos["shares"] * price,
            "decision": decision,
            "avg_cost": pos["avg_cost"],
            "trade_amount": amount,
            "trade_profit": 0.0,
            "trade_pl_pct": 0.0,
            "note": "",
            "score": score_val,
            "rsi_score": rsi_val,
            "signal1": ma_val,
            "filter": filter_val,
        }
        state.daily_records.setdefault(ticker, []).append(record)

    if record:
        # Update existing or newly created record
        record["decision"] = decision
        record["trade_amount"] = amount
        # Ensure shares/pv reflect post-trade
        record["shares"] = pos["shares"]
        record["avg_cost"] = pos["avg_cost"]
        # Recalculate PV?
        # Logic: PV should be Close * Shares.
        # If record existed, it had 'price' (Close).
        # We keep 'price' but update shares.
        current_price = record["price"]
        record["pv"] = record["shares"] * current_price

        if "profit" in stats:
            record["trade_profit"] = stats["profit"]
            record["trade_pl_pct"] = stats["pct"]


def execute_rebalance_sell(
    i: int,
    total_days: int,
    state: PortfolioState,
    ticker: str,
    target_shares: float,
    data: MarketData,
    config: BacktestConfig,
    dt: pd.Timestamp,
) -> float:
    """Executes sell if current shares > target shares. Returns proceeds."""
    current_shares = state.get_shares(ticker)
    if current_shares <= target_shares:
        return 0.0

    sell_qty = current_shares - target_shares
    sell_price = calculate_trade_price(
        i, total_days, data.open_prices[ticker], data.close_prices[ticker], config.country_code, is_buy=False
    )

    if sell_price <= 0:
        return 0.0

    amount = -(sell_qty * sell_price)  # Negative for cash flow out (but here inflow)

    apply_trade(state, ticker, amount, sell_price, -sell_qty, "SELL_REBALANCE", dt, data=data, i=i)
    return -amount  # Return positive cash inflow


def execute_rebalance_buy(
    i: int,
    total_days: int,
    state: PortfolioState,
    ticker: str,
    target_shares: float,
    data: MarketData,
    config: BacktestConfig,
    dt: pd.Timestamp,
) -> float:
    """Executes buy if current shares < target shares. Returns cost."""
    current_shares = state.get_shares(ticker)
    if current_shares >= target_shares:
        return 0.0

    buy_qty = target_shares - current_shares
    buy_price = calculate_trade_price(
        i, total_days, data.open_prices[ticker], data.close_prices[ticker], config.country_code, is_buy=True
    )

    if buy_price <= 0:
        return 0.0

    cost = buy_qty * buy_price

    if cost > state.cash:
        # Cap by cash
        buy_qty = int(state.cash / buy_price)
        cost = buy_qty * buy_price

    if buy_qty <= 0:
        return 0.0

    decision = "BUY_REBALANCE"
    if current_shares == 0:
        decision = "BUY"  # Maps to "✅ 신규 매수"

    apply_trade(state, ticker, cost, buy_price, buy_qty, decision, dt, data=data, i=i)
    return cost
