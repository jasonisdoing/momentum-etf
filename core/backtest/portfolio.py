from typing import Any

import pandas as pd

from .domain import BacktestConfig, PortfolioState


def get_current_holdings_value(state: PortfolioState, current_prices: dict[str, float]) -> float:
    """Calculates total value of current holdings."""
    total_val = 0.0
    for ticker, pos in state.positions.items():
        shares = pos.get("shares", 0)
        if shares > 0:
            price = current_prices.get(ticker)
            if price and price > 0:
                total_val += shares * price
    return total_val


def calculate_rebalance_targets(
    config: BacktestConfig,
    current_equity: float,
    available_tickers: list[str],
    today_scores: dict[str, float],
    today_prices: dict[str, float],
) -> dict[str, float]:
    """
    Calculates target shares for quarterly rebalancing.
    Logic:
    1. Group available tickers by Bucket.
    2. Select Top N per bucket based on Score (allow negative if configured).
    3. Equal weight allocation among selected tickers.
    """
    targets = {}

    # 1. Group by Bucket
    bucket_scores = {}  # bucket_id -> list of (score, ticker)
    for ticker in available_tickers:
        b_id = config.bucket_map.get(ticker, 0)
        if b_id == 0:
            continue

        score = today_scores.get(ticker, float("-inf"))
        if not config.allow_negative_score and score <= 0:
            continue

        bucket_scores.setdefault(b_id, []).append((score, ticker))

    # 2. Select Top N per bucket
    selected_tickers = []
    # Assumes buckets 1 to 5 (or however many defined)
    # We iterate through keys present in data
    for b_id in bucket_scores:
        candidates = bucket_scores[b_id]
        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        # Select Top N
        selected = [t for s, t in candidates[: config.bucket_topn]]
        selected_tickers.extend(selected)

    # 3. Allocation (Equal Weight)
    total_selected = len(selected_tickers)
    if total_selected == 0:
        return {}

    target_val_per_stock = current_equity / total_selected

    for ticker in selected_tickers:
        price = today_prices.get(ticker)
        if price and price > 0:
            targets[ticker] = target_val_per_stock / price

    return targets


# --- Helpers for Compatibility & External Use ---


def get_sell_states() -> set[str]:
    """Returns set of sell states."""
    return {"SELL_TREND", "SELL_REPLACE", "CUT_STOPLOSS", "SELL_RSI"}


def get_hold_states() -> set[str]:
    """Returns set of hold states (including pending sell)."""
    return {
        "HOLD",
        "SELL_TREND",
        "SELL_REPLACE",
        "CUT_STOPLOSS",
        "SELL_RSI",
    }


def count_current_holdings(items: list, *, get_state_func=None) -> int:
    """Counts currently held items (including pending sells)."""
    hold_states = get_hold_states()
    if get_state_func:
        return sum(1 for item in items if get_state_func(item) in hold_states)
    else:
        return sum(1 for item in items if isinstance(item, dict) and str(item.get("state", "")).upper() in hold_states)


def check_buy_candidate_filters(rsi_score: float, rsi_sell_threshold: float) -> tuple[bool, str]:
    """Checks buy candidate filters (RSI)."""
    if rsi_score >= rsi_sell_threshold:
        return False, f"RSI 과매수 (RSI점수: {rsi_score:.1f})"
    return True, ""


def calculate_buy_budget(cash: float, current_holdings_value: float, top_n: int) -> float:
    """Calculates buy budget per stock."""
    if cash <= 0 or top_n <= 0:
        return 0.0
    total_equity = cash + max(current_holdings_value, 0.0)
    if total_equity <= 0:
        return 0.0
    target_value = total_equity / top_n
    if target_value <= 0:
        return 0.0
    return min(target_value, cash)


def calculate_held_count(position_state: dict) -> int:
    """Calculates number of held positions from state."""
    return sum(1 for pos in position_state.values() if pos.get("shares", 0) > 0)


def validate_portfolio_topn(topn: int, account_id: str = "") -> None:
    """Validates portfolio TopN."""
    if topn <= 0:
        msg = f"'{account_id}' 계정의 " if account_id else ""
        raise ValueError(f"{msg}PORTFOLIO_TOPN은 0보다 커야 합니다.")


def calculate_cooldown_blocks(
    trade_cooldown_info: dict[str, dict[str, Any]],
    cooldown_days: int,
    base_date: Any,
    country_code: str,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Calculates cooldown blocks."""
    from utils.data_loader import count_trading_days

    sell_cooldown_block = {}
    buy_cooldown_block = {}

    try:
        base_date_norm = pd.to_datetime(base_date).normalize()
    except Exception:
        return {}, {}

    if not (cooldown_days and cooldown_days > 0):
        return {}, {}

    # Optimized lookup logic simplified for porting
    # Assuming caching handled by caller or simple repeated calls

    for tkr, info in (trade_cooldown_info or {}).items():
        if not isinstance(info, dict):
            continue
        last_sell = info.get("last_sell")
        last_buy = info.get("last_buy")

        # 1. Sell Cooldown
        if last_buy:
            try:
                dt = pd.to_datetime(last_buy).normalize()
                if dt <= base_date_norm:
                    days = count_trading_days(country_code, dt, base_date_norm)
                    if days <= cooldown_days:
                        sell_cooldown_block[tkr] = {"last_buy": dt, "days_since": days}
            except Exception:
                pass

        # 2. Buy Cooldown
        if last_sell:
            try:
                dt = pd.to_datetime(last_sell).normalize()
                if dt <= base_date_norm:
                    days = count_trading_days(country_code, dt, base_date_norm)
                    if days <= cooldown_days:
                        buy_cooldown_block[tkr] = {"last_sell": dt, "days_since": days}
            except Exception:
                pass

    return sell_cooldown_block, buy_cooldown_block
