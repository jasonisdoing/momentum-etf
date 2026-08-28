"""포트폴리오 전략 백테스트 — 정한 비중으로 시작해 주기마다 그 비중으로 되돌린다.

모멘텀·신고가와 달리 **종목을 고르지 않으므로 체결 내역이 종목 교체가 아니다.** 리밸런싱
때 리밸런싱 기준을 넘긴 종목만 사고팔며, 그 매매만 비용(슬리피지)을 문다. 그 사이에는 시세대로
흘러가게 둔다 — 그게 이 전략의 전부다.

결과 형태는 신고가·모멘텀과 같다(`start_date`·`strategy_total_pct`·`daily` …). 화면과
합성 어댑터가 전략을 가리지 않고 같은 키를 읽기 때문이다.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from utils.logger import get_app_logger
from utils.pool_settings_store import get_pool_slippage
from utils.portfolio_service import (
    DEFAULT_BACKTEST_MONTHS,
    benchmark_info,
    load_settings,
    validate_settings,
)

logger = get_app_logger()

MAX_BACKTEST_MONTHS = 60

# 리밸런싱 주기 → pandas 기간 라벨. 'none' 은 되돌리지 않는다(최초 매수 후 그대로).
_PERIOD_FREQ: dict[str, str] = {"monthly": "%Y-%m", "quarterly": "%Y-Q", "yearly": "%Y"}


def _period_key(day: pd.Timestamp, rebalance: str) -> str | None:
    """그 날짜가 속한 리밸런싱 구간의 키. 'none' 이면 None(되돌리지 않음)."""
    if rebalance == "monthly":
        return day.strftime("%Y-%m")
    if rebalance == "quarterly":
        return f"{day.year}-Q{(day.month - 1) // 3 + 1}"
    if rebalance == "yearly":
        return str(day.year)
    return None


def _load_close_frame(pool: str, tickers: list[str]) -> pd.DataFrame:
    """[일자 × 티커] 종가 프레임. 가격이 없는 종목은 명시적으로 에러 — 조용히 빼지 않는다."""
    from utils.cache_utils import load_cached_frames_bulk_from_ticker_types

    frames = load_cached_frames_bulk_from_ticker_types([pool], tickers)
    series: dict[str, pd.Series] = {}
    missing: list[str] = []
    for ticker in tickers:
        frame = frames.get(ticker)
        if frame is None or frame.empty or "Close" not in frame.columns:
            missing.append(ticker)
            continue
        close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
        if close.empty:
            missing.append(ticker)
            continue
        series[ticker] = close
    if missing:
        raise RuntimeError(f"가격 캐시가 없어 백테스트를 돌릴 수 없습니다: {', '.join(missing)}")
    return pd.DataFrame(series).sort_index().dropna(how="all")


def _cagr_pct(total_pct: float, months: int) -> float:
    if months <= 0:
        return 0.0
    return ((1 + total_pct / 100) ** (12 / months) - 1) * 100


def _drawdown_pct(curve: pd.Series) -> float:
    return float(((curve / curve.cummax()) - 1).min() * 100)


def _sortino(returns: pd.Series) -> float | None:
    downside = returns[returns < 0]
    deviation = float((downside**2).mean() ** 0.5) if not downside.empty else 0.0
    if deviation <= 0 or len(returns) < 2:
        return None
    return round(float(returns.mean()) / deviation * (252**0.5), 2)


def run_backtest(
    months: int | None = None,
    settings: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """고정 비중 리밸런싱 백테스트. 일별 자산곡선과 리밸런싱 내역을 함께 돌려준다.

    `context` 는 어댑터 계약을 맞추기 위한 자리다 — 이 전략은 무거운 준비물이 없어 쓰지 않는다.
    """
    del context  # 이 전략은 사전 준비물이 없다(계약만 맞춘다)

    settings = validate_settings(settings or load_settings())
    months = int(months or DEFAULT_BACKTEST_MONTHS)
    if not 1 <= months <= MAX_BACKTEST_MONTHS:
        raise ValueError(f"'months' 는 1~{MAX_BACKTEST_MONTHS} 사이여야 합니다.")

    pool = settings["pool"]
    weights = settings["weights"]
    if not weights:
        raise ValueError("담긴 종목이 없습니다 — 화면에서 종목과 비중을 먼저 저장하세요.")

    rebalance = settings["rebalance"]
    band_pct = float(settings["band_pct"])
    buy_slippage, sell_slippage = get_pool_slippage(pool)

    target_by_ticker = {row["ticker"]: float(row["weight_pct"]) / 100.0 for row in weights}
    tickers = list(target_by_ticker)

    close_df = _load_close_frame(pool, tickers)
    # 벤치마크는 종목풀 설정 것 — 다른 전략 화면과 같은 대조군이다.
    from utils.benchmark_curve import load_benchmark_frame

    benchmark_frame = load_benchmark_frame(pool)
    benchmark_close = pd.to_numeric(benchmark_frame["Close"], errors="coerce").dropna()

    # 구간 — 종목·벤치마크가 모두 있는 날만 쓴다.
    index = close_df.dropna().index.intersection(benchmark_close.index)
    if len(index) < 2:
        raise RuntimeError("종목과 벤치마크의 공통 가격 구간이 부족합니다.")
    start = index[-1] - pd.DateOffset(months=months)
    index = index[index >= start]
    if len(index) < 2:
        raise RuntimeError(f"{months}개월치 가격이 부족합니다.")
    close_df = close_df.loc[index, tickers]
    benchmark_close = benchmark_close.loc[index]

    # ── 시뮬레이션 ──────────────────────────────────────────────────────────
    # 자산 1.0 에서 시작해 목표 비중대로 산다. 매수는 슬리피지만큼 비싸게 체결된다.
    # 현금은 사용자가 정한 값이다(종목합 + 현금 = 100% 는 저장 때 검증했다).
    cash_target = float(settings["cash_weight_pct"]) / 100.0
    shares: dict[str, float] = {}
    cash = 1.0
    trades: list[dict[str, Any]] = []

    def rebalance_to_target(day: pd.Timestamp, reason: str) -> None:
        """목표 비중으로 되돌린다 — 리밸런싱 기준을 넘긴 종목만 사고판다."""
        nonlocal cash
        prices = close_df.loc[day]
        total = cash + sum(shares.get(t, 0.0) * float(prices[t]) for t in tickers)
        if total <= 0:
            return
        for ticker in tickers:
            price = float(prices[ticker])
            if price <= 0:
                continue
            held_value = shares.get(ticker, 0.0) * price
            current_pct = held_value / total * 100.0
            target_pct = target_by_ticker[ticker] * 100.0
            gap = target_pct - current_pct
            # 기준 안이면 두고 본다 — 가격 드리프트로 매매하지 않는 것이 이 전략의 규칙이다.
            if abs(gap) < band_pct:
                continue
            target_value = total * target_by_ticker[ticker]
            diff_value = target_value - held_value
            # 체결가 — 사면 비싸게, 팔면 싸게(편도 슬리피지).
            fill = price * (1 + buy_slippage / 100.0) if diff_value > 0 else price * (1 - sell_slippage / 100.0)
            delta_shares = diff_value / fill
            shares[ticker] = shares.get(ticker, 0.0) + delta_shares
            cash -= delta_shares * fill
            trades.append(
                {
                    "date": str(day.date()),
                    "ticker": ticker,
                    "side": "buy" if diff_value > 0 else "sell",
                    "reason": reason,
                    "price": round(fill, 4),
                    "weight_before_pct": round(current_pct, 2),
                    "weight_after_pct": round(target_pct, 2),
                }
            )

    rebalance_to_target(index[0], "최초 매수")
    current_period = _period_key(index[0], rebalance)

    curve: list[float] = []
    for day in index:
        period = _period_key(day, rebalance)
        if period is not None and period != current_period:
            rebalance_to_target(day, "리밸런싱")
            current_period = period
        prices = close_df.loc[day]
        curve.append(cash + sum(shares.get(t, 0.0) * float(prices[t]) for t in tickers))

    strategy = pd.Series(curve, index=index)
    # 벤치마크는 **시작일 시가**를 1 로 둔다 — 전략도 그날 시가에 사기 때문이다(공용 함수).
    from utils.benchmark_curve import benchmark_growth

    benchmark = benchmark_growth(pool, index)
    strategy_total = float((strategy.iloc[-1] / strategy.iloc[0] - 1) * 100)
    benchmark_total = float((benchmark.iloc[-1] - 1) * 100)
    strategy_norm = strategy / float(strategy.iloc[0])

    return {
        "start_date": str(index[0].date()),
        "end_date": str(index[-1].date()),
        "months": months,
        "strategy_total_pct": round(strategy_total, 2),
        "strategy_cagr_pct": round(_cagr_pct(strategy_total, months), 2),
        "strategy_mdd_pct": round(_drawdown_pct(strategy_norm), 2),
        "strategy_sortino": _sortino(strategy_norm.pct_change().dropna()),
        "benchmark_total_pct": round(benchmark_total, 2),
        "benchmark_cagr_pct": round(_cagr_pct(benchmark_total, months), 2),
        "benchmark_mdd_pct": round(_drawdown_pct(benchmark), 2),
        "benchmark_sortino": _sortino(benchmark.pct_change().dropna()),
        "benchmark_name": benchmark_info(pool)["name"],
        # 이 전략의 '체결'은 종목 교체가 아니라 비중 되돌리기다 — 승률·평균손익 개념이 없다.
        "trades": list(reversed(trades)),
        "rebalance_count": sum(1 for trade in trades if trade["reason"] == "리밸런싱"),
        "cash_weight_pct": round(cash_target * 100, 2),
        "daily": [
            {
                "date": str(day.date()),
                "strategy_pct": round((float(strategy_norm.loc[day]) - 1) * 100, 2),
                "benchmark_pct": round((float(benchmark.loc[day]) - 1) * 100, 2),
            }
            for day in index
        ],
    }
