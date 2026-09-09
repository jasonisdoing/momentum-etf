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

from config import CACHE_TTL_COMPUTE
from core.strategy.portfolio.backtest import simulate_portfolio
from utils.logger import get_app_logger
from utils.pool_settings_store import get_pool_slippage
from utils.pool_signal_backtest_service import validate_backtest_months
from utils.portfolio_service import (
    DEFAULT_BACKTEST_MONTHS,
    benchmark_info,
    load_settings,
    validate_settings,
)
from utils.ttl_cache import TtlCache

logger = get_app_logger()


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


def _overlay_live_last_bar(
    pool: str, close_df: pd.DataFrame, benchmark_close: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """장중이면 실시간 가격을 **마지막 봉**으로 얹는다(AGENTS.md §10-6) — 운용 현황 전용.

    규칙·수식은 그대로고 입력만 잠정이라, 종가가 확정되면 확정 계산과 일치한다.
    전 종목의 실시간 시세가 있어야 얹는다 — 일부만 잠정이면 비중 판정이 뒤섞인다.
    벤치마크는 오늘 값이 없어 마지막 확정값을 이월한다 — 현황(보유·지시)만 읽는 경로라
    성과 숫자에는 쓰이지 않는다.
    """
    from utils.slot_positions import _live_quotes

    tickers = list(close_df.columns)
    quotes = _live_quotes(pool, tickers, close_df.index[-1])
    if not quotes["live"]:
        return close_df, benchmark_close
    prices = {t: (quotes["by_ticker"].get(t) or {}).get("price") for t in tickers}
    if any(p is None for p in prices.values()):
        return close_df, benchmark_close
    session_ts = pd.Timestamp(str(quotes["traded_at"])[:10])
    close_df = close_df.copy()
    close_df.loc[session_ts] = pd.Series(prices)
    benchmark_close = benchmark_close.copy()
    benchmark_close.loc[session_ts] = float(benchmark_close.iloc[-1])
    return close_df, benchmark_close


def run_backtest(
    months: int | None = None,
    settings: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    *,
    start_date: str | None = None,
    with_live_last_bar: bool = False,
) -> dict[str, Any]:
    """고정 비중 리밸런싱 백테스트. 일별 자산곡선과 리밸런싱 내역을 함께 돌려준다.

    `context` 는 어댑터 계약을 맞추기 위한 자리다 — 이 전략은 무거운 준비물이 없어 쓰지 않는다.
    ``with_live_last_bar`` 는 운용 현황 전용이다 — 성과 비교 백테스트는 확정 데이터만 쓴다.
    """
    del context  # 이 전략은 사전 준비물이 없다(계약만 맞춘다)

    settings = validate_settings(settings or load_settings())
    months = int(months or DEFAULT_BACKTEST_MONTHS)
    validate_backtest_months(months)

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

    if with_live_last_bar:
        close_df, benchmark_close = _overlay_live_last_bar(pool, close_df, benchmark_close)

    # 구간 — 종목·벤치마크가 모두 있는 날만 쓴다.
    index = close_df.dropna().index.intersection(benchmark_close.index)
    if len(index) < 2:
        raise RuntimeError("종목과 벤치마크의 공통 가격 구간이 부족합니다.")
    start = pd.Timestamp(start_date) if start_date is not None else index[-1] - pd.DateOffset(months=months)
    index = index[index >= start]
    if len(index) < 2:
        raise RuntimeError(f"{months}개월치 가격이 부족합니다.")
    close_df = close_df.loc[index, tickers]
    benchmark_close = benchmark_close.loc[index]

    # ── 시뮬레이션 — 핵심 계산은 core 로 분리했다(외부 조회 없는 순수 함수) ──
    # 자산 1.0 에서 시작해 목표 비중대로 산다. 매수는 슬리피지만큼 비싸게 체결된다.
    # 현금은 사용자가 정한 값이다(종목합 + 현금 = 100% 는 저장 때 검증했다).
    cash_target = float(settings["cash_weight_pct"]) / 100.0
    simulated = simulate_portfolio(
        close_df=close_df,
        target_by_ticker=target_by_ticker,
        cash_target=cash_target,
        band_pct=band_pct,
        rebalance=rebalance,
        buy_slippage=buy_slippage,
        sell_slippage=sell_slippage,
    )
    strategy = simulated["curve"]
    cash_curve = simulated["cash_curve"]
    trades = simulated["trades"]
    shares = simulated["shares"]
    cash = simulated["cash"]
    # 벤치마크는 **시작일 시가**를 1 로 둔다 — 전략도 그날 시가에 사기 때문이다(공용 함수).
    from utils.benchmark_curve import benchmark_growth

    benchmark = benchmark_growth(pool, index)
    strategy_total = float((strategy.iloc[-1] / strategy.iloc[0] - 1) * 100)
    benchmark_total = float((benchmark.iloc[-1] - 1) * 100)
    strategy_norm = strategy / float(strategy.iloc[0])

    return {
        "start_date": str(index[0].date()),
        # 합성은 저장 비중 대신 이 최종 상태를 읽는다. 주기·밴드 판정을 다시 만들지 않는다.
        "as_of": str(index[-1].date()),
        "open_positions": [
            {
                "ticker": ticker,
                "shares": shares.get(ticker, 0.0),
                "price": float(close_df.at[index[-1], ticker]),
                "sleeve_weight_pct": shares.get(ticker, 0.0)
                * float(close_df.at[index[-1], ticker])
                / float(strategy.iloc[-1])
                * 100.0,
            }
            for ticker in tickers
        ],
        "sleeve_cash_weight_pct": cash / float(strategy.iloc[-1]) * 100.0,
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
                "cash_weight_pct": cash_curve[day],
                "strategy_pct": round((float(strategy_norm.loc[day]) - 1) * 100, 2),
                "benchmark_pct": round((float(benchmark.loc[day]) - 1) * 100, 2),
            }
            for day in index
        ],
    }


# 설정이 같으면 결과도 같으므로 짧게 재사용한다 — 개별 화면과 합성(목표·슬리브 몫)이
# 같은 실행 결과를 읽고, 같은 요청 흐름에서 엔진을 여러 번 돌리지 않는다.
_POSITIONS_CACHE = TtlCache(CACHE_TTL_COMPUTE, name="portfolio_positions")


def current_positions(settings: dict[str, Any]) -> dict[str, Any]:
    """개별 운용 현황과 합성이 공유하는 고정 시작일 기준 포트폴리오 상태.

    장중이면 실시간 가격을 마지막 봉으로 쓴 같은 엔진의 상태다(AGENTS.md §10-6) —
    보유 비중·리밸런싱 지시가 실시간 기준으로 움직이고, 종가 확정 후 백테스트와 일치한다.
    ``daily`` 는 이 상태를 만든 실행의 일별 곡선 — 합성 슬리브 몫이 같은 실행 결과를 읽는다.
    """
    from utils.strategy_settings import require_start_date

    start_date = require_start_date(settings)

    def compute() -> dict[str, Any]:
        result = run_backtest(DEFAULT_BACKTEST_MONTHS, settings, start_date=start_date, with_live_last_bar=True)
        return {key: result[key] for key in ("as_of", "open_positions", "sleeve_cash_weight_pct", "trades", "daily")}

    # 키에 슬리피지까지 넣는다 — 풀 설정을 바꾸면 즉시 새 값으로 계산돼야 한다.
    key = _POSITIONS_CACHE.make_key(settings, start_date, get_pool_slippage(settings["pool"]))
    return _POSITIONS_CACHE.get_or_compute(key, compute)
