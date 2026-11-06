"""사고팔기 백테스트 — 종목풀 하나에 켠 전략 전체를 함께 굴린 결과 (합성 슬리브가 쓴다).

종목별 사다리 시뮬레이션(`core.strategy.trade.backtest`)을 전략마다 돌려 손익을 합친다.

- 기준 자본 = 합성 슬리브 한도(슬리브 비중 × 합성 운용 기준금액)다. 전략마다 한도를 균등 분할해
  그 한도를 넘는 회차부터는 사지 않고, 일별 수익률은 기준 자본 대비 손익으로 낸다.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.strategy.trade.backtest import LadderSpec, simulate_ladder
from utils.cache_utils import load_cached_frames_bulk_from_ticker_types
from utils.effective_prices import confirmed_bars
from utils.pool_signal_backtest_service import validate_backtest_months
from utils.settings_loader import get_ticker_type_settings
from utils.strategy_trade_service import load_settings, sleeve_for_pool, strategy_budget

COUNTRY = "kor"
_OHLC = ["Open", "High", "Low", "Close"]


def _load_bars(pool: str, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """종목별 확정 일봉 — 진행 중인 날의 봉은 뺀다(판정이 흔들리지 않게)."""
    frames = load_cached_frames_bulk_from_ticker_types([pool], tickers)
    bars: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        frame = frames.get(ticker)
        if frame is None or frame.empty or not set(_OHLC) <= set(frame.columns):
            raise ValueError(f"{ticker} 가격 캐시(시가·고가·저가·종가)를 불러올 수 없습니다.")
        confirmed = confirmed_bars(frame, COUNTRY)[_OHLC].apply(pd.to_numeric, errors="coerce").dropna()
        confirmed.index = pd.DatetimeIndex(confirmed.index).normalize()
        bars[ticker] = confirmed.sort_index()
    return bars


def _simulation_input(bars: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """기간 시작 직전 봉(기준 봉)부터 끝까지 — 직전 봉이 없으면 가진 첫 봉이 기준이다."""
    before = bars[bars.index < start]
    window = bars[bars.index >= start]
    return pd.concat([before.tail(1), window]) if not before.empty else window


def _simulate_pool(pool: str, months: int) -> dict[str, Any]:
    """종목풀의 사고팔기 전략 전체를 최근 ``months`` 개월 시뮬레이션해 손익(원)과 체결을 모은다."""
    validate_backtest_months(months)
    strategies = [entry for entry in load_settings()["strategies"].values() if entry["pool"] == pool]
    if not strategies:
        raise ValueError(f"{pool} 종목풀에 사고팔기 전략이 없습니다.")
    sleeve = sleeve_for_pool(pool)
    if sleeve is None:
        raise ValueError(f"{pool} 종목풀의 사고팔기 슬리브가 어느 계좌에도 붙어 있지 않습니다 — 합성에서 붙이세요.")
    budget = strategy_budget(sleeve, len(strategies))

    pool_settings = get_ticker_type_settings(pool) or {}
    slippage = pool_settings.get("BUY_SLIPPAGE_PCT")
    if slippage is None:
        raise ValueError(f"{pool} 종목풀에 매수 슬리피지가 설정돼 있지 않습니다.")

    tickers = sorted({entry["ticker"] for entry in strategies})
    bars = _load_bars(pool, tickers)
    end = max(bars[entry["ticker"]].index[-1] for entry in strategies)
    start = end - pd.DateOffset(months=months)

    results = []
    for entry in strategies:
        data = _simulation_input(bars[entry["ticker"]], start)
        spec = LadderSpec(
            entry["ticker"], entry["name"], entry["trigger_pct"], entry["round_quantity"], entry["rounds"], budget
        )
        results.append(simulate_ladder(data, spec, market_buy_slippage_pct=float(slippage)))

    calendar = pd.DatetimeIndex(sorted(set().union(*(result["pnl"].index for result in results))))
    total_pnl = pd.Series(0.0, index=calendar)
    for result in results:
        # 상장 전·거래 정지일은 직전 손익을 그대로 잇는다(상장 전은 0).
        total_pnl += result["pnl"].reindex(calendar).ffill().fillna(0.0)
    trades = [trade for result in results for trade in result["trades"]]
    return {
        "pnl": total_pnl,
        "base_capital": float(sleeve["limit_krw"]),
        "trades": trades,
    }


def _order_trades(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """보유중 행이 위, 그 아래 매도일 최신순."""
    holding = sorted((row for row in trades if row["exit_date"] is None), key=lambda row: (row["ticker"], row["round"]))
    closed = sorted(
        (row for row in trades if row["exit_date"] is not None),
        key=lambda row: (row["exit_date"], row["entry_date"]),
        reverse=True,
    )
    return holding + closed


def run_for_mix(pool: str, months: int) -> dict[str, Any]:
    """합성 슬리브용 결과 — 슬리브 한도 대비 누적 수익률(%)의 일별 곡선, 체결, 원화 손익.

    합성은 이 손익을 자기 곡선에 더한다(슬리브 몫은 합성 재생에서 현금으로 남아 있다).
    """
    simulated = _simulate_pool(pool, months)
    base_capital = simulated["base_capital"]
    pnl = simulated["pnl"]
    return {
        "daily": [
            {"date": str(day.date()), "strategy_pct": round(float(value) / base_capital * 100.0, 4)}
            for day, value in pnl.items()
        ],
        "pnl_krw": {str(day.date()): float(value) for day, value in pnl.items()},
        "trades": _order_trades(simulated["trades"]),
    }
