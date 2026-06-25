"""무한매수법(buy) 전략 엔진 - 1단계 단순화 버전.

규칙(단순화):
  - 단일 대상 종목을 원금에서 매일 1/divisions 씩 분할 매수한다.
  - 매수/익절 체결은 그날 시초가(open) 기준 (당일 종가 신호 가정 없음, 룩어헤드 없음).
  - 보유 전량 평가가 평단 × (1 + take_profit_pct/100) 에 도달하면 그날 시초가에 전량 익절하고
    다음 거래일부터 새 사이클(1회차)을 다시 시작한다.
  - divisions 회까지 매수하면(원금 소진) 추가 매수를 멈추고 보유를 유지(존버)하며 익절만 대기한다.
  - 미투입 현금은 수익률 0으로 보유한다. CAGR/MDD 등은 원금 전체(현금+주식) 기준으로 산출한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from leverage.constants import INITIAL_CAPITAL_KRW
from leverage.data_adapter import compute_bounds, current_trading_day, download_opens, download_prices
from leverage.report import format_kr_money, render_table_eaw


def _annualized_metrics(equity: pd.Series) -> dict:
    """일별 평가액 시계열에서 수익률 지표를 계산한다."""
    if len(equity) < 2:
        return {"period_return": 0.0, "cagr": 0.0, "vol": 0.0, "sharpe": 0.0, "mdd": 0.0}

    initial = equity.iloc[0]
    final = equity.iloc[-1]
    period_return = final / initial - 1.0
    cagr = (final / initial) ** (252 / len(equity)) - 1.0

    daily_rets = equity.pct_change(fill_method=None).dropna()
    vol = daily_rets.std() * np.sqrt(252) if not daily_rets.empty else 0.0
    sharpe = cagr / vol if vol and not np.isnan(vol) else 0.0

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    mdd = drawdown.min()

    return {
        "period_return": float(period_return),
        "cagr": float(cagr),
        "vol": float(vol) if not np.isnan(vol) else 0.0,
        "sharpe": float(sharpe) if not np.isnan(sharpe) else 0.0,
        "mdd": float(mdd),
    }


def simulate_infinite_buy(
    opens: pd.Series,
    closes: pd.Series,
    *,
    divisions: int,
    take_profit_pct: float,
    slippage: float,
    initial_capital: float = INITIAL_CAPITAL_KRW,
) -> dict:
    """무한매수법 시뮬레이션. opens/closes 는 동일 인덱스의 대상 종목 시가/종가."""
    index = opens.index.intersection(closes.index)
    opens = opens.loc[index]
    closes = closes.loc[index]

    slip = slippage / 100.0
    tp = take_profit_pct / 100.0
    daily_budget = initial_capital / divisions

    cash = initial_capital
    qty = 0.0
    avg = 0.0  # 평단가 (체결가 기준, 슬리피지 포함)
    buys_done = 0
    cycles = 0  # 완료된 익절 사이클 수
    win_cycles = 0

    equity_values: list[float] = []
    daily_records: list[dict] = []

    for date in index:
        open_px = float(opens.loc[date])
        close_px = float(closes.loc[date])
        action = "HOLD"
        traded_qty = 0.0

        # 1) 익절 판정: 보유분이 평단 목표가에 도달하면 시초가 전량 매도
        sold_today = False
        if qty > 0 and open_px >= avg * (1 + tp):
            sell_px = open_px * (1 - slip)
            cash += qty * sell_px
            cycles += 1
            win_cycles += 1  # 익절은 항상 이익 실현
            action = "SELL"
            traded_qty = qty
            qty = 0.0
            avg = 0.0
            buys_done = 0
            sold_today = True

        # 2) 매수: 익절한 날은 쉬고 다음 거래일부터 새 사이클을 시작
        if not sold_today and buys_done < divisions and cash > 0:
            spend = min(daily_budget, cash)
            buy_px = open_px * (1 + slip)
            bought = spend / buy_px
            new_qty = qty + bought
            avg = (avg * qty + bought * buy_px) / new_qty if new_qty > 0 else 0.0
            qty = new_qty
            cash -= spend
            buys_done += 1
            action = "BUY"
            traded_qty = bought

        equity = cash + qty * close_px
        equity_values.append(equity)
        daily_records.append(
            {
                "date": date,
                "action": action,
                "open": open_px,
                "close": close_px,
                "traded_qty": traded_qty,
                "qty": qty,
                "avg": avg,
                "buys_done": buys_done,
                "cash": cash,
                "equity": equity,
            }
        )

    equity_series = pd.Series(equity_values, index=index, name="equity")
    metrics = _annualized_metrics(equity_series)

    last = daily_records[-1] if daily_records else None
    return {
        "equity": equity_series,
        "daily_records": daily_records,
        "cycles": cycles,
        "win_cycles": win_cycles,
        "final_qty": qty,
        "final_avg": avg,
        "final_cash": cash,
        "final_buys_done": buys_done,
        "last": last,
        **metrics,
    }


def _load_target_series(settings: dict, drop_today: bool = False):
    """대상 종목의 시가/종가 시계열과 기간 메타를 반환한다.

    drop_today=True 이면 오늘(현지 기준) 미완성 세션을 제외하고
    마지막으로 닫힌 거래일까지만 사용한다(장중 추천용).
    """
    start_bound, _warmup, _end = compute_bounds(settings)
    # 무한매수법은 신호 워밍업이 필요 없으므로 start_bound 부터 사용
    prices = download_prices(settings, start_bound)
    opens = download_opens(settings, start_bound)

    # buy 설정은 offense_ticker=signal_ticker(=대상 종목)이라 컬럼이 중복될 수 있다.
    prices = prices.loc[:, ~prices.columns.duplicated()]
    opens = opens.loc[:, ~opens.columns.duplicated()]

    ticker = settings["target_ticker"]
    closes = prices[ticker]
    open_series = opens[ticker]

    full_index = open_series.index.intersection(closes.index)
    full_index = full_index[full_index >= start_bound]
    # 장중 정보용: drop_today 로 신호에서 제외되더라도 '오늘 포함' 최신 종가를 보존한다.
    live_close = float(closes.loc[full_index[-1]]) if len(full_index) > 0 else None
    live_date = full_index[-1].date().isoformat() if len(full_index) > 0 else None

    index = full_index
    if drop_today and len(index) > 0:
        cutoff = current_trading_day(settings.get("market", "kor"))
        index = index[index < cutoff]
    if len(index) == 0:
        raise ValueError("대상 종목의 거래 데이터가 비어 있습니다. 기간/티커 설정을 확인하세요.")

    meta = {
        "period_start": index[0].date().isoformat(),
        "period_end": index[-1].date().isoformat(),
        "period_days": len(index),
        "live_close": live_close,
        "live_date": live_date,
    }
    return open_series.loc[index], closes.loc[index], meta


def run_buy_tuning(
    settings: dict,
    divisions_grid,
    take_profit_grid,
    *,
    progress_cb=None,
) -> tuple[list[dict], dict]:
    """divisions × take_profit_pct 격자를 전수 탐색한다. 데이터는 한 번만 받는다."""
    opens, closes, meta = _load_target_series(settings)
    slippage = settings["slippage"]

    combos = [(int(d), float(t)) for d in divisions_grid for t in take_profit_grid]
    total = len(combos)
    results: list[dict] = []

    for i, (divisions, tp) in enumerate(combos, start=1):
        sim = simulate_infinite_buy(opens, closes, divisions=divisions, take_profit_pct=tp, slippage=slippage)
        results.append(
            {
                "params": {"divisions": divisions, "take_profit_pct": tp},
                "cagr": sim["cagr"],
                "mdd": sim["mdd"],
                "vol": sim["vol"],
                "sharpe": sim["sharpe"],
                "period_return": sim["period_return"],
                "cycles": sim["cycles"],
            }
        )
        if progress_cb:
            progress_cb(i, total)

    return results, meta


def render_buy_top_table(results: list[dict], top_n: int = 100, period_days: int | None = None) -> list[str]:
    """튜닝 결과 상위 표를 렌더링한다."""
    pr_label = "기간 수익률(%)"
    headers = ["분할", "익절률(%)", pr_label, "CAGR(%)", "MDD(%)", "Sharpe", "Vol(%)", "익절횟수"]
    aligns = ["right"] * len(headers)
    rows: list[list[str]] = []
    for row in results[:top_n]:
        p = row["params"]
        rows.append(
            [
                f"{p['divisions']}",
                f"{p['take_profit_pct']:.1f}",
                f"{row.get('period_return', 0.0) * 100:.2f}",
                f"{row['cagr'] * 100:.2f}",
                f"{row['mdd'] * 100:.2f}",
                f"{row['sharpe']:.2f}",
                f"{row['vol'] * 100:.2f}",
                f"{row.get('cycles', 0)}",
            ]
        )
    return render_table_eaw(headers, rows, aligns)


def _fmt_price(value: float, market: str) -> str:
    if market == "kor":
        return f"{value:,.0f}원"
    return f"{value:,.2f}"


def run_buy_backtest(settings: dict, drop_today: bool = False) -> dict:
    """무한매수법 백테스트를 실행하고 리포트 dict 를 반환한다.

    drop_today=True 이면 오늘 미완성 세션을 제외하고 마지막 닫힌 거래일까지로 계산한다(장중 추천용).
    """
    opens, closes, meta = _load_target_series(settings, drop_today=drop_today)
    divisions = int(settings["divisions"])
    take_profit_pct = float(settings["take_profit_pct"])
    slippage = settings["slippage"]
    market = settings.get("market", "kor")

    sim = simulate_infinite_buy(opens, closes, divisions=divisions, take_profit_pct=take_profit_pct, slippage=slippage)

    weekday_map = ["월", "화", "수", "목", "금", "토", "일"]

    def _fmt_money(val: float) -> str:
        return format_kr_money(val) if market == "kor" else f"{val:,.2f}"

    # 일자별 로그
    daily_log: list[str] = []
    for rec in sim["daily_records"]:
        d = rec["date"]
        date_str = f"{d.date()}({weekday_map[d.weekday()]})"
        line = (
            f"{date_str} {rec['action']:4s} "
            f"종가 {_fmt_price(rec['close'], market)} | "
            f"회차 {rec['buys_done']}/{divisions} | "
            f"보유 {rec['qty']:,.2f} | "
            f"평단 {_fmt_price(rec['avg'], market) if rec['avg'] else '-'} | "
            f"평가액 {_fmt_money(rec['equity'])}"
        )
        daily_log.append(line)

    # 대상 종목 단순 보유(Buy&Hold) 기준선
    bh_metrics = _annualized_metrics(closes / closes.iloc[0] * INITIAL_CAPITAL_KRW)

    # 오늘의 추천(마지막 날 상태 기준)
    recommendation = build_buy_recommendation(sim, settings)

    asset_summary_lines = [
        f"대상: {settings['target_name']}({settings['target_ticker']})",
        f"분할 수: {divisions} | 익절률: {take_profit_pct:.1f}% | 슬리피지: {slippage}%",
        f"완료 익절 사이클: {sim['cycles']}회",
    ]

    summary_lines = [
        f"| 기간: {meta['period_start']} ~ {meta['period_end']} ({meta['period_days']} 거래일)",
        f"| 기간 수익률: {sim['period_return'] * 100:.2f}%",
        f"| CAGR: {sim['cagr'] * 100:.2f}%",
        f"| MDD: {sim['mdd'] * 100:.2f}%",
        f"| Sharpe: {sim['sharpe']:.2f}",
        f"| Vol: {sim['vol'] * 100:.2f}%",
        f"| 최종 평가액: {_fmt_money(sim['equity'].iloc[-1])}",
        "",
        f"[대상 종목 단순보유 비교] CAGR: {bh_metrics['cagr'] * 100:.2f}% | MDD: {bh_metrics['mdd'] * 100:.2f}%",
    ]

    used_settings_lines = [
        "=== 적용 설정 ===",
        f"전략: 무한매수법(buy) | 시장: {market}",
        f"분할 수: {divisions} | 익절률: {take_profit_pct:.1f}%",
        f"초기자본: {INITIAL_CAPITAL_KRW:,}",
    ]

    return {
        "start": meta["period_start"],
        "end": meta["period_end"],
        "daily_log": daily_log,
        "asset_summary_lines": asset_summary_lines,
        "summary_lines": summary_lines,
        "used_settings_lines": used_settings_lines,
        "cagr": sim["cagr"],
        "mdd": sim["mdd"],
        "period_return": sim["period_return"],
        "cycles": sim["cycles"],
        "recommendation": recommendation,
        "meta": meta,
    }


def build_buy_recommendation(sim: dict, settings: dict) -> dict:
    """마지막 날 상태로부터 '오늘의 행동'을 산출한다."""
    divisions = int(settings["divisions"])
    take_profit_pct = float(settings["take_profit_pct"])
    market = settings.get("market", "kor")
    last = sim["last"]

    qty = sim["final_qty"]
    avg = sim["final_avg"]
    buys_done = sim["final_buys_done"]
    daily_budget = INITIAL_CAPITAL_KRW / divisions
    last_close = last["close"] if last else 0.0
    target_price = avg * (1 + take_profit_pct / 100.0) if avg else 0.0

    if qty > 0 and last_close >= target_price > 0:
        action = "SELL"
        message = f"전량 익절 (목표가 {_fmt_price(target_price, market)} 도달)"
    elif buys_done < divisions:
        action = "BUY"
        message = f"{_fmt_price(daily_budget, market)} 매수 ({buys_done + 1}/{divisions}회차)"
    else:
        action = "HOLD"
        message = f"분할 소진 → 추가매수 없이 보유 유지 (익절 목표 {_fmt_price(target_price, market)})"

    return {
        "action": action,
        "message": message,
        "qty": qty,
        "avg": avg,
        "buys_done": buys_done,
        "divisions": divisions,
        "target_price": target_price,
        "last_close": last_close,
        "daily_budget": daily_budget,
    }
