"""SMA 크로스 전략 백테스트 엔진.

지수 종가가 SMA(N일) **위면 레버리지 티커**, **아래면 방어 티커(현금 또는 종목)**를 보유한다.
기존 switch(드로다운 컷) 전략과 독립이며, 종가 시리즈만 받는 순수 함수라 테스트가 쉽다.

신호는 look-ahead 방지를 위해 **전일 종가 기준**으로 그날 보유를 결정한다(당일 신호로 당일
수익을 먹지 않는다). 전환(방어↔레버리지)이 일어난 날에만 왕복 슬리피지를 물린다.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

# 튜닝 sweep 기본 후보(SMA 일수). 과적합 경계: '최적 1개'가 아니라 이 분포 전체를 보여줘
# 특정 값만 튀는지 vs 넓게 완만한지(강건성)를 판단하게 한다.
DEFAULT_SMA_CANDIDATES: tuple[int, ...] = (5, 10, 20, 40, 60, 90, 120, 150, 200, 240)


def sortino(daily_returns: pd.Series, periods_per_year: int = 252) -> float | None:
    """일별 수익률의 소르티노 지수(연율). 하방편차(0 미만 수익) 기준.

    표본이 적거나 하락일이 없으면 값을 내지 않는다(None) — 억지로 채우지 않는다.
    """
    returns = daily_returns.dropna()
    if len(returns) < 3:
        return None
    downside = returns[returns < 0.0]
    if downside.empty:
        return None
    downside_dev = float(np.sqrt((downside**2).mean()))
    if downside_dev == 0.0:
        return None
    return round(float(returns.mean() / downside_dev) * float(np.sqrt(periods_per_year)), 2)


def max_drawdown_pct(curve: pd.Series) -> float | None:
    """자산 곡선의 최대낙폭(MDD, 음수 %). 일별 곡선으로 계산해야 실제를 반영한다."""
    curve = curve.dropna()
    if len(curve) < 2:
        return None
    drawdown = curve / curve.cummax() - 1.0
    return round(float(drawdown.min()) * 100.0, 1)


def cagr_pct(curve: pd.Series, periods_per_year: int = 252) -> float | None:
    """자산 곡선의 연율 수익률(CAGR, %). 거래일 수 기준으로 연율화한다."""
    curve = curve.dropna()
    if len(curve) < 2:
        return None
    final_value = float(curve.iloc[-1])
    if final_value <= 0:
        return None
    years = len(curve) / float(periods_per_year)
    if years <= 0:
        return None
    return round((final_value ** (1.0 / years) - 1.0) * 100.0, 1)


def run_sma_cross(
    index_close: pd.Series,
    leverage_close: pd.Series,
    defense_close: pd.Series | None,
    sma_days: int,
    *,
    peak_drawdown_pct: float | None,
    buy_pct: float,
    sell_pct: float,
    eval_start: pd.Timestamp | None = None,
) -> dict[str, object] | None:
    """SMA 크로스 전략을 일별로 시뮬레이션해 지표를 반환한다.

    ``peak_drawdown_pct`` 가 있으면 지수가 전고점 대비 해당 % 이내일 때만 레버리지를 보유한다.
    ``defense_close`` 가 None 이면 방어는 **현금**(그날 수익 0)이다.
    ``buy_pct``/``sell_pct`` 는 편도 슬리피지 비율(소수, 예 0.001 = 0.1%).
    ``eval_start`` 가 있으면 SMA 는 전체(워밍업 포함) 구간으로 계산하되 **성과는 그 날짜
    이후만** 집계한다 — SMA(장기)도 창 시작부터 신호가 살아 있어 기간 비교가 공정해진다.
    반환: ``{sma_days, cumulative_pct, cagr_pct, mdd_pct, sortino, switches, days, leverage_days}`` 또는 데이터 부족 시 None.
    """
    frames: dict[str, pd.Series] = {"index": index_close, "leverage": leverage_close}
    if defense_close is not None:
        frames["defense"] = defense_close
    df = pd.DataFrame(frames).dropna(subset=["index", "leverage"]).sort_index()
    if len(df) < sma_days + 5:
        return None

    sma = df["index"].rolling(sma_days).mean()
    valid = sma.notna()
    df = df[valid]
    sma = sma[valid]
    if len(df) < 5:
        return None

    # 전일 종가 신호로 그날 보유를 결정한다(look-ahead 방지).
    above_sma = df["index"] > sma
    if peak_drawdown_pct is None:
        want_leverage_by_index = above_sma
    else:
        peak_drawdown = (df["index"] / df["index"].cummax() - 1.0) * 100.0
        want_leverage_by_index = above_sma & (peak_drawdown >= -float(peak_drawdown_pct))
    want_leverage = want_leverage_by_index.shift(1)

    leverage_ret = df["leverage"].pct_change()
    defense_ret = df["defense"].pct_change() if defense_close is not None else pd.Series(0.0, index=df.index)
    gross = pd.Series(np.where(want_leverage, leverage_ret, defense_ret), index=df.index)

    # 보유 대상이 바뀐 날에만 왕복 슬리피지(직전 매도 + 신규 매수).
    switched = want_leverage.ne(want_leverage.shift(1)) & want_leverage.notna()
    cost = switched.astype(float) * (float(buy_pct) + float(sell_pct))

    net = (gross - cost).dropna()
    if eval_start is not None:
        net = net[net.index >= eval_start]
        switched = switched.reindex(net.index, fill_value=False)
    if net.empty:
        return None
    leverage_days = int(want_leverage.reindex(net.index, fill_value=False).fillna(False).astype(bool).sum())
    curve = (1.0 + net).cumprod()
    return {
        "sma_days": int(sma_days),
        "peak_drawdown_pct": None if peak_drawdown_pct is None else float(peak_drawdown_pct),
        "cumulative_pct": round((float(curve.iloc[-1]) - 1.0) * 100.0, 1),
        "cagr_pct": cagr_pct(curve),
        "mdd_pct": max_drawdown_pct(curve),
        "sortino": sortino(net),
        "switches": int(switched.sum()),
        "days": int(len(net)),
        "leverage_days": leverage_days,
    }


def run_buy_hold(close: pd.Series, *, eval_start: pd.Timestamp | None = None) -> dict[str, object] | None:
    """단일 자산을 평가 기간 동안 그대로 보유했을 때의 성과 지표를 반환한다.

    SMA 전략과 비교 기준을 맞추기 위해 일별 수익률 곡선으로 MDD/소르티노를 계산한다.
    단순 보유는 매매 전환이 없으므로 슬리피지와 전환수는 적용하지 않는다.
    """
    series = close.dropna().sort_index()
    if len(series) < 2:
        return None

    net = series.pct_change().dropna()
    if eval_start is not None:
        net = net[net.index >= eval_start]
    if net.empty:
        return None

    curve = (1.0 + net).cumprod()
    return {
        "cumulative_pct": round((float(curve.iloc[-1]) - 1.0) * 100.0, 1),
        "cagr_pct": cagr_pct(curve),
        "mdd_pct": max_drawdown_pct(curve),
        "sortino": sortino(net),
        "switches": 0,
        "days": int(len(net)),
    }


def tune_sma_cross(
    index_close: pd.Series,
    leverage_close: pd.Series,
    defense_close: pd.Series | None,
    *,
    buy_pct: float,
    sell_pct: float,
    candidates: Sequence[int] = DEFAULT_SMA_CANDIDATES,
    peak_drawdown_candidates: Sequence[float],
    eval_start: pd.Timestamp | None = None,
) -> list[dict[str, object]]:
    """SMA 후보와 고점대비 후보를 sweep 해 소르티노, CAGR 내림차순으로 반환한다.

    '최적 1개'만 강조하지 않는다 — 전체 후보의 소르티노/수익/MDD 를 나란히 반환해
    특정 SMA 만 튀는지(과적합 위험) vs 넓게 완만한지(강건)를 화면에서 판단하게 한다.
    소르티노/CAGR 이 None 인 후보는 정렬에서 맨 뒤로 보낸다.
    """
    rows: list[dict[str, object]] = []
    for sma_days in candidates:
        for peak_drawdown_pct in peak_drawdown_candidates:
            result = run_sma_cross(
                index_close, leverage_close, defense_close, sma_days,
                peak_drawdown_pct=peak_drawdown_pct,
                buy_pct=buy_pct, sell_pct=sell_pct, eval_start=eval_start,
            )
            if result is not None:
                rows.append(result)
    rows.sort(
        key=lambda r: (
            r["sortino"] if r["sortino"] is not None else float("-inf"),
            r["cagr_pct"] if r["cagr_pct"] is not None else float("-inf"),
        ),
        reverse=True,
    )
    return rows


def current_index_judgment(index_close: pd.Series, sma_days: int, *, peak_drawdown_pct: float) -> dict[str, object] | None:
    """가장 최근 종가와 SMA(N일), 고점대비 조건으로 지금 보유해야 할 쪽을 판정한다.

    반환: ``{as_of, index_close, sma, gap_pct, peak_drawdown_pct, want_leverage, required_index_close}``.
    ``gap_pct`` 는 (종가/SMA-1)% 이고, ``peak_drawdown_pct`` 는 전고점 대비 현재 하락률(%).
    데이터가 SMA 계산에 부족하면 None.
    """
    close = index_close.dropna().sort_index()
    if len(close) < sma_days:
        return None
    sma_value = float(close.iloc[-sma_days:].mean())
    if sma_value == 0.0:
        return None
    last_close = float(close.iloc[-1])
    high = float(close.cummax().iloc[-1])
    if high == 0.0:
        return None
    current_peak_drawdown_pct = (last_close / high - 1.0) * 100.0
    is_above_sma = last_close > sma_value
    is_within_peak = current_peak_drawdown_pct >= -float(peak_drawdown_pct)
    peak_threshold = high * (1.0 - float(peak_drawdown_pct) / 100.0)
    required_index_close = max(sma_value, peak_threshold)
    return {
        "as_of": close.index[-1],
        "index_close": last_close,
        "sma": round(sma_value, 4),
        "gap_pct": round((last_close / sma_value - 1.0) * 100.0, 2),
        "peak_drawdown_pct": round(current_peak_drawdown_pct, 2),
        "peak_drawdown_limit_pct": float(peak_drawdown_pct),
        "sma_threshold_close": round(sma_value, 4),
        "peak_threshold_close": round(peak_threshold, 4),
        "required_index_close": round(required_index_close, 4),
        "want_leverage": bool(is_above_sma and is_within_peak),
    }
