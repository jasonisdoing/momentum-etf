"""수익 곡선 공통 성과 지표 (백테스트 엔진·포트폴리오 실험 공용)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_return_metrics(returns_pct: pd.Series) -> dict[str, float | None]:
    """날짜 인덱스의 일별 수익률(%)에서 총수익·CAGR·MDD·소르티노를 계산한다."""
    if returns_pct.empty:
        return {"total_pct": 0.0, "cagr_pct": None, "mdd_pct": 0.0, "sortino": None}
    if not isinstance(returns_pct.index, pd.DatetimeIndex):
        raise ValueError("일별 성과 계산에는 DatetimeIndex가 필요합니다.")
    if returns_pct.isna().any():
        raise ValueError("일별 성과 계산에는 빈 수익률이 없어야 합니다.")

    ordered = returns_pct.sort_index().astype(float)
    returns = ordered / 100.0
    growth = (1.0 + returns).cumprod()
    total_pct = float((growth.iloc[-1] - 1.0) * 100.0)
    # 첫 거래일 손실도 시작 자산 1.0 대비 낙폭으로 잡혀야 한다.
    curve = pd.concat([pd.Series([1.0]), growth.reset_index(drop=True)], ignore_index=True)
    mdd_pct = float((((curve / curve.cummax()) - 1.0) * 100.0).min())

    downside = returns[returns < 0]
    downside_deviation = float((downside**2).mean() ** 0.5) if not downside.empty else 0.0
    sortino_value = float(returns.mean()) / downside_deviation * float(252**0.5) if downside_deviation > 0 else None

    days = int((ordered.index[-1] - ordered.index[0]).days)
    final_growth = float(growth.iloc[-1])
    cagr_value = (final_growth ** (365.0 / days) - 1.0) * 100.0 if final_growth > 0 and days > 0 else None
    return {
        "total_pct": total_pct,
        "cagr_pct": cagr_value,
        "mdd_pct": mdd_pct,
        "sortino": sortino_value,
    }


def sharpe_from_curve(start_val: float, values: np.ndarray, cagr_pct: float = 0.0) -> float:
    """Sharpe = 연율화 평균 수익률(일간 평균 수익률 × 252) ÷ 연율화 변동성(일간 수익률 표준편차 × √252). 계산 불가 시 0."""
    curve = np.concatenate(([start_val], np.asarray(values, dtype=np.float64)))
    if curve.size < 3 or np.any(curve[:-1] <= 0):
        return 0.0
    daily_rets = np.diff(curve) / curve[:-1]

    # 일간 평균 수익률의 단순 연율화 (분자)
    ann_ret = float(np.mean(daily_rets)) * 252.0

    # 일간 변동성의 연율화 (분모)
    vol = float(np.std(daily_rets, ddof=1)) * float(np.sqrt(252.0))

    if vol <= 0:
        return 0.0
    return ann_ret / vol


def sortino_from_curve(start_val: float, values: np.ndarray) -> float:
    """Sortino = 연율화 평균 수익률 ÷ 연율화 하방 변동성 (일간 음수 수익률 표준편차 × √252). 계산 불가 시 0."""
    curve = np.concatenate(([start_val], np.asarray(values, dtype=np.float64)))
    if curve.size < 3 or np.any(curve[:-1] <= 0):
        return 0.0
    daily_rets = np.diff(curve) / curve[:-1]

    # 일간 평균 수익률의 단순 연율화 (분자)
    ann_ret = float(np.mean(daily_rets)) * 252.0

    # 하방 변동성(Downside Deviation) 계산: 음수 수익률의 제곱합 기준 (분모)
    downside_rets = np.minimum(0, daily_rets)
    downside_std = float(np.sqrt(np.sum(downside_rets**2) / (daily_rets.size - 1))) * float(np.sqrt(252.0))

    if downside_std <= 0:
        return 0.0
    return ann_ret / downside_std


def mdd_span(values: np.ndarray) -> tuple[int, int, float]:
    """곡선에서 최대낙폭(MDD)의 (고점 인덱스, 저점 인덱스, MDD%)를 반환한다.

    MDD 저점 = drawdown 최소 지점, 고점 = 그 저점 직전까지의 최고값 지점.
    빈 곡선이면 (0, 0, 0.0).
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0, 0, 0.0
    running_max = np.maximum.accumulate(values)
    drawdown = values / running_max - 1.0
    trough = int(np.argmin(drawdown))
    peak = int(np.argmax(values[: trough + 1])) if trough > 0 else 0
    return peak, trough, float(drawdown[trough] * 100.0)


def curve_metrics(start_val: float, values: np.ndarray) -> dict[str, float]:
    """KRW 평가 곡선에서 (총수익률%, CAGR%, MDD%, Sortino) 를 계산한다."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or start_val <= 0:
        return {"total_return_pct": 0.0, "cagr_pct": 0.0, "mdd_pct": 0.0, "sortino": 0.0}

    end_val = float(values[-1])
    total_return_pct = (end_val / start_val - 1.0) * 100.0

    years = max(1, values.size) / 252.0
    cagr_pct = (pow(end_val / start_val, 1.0 / years) - 1.0) * 100.0 if end_val > 0 and years > 0 else 0.0

    curve = np.concatenate(([start_val], values))
    running_max = np.maximum.accumulate(curve)
    drawdown = (curve / running_max - 1.0) * 100.0
    mdd_pct = float(np.min(drawdown)) if drawdown.size else 0.0

    return {
        "total_return_pct": float(total_return_pct),
        "cagr_pct": float(cagr_pct),
        "mdd_pct": mdd_pct,
        "sortino": sortino_from_curve(start_val, values),
    }


def single_stock_backtest_stats(close_prices, lookback_months: int) -> dict:
    """단일 종목 종가 시계열(pd.Series)의 최근 N개월 단순 보유 성과(수익률·MDD·소르티노)를 구한다.

    상장일이 시작일보다 뒤라 가용 기간이 N개월에 못 미치면 전체 기간으로 계산하고
    is_partial=True 를 표시한다(화면에서 노란색·🆕 강조 기준).
    순위 화면과 공유하는 단일 소스 — 기간은 호출부가 장기 이평선에서 파생해 넘긴다.
    """
    import pandas as pd

    empty = {"cagr": 0.0, "mdd": 0.0, "sortino": 0.0, "is_partial": False}
    if close_prices is None or len(close_prices) == 0:
        return empty

    try:
        last_date = close_prices.index[-1]
        start_date = last_date - pd.DateOffset(months=int(lookback_months))

        # 상장일이 시작일보다 나중인지 여부 판정
        is_partial = close_prices.index[0] > start_date

        target_series = close_prices.loc[start_date:]
        if len(target_series) < 2:
            return {**empty, "is_partial": is_partial}

        start_val = float(target_series.iloc[0])
        values = target_series.iloc[1:].to_numpy()

        metrics = curve_metrics(start_val, values)
        return {
            "cagr": round(float(metrics.get("total_return_pct", 0.0)), 2),  # CAGR 대신 단순 누적 수익률(%) 저장
            "mdd": round(float(metrics.get("mdd_pct", 0.0)), 2),
            "sortino": round(float(metrics.get("sortino", 0.0)), 2),
            "is_partial": is_partial,
        }
    except Exception:
        return empty
