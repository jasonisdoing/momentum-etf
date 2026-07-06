"""수익 곡선 공통 성과 지표 (백테스트 엔진·포트폴리오 실험 공용)."""

from __future__ import annotations

import numpy as np


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
    downside_std = float(np.sqrt(np.sum(downside_rets ** 2) / (daily_rets.size - 1))) * float(np.sqrt(252.0))
    
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
    """KRW 평가 곡선에서 (총수익률%, CAGR%, MDD%, Sharpe) 를 계산한다."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or start_val <= 0:
        return {"total_return_pct": 0.0, "cagr_pct": 0.0, "mdd_pct": 0.0, "sharpe": 0.0, "sortino": 0.0}

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
        "sharpe": sharpe_from_curve(start_val, values, cagr_pct),
        "sortino": sortino_from_curve(start_val, values),
    }
