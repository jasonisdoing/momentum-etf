"""전략 튜닝 공용 — 여러 설정 조합의 백테스트 결과를 같은 잣대로 요약한다.

모멘텀·신고가 화면의 '튜닝' 섹션이 함께 쓴다. 엔진은 각 전략 것을 그대로 호출하고,
여기서는 일별 수익률 → 지표(수익·CAGR·MDD·소르티노), 분기별 수익률,
**분기 승수**(조합들 중 그 분기에 상위 절반에 든 횟수 — 구간 일관성), **축별 평균**
(어느 값이 일관되게 나은가)만 계산한다.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Any

import pandas as pd


def tuning_workers(task_count: int) -> int:
    """병렬 프로세스 수 — 코어 수 전부(최대 32), 작업 수보다 크지 않게. TUNING_WORKERS 로 덮어쓸 수 있다."""
    override = os.environ.get("TUNING_WORKERS")
    if override and override.isdigit():
        workers = int(override)
    else:
        workers = min(32, max(1, os.cpu_count() or 1))
    return max(1, min(workers, task_count))


def seed_worker_caches(pool_configs: list[dict[str, Any]], stocks_by_pool: dict[str, list[dict[str, Any]]]) -> None:
    """워커 프로세스가 DB 를 건드리지 않게, 부모가 읽어 둔 풀 설정·종목 목록으로 캐시를 채운다.

    워커 여러 개가 동시에 MongoDB 를 읽으면(가격·종목·설정) 로컬 mongod 가 CPU 를 뺏겨
    타임아웃이 나고 화면의 다른 요청까지 실패한다 — 그래서 워커는 읽기 전용 사본만 쓴다.
    """
    import os
    from time import monotonic

    from utils import settings_loader, stock_list_io

    settings_loader._load_pool_configs = lambda: pool_configs  # type: ignore[assignment]
    for pool, docs in stocks_by_pool.items():
        stock_list_io._TICKER_TYPE_STOCKS_CACHE[pool] = (monotonic(), docs)
    try:
        os.nice(5)  # 화면·DB 가 워커에 밀리지 않게 우선순위를 조금 낮춘다
    except OSError:
        pass


def run_groups(
    worker: Callable[[Any], Any],
    tasks: Iterable[Any],
    *,
    initializer: Callable[..., None] | None = None,
    initargs: tuple = (),
) -> list[Any]:
    """작업 그룹을 별도 프로세스(spawn)에서 병렬로 돌린다.

    ``initializer`` 는 워커마다 한 번 실행돼 부모가 읽어 둔 데이터(가격·종목·설정)를
    넘긴다 — 워커는 DB 를 건드리지 않는다. fork 대신 spawn 을 써서 macOS 에서도 안전하다.
    작업이 하나뿐이거나 워커가 1개면 현재 프로세스에서 그대로 돈다(초기화도 여기서).
    """
    tasks = list(tasks)
    workers = tuning_workers(len(tasks))
    if workers <= 1:
        if initializer is not None:
            initializer(*initargs)
        return [worker(task) for task in tasks]
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn"), initializer=initializer, initargs=initargs
    ) as executor:
        return list(executor.map(worker, tasks))


def metrics_from_returns(returns_pct: pd.Series) -> dict[str, float | None]:
    """일별 수익률(%) 시계열 → 총수익·MDD·소르티노(연환산)."""
    if returns_pct.empty:
        return {"total_pct": 0.0, "mdd_pct": 0.0, "sortino": None}
    growth = (1 + returns_pct / 100).cumprod()
    returns = returns_pct / 100
    downside = returns[returns < 0]
    dev = float((downside**2).mean() ** 0.5) if not downside.empty else 0.0
    return {
        "total_pct": round(float((growth.iloc[-1] - 1) * 100), 1),
        "mdd_pct": round(float((((growth / growth.cummax()) - 1) * 100).min()), 1),
        "sortino": round(float(returns.mean()) / dev * (252**0.5), 2) if dev > 0 else None,
    }


def cumulative_to_returns(cumulative_pct: pd.Series) -> pd.Series:
    """누적 수익률(%) 곡선 → 일별 수익률(%)."""
    level = 1 + cumulative_pct.sort_index() / 100
    return ((level / level.shift(1)) - 1).dropna() * 100


def quarterly_returns(returns_pct: pd.Series) -> dict[str, float]:
    return {
        str(quarter): round(float(((1 + part / 100).prod() - 1) * 100), 1)
        for quarter, part in returns_pct.groupby(returns_pct.index.to_period("Q"))
    }


def cagr_pct(returns_pct: pd.Series) -> float | None:
    """연환산 수익률(%) — 총수익을 실제 달력 기간(일수)으로 연환산한다."""
    if len(returns_pct) < 2:
        return None
    growth = float((1 + returns_pct / 100).prod())
    days = (returns_pct.index[-1] - returns_pct.index[0]).days
    if growth <= 0 or days <= 0:
        return None
    return round((growth ** (365.0 / days) - 1) * 100, 1)


def summarize_combo(
    params: dict[str, Any], returns_pct: pd.Series, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """조합 하나의 요약 행. ``extra`` 는 전략별 부가 지표(거래 수·승률 등)."""
    return {
        "params": params,
        **metrics_from_returns(returns_pct),
        "cagr_pct": cagr_pct(returns_pct),
        "quarters": quarterly_returns(returns_pct),
        **(extra or {}),
    }


def finalize(rows: list[dict[str, Any]], axes: list[str]) -> dict[str, Any]:
    """분기 승수와 축별 평균을 붙여 화면 페이로드로 만든다. 행은 소르티노 내림차순."""
    quarter_keys = sorted({key for row in rows for key in row["quarters"]})
    half = len(rows) / 2
    wins: dict[int, int] = {}
    for key in quarter_keys:
        ordered = sorted(rows, key=lambda r: r["quarters"].get(key, -999), reverse=True)
        for rank, row in enumerate(ordered):
            wins.setdefault(id(row), 0)
            if rank < half:
                wins[id(row)] += 1
    for row in rows:
        row["quarter_wins"] = wins.get(id(row), 0)
    rows.sort(key=lambda r: -(r["sortino"] if r["sortino"] is not None else -999))

    axis_summary: dict[str, list[dict[str, Any]]] = {}
    for axis in axes:
        values = []
        seen: list[Any] = []
        for row in rows:
            value = row["params"].get(axis)
            if value not in seen:
                seen.append(value)
        # 축 값 표시 순서 — 문자열(미사용·없음 등)은 주어진 순서대로 앞에, 숫자는 절댓값 오름차순
        # (손절선 -5, -6, … -10 / 종목수 5, 6, …), None 은 맨 뒤.
        order = {v: i for i, v in enumerate(seen)}
        seen.sort(key=lambda v: (2, 0) if v is None else ((0, order[v]) if isinstance(v, str) else (1, abs(v))))
        for value in seen:
            subset = [r for r in rows if r["params"].get(axis) == value]
            values.append(
                {
                    "value": value,
                    "count": len(subset),
                    "sortino": round(sum((r["sortino"] or 0) for r in subset) / len(subset), 2),
                    "total_pct": round(sum(r["total_pct"] for r in subset) / len(subset), 1),
                    "mdd_pct": round(sum(r["mdd_pct"] for r in subset) / len(subset), 1),
                }
            )
        axis_summary[axis] = values
    return {"rows": rows, "quarter_keys": quarter_keys, "quarter_count": len(quarter_keys), "axes": axis_summary}
