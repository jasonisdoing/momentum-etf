"""신고가 전략 튜닝 — 설정 항목들의 범위 조합을 한 번에 백테스트해 비교한다.

화면 '튜닝' 섹션용. 축(화면 순서): 이탈 이평선 · 거래대금 하한 · ADR 하한 · 손절선.
종목 수는 풀 설정(`pool_settings.TOP_N_HOLD`) 고정, 업종 상한은 폐기 — 튜닝은 시장의 이평 반응과
급증·손절 기준을 재는 용도로만 쓴다. 축 밖의 설정은 전달받은 설정으로 고정하고, (이탈 이평선, 손절선)을 작업
단위로 별도 프로세스에서 병렬로 돌리며, 각 프로세스가 패널·신호를 한 번 만들어 그 안의 조합에
공유한다. 병렬 수(코어 수)만큼 전체 시간이 줄어든다.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import pandas as pd

from config import ADR_FLOOR_OPTIONS
from utils.new_high_service import (
    EXIT_MA_OPTIONS,
    MIN_VALUE_MULT_OPTIONS,
    STOP_LOSS_OPTIONS,
    build_price_panel,
    compute_signals,
    load_price_frames,
    load_settings,
    load_universe,
    validate_settings,
)
from utils.strategy_tuning import cumulative_to_returns, finalize, run_groups, seed_worker_caches, summarize_combo

TUNING_AXES = ("exit_ma_days", "min_value_mult", "adr_floor", "stop_loss_pct")


def _checked(values: list[Any], options: tuple, label: str, *, cast) -> list[Any]:
    cleaned = list(dict.fromkeys(None if v is None else cast(v) for v in values))
    bad = [v for v in cleaned if v not in options]
    if not cleaned or bad:
        raise ValueError(f"'{label}' 범위가 올바르지 않습니다: {bad or cleaned}")
    return cleaned


# 워커 프로세스 안에서만 쓰는 상태 — 부모가 넘긴 유니버스·패널·설정(프리로드)과
# 이탈선별 신호 캐시. 워커는 DB 를 건드리지 않는다.
_PRELOAD: dict[str, Any] = {}
_WORKER_SIGNALS: dict[int, Any] = {}


def _init_worker(bundle: dict[str, Any]) -> None:
    seed_worker_caches(bundle["pool_configs"], bundle["stocks_by_pool"])
    _PRELOAD.update(bundle)
    if bundle.get("adr_market") and bundle.get("adr_series") is not None:
        from utils.momentum_service import seed_adr_series

        seed_adr_series(bundle["adr_market"], bundle["adr_series"])


def _preload(pool: str) -> dict[str, Any]:
    """부모가 한 번 읽는 공유 데이터 — 워커 초기화 때 통째로 넘긴다."""
    from utils.momentum_service import adr_market_of_pool, load_adr_series
    from utils.settings_loader import _load_pool_configs
    from utils.stock_list_io import _load_ticker_type_stocks_raw

    universe = load_universe(pool)
    # ADR 시계열 — 워커는 DB 를 안 건드리므로 부모가 읽어서 심는다(레짐 미설정 풀은 None).
    adr_market = adr_market_of_pool(pool)
    return {
        "pool_configs": _load_pool_configs(),
        "stocks_by_pool": {pool: _load_ticker_type_stocks_raw(pool)},
        "pool": pool,
        "universe": universe,
        "panel": build_price_panel(universe, load_price_frames(universe)),
        "adr_market": adr_market,
        "adr_series": load_adr_series(adr_market) if adr_market else None,
    }


def _worker_context(exit_ma: int) -> dict[str, Any]:
    universe = _PRELOAD["universe"]
    if exit_ma not in _WORKER_SIGNALS:
        _WORKER_SIGNALS[exit_ma] = compute_signals(_PRELOAD["panel"], int(exit_ma))
    return {
        "pool": _PRELOAD["pool"],
        "universe": universe,
        "name_by": {row["ticker"]: row["name"] for row in universe},
        "industry_by": {row["ticker"]: row.get("industry", "") for row in universe},
        "panel": _PRELOAD["panel"],
        "signals": _WORKER_SIGNALS[exit_ma],
    }


def _run_group(task: tuple) -> list[dict[str, Any]]:
    """(이탈 이평선, 손절선) 하나의 조합 — 별도 프로세스에서 돈다."""
    months, base, exit_ma, stops, mults, adr_floors = task
    from utils.new_high_backtest import run_backtest

    context = _worker_context(int(exit_ma))
    rows: list[dict[str, Any]] = []
    for stop in stops:
        for mult, adr_floor in product(mults, adr_floors):
            combo = dict(
                base,
                stop_loss_pct=stop,
                exit_ma_days=exit_ma,
                min_value_mult=mult,
                adr_floor=adr_floor,
            )
            result = run_backtest(months, combo, context)
            daily = pd.DataFrame(result["daily"])
            daily["date"] = pd.to_datetime(daily["date"])
            returns = cumulative_to_returns(daily.set_index("date")["strategy_pct"])
            rows.append(
                summarize_combo(
                    {
                        "stop_loss_pct": stop,
                        "exit_ma_days": exit_ma,
                        "min_value_mult": mult,
                        "adr_floor": adr_floor,
                    },
                    returns,
                    {"trade_count": result["trade_count"], "win_rate_pct": result["win_rate_pct"]},
                )
            )
    return rows


def run_tuning(months: int, settings: dict[str, Any] | None, ranges: dict[str, list[Any]]) -> dict[str, Any]:
    base = validate_settings(settings or load_settings())
    stops = _checked(ranges.get("stop_loss_pct", []), STOP_LOSS_OPTIONS, "손절선", cast=float)
    exit_mas = _checked(ranges.get("exit_ma_days", []), EXIT_MA_OPTIONS, "이탈 이평선", cast=int)
    mults = _checked(ranges.get("min_value_mult", []), MIN_VALUE_MULT_OPTIONS, "거래대금 하한", cast=float)
    adr_floors = _checked(ranges.get("adr_floor", []), ADR_FLOOR_OPTIONS, "ADR 하한", cast=int)

    # 작업을 잘게 쪼개 코어가 놀지 않게 한다 — (이탈선, 손절선)마다 하나.
    tasks = [(months, base, exit_ma, [stop], mults, adr_floors) for exit_ma in exit_mas for stop in stops]
    rows: list[dict[str, Any]] = []
    bundle = _preload(str(base["pool"]))
    for group_rows in run_groups(_run_group, tasks, initializer=_init_worker, initargs=(bundle,)):
        rows.extend(group_rows)

    payload = finalize(rows, list(TUNING_AXES))
    payload["months"] = int(months)
    return payload
