"""모멘텀 전략 튜닝 — 설정 항목들의 범위 조합을 한 번에 백테스트해 비교한다.

화면 '튜닝' 섹션용. 축(화면 순서): 종목 수 · 업종 상한 · 선정 이평(단기·장기) · 주중 손절선.
(단기, 장기) 쌍을 작업 단위로 별도 프로세스에서 병렬로 돌린다 — 각 프로세스는 가격·판정일별
후보를 한 번 읽어 그 쌍의 전 조합에 공유(run_backtest 의 context)하므로 조합당 0.5초 수준이고,
병렬 수(코어 수 − 1)만큼 전체 시간이 줄어든다.

주중 손절선 축의 값
  "off"  → 주중 이탈 미사용
  "none" → 주중 이탈 사용, 손절선 없음(이평선 이탈만)
  숫자   → 주중 이탈 사용 + 그 손절선(%)
"""

from __future__ import annotations

from itertools import product
from typing import Any

import pandas as pd

from utils.momentum_service import (
    INTRAWEEK_STOP_OPTIONS,
    LONG_MA_OPTIONS,
    MAX_PER_INDUSTRY_OPTIONS,
    SHORT_MA_OPTIONS,
    TOP_N_OPTIONS,
    load_settings,
    validate_settings,
)
from utils.strategy_tuning import finalize, run_groups, seed_worker_caches, summarize_combo

TUNING_AXES = ("top_n", "max_per_industry", "short_ma_days", "long_ma_days", "intraweek")


def _intraweek_settings(value: Any) -> dict[str, Any]:
    if value == "off":
        return {"intraweek_exit": False, "intraweek_stop_pct": None}
    if value == "none" or value is None:
        return {"intraweek_exit": True, "intraweek_stop_pct": None}
    stop = float(value)
    if stop not in INTRAWEEK_STOP_OPTIONS:
        raise ValueError(f"주중 손절선 값이 올바르지 않습니다: {value}")
    return {"intraweek_exit": True, "intraweek_stop_pct": stop}


def _checked(values: list[Any], options: tuple, label: str) -> list[Any]:
    # None(제한없음)은 그대로 두고 숫자만 정수화한다. 순서는 선택지 순서.
    cleaned = list(dict.fromkeys(None if v is None else int(v) for v in values))
    bad = [v for v in cleaned if v not in options]
    if not cleaned or bad:
        raise ValueError(f"'{label}' 범위가 올바르지 않습니다: {bad or cleaned}")
    return cleaned


# 워커 프로세스 안에서만 쓰는 상태 — 부모가 넘긴 가격·종목·설정(프리로드)과, 작업 사이에
# 재사용하는 백테스트 컨텍스트(후보 캐시 포함). 워커는 DB 를 건드리지 않는다.
_PRELOAD: dict[str, Any] = {}
_WORKER_CONTEXT: dict[str, Any] = {}


def _init_worker(bundle: dict[str, Any]) -> None:
    seed_worker_caches(bundle["pool_configs"], bundle["stocks_by_pool"])
    _PRELOAD.update(bundle)


def _preload(pool: str) -> dict[str, Any]:
    """부모가 한 번 읽는 공유 데이터 — 워커 초기화 때 통째로 넘긴다."""
    from utils.industry_map import industry_map
    from utils.momentum_service import load_benchmark_close, load_price_frames, load_universe
    from utils.settings_loader import _load_pool_configs
    from utils.stock_list_io import _load_ticker_type_stocks_raw

    universe = load_universe(pool)
    return {
        "pool_configs": _load_pool_configs(),
        "stocks_by_pool": {pool: _load_ticker_type_stocks_raw(pool)},
        "universe": universe,
        "frames": load_price_frames(universe),
        "benchmark_close": load_benchmark_close(pool),
        "industry_by": industry_map(pool),
    }


def _run_ma_group(task: tuple) -> tuple[list[dict[str, Any]], list[str]]:
    """(단기, 장기) 쌍 × 종목수 묶음 하나의 조합 — 별도 프로세스에서 돈다."""
    months, base, short, long, top_ns, caps, intraweeks = task
    from utils.momentum_backtest import run_backtest

    context = _WORKER_CONTEXT
    if not context:
        context.update({k: _PRELOAD[k] for k in ("universe", "frames", "benchmark_close", "industry_by")})
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for top_n, cap, value in product(top_ns, caps, intraweeks):
        combo = dict(base, top_n=top_n, max_per_industry=cap, short_ma_days=short, long_ma_days=long, **_intraweek_settings(value))
        try:
            result = run_backtest(months, combo, include_daily=True, context=context)
        except ValueError as error:  # 장기 이평이 길어 기간이 모자라는 조합 — 그 이평 쌍은 통째로 건너뛴다
            skipped.append(f"단기 {short}/장기 {long}: {error}")
            break
        daily = pd.DataFrame(sorted(result["daily"], key=lambda r: r["date"]))
        daily["date"] = pd.to_datetime(daily["date"])
        returns = daily.set_index("date")["strategy_pct"].dropna()
        rows.append(
            summarize_combo(
                {"top_n": top_n, "max_per_industry": cap, "short_ma_days": short, "long_ma_days": long, "intraweek": value},
                returns,
            )
        )
    return rows, skipped


def run_tuning(months: int, settings: dict[str, Any] | None, ranges: dict[str, list[Any]]) -> dict[str, Any]:
    base = validate_settings(settings or load_settings())
    top_ns = _checked(ranges.get("top_n", []), TOP_N_OPTIONS, "종목 수")
    caps = _checked(ranges.get("max_per_industry", []), MAX_PER_INDUSTRY_OPTIONS, "업종별 최대 보유")
    shorts = _checked(ranges.get("short_ma_days", []), SHORT_MA_OPTIONS, "단기 이평")
    longs = _checked(ranges.get("long_ma_days", []), LONG_MA_OPTIONS, "장기 이평")
    intraweeks = list(dict.fromkeys(ranges.get("intraweek", [])))
    if not intraweeks:
        raise ValueError("'주중 손절선' 범위가 비어 있습니다.")
    for value in intraweeks:
        _intraweek_settings(value)  # 검증
    ma_pairs = [(short, long) for short in shorts for long in longs if short < long]
    if not ma_pairs:
        raise ValueError("단기 이평이 장기 이평보다 작은 조합이 없습니다.")

    # 작업을 잘게 쪼개 코어가 놀지 않게 한다 — 이평 쌍마다 종목수를 두 묶음으로.
    halves = [top_ns[: (len(top_ns) + 1) // 2], top_ns[(len(top_ns) + 1) // 2 :]]
    tasks = [
        (months, base, short, long, part, caps, intraweeks)
        for short, long in ma_pairs
        for part in halves
        if part
    ]
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    bundle = _preload(str(base["pool"]))
    for group_rows, group_skipped in run_groups(_run_ma_group, tasks, initializer=_init_worker, initargs=(bundle,)):
        rows.extend(group_rows)
        skipped.extend(group_skipped)

    if not rows:
        raise ValueError("돌릴 수 있는 조합이 없습니다. " + (skipped[0] if skipped else ""))
    payload = finalize(rows, list(TUNING_AXES))
    payload["months"] = int(months)
    payload["fixed"] = {}
    payload["skipped"] = list(dict.fromkeys(skipped))
    return payload
