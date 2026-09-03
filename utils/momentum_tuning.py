"""모멘텀 전략 튜닝 — 설정 항목들의 범위 조합을 한 번에 백테스트해 비교한다.

화면 '튜닝' 섹션용. 축(화면 순서): 선정 이평(단기·장기) · ADR 하한.
종목 수는 풀 설정(`pool_settings.TOP_N_HOLD`) 고정, 업종 상한은 폐기, 교체 규칙(자격 유지)과
주중 이탈(사용)은 전략에 고정 — 튜닝은 "그 시장이 어떤 이동평균에 반응하는가"를 재는 용도로만
쓴다(종목 수까지 돌리면 과적합 탐색이 된다).
(단기, 장기) 쌍을 작업 단위로 별도 프로세스에서 병렬로 돌린다 — 각 프로세스는 가격·판정일별
후보를 한 번 읽어 그 쌍의 전 조합에 공유(run_backtest 의 context)하므로 조합당 0.5초 수준이고,
병렬 수(코어 수 − 1)만큼 전체 시간이 줄어든다.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pandas as pd

from config import ADR_FLOOR_OPTIONS
from utils.momentum_service import (
    LONG_MA_OPTIONS,
    SHORT_MA_OPTIONS,
    load_settings,
    validate_settings,
)
from utils.strategy_tuning import (
    TuningRun,
    begin_tuning,
    finalize,
    iter_groups,
    managed_tuning_events,
    seed_worker_caches,
    summarize_combo,
    tuning_cancelled,
)

TUNING_AXES = ("short_ma_days", "long_ma_days", "adr_floor")


def _checked_optional_ints(values: list[Any], options: tuple, label: str) -> list[Any]:
    """None(없음)이 섞인 정수 선택지 축 검증."""
    cleaned = list(dict.fromkeys(None if v in (None, "", "none") else int(v) for v in values))
    bad = [v for v in cleaned if v not in options]
    if not cleaned or bad:
        raise ValueError(f"'{label}' 범위가 올바르지 않습니다: {bad or cleaned}")
    return cleaned


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
    if bundle.get("adr_market") and bundle.get("adr_series") is not None:
        from utils.momentum_service import seed_adr_series

        seed_adr_series(bundle["adr_market"], bundle["adr_series"])


def _preload(pool: str) -> dict[str, Any]:
    """부모가 한 번 읽는 공유 데이터 — 워커 초기화 때 통째로 넘긴다."""
    from utils.momentum_service import (
        adr_market_of_pool,
        load_adr_series,
        load_benchmark_close,
        load_price_frames,
        load_universe,
    )
    from utils.settings_loader import _load_pool_configs
    from utils.stock_list_io import _load_ticker_type_stocks_raw

    universe = load_universe(pool)
    # ADR 시계열 — 워커는 DB 를 안 건드리므로 부모가 읽어서 심는다(레짐 미설정 풀은 None).
    adr_market = adr_market_of_pool(pool)
    return {
        "pool_configs": _load_pool_configs(),
        "stocks_by_pool": {pool: _load_ticker_type_stocks_raw(pool)},
        "universe": universe,
        "frames": load_price_frames(universe),
        "benchmark_close": load_benchmark_close(pool),
        "adr_market": adr_market,
        "adr_series": load_adr_series(adr_market) if adr_market else None,
    }


def _run_ma_group(task: tuple) -> tuple[list[dict[str, Any]], list[str]]:
    """(단기, 장기) 쌍 하나의 조합 — 별도 프로세스에서 돈다."""
    months, base, short, long, adr_floors = task
    from utils.momentum_backtest import run_backtest

    context = _WORKER_CONTEXT
    if not context:
        context.update({k: _PRELOAD[k] for k in ("universe", "frames", "benchmark_close")})
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for adr_floor in adr_floors:
        if tuning_cancelled():
            break
        combo = dict(base, short_ma_days=short, long_ma_days=long, adr_floor=adr_floor)
        try:
            result = run_backtest(months, combo, include_daily=True, tuning_only=True, context=context)
        except ValueError as error:  # 장기 이평이 길어 기간이 모자라는 조합 — 그 이평 쌍은 통째로 건너뛴다
            skipped.append(f"단기 {short}/장기 {long}: {error}")
            break
        daily = pd.DataFrame(sorted(result["daily"], key=lambda r: r["date"]))
        daily["date"] = pd.to_datetime(daily["date"])
        returns = daily.set_index("date")["strategy_pct"].dropna()
        rows.append(
            summarize_combo(
                {"short_ma_days": short, "long_ma_days": long, "adr_floor": adr_floor},
                returns,
                {"trade_count": result["trade_count"], "win_rate_pct": result["win_rate_pct"]},
            )
        )
    return rows, skipped


def run_tuning(months: int, settings: dict[str, Any] | None, ranges: dict[str, list[Any]]) -> dict[str, Any]:
    """전 조합을 돌려 결과 페이로드를 만든다(진행 없이 한 번에)."""
    run = begin_tuning(f"모멘텀 직접 실행 {months}개월")
    for event in stream_tuning(months, settings, ranges, run):
        if event["type"] == "result":
            return event["payload"]
    raise RuntimeError("튜닝이 결과를 내지 못했습니다.")


def stream_tuning(
    months: int,
    settings: dict[str, Any] | None,
    ranges: dict[str, list[Any]],
    run: TuningRun,
) -> Iterator[dict[str, Any]]:
    yield from managed_tuning_events(run, _stream_tuning(months, settings, ranges, run))


def _stream_tuning(
    months: int,
    settings: dict[str, Any] | None,
    ranges: dict[str, list[Any]],
    run: TuningRun,
) -> Iterator[dict[str, Any]]:
    """묶음이 끝날 때마다 진행을, 마지막에 결과를 내보낸다.

    화면이 이 흐름을 그대로 진행 바에 쓴다 — 다 끝난 뒤 한 번에 응답하면 10분 넘게
    아무 소식이 없어 화면이 죽은 것처럼 보인다.
    """
    base = validate_settings(settings or load_settings())
    shorts = _checked(ranges.get("short_ma_days", []), SHORT_MA_OPTIONS, "단기 이평")
    adr_floors = _checked_optional_ints(ranges.get("adr_floor", []), ADR_FLOOR_OPTIONS, "ADR 하한")
    longs = _checked(ranges.get("long_ma_days", []), LONG_MA_OPTIONS, "장기 이평")
    ma_pairs = [(short, long) for short in shorts for long in longs if short < long]
    if not ma_pairs:
        raise ValueError("단기 이평이 장기 이평보다 작은 조합이 없습니다.")

    tasks = [(months, base, short, long, adr_floors) for short, long in ma_pairs]
    combos_per_group = len(adr_floors)
    total_combos = len(tasks) * combos_per_group
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    yield {"type": "progress", "phase": "prepare", "done": 0, "total": total_combos}
    bundle = _preload(str(base["pool"]))
    yield {"type": "progress", "phase": "backtest", "done": 0, "total": total_combos}
    for done, total, (group_rows, group_skipped) in iter_groups(
        _run_ma_group,
        tasks,
        run=run,
        initializer=_init_worker,
        initargs=(bundle,),
    ):
        rows.extend(group_rows)
        skipped.extend(group_skipped)
        yield {
            "type": "progress",
            "phase": "backtest",
            "done": min(total_combos, done * combos_per_group),
            "total": total * combos_per_group,
        }

    if not rows:
        raise ValueError("돌릴 수 있는 조합이 없습니다. " + (skipped[0] if skipped else ""))
    yield {"type": "progress", "phase": "finalize", "done": total_combos, "total": total_combos}
    payload = finalize(rows, list(TUNING_AXES))
    payload["months"] = int(months)
    payload["fixed"] = {}
    payload["skipped"] = list(dict.fromkeys(skipped))
    yield {"type": "result", "payload": payload}
