"""전략 튜닝 공용 — 여러 설정 조합의 백테스트 결과를 같은 잣대로 요약한다.

모멘텀·신고가 화면의 '튜닝' 섹션이 함께 쓴다. 엔진은 각 전략 것을 그대로 호출하고,
여기서는 일별 수익률 → 지표(수익·CAGR·MDD·소르티노), 분기별 수익률,
**분기 승수**(조합들 중 그 분기에 상위 절반에 든 횟수 — 구간 일관성), **축별 평균**
(어느 값이 일관되게 나은가)만 계산한다.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator
from multiprocessing import get_context
from multiprocessing.pool import Pool
from threading import Event, Lock
from typing import Any

import pandas as pd

from config import TUNING_WORKERS
from utils.perf_metrics import daily_return_metrics

# 튜닝 스트림이 끊기면 워커가 현재 조합을 마친 뒤 다음 조합으로 넘어가지 않게 하는 신호.
# 프로세스 시작 때 initializer 로 전달하므로 spawn 환경에서도 같은 이벤트를 본다.
_CANCEL_EVENT: Any = None


class TuningCancelledError(RuntimeError):
    """사용자 중단 또는 새 튜닝 시작으로 현재 실행이 교체됐다."""


class TuningRun:
    """서버 전체에서 실행 중인 튜닝 하나와 그 프로세스 풀의 종료 핸들."""

    def __init__(self, label: str, replaced_label: str | None) -> None:
        self.label = label
        self.replaced_label = replaced_label
        self._cancelled = Event()
        self._lock = Lock()
        self._pool: Pool | None = None
        self._worker_cancel_event: Any = None
        self._pool_stopped = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def wait(self, seconds: float) -> None:
        self._cancelled.wait(seconds)

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise TuningCancelledError("튜닝이 중단되었습니다.")

    def attach_pool(self, pool: Pool, worker_cancel_event: Any) -> None:
        """새 풀을 연결한다. 연결 전에 취소됐다면 풀을 즉시 끝낸다."""
        with self._lock:
            self._pool = pool
            self._worker_cancel_event = worker_cancel_event
            self._pool_stopped = False
            cancelled = self.cancelled
        if cancelled:
            self.cancel()
            raise TuningCancelledError("튜닝이 중단되었습니다.")

    def close_pool(self, pool: Pool) -> None:
        """정상 완료된 풀을 정리한다."""
        with self._lock:
            if self._pool is not pool or self._pool_stopped:
                return
            self._pool_stopped = True
        pool.close()
        pool.join()

    def detach_pool(self, pool: Pool) -> None:
        with self._lock:
            if self._pool is pool:
                self._pool = None
                self._worker_cancel_event = None

    def cancel(self) -> None:
        """실행 중인 워커를 기다리지 않고 실제 프로세스 풀까지 종료한다."""
        self._cancelled.set()
        with self._lock:
            worker_cancel_event = self._worker_cancel_event
            pool = self._pool
            should_stop = pool is not None and not self._pool_stopped
            if should_stop:
                self._pool_stopped = True
        if worker_cancel_event is not None:
            worker_cancel_event.set()
        if should_stop and pool is not None:
            pool.terminate()
            pool.join()


_ACTIVE_TUNING_LOCK = Lock()
_ACTIVE_TUNING: TuningRun | None = None


def begin_tuning(label: str) -> TuningRun:
    """기존 튜닝을 종료하고 서버 전체의 새 활성 튜닝을 등록한다."""
    global _ACTIVE_TUNING
    with _ACTIVE_TUNING_LOCK:
        previous = _ACTIVE_TUNING
        if previous is not None:
            previous.cancel()
        run = TuningRun(label, previous.label if previous is not None else None)
        _ACTIVE_TUNING = run
        return run


def finish_tuning(run: TuningRun) -> None:
    """자신이 아직 활성 실행일 때만 전역 슬롯을 비운다."""
    global _ACTIVE_TUNING
    with _ACTIVE_TUNING_LOCK:
        if _ACTIVE_TUNING is run:
            _ACTIVE_TUNING = None


def cancel_active_tuning() -> dict[str, Any]:
    """브라우저의 중단 버튼용 — 현재 튜닝과 워커를 즉시 종료한다."""
    global _ACTIVE_TUNING
    with _ACTIVE_TUNING_LOCK:
        run = _ACTIVE_TUNING
        if run is None:
            return {"cancelled": False, "message": "실행 중인 튜닝이 없습니다."}
        run.cancel()
        if _ACTIVE_TUNING is run:
            _ACTIVE_TUNING = None
        return {"cancelled": True, "label": run.label, "message": f"{run.label} 튜닝을 중단했습니다."}


def managed_tuning_events(run: TuningRun, source: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """교체·중단 알림과 활성 슬롯 정리를 전략별 이벤트 흐름에 공통 적용한다."""
    iterator = iter(source)
    completed = False
    try:
        if run.replaced_label is not None:
            yield {
                "type": "notice",
                "message": f"기존 튜닝({run.replaced_label})을 중단하고 새 튜닝을 시작했습니다.",
            }
        for event in iterator:
            run.raise_if_cancelled()
            if event.get("type") == "result":
                completed = True
            yield event
    except TuningCancelledError:
        yield {"type": "cancelled", "message": "튜닝이 중단되었습니다."}
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
        if not completed:
            run.cancel()
        finish_tuning(run)


def _init_cancelable_worker(
    initializer: Callable[..., None] | None,
    initargs: tuple,
    cancel_event: Any,
) -> None:
    global _CANCEL_EVENT
    _CANCEL_EVENT = cancel_event
    if initializer is not None:
        initializer(*initargs)


def tuning_cancelled() -> bool:
    """현재 튜닝 요청이 취소됐는지 워커 안에서 확인한다."""
    return bool(_CANCEL_EVENT is not None and _CANCEL_EVENT.is_set())


def tuning_workers(task_count: int) -> int:
    """병렬 프로세스 수 — `config.TUNING_WORKERS`(None 이면 코어 수), 작업 수보다 크지 않게.

    왜 코어 수 전부가 기본이 아닌지는 config 쪽 주석에 적어 두었다.
    """
    workers = TUNING_WORKERS if TUNING_WORKERS is not None else (os.cpu_count() or 1)
    return max(1, min(workers, task_count))


def seed_worker_caches(pool_configs: list[dict[str, Any]], stocks_by_pool: dict[str, list[dict[str, Any]]]) -> None:
    """워커 프로세스가 DB 를 건드리지 않게, 부모가 읽어 둔 풀 설정·종목 목록으로 캐시를 채운다.

    워커 여러 개가 동시에 MongoDB 를 읽으면(가격·종목·설정) 로컬 mongod 가 CPU 를 뺏겨
    타임아웃이 나고 화면의 다른 요청까지 실패한다 — 그래서 워커는 읽기 전용 사본만 쓴다.

    **메인 프로세스에서는 아무것도 하지 않는다.** 작업이 하나뿐이면 `iter_groups` 가
    별도 프로세스를 만들지 않고 여기서 초기화하는데, 그때 아래 것들을 그대로 실행하면
    서버 프로세스가 오염된다:

      · `_load_pool_configs` 를 스냅샷으로 갈아끼우므로, 그 뒤 종목풀 설정을 저장해도
        서버가 계속 옛 값을 본다(종목 수를 6→5 로 바꿔도 6 으로 굳었다).
      · `os.nice(5)` 는 되돌릴 수 없어 서버 우선순위가 영구히 내려간다.

    메인은 DB 를 읽을 수 있으니 시딩 자체가 필요 없다 — 건너뛰는 것이 곧 정답이다.
    """
    import os
    from multiprocessing import current_process

    if current_process().name == "MainProcess":
        return

    from utils import settings_loader, stock_list_io

    settings_loader._load_pool_configs = lambda: pool_configs  # type: ignore[assignment]
    stock_list_io.seed_pool_stocks(stocks_by_pool)
    try:
        os.nice(5)  # 화면·DB 가 워커에 밀리지 않게 우선순위를 조금 낮춘다
    except OSError:
        pass


def iter_groups(
    worker: Callable[[Any], Any],
    tasks: Iterable[Any],
    *,
    run: TuningRun,
    initializer: Callable[..., None] | None = None,
    initargs: tuple = (),
) -> Iterator[tuple[int, int, Any]]:
    """작업 그룹을 병렬 실행하고 결과를 **끝나는 대로 하나씩** 흘린다.

    `(완료 수, 전체 수, 결과)` 를 내보낸다 — 호출자가 진행을 그대로 스트리밍할 수 있다.
    튜닝은 10분 넘게 걸려서, 다 끝난 뒤 한 번에 응답하면 화면이 죽은 것처럼 보인다.
    """
    tasks = list(tasks)
    total = len(tasks)
    workers = tuning_workers(total)
    context = get_context("spawn")
    cancel_event = context.Event()
    pool = context.Pool(
        processes=workers,
        initializer=_init_cancelable_worker,
        initargs=(initializer, initargs, cancel_event),
    )
    run.attach_pool(pool, cancel_event)
    pending = [pool.apply_async(worker, (task,)) for task in tasks]
    completed = False
    try:
        done = 0
        while pending:
            run.raise_if_cancelled()
            ready = [result for result in pending if result.ready()]
            if not ready:
                # 취소 API가 0.1초 안에 이 루프를 깨울 수 있게 결과를 무한 대기하지 않는다.
                run.wait(0.1)
                continue
            for result in ready:
                pending.remove(result)
                done += 1
                yield done, total, result.get()
        completed = True
    finally:
        if completed:
            run.close_pool(pool)
        else:
            run.cancel()
        run.detach_pool(pool)


def cumulative_to_returns(cumulative_pct: pd.Series) -> pd.Series:
    """누적 수익률(%) 곡선 → 일별 수익률(%)."""
    level = 1 + cumulative_pct.sort_index() / 100
    return ((level / level.shift(1)) - 1).dropna() * 100


def quarterly_returns(returns_pct: pd.Series) -> dict[str, float]:
    return {
        str(quarter): round(float(((1 + part / 100).prod() - 1) * 100), 1)
        for quarter, part in returns_pct.groupby(returns_pct.index.to_period("Q"))
    }


def summarize_combo(
    params: dict[str, Any], returns_pct: pd.Series, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """조합 하나의 요약 행. ``extra`` 는 전략별 부가 지표(거래 수·승률 등)."""
    metrics = daily_return_metrics(returns_pct)
    return {
        "params": params,
        "total_pct": round(float(metrics["total_pct"]), 1),
        "cagr_pct": round(float(metrics["cagr_pct"]), 1) if metrics["cagr_pct"] is not None else None,
        "mdd_pct": round(float(metrics["mdd_pct"]), 1),
        "sortino": round(float(metrics["sortino"]), 2) if metrics["sortino"] is not None else None,
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
        # (종목수 5, 6, … / ADR 90, 100), None 은 맨 뒤.
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
