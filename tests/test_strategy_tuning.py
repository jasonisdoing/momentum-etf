from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from utils import momentum_service, momentum_tuning, strategy_tuning
from utils.moving_averages import calculate_moving_average
from utils.perf_metrics import daily_return_metrics


class _FakeAsyncResult:
    def __init__(self, value: object) -> None:
        self.value = value

    def ready(self) -> bool:
        return True

    def get(self) -> object:
        return self.value


class _FakePool:
    latest: _FakePool | None = None

    def __init__(self, *, initializer=None, initargs=(), **_kwargs) -> None:
        self.closed = False
        self.terminated = False
        self.joined = False
        self.cancel_event = initargs[-1]
        _FakePool.latest = self
        if initializer is not None:
            initializer(*initargs)

    def apply_async(self, worker, args) -> _FakeAsyncResult:
        return _FakeAsyncResult(worker(*args))

    def close(self) -> None:
        self.closed = True

    def terminate(self) -> None:
        self.terminated = True

    def join(self) -> None:
        self.joined = True


class _FakeContext:
    def Event(self):  # noqa: N802 - multiprocessing 컨텍스트 API 이름을 그대로 흉내 낸다.
        from threading import Event

        return Event()

    def Pool(self, **kwargs) -> _FakePool:  # noqa: N802 - multiprocessing 컨텍스트 API 이름이다.
        return _FakePool(**kwargs)


class StrategyTuningProgressTest(unittest.TestCase):
    def tearDown(self) -> None:
        strategy_tuning.cancel_active_tuning()

    @patch.object(strategy_tuning, "get_context", return_value=_FakeContext())
    @patch.object(strategy_tuning, "tuning_workers", return_value=2)
    def test_iter_groups_closes_pool_after_completion(self, _workers, _context) -> None:
        run = strategy_tuning.begin_tuning("테스트")
        events = list(strategy_tuning.iter_groups(lambda value: value, [1, 2, 3], run=run))

        self.assertEqual([event[2] for event in events], [1, 2, 3])
        assert _FakePool.latest is not None
        self.assertTrue(_FakePool.latest.closed)
        self.assertTrue(_FakePool.latest.joined)
        self.assertFalse(_FakePool.latest.terminated)

    @patch.object(strategy_tuning, "get_context", return_value=_FakeContext())
    @patch.object(strategy_tuning, "tuning_workers", return_value=2)
    def test_iter_groups_terminates_pool_when_stream_closes(self, _workers, _context) -> None:
        run = strategy_tuning.begin_tuning("테스트")
        iterator = strategy_tuning.iter_groups(lambda value: value, [1, 2, 3], run=run)
        next(iterator)
        iterator.close()

        assert _FakePool.latest is not None
        self.assertTrue(_FakePool.latest.cancel_event.is_set())
        self.assertTrue(_FakePool.latest.terminated)
        self.assertTrue(_FakePool.latest.joined)

    def test_begin_tuning_terminates_previous_pool(self) -> None:
        previous = strategy_tuning.begin_tuning("모멘텀 us_stock 18개월")
        pool = _FakePool(initargs=(None, (), _FakeContext().Event()))
        previous.attach_pool(pool, pool.cancel_event)

        current = strategy_tuning.begin_tuning("모멘텀 us_stock 3개월")

        self.assertTrue(previous.cancelled)
        self.assertTrue(pool.terminated)
        self.assertTrue(pool.joined)
        self.assertEqual(current.replaced_label, "모멘텀 us_stock 18개월")

    @patch.object(momentum_tuning, "finalize", return_value={"rows": [], "axes": {}, "quarter_count": 0})
    @patch.object(momentum_tuning, "iter_groups", return_value=[(1, 1, ([{"params": {}}], []))])
    @patch.object(momentum_tuning, "_preload", return_value={})
    @patch.object(momentum_tuning, "validate_settings", return_value={"pool": "us_stock"})
    def test_momentum_progress_reports_combinations_and_phases(
        self,
        _validate_settings,
        _preload,
        _iter_groups,
        _finalize,
    ) -> None:
        run = strategy_tuning.begin_tuning("모멘텀 us_stock 12개월")
        events = list(
            momentum_tuning.stream_tuning(
                12,
                {"pool": "us_stock"},
                {
                    "short_ma_days": [20],
                    "long_ma_days": [50],
                    "adr_floor": [90, 100],
                },
                run,
            )
        )

        progress = [event for event in events if event["type"] == "progress"]
        self.assertEqual([event["phase"] for event in progress], ["prepare", "backtest", "backtest", "finalize"])
        self.assertEqual(progress[0]["total"], 2)
        self.assertEqual(progress[2]["done"], 2)


class DailyPerformanceMetricsTest(unittest.TestCase):
    def test_mdd_includes_loss_on_first_day(self) -> None:
        returns = pd.Series([-10.0, 5.0], index=pd.to_datetime(["2026-01-02", "2026-01-05"]))

        metrics = daily_return_metrics(returns)

        self.assertAlmostEqual(float(metrics["mdd_pct"]), -10.0)

    def test_tuning_summary_uses_shared_daily_metrics(self) -> None:
        returns = pd.Series(
            [10.0, -20.0, 5.0],
            index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
        )

        metrics = daily_return_metrics(returns)
        summary = strategy_tuning.summarize_combo({}, returns)

        self.assertEqual(summary["total_pct"], round(float(metrics["total_pct"]), 1))
        self.assertEqual(summary["cagr_pct"], round(float(metrics["cagr_pct"]), 1))
        self.assertEqual(summary["mdd_pct"], round(float(metrics["mdd_pct"]), 1))
        self.assertEqual(summary["sortino"], round(float(metrics["sortino"]), 2))


class MomentumIntraweekCacheTest(unittest.TestCase):
    def test_candidate_dates_reuse_same_moving_average_series(self) -> None:
        dates = pd.date_range("2026-01-02", periods=12, freq="B")
        close = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111], index=dates)
        frames = {"AAA": pd.DataFrame({"Open": close, "Close": close}, index=dates)}
        universe = [{"ticker": "AAA", "name": "테스트", "pool": "us_stock"}]
        settings = {
            "short_ma_days": 2,
            "long_ma_days": 3,
            "adr_floor": None,
        }
        series_cache: momentum_service.IntraweekSeriesCache = {}

        with patch(
            "utils.moving_averages.calculate_moving_average",
            wraps=calculate_moving_average,
        ) as moving_average:
            first = momentum_service.select_candidates(
                universe,
                frames,
                settings,
                as_of=dates[8],
                series_cache=series_cache,
            )
            second = momentum_service.select_candidates(
                universe,
                frames,
                settings,
                as_of=dates[10],
                series_cache=series_cache,
            )

        expected_first = momentum_service.momentum_metrics(
            close,
            short_ma_days=2,
            long_ma_days=3,
            as_of=dates[8],
        )
        expected_second = momentum_service.momentum_metrics(
            close,
            short_ma_days=2,
            long_ma_days=3,
            as_of=dates[10],
        )
        assert expected_first is not None and expected_second is not None
        self.assertEqual(moving_average.call_count, 2)
        self.assertAlmostEqual(first[0]["disparity_pct"], expected_first["disparity_pct"])
        self.assertAlmostEqual(second[0]["disparity_pct"], expected_second["disparity_pct"])

    def test_intraweek_exits_only_fire_on_adr_gate(self) -> None:
        """주중 매도는 ADR 게이트만 — 게이트가 없으면(하한 미설정) 주중 매도가 없다."""
        dates = pd.date_range("2026-01-02", periods=10, freq="B")
        settings = {"adr_floor": None, "pool": "us_stock"}

        exits = momentum_service.simulate_intraweek_exits(settings, {"AAA"}, dates, dates[5], dates[8])

        self.assertEqual(exits, [])


if __name__ == "__main__":
    unittest.main()
