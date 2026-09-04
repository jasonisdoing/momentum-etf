from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from core.strategy.scoring import hold_eligible
from utils import momentum_backtest, momentum_tuning, strategy_tuning
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


class MomentumSignalsTest(unittest.TestCase):
    """모멘텀 신호 표 — 진입·청산·우선순위가 공용 규칙과 같은지."""

    def _panel(self) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
        dates = pd.date_range("2026-01-02", periods=12, freq="B")
        rising = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111], index=dates, dtype=float)
        falling = pd.Series([120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109], index=dates, dtype=float)
        panel = {
            "close": pd.DataFrame({"UP": rising, "DOWN": falling}),
            "open": pd.DataFrame({"UP": rising, "DOWN": falling}),
        }
        return panel, dates

    def test_entry_matches_shared_hold_rule(self) -> None:
        """진입 자격은 순위 화면과 **같은 공용 함수**(`hold_eligible`)여야 한다."""
        panel, dates = self._panel()
        signals = momentum_backtest.compute_signals(panel, short_ma_days=2, long_ma_days=3)
        last = dates[-1]

        # 오르는 종목은 두 이평선 위 — 진입 가능. 내리는 종목은 아래 — 청산 신호.
        self.assertTrue(bool(signals["eligible"].at[last, "UP"]))
        self.assertFalse(bool(signals["eligible"].at[last, "DOWN"]))
        self.assertTrue(bool(signals["exit"].at[last, "DOWN"]))
        self.assertFalse(bool(signals["exit"].at[last, "UP"]))
        self.assertTrue(
            bool(hold_eligible(signals["long"].at[last, "UP"], signals["short"].at[last, "UP"])),
        )

    def test_unknown_days_are_neither_entry_nor_exit(self) -> None:
        """이평선을 못 채운 날은 사지도 팔지도 않는다 — 값을 추정하지 않는다."""
        panel, dates = self._panel()
        signals = momentum_backtest.compute_signals(panel, short_ma_days=2, long_ma_days=3)
        first = dates[0]

        self.assertFalse(bool(signals["known"].at[first, "UP"]))
        self.assertFalse(bool(signals["eligible"].at[first, "UP"]))
        self.assertFalse(bool(signals["exit"].at[first, "UP"]))

    def test_priority_is_long_disparity(self) -> None:
        """자리 경쟁 우선순위는 순위 화면과 같은 기준(`rank_score`, 장기 이격률)이다."""
        panel, dates = self._panel()
        signals = momentum_backtest.compute_signals(panel, short_ma_days=2, long_ma_days=3)
        last = dates[-1]

        self.assertAlmostEqual(signals["priority"].at[last, "UP"], signals["long"].at[last, "UP"])
        self.assertGreater(signals["priority"].at[last, "UP"], signals["priority"].at[last, "DOWN"])


if __name__ == "__main__":
    unittest.main()
