"""합성의 고정 시작일이 날짜 이동으로 보유 경로를 바꾸지 않는지 검증한다."""

import unittest
from unittest.mock import patch

import pandas as pd

from utils.slot_backtest import run_slot_backtest


class FixedStartDateTest(unittest.TestCase):
    def run_engine(self, end, start):
        days = pd.to_datetime(["2025-09-03", "2025-09-04", "2026-09-03", "2026-09-04"])
        prices = pd.DataFrame(100.0, index=days, columns=["A", "B"]).loc[:end]
        entry = pd.DataFrame(False, index=days, columns=["A", "B"])
        entry.loc[days[0], "A"] = True
        entry.loc[days[1] :, "B"] = True
        with (
            patch("utils.slot_backtest.adr_entry_gate", return_value=(lambda day: False, lambda day: None)),
            patch("utils.slot_backtest.get_pool_slippage", return_value=(0.0, 0.0)),
            patch("utils.slot_backtest.backtest_initial_capital", return_value=1000.0),
            patch(
                "utils.benchmark_curve.benchmark_growth", side_effect=lambda pool, index: pd.Series(1.0, index=index)
            ),
            patch("utils.new_high_service.benchmark_info", return_value={"name": "기준"}),
        ):
            return run_slot_backtest(
                pool="test",
                months=12,
                panel={"close": prices, "open": prices},
                entry=entry.loc[prices.index],
                exit_signal=prices < 0,
                priority=prices,
                slots=1,
                adr_floor=None,
                name_by={},
                industry_by={},
                exit_reason="이탈",
                start_date=start,
            )

    def test_fixed_start_keeps_holdings_when_day_advances(self):
        before = self.run_engine("2026-09-03", "2025-09-03")
        after = self.run_engine("2026-09-04", "2025-09-03")
        for result in (before, after):
            self.assertEqual([row["ticker"] for row in result["open_positions"]], ["A"])
            self.assertEqual(result["start_date"], "2025-09-03")
        rolling = self.run_engine("2026-09-04", None)
        self.assertEqual([row["ticker"] for row in rolling["open_positions"]], ["B"])

    def test_explicit_start_change_recomputes_holdings(self):
        result = self.run_engine("2026-09-04", "2025-09-04")
        self.assertEqual([row["ticker"] for row in result["open_positions"]], ["B"])

    def test_future_start_fails_instead_of_using_rolling_window(self):
        with self.assertRaises(RuntimeError):
            self.run_engine("2026-09-04", "2026-09-05")


class StrategyStartSettingsTest(unittest.TestCase):
    def test_unset_and_invalid_dates(self):
        from utils.strategy_settings import require_start_date, validate_start_date

        self.assertIsNone(validate_start_date(None))
        self.assertEqual(validate_start_date("2024-02-29"), "2024-02-29")
        for invalid in ("", "2025-02-29", "2025-9-3", 20250903):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                validate_start_date(invalid)
        with self.assertRaisesRegex(ValueError, "선택하고 저장"):
            require_start_date({"pool": "test"})

    def test_unset_start_does_not_run_positions(self):
        from utils import momentum_backtest, new_high_backtest

        for module in (momentum_backtest, new_high_backtest):
            with (
                self.subTest(module=module.__name__),
                patch.object(module, "validate_settings", return_value={"pool": "test", "start_date": None}),
                patch.object(module, "load_context") as load,
            ):
                with self.assertRaisesRegex(ValueError, "선택하고 저장"):
                    module.current_positions({"pool": "test"})
                load.assert_not_called()

    def test_mix_reads_each_strategy_start(self):
        from utils import momentum_backtest, new_high_backtest
        from utils.mix_sleeve import SleeveSpec, current_state

        for strategy, module, start in (
            ("momentum", momentum_backtest, "2025-09-03"),
            ("new_high", new_high_backtest, "2026-01-02"),
        ):
            settings = {"pool": "test", "start_date": start}
            spec = SleeveSpec(key="a", strategy=strategy, pool="test", settings=settings)
            with patch.object(module, "current_positions", return_value={}) as positions:
                current_state(spec)
                positions.assert_called_once_with(settings)
