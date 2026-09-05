"""포트폴리오의 주기·밴드·가격 드리프트가 합성 목표까지 유지되는지 검증한다."""

import unittest
from unittest.mock import patch

import pandas as pd

from utils.mix_sleeve import SleeveSpec, slot_state
from utils.portfolio_backtest import run_backtest


class PortfolioMixStateTest(unittest.TestCase):
    def simulate(self, dates, a_prices, rebalance, band, cash):
        index = pd.to_datetime(dates)
        frame = pd.DataFrame({"A": a_prices, "B": [100.0] * len(dates)}, index=index)
        settings = {
            "pool": "test",
            "start_date": dates[0],
            "rebalance": rebalance,
            "band_pct": band,
            "cash_weight_pct": cash,
            "weights": [{"ticker": t, "weight_pct": (100 - cash) / 2} for t in ("A", "B")],
        }
        with (
            patch("utils.portfolio_backtest.validate_settings", side_effect=lambda s: s),
            patch("utils.portfolio_backtest._load_close_frame", return_value=frame),
            patch("utils.portfolio_backtest.get_pool_slippage", return_value=(0, 0)),
            patch("utils.benchmark_curve.load_benchmark_frame", return_value=pd.DataFrame({"Close": 100}, index=index)),
            patch("utils.benchmark_curve.benchmark_growth", side_effect=lambda pool, ix: pd.Series(1.0, index=ix)),
            patch("utils.portfolio_backtest.benchmark_info", return_value={"name": "기준"}),
        ):
            result = run_backtest(12, settings, start_date=dates[0])
        spec = SleeveSpec("a", "portfolio", "test", settings)
        with (
            patch("utils.mix_sleeve.current_state", return_value=result),
            patch("utils.portfolio_service.universe_metrics", return_value=[]),
            patch("utils.settings_loader.get_ticker_type_settings", return_value={"currency": "USD"}),
        ):
            state = slot_state(spec)
        self.assertEqual(state.engine_trades, [t for t in result["trades"] if t["date"] == result["as_of"]])
        for source, target in zip(result["open_positions"], state.targets, strict=True):
            self.assertEqual(source["sleeve_weight_pct"], target["drift_pct"])
            self.assertEqual(source["price"], target["price"])
        self.assertAlmostEqual(sum(t["drift_pct"] for t in state.targets) + result["sleeve_cash_weight_pct"], 100)
        return result, state

    def test_between_monthly_rebalances_keeps_drift(self):
        result, state = self.simulate(["2026-01-02", "2026-01-30"], [100, 200], "monthly", 3, 0)
        self.assertEqual(result["rebalance_count"], 0)
        self.assertAlmostEqual(state.targets[0]["drift_pct"], 200 / 3)

    def test_monthly_rebalance_below_band_keeps_drift(self):
        result, state = self.simulate(["2026-01-02", "2026-02-02"], [100, 110], "monthly", 3, 0)
        self.assertEqual(result["rebalance_count"], 0)
        self.assertAlmostEqual(state.targets[0]["drift_pct"], 110 / 210 * 100)

    def test_monthly_rebalance_above_band_resets_weights(self):
        result, state = self.simulate(["2026-01-02", "2026-02-02"], [100, 200], "monthly", 3, 0)
        self.assertEqual(result["rebalance_count"], 2)
        self.assertAlmostEqual(state.targets[0]["drift_pct"], 50)

    def test_other_frequencies_do_not_rebalance_in_february(self):
        for frequency in ("none", "quarterly", "yearly"):
            with self.subTest(frequency=frequency):
                result, state = self.simulate(["2026-01-02", "2026-02-02"], [100, 200], frequency, 3, 0)
                self.assertEqual(result["rebalance_count"], 0)
                self.assertAlmostEqual(state.targets[0]["drift_pct"], 200 / 3)

    def test_cash_drifts_with_actual_holdings(self):
        result, _ = self.simulate(["2026-01-02", "2026-01-30"], [100, 200], "monthly", 3, 20)
        self.assertAlmostEqual(result["sleeve_cash_weight_pct"], 0.2 / 1.4 * 100)
