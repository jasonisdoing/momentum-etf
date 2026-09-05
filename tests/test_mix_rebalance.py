"""합성 배분의 현금 보존과 월초 재생 일치 검증."""

import unittest
from unittest.mock import patch

from core.strategy.mix_rebalance import rebalance_sleeves
from utils.mix_sleeve import SleeveSpec
from utils.strategy_mix_service import _simulate_mix


class MixRebalanceTests(unittest.TestCase):
    def test_cash_transfer_has_no_cost(self):
        values, cash, cost = rebalance_sleeves(
            {"a": 80, "b": 10},
            10,
            {"a": 0.4, "b": 0.4, "cash": 0.2},
            {"a": 0, "b": 0},
            {"a": (0.01, 0.02), "b": (0.01, 0.02)},
        )
        self.assertEqual(values, {"a": 40, "b": 40})
        self.assertEqual(cash, 20)
        self.assertEqual(cost, 0)

    def test_only_stock_changes_pay_cost(self):
        values, cash, cost = rebalance_sleeves(
            {"a": 80, "b": 10},
            10,
            {"a": 0.4, "b": 0.4, "cash": 0.2},
            {"a": 0.5, "b": 0.25},
            {"a": (0.01, 0.02), "b": (0.01, 0.02)},
        )
        expected = (80 - values["a"]) * 0.5 * 0.02 + (values["b"] - 10) * 0.25 * 0.01
        self.assertAlmostEqual(cost, expected)
        self.assertAlmostEqual(sum(values.values()) + cash + cost, 100)
        self.assertAlmostEqual(cash / (100 - cost), 0.2)

    def test_pending_month_start_matches_backtest(self):
        slots = [SleeveSpec("a", "momentum", "us_stock", {}), SleeveSpec("b", "momentum", "us_stock", {})]
        ctx = {"account_id": "test", "slots": slots}
        results = {
            key: {
                "daily": [
                    {"date": "2026-08-28", "strategy_pct": 0, "cash_weight_pct": 50},
                    {"date": "2026-08-31", "strategy_pct": pct, "cash_weight_pct": 50},
                ]
            }
            for key, pct in [("a", 20), ("b", 0)]
        }
        with (
            patch(
                "utils.strategy_mix_service.mix_weights_for_account",
                return_value={"a_pct": 40, "b_pct": 40, "cash_pct": 20},
            ),
            patch("utils.pool_settings_store.get_pool_slippage", return_value=(0.1, 0.2)),
        ):
            pending = _simulate_mix(ctx, results, through_date="2026-09-01")
            for result in results.values():
                result["daily"].append({**result["daily"][-1], "date": "2026-09-01"})
            actual = _simulate_mix(ctx, results, through_date=None)
        self.assertEqual(pending["values"], actual["values"])
        self.assertEqual(pending["cash"], actual["cash"])
        total = sum(actual["values"].values()) + actual["cash"]
        self.assertAlmostEqual(actual["values"]["a"] / total, 0.4)
        self.assertAlmostEqual(actual["cash"] / total, 0.2)
