"""설정 변경은 재계산하고 금액 변경은 엔진 목표를 그대로 환산하는지 검증한다."""

import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from utils.mix_sleeve import SleeveSpec
from utils.strategy_mix_service import _sleeve_shares, _sleeve_target_shares
from utils.ttl_cache import TtlCache


class SettingsChangesTest(unittest.TestCase):
    def test_start_conditions_pool_and_allocation_invalidate_cached_mix(self):
        settings = {"start_date": "2026-01-02", "top_n": 2}
        ctx = {"account_id": "test", "slots": [SleeveSpec("a", "momentum", "us_stock", settings)]}
        weights = {"a_pct": 80, "cash_pct": 20}
        with (
            patch("utils.strategy_mix_service._SHARES_CACHE", TtlCache(60)),
            patch("utils.strategy_mix_service.mix_weights_for_account", side_effect=lambda account: dict(weights)),
            patch(
                "utils.strategy_mix_service._compute_sleeve_shares", return_value={"a_pct": 80, "cash_pct": 20}
            ) as compute,
        ):
            _sleeve_shares(ctx)
            _sleeve_shares(ctx)
            self.assertEqual(compute.call_count, 1)
            settings["start_date"] = "2026-02-02"
            _sleeve_shares(ctx)
            self.assertEqual(compute.call_count, 2)
            settings["top_n"] = 3
            _sleeve_shares(ctx)
            self.assertEqual(compute.call_count, 3)
            ctx["slots"] = [SleeveSpec("a", "new_high", "us_etf", settings)]
            _sleeve_shares(ctx)
            self.assertEqual(compute.call_count, 4)
            weights.update(a_pct=60, cash_pct=40)
            _sleeve_shares(ctx)
            self.assertEqual(compute.call_count, 5)

    def test_amount_and_allocation_changes_preserve_engine_internal_targets(self):
        states = {
            "a": SimpleNamespace(top_n=2, targets=[{"ticker": "X", "price": 100, "drift_pct": 50}]),
            "b": SimpleNamespace(top_n=2, targets=[{"ticker": "Y", "price": 100, "drift_pct": 25}]),
        }
        before = deepcopy(states)
        for budgets, expected in [
            ({"a": 4000, "b": 4000}, {"X": 20, "Y": 10}),
            ({"a": 8000, "b": 8000}, {"X": 40, "Y": 20}),
            ({"a": 6000, "b": 2000}, {"X": 30, "Y": 5}),
        ]:
            with self.subTest(budgets=budgets):
                self.assertEqual(_sleeve_target_shares(states, budgets, 1), expected)
                for key in states:
                    self.assertEqual(states[key].targets, before[key].targets)
