"""목표 배분은 보유와 무관하고, 실제 보유는 매매 차이에만 영향을 줘야 한다."""

import unittest
from itertools import permutations
from math import fsum

from utils.share_allocation import ShareTarget, allocate_integer_shares


class ShareAllocationBudgetTest(unittest.TestCase):
    def test_fractional_priority_is_independent_of_input_order(self):
        targets = [ShareTarget("A", 190, 100), ShareTarget("B", 110, 100)]
        for ordered in permutations(targets):
            self.assertEqual(allocate_integer_shares(list(ordered), 300), {"A": 2, "B": 1})

    def test_exact_target_is_not_rounded_up(self):
        self.assertEqual(allocate_integer_shares([ShareTarget("A", 200, 100)], 1000), {"A": 2})

    def test_unaffordable_next_share_does_not_favor_cheaper_stock(self):
        targets = [ShareTarget("A", 190, 100), ShareTarget("B", 12, 10)]
        self.assertEqual(allocate_integer_shares(targets, 150), {"A": 1, "B": 1})

    def test_invalid_floor_budget_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "내림"):
            allocate_integer_shares([ShareTarget("A", 200, 100)], 100)

    def test_decimal_prices_never_exceed_budget(self):
        targets = [ShareTarget("A", 0.2, 0.1), ShareTarget("B", 0.2, 0.1)]
        result = allocate_integer_shares(targets, 0.4)
        self.assertLessEqual(fsum(result[t.key] * t.price for t in targets), 0.4)

    def test_invalid_budget_is_rejected(self):
        for value in (-1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                allocate_integer_shares([], value)

    def test_inventory_changes_only_trade_quantity(self):
        from utils.strategy_mix_service import _attach_account_targets

        target = allocate_integer_shares([ShareTarget("A", 500, 100)], 500)
        for held, trade in ((2, 3), (8, -3)):
            rows = [{"ticker": "A", "price": 100, "weight_pct": 50}]
            account = {"total_assets": 1000, "holdings": {"A": {"quantity": held, "value": held * 100}}}
            _attach_account_targets(rows, account, krw_rate=1, slot_keys=(), target_shares=target)
            self.assertEqual(rows[0]["target_quantity"], 5)
            self.assertEqual(rows[0]["trade_quantity"], trade)
