"""보유 단주 유지가 배분 예산을 초과하지 않는지 검증한다."""

import unittest
from itertools import permutations
from math import fsum

from utils.share_allocation import ShareTarget, allocate_integer_shares


class ShareAllocationBudgetTest(unittest.TestCase):
    def test_retained_extra_shares_are_trimmed_by_rounding_excess(self):
        targets = [ShareTarget("A", 190, 100, 2), ShareTarget("B", 110, 100, 2)]
        for ordered in permutations(targets):
            self.assertEqual(allocate_integer_shares(list(ordered), 300), {"A": 2, "B": 1})

    def test_affordable_existing_holdings_are_unchanged(self):
        targets = [ShareTarget("A", 190, 100, 2), ShareTarget("B", 110, 100, 2)]
        self.assertEqual(allocate_integer_shares(targets, 400), {"A": 2, "B": 2})

    def test_multiple_retained_extras_can_be_removed(self):
        targets = [ShareTarget("A", 110, 100, 2), ShareTarget("B", 120, 100, 2)]
        self.assertEqual(allocate_integer_shares(targets, 230), {"A": 1, "B": 1})

    def test_remaining_budget_can_still_fill_new_target(self):
        targets = [ShareTarget("A", 110, 100, 2), ShareTarget("B", 54, 60)]
        self.assertEqual(allocate_integer_shares(targets, 164), {"A": 1, "B": 1})

    def test_unaffordable_next_share_does_not_favor_cheaper_stock(self):
        targets = [ShareTarget("A", 190, 100), ShareTarget("B", 12, 10)]
        self.assertEqual(allocate_integer_shares(targets, 150), {"A": 1, "B": 1})

    def test_invalid_floor_budget_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "내림"):
            allocate_integer_shares([ShareTarget("A", 200, 100)], 100)

    def test_decimal_prices_never_exceed_budget(self):
        targets = [ShareTarget("A", 0.2, 0.1, 3), ShareTarget("B", 0.2, 0.1, 3)]
        result = allocate_integer_shares(targets, 0.4)
        self.assertLessEqual(fsum(result[t.key] * t.price for t in targets), 0.4)

    def test_invalid_budget_is_rejected(self):
        for value in (-1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                allocate_integer_shares([], value)
