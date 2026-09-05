"""전략 신호와 계좌 차이를 구분하되 목표·주문을 바꾸지 않는지 검증한다."""

import unittest

from utils.strategy_mix_service import _action_reasons, _build_action_groups


class ActionReasonsTest(unittest.TestCase):
    def actions(self):
        return {
            "slots": {
                "a": {
                    "label": "A",
                    "entries": [{"ticker": "X"}],
                    "sells": [],
                    "rebalance": None,
                    "exit_forecast": [],
                    "live": False,
                }
            }
        }

    def test_signal_is_directional(self):
        self.assertEqual(
            [r["code"] for r in _action_reasons("X", "buy", self.actions())], ["strategy_signal", "target_difference"]
        )
        self.assertEqual([r["code"] for r in _action_reasons("X", "sell", self.actions())], ["target_difference"])

    def test_filled_rebalance_is_not_new_signal(self):
        actions = self.actions()
        actions["slots"]["a"]["rebalance"] = {"is_filled": True, "buys": [{"ticker": "Y"}], "sells": []}
        self.assertEqual(len(_action_reasons("Y", "buy", actions)), 1)
        actions["slots"]["a"]["rebalance"]["is_filled"] = False
        self.assertEqual(_action_reasons("Y", "buy", actions)[0]["code"], "strategy_rebalance")

    def test_month_start_context_does_not_replace_signal(self):
        actions = self.actions()
        actions["sleeve_rebalance_today"] = True
        self.assertEqual(
            [r["code"] for r in _action_reasons("X", "buy", actions)],
            ["strategy_signal", "mix_rebalance", "target_difference"],
        )

    def test_engine_trade_keeps_date_and_direction(self):
        actions = self.actions()
        actions["slots"]["a"]["engine_trades"] = [
            {"ticker": "Y", "side": "sell", "date": "2026-09-01", "reason": "리밸런싱"}
        ]
        reasons = _action_reasons("Y", "sell", actions)
        self.assertEqual(reasons[0]["code"], "engine_trade")
        self.assertIn("2026-09-01", reasons[0]["label"])
        self.assertEqual(len(_action_reasons("Y", "buy", actions)), 1)

    def test_rendered_action_retains_order_quantity(self):
        rows = [
            {
                "ticker": "X",
                "price": 100,
                "trade_quantity": 3,
                "target_quantity": 5,
                "held_quantity": 2,
                "weight_pct": 50,
                "sources": ["a"],
            }
        ]
        item = _build_action_groups(rows, self.actions(), "2026-09-08", currency="USD")[0]["items"][0]
        self.assertEqual(item["quantity"], 3)
        self.assertEqual(item["date"], "2026-09-08")
        self.assertIn("A 진입", item["text"])
        self.assertIn("실제 보유 차이", item["text"])
