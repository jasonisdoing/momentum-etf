"""지수 구성종목 배치의 종목풀 미등록 알림 검증."""

import unittest
from unittest.mock import patch

from utils.index_constituents_loader import us_market_constituents
from utils.index_pool_alert import notify_unregistered_index_stocks


class IndexPoolAlertTests(unittest.TestCase):
    def test_us_alert_uses_sp500_top_300_and_full_nasdaq_100(self):
        sp500 = [{"ticker": f"T{i:03d}", "market_cap": i} for i in range(500)]
        targets = {
            "SP500": us_market_constituents("SP500", sp500),
            "NDX100": [{"ticker": "T010"}, {"ticker": "N1"}],
        }
        with (
            patch(
                "utils.index_pool_alert.load_ticker_pool_type_map",
                return_value={"T499": ["us_stock"], "T010": ["us_stock"]},
            ) as pools,
            patch("utils.index_pool_alert.app_link", return_value="화면"),
            patch("utils.index_pool_alert.send_slack_message_v2", return_value="sent") as send,
        ):
            count = notify_unregistered_index_stocks("us", targets)

        self.assertEqual(count, 300)
        pools.assert_called_once_with("us")
        message = send.call_args.args[0]
        self.assertIn("T200", message)
        self.assertIn("N1", message)
        self.assertNotIn("T199", message)
        self.assertNotIn("T499", message)

    def test_no_missing_stocks_sends_no_message(self):
        with (
            patch("utils.index_pool_alert.load_ticker_pool_type_map", return_value={"005930": ["kor_stock"]}),
            patch("utils.index_pool_alert.send_slack_message_v2") as send,
        ):
            count = notify_unregistered_index_stocks("kor", {"KOSPI200": [{"ticker": "005930"}]})

        self.assertEqual(count, 0)
        send.assert_not_called()

    def test_slack_failure_fails_batch(self):
        with (
            patch("utils.index_pool_alert.load_ticker_pool_type_map", return_value={}),
            patch("utils.index_pool_alert.app_link", return_value="화면"),
            patch("utils.index_pool_alert.send_slack_message_v2", return_value=None),
        ):
            with self.assertRaisesRegex(RuntimeError, "슬랙으로 보내지 못했습니다"):
                notify_unregistered_index_stocks("kor", {"KOSPI200": [{"ticker": "005930"}]})
