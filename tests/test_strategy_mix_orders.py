"""합성 액션의 Slack 발송과 기존 주문 스레드 연결을 외부 접속 없이 검증한다."""

import json
from datetime import datetime
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

from services.broker_api_service import BrokerApiError, OrderUncertainError
from utils import slack_mock_orders as slack
from utils import strategy_mix_orders as mix
from utils import strategy_trade_orders as order_store
from utils import strategy_trade_threads as threads


class MixOrderTest(TestCase):
    def setUp(self):
        self.request = {
            "env": "live",
            "account_id": "a1",
            "ticker": "ETF",
            "side": "buy",
            "mix_action_key": "action",
            "label": "종목",
            "quantity": 36,
            "round": None,
            "rebuy": False,
        }
        self.item = {
            "key": "action",
            "ticker": "ETF",
            "side": "buy",
            "quantity": 36,
            "title": "매수",
            "text": "종목 36주",
        }
        self.group = {"section": "today", "title": "장중", "items": [self.item]}
        self.positions = {
            "account_id": "a1",
            "country": "kor",
            "account": {"holdings": {"ETF": {"quantity": 100}}},
            "holdings": [{"ticker": "ETF", "name": "종목"}],
            "actions": {"groups": [self.group]},
        }
        self.target = {"request": self.request, "thread_ts": "root", "message_ts": "reply", "channel": "test"}
        self.signal = {"thread_ts": "root", "action_key": "action", "side": "buy", "state": "sent"}
        self.db = Mock()
        self.db.find_one.return_value = self.signal
        self.db.update_one.return_value = SimpleNamespace(matched_count=1)
        self.orders = Mock()
        self.orders.find_one.return_value = None
        self.balance = {"holdings": [{"ticker": "ETF", "quantity": 100}], "cash": 50000}
        patches = [
            patch(
                "utils.db_manager.get_db_connection", side_effect=AssertionError("테스트에서는 DB 연결을 열지 않는다.")
            ),
            patch.object(slack, "_account_line", return_value="계좌 a1"),
            patch.object(mix, "_signals", return_value=self.db),
            patch.object(mix, "_orders", return_value=self.orders),
            patch.object(mix, "can_withdraw_signal", return_value=True),
            patch.object(mix, "mix_positions", return_value=self.positions),
            patch.object(mix, "account_env", return_value="live"),
            patch.object(mix, "_require_order_allowed"),
            patch.object(mix, "account_country", return_value="kor"),
            patch.object(order_store, "is_instrument_trading_time", return_value=True),
            patch.object(order_store, "_order_venue", return_value="KRX"),
            patch.object(mix, "fetch_broker_balance", return_value=self.balance),
            patch.object(mix, "list_day_orders", return_value=[]),
            patch.object(mix, "resolve_ticker_meta", return_value={"is_etf": True, "name": "종목"}),
            patch.object(mix, "resolve_order_quote", return_value=("KRX", "매도 1호가", 100, None)),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_send_uses_common_order_button_without_submitting(self):
        self.db.find_one.return_value = None
        with (
            patch.object(mix, "get_account_settings", return_value={"name": "계좌"}),
            patch.object(mix, "get_slack_channel", return_value="test"),
            patch.object(mix, "send_thread_message", return_value="root") as main,
            patch.object(slack, "send_thread_message", return_value="reply") as reply,
            patch.object(mix, "register_signal_thread") as register,
            patch.object(mix, "submit_order_record") as submit,
        ):
            result = mix.send_mix_order_signal.__wrapped__("a1", "action", "buy", 36)
        self.assertFalse(result["already_sent"])
        self.assertIn("합성 오늘의 액션", main.call_args.args[0])
        self.assertEqual(reply.call_args.args[2], "root")
        self.assertEqual(register.call_args.args[2][0]["request"]["mix_action_key"], "action")
        submit.assert_not_called()

    def test_duplicate_and_unconfirmed_signal_do_not_send(self):
        with patch.object(mix, "send_thread_message") as send:
            self.assertTrue(mix.send_mix_order_signal.__wrapped__("a1", "action", "buy", 36)["already_sent"])
            self.signal["state"] = "sending"
            with self.assertRaisesRegex(BrokerApiError, "발송 결과"):
                mix.send_mix_order_signal.__wrapped__("a1", "action", "buy", 36)
        send.assert_not_called()

    def test_withdraw_reuses_slack_path_for_existing_signal(self):
        self.signal["name"] = "종목"
        messages = Mock()
        messages.find_one.return_value = {}
        messages.update_one.return_value = SimpleNamespace(matched_count=1)
        with (
            patch.object(threads, "_messages", return_value=messages),
            patch.object(threads, "_signal_thread_has_orders", return_value=False),
            patch.object(threads, "resume_withdrawal") as resume,
        ):
            mix.withdraw_mix_order_signal.__wrapped__("a1", "ETF", "root")
        claim = messages.update_one.call_args.args[1]["$set"]
        self.assertIn("종목 매수", claim["withdrawal_text"])
        self.assertNotIn("None호", claim["withdrawal_text"])
        resume.assert_called_once_with("root")
        self.db.delete_one.assert_called_once_with(
            {
                "_id": {"account_id": "a1", "ticker": "ETF"},
                "thread_ts": "root",
                "state": "sent",
            }
        )

    def test_stale_or_ordered_signal_cannot_be_withdrawn(self):
        with patch.object(mix, "withdraw_signal") as withdraw:
            self.db.find_one.return_value = None
            with self.assertRaisesRegex(BrokerApiError, "다른 신호"):
                mix.withdraw_mix_order_signal.__wrapped__("a1", "ETF", "old")
            withdraw.assert_not_called()
            self.db.find_one.return_value = {**self.signal, "name": "종목"}
            withdraw.return_value = False
            with self.assertRaisesRegex(BrokerApiError, "회수할 수 없습니다"):
                mix.withdraw_mix_order_signal.__wrapped__("a1", "ETF", "root")
        self.db.delete_one.assert_not_called()

    def test_screen_hides_withdraw_when_order_already_exists(self):
        self.db.find.return_value = [self.signal]
        with patch.object(mix, "can_withdraw_signal", return_value=False):
            self.assertEqual(
                mix.mix_order_signals("a1", [self.group]),
                {"action": {"state": "sent", "withdraw_key": None}},
            )

    def test_closed_session_blocks_send_and_previously_sent_order(self):
        with (
            patch.object(order_store, "is_instrument_trading_time", return_value=False),
            patch.object(mix, "send_thread_message") as send,
            patch.object(mix, "resolve_order_quote") as quote,
            patch.object(mix, "submit_order_record") as submit,
        ):
            with self.assertRaisesRegex(BrokerApiError, "거래 시간이 아닙니다"):
                mix.send_mix_order_signal.__wrapped__("a1", "action", "buy", 36)
            with self.assertRaisesRegex(BrokerApiError, "거래 시간이 아닙니다"):
                mix.resolve_mix_order(self.request, "root")
            with self.assertRaisesRegex(BrokerApiError, "거래 시간이 아닙니다"):
                mix.place_mix_order(self.request, self.target, {}, "view")
        send.assert_not_called()
        quote.assert_not_called()
        submit.assert_not_called()

    def test_us_extended_sessions_use_common_schedule(self):
        from utils.market_session import is_instrument_trading_time

        zone = ZoneInfo("America/New_York")
        self.positions["country"] = "us"
        for hour, minute in ((7, 0), (17, 0), (21, 0)):
            with (
                self.subTest(hour=hour),
                patch.object(mix, "account_country", return_value="us"),
                patch.object(mix, "_now", return_value=datetime(2026, 11, 17, hour, minute, tzinfo=zone)),
                patch("utils.trading_calendar.get_trading_days", return_value=[True]),
                patch.object(order_store, "is_instrument_trading_time", side_effect=is_instrument_trading_time),
                patch.object(mix, "resolve_us_exchange", return_value="OQ"),
            ):
                self.assertEqual(mix.resolve_mix_order(self.request, "root")["quantity"], 36)

    def test_submit_rechecks_venue_after_confirmation(self):
        with (
            patch.object(order_store, "_order_venue", return_value="KRX"),
            patch.object(order_store, "_orders") as records,
            patch.object(order_store, "place_limit_order") as place,
        ):
            with self.assertRaisesRegex(BrokerApiError, "거래소가 바뀌었습니다"):
                order_store.submit_order_record({"venue": "NXT", "strategy_label": "종목"}, "buy", 100)
        records.assert_not_called()
        place.assert_not_called()

    def test_tomorrows_action_can_be_ordered(self):
        self.group["section"] = "later"
        self.assertEqual(mix.resolve_mix_order(self.request, "root")["quantity"], 36)

    def test_common_slack_modal_uses_latest_quantity_and_terms(self):
        self.item["quantity"] = 16
        with patch.object(slack, "_update_view") as update:
            slack.fill_confirm_modal({"target": self.target, "view_id": "view"})
        view = update.call_args.args[1]
        summary = view["blocks"][0]["text"]["text"]
        self.assertIn("16주 매수", summary)
        self.assertIn("지정가 100원", summary)
        self.assertNotIn("1호가", summary)
        self.assertNotIn("신호 가격 비교 없음", summary)
        metadata = json.loads(view["private_metadata"])
        self.assertEqual(metadata["confirmed"], {"quantity": 16, "price": 100, "venue": "KRX"})
        self.assertEqual(metadata["confirmation_id"], "view")

    def test_recheck_and_one_use_confirmation(self):
        terms = {"quantity": 36, "price": 100, "venue": "KRX"}
        with patch.object(mix, "submit_order_record") as submit:
            self.item["quantity"] = 16
            with self.assertRaises(mix.MixOrderChangedError):
                mix.place_mix_order(self.request, self.target, terms, "view")
            self.item["quantity"] = 36
            mix.place_mix_order(self.request, self.target, terms, "view")
            self.assertEqual(submit.call_args.args[0]["quantity"], 36)
            self.db.update_one.return_value = SimpleNamespace(matched_count=0)
            with self.assertRaises(mix.MixOrderChangedError):
                mix.place_mix_order(self.request, self.target, terms, "view")
        self.assertEqual(submit.call_count, 1)

    def test_partial_fill_and_out_of_sync_balance_are_blocked(self):
        with patch.object(mix, "list_day_orders", return_value=[{"ticker": "ETF", "status": "partial"}]):
            with self.assertRaisesRegex(BrokerApiError, "미체결"):
                mix.resolve_mix_order(self.request, "root")
        self.balance["holdings"][0]["quantity"] = 120
        with self.assertRaisesRegex(BrokerApiError, "잔고 동기화"):
            mix.resolve_mix_order(self.request, "root")
        self.positions["account"]["holdings"]["ETF"]["quantity"] = 120
        self.item["quantity"] = 16
        self.assertEqual(mix.resolve_mix_order(self.request, "root")["quantity"], 16)

    def test_sell_uses_common_submission_and_cancel_controls(self):
        self.request["side"] = self.signal["side"] = self.item["side"] = "sell"
        terms = {"quantity": 36, "price": 100, "venue": "KRX"}
        target = {**self.target, "confirmed": terms, "confirmation_id": "view"}
        inserted = SimpleNamespace(inserted_id="record")
        store = Mock()
        store.insert_one.return_value = inserted
        store.find_one.return_value = {
            "_id": "record",
            "env": "live",
            "round": None,
            "created_at": order_store._now().isoformat(),
            "strategy_label": "종목",
            "side": "sell",
            "ticker": "ETF",
            "country": "kor",
            "quantity": 36,
            "price": 100,
            "filled_quantity": 0,
            "average_fill_price": 0,
            "status": "pending",
            "venue": "KRX",
        }
        with (
            patch.object(mix, "resolve_order_quote", return_value=("KRX", "매수 1호가", 100, None)),
            patch.object(order_store, "_orders", return_value=store),
            patch.object(order_store, "place_limit_order", return_value={"order_no": "123", "message": ""}) as broker,
            patch.object(slack, "_update_message") as update,
            patch.object(slack, "send_thread_message") as send,
            patch.object(slack, "_sync_thread"),
        ):
            slack._place(target)
        broker.assert_called_once_with("a1", "sell", "ETF", 36, 100, "KRX")
        self.assertEqual(store.insert_one.call_args.args[0]["mix_action_key"], "action")
        self.assertIn("주문 접수", send.call_args.args[0])
        # 접수 뒤 메시지에는 공용 주문 조작 버튼(호가 정정·주문 취소)이 남는다.
        actions = [element["action_id"] for element in update.call_args.args[1][-1]["elements"]]
        self.assertEqual(actions, [slack.MODIFY_ACTION_ID, slack.CANCEL_ACTION_ID])

    def test_indeterminate_record_remains_active_after_refresh(self):
        self.orders.find.return_value = [
            {
                "_id": "uncertain",
                "account_id": "a1",
                "country": "kor",
                "mix_action_key": "action",
                "status": "submitting",
                "order_no": None,
                "created_at": "2025-01-03T10:00:00+09:00",
            }
        ]
        with (
            patch.object(order_store, "_orders", return_value=self.orders),
            patch.object(order_store, "list_day_orders", return_value=[]),
        ):
            order_store.refresh_account_orders("a1")
        self.orders.update_one.assert_not_called()

    def test_indeterminate_submission_keeps_button_off(self):
        target = {**self.target, "confirmed": {}, "confirmation_id": "view"}
        with (
            patch.object(mix, "place_mix_order", side_effect=OrderUncertainError("접수 불명")),
            patch.object(slack, "_update_message") as update,
            patch.object(slack, "send_thread_message") as send,
        ):
            slack._place(target)
        self.assertIn("접수 여부", send.call_args.args[0])
        self.assertTrue(all(block["type"] != "actions" for block in update.call_args.args[1]))
