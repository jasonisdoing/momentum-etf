"""사용자가 요청한 수동 Slack 신호 생명주기 회귀 검증. 외부 전송·주문·DB는 사용하지 않는다."""

from copy import deepcopy
from datetime import datetime
from threading import Event, Lock, Thread
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

from pymongo.errors import DuplicateKeyError

from utils import slack_mock_orders as slack
from utils import strategy_trade_notify as notify
from utils import strategy_trade_orders as orders
from utils import strategy_trade_threads as threads
from utils.strategy_trade_ledger import _day_trades

_REAL_SEND_ORDER_MESSAGES = slack.send_signal_order_messages
_REAL_CLOSE_THREADS = slack.close_session_threads
_REAL_SEND_THREAD_MESSAGE = threads.send_thread_message


class MemoryCollection:
    """테스트에서 실제 선점 경합을 만드는 최소 원자 저장소."""

    def __init__(self):
        self.docs = []
        self.lock = Lock()

    def matches(self, doc, query):
        for key, expected in query.items():
            if key == "$nor":
                if any(self.matches(doc, item) for item in expected):
                    return False
                continue
            value = doc
            for part in key.split("."):
                value = value.get(part) if isinstance(value, dict) else None
            if isinstance(expected, dict):
                for op, target in expected.items():
                    if op == "$ne" and value == target:
                        return False
                    if op == "$exists" and (value is not None) != target:
                        return False
                    if op == "$gt" and (value is None or value <= target):
                        return False
                    if op == "$lt" and (value is None or value >= target):
                        return False
            elif value != expected:
                return False
        return True

    def find(self, query):
        with self.lock:
            return deepcopy([d for d in self.docs if self.matches(d, query)])

    def find_one(self, query, projection=None):
        result = self.find(query)
        return result[0] if result else None

    def insert_one(self, doc):
        with self.lock:
            if "_id" in doc and any(d.get("_id") == doc["_id"] for d in self.docs):
                raise DuplicateKeyError("중복")
            self.docs.append(deepcopy(doc))

    def delete_one(self, query):
        with self.lock:
            index = next((i for i, d in enumerate(self.docs) if self.matches(d, query)), None)
            if index is None:
                return SimpleNamespace(deleted_count=0)
            del self.docs[index]
            return SimpleNamespace(deleted_count=1)

    def update_one(self, query, update):
        with self.lock:
            for doc in self.docs:
                if self.matches(doc, query):
                    doc.update(deepcopy(update.get("$set", {})))
                    for key in update.get("$unset", {}):
                        doc.pop(key, None)
                    return SimpleNamespace(matched_count=1)
        return SimpleNamespace(matched_count=0)


class ManualSignalsTest(TestCase):
    def setUp(self):
        self.config = MemoryCollection()
        self.messages = MemoryCollection()
        self.orders = MemoryCollection()
        self.sent = {}
        self.strategy = {
            "strategy_id": "s1",
            "account_id": "a1",
            "account_name": "계좌",
            "pool": "test",
            "ticker": "ETF",
            "label": "종목",
            "country": "kor",
            "instrument_kind": "etf",
            "arbitrary_lots": [],
            "config": {"trigger_pct": 8},
            "index": {"close": 100, "prev_close": 110, "name": "지수"},
            "status": {"held_count": 0, "last_buy_price": None, "first_entry_time": "09:20"},
            "manual_orders": [],
            "fill_events": [],
            "rounds": [
                {
                    "round": 1,
                    "held": False,
                    "is_next": True,
                    "ticker": "ETF",
                    "name": "종목",
                    "buy_limit": None,
                    "buy_index": None,
                    "sell_limit": None,
                    "close": 100,
                    "quantity": 10,
                    "buy_reached": True,
                    "sell_reached": False,
                    "close_entry": True,
                    "rebuy_after_sell": False,
                    "cash_short": False,
                    "budget_short": False,
                    "fill_price": None,
                }
            ],
        }
        self.payload = {"account_id": "a1", "strategy_id": "s1", "side": "buy", "round": 1}
        patches = [
            patch("utils.db_manager.get_db_connection", side_effect=AssertionError("외부 DB 금지")),
            patch("utils.strategy_trade_signal_lock.get_db_connection", return_value={"system_config": self.config}),
            patch.object(threads, "_messages", return_value=self.messages),
            patch.object(orders, "_orders", return_value=self.orders),
            patch.object(orders, "_order_strategy", side_effect=lambda *a, **k: deepcopy(self.strategy)),
            patch.object(
                notify,
                "load_settings",
                return_value={
                    "strategies": {"s1": {"account_id": "a1", "pool": "test"}},
                    "accounts": {"a1": {"slack_enabled": True}},
                },
            ),
            patch.object(notify, "_load_sent_log", side_effect=lambda: deepcopy(self.sent)),
            patch.object(notify, "_save_sent_log", side_effect=self.save),
            patch.object(notify, "build_message", return_value="신호"),
            patch.object(orders, "find_duplicate_order", return_value=None),
            patch.object(orders, "is_instrument_trading_time", return_value=True),
            patch.object(orders, "active_order_keys", return_value=set()),
            patch.object(notify, "refresh_all_ledgers", return_value={}),
            patch.object(
                notify,
                "load_views_where",
                side_effect=lambda *a, **k: {"strategies": [deepcopy(self.strategy)], "errors": {}},
            ),
            patch.object(notify, "_can_notify", return_value=True),
            patch.object(notify, "_session_closed", return_value=False),
            patch.object(
                threads,
                "open_signal_threads",
                side_effect=lambda roots: self.messages.find(
                    {"signal_used": {"$ne": True}, "withdrawal_done": {"$ne": True}}
                ),
            ),
            patch.object(slack, "close_session_threads", return_value=0),
            patch.object(slack, "send_signal_order_messages", return_value=1),
            patch.object(orders, "place_limit_order", side_effect=AssertionError("실제 주문 금지")),
            patch.object(threads, "send_slack_message_v2", side_effect=AssertionError("실제 Slack 금지")),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.sender = Mock(side_effect=self.send)
        for p in (
            patch.object(threads, "send_thread_message", self.sender),
            patch.object(notify, "_send_live_message", self.sender),
        ):
            p.start()
            self.addCleanup(p.stop)

    def save(self, value):
        self.sent = deepcopy(value)

    def send(self, *args, **kwargs):
        ts = str(len(self.messages.docs) + 1)
        self.messages.insert_one({"thread_ts": ts, "ts": ts, "channel": "test"})
        return ts

    def manual(self):
        preview = notify.preview_manual_signal(self.payload)
        return notify.send_manual_signal(preview)

    def test_manual_then_auto_and_repeated_manual(self):
        self.assertTrue(self.manual()["sent"])
        self.assertTrue(self.manual()["duplicate"])
        notify.notify_strategy_trade()
        self.assertEqual(self.sender.call_count, 1)

    def test_auto_then_manual(self):
        notify.notify_strategy_trade()
        self.assertTrue(self.manual()["duplicate"])
        self.assertEqual(self.sender.call_count, 1)

    def test_simultaneous_manual_auto_only_one_send(self):
        entered, release = Event(), Event()
        failures = []
        original = self.sender.side_effect

        def wait_send(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise AssertionError("대기 시간 초과")
            return original(*args, **kwargs)

        self.sender.side_effect = wait_send

        def worker():
            try:
                notify.notify_strategy_trade()
            except Exception as exc:
                failures.append(exc)

        insert = self.config.insert_one

        def release_on_contention(doc):
            try:
                insert(doc)
            except DuplicateKeyError:
                release.set()
                raise

        worker_thread = Thread(target=worker)
        worker_thread.start()
        try:
            self.assertTrue(entered.wait(5))
            with patch.object(self.config, "insert_one", side_effect=release_on_contention) as attempted:
                self.assertTrue(self.manual()["duplicate"])
            self.assertGreaterEqual(attempted.call_count, 2)
        finally:
            release.set()
            worker_thread.join(5)
        self.assertFalse(failures)
        self.assertEqual(self.sender.call_count, 1)

    def test_batch_preparation_does_not_block_manual_send(self):
        preparing, release = Event(), Event()
        failures = []

        def refresh():
            preparing.set()
            if not release.wait(5):
                raise AssertionError("대기 시간 초과")
            return {}

        def worker():
            try:
                notify.notify_strategy_trade()
            except Exception as exc:
                failures.append(exc)

        with patch.object(notify, "refresh_all_ledgers", side_effect=refresh):
            batch = Thread(target=worker)
            batch.start()
            try:
                self.assertTrue(preparing.wait(5))
                self.assertEqual(self.config.docs, [])
                self.assertTrue(self.manual()["sent"])
            finally:
                release.set()
                batch.join(5)
        self.assertFalse(failures)
        self.assertEqual(self.sender.call_count, 1)

    def test_lock_timeout_does_not_retry_sending(self):
        from utils import strategy_trade_signal_lock as locking

        with locking.signal_lock():
            # 대기 한도를 넘긴 시각을 바로 돌려줘 실제로 기다리지 않는다.
            with patch.object(locking, "monotonic", side_effect=[0, locking._LOCK_WAIT_SECONDS + 1]):
                with self.assertRaisesRegex(ValueError, "다른 신호 작업"):
                    self.manual()
        self.sender.assert_not_called()

    def test_next_day_price_return_and_settings_keep_manual(self):
        self.manual()
        self.strategy["index"]["close"] = 200
        self.strategy["rounds"][0]["buy_reached"] = False
        tomorrow = datetime(2030, 1, 2, 20, tzinfo=notify.KST)
        with patch.object(notify, "_session_closed", return_value=True):
            self.assertEqual(notify._settle_unused_signals([self.strategy], self.sent, tomorrow, {"s1"}), (0, 0))
            self.assertFalse(notify._close_sessions([self.strategy], self.sent, tomorrow, {"s1"}))
            self.assertFalse(notify._drop_closed_session_signals(self.sent, {"s1"}))
        self.assertFalse(threads.withdraw_signal("1", "settings_changed", "간격 변경"))
        self.assertEqual(len(self.sent), 1)

    def test_direct_withdraw_prevents_old_message_order(self):
        self.manual()
        with patch.object(threads, "resume_withdrawal") as resume:
            result = threads.withdraw_signal_by_key("s1:1-buy")
        self.assertEqual(result["records_deleted"], 1)
        resume.assert_called_once_with("1")
        self.assertFalse(self.sent)
        self.assertFalse(threads.reserve_signal_for_order("1", "1"))
        self.assertTrue(self.manual()["sent"])

    def test_filled_old_message_blocked_and_new_cycle_allowed(self):
        self.manual()
        self.orders.insert_one(
            {
                "round": 1,
                "side": "buy",
                "slack": {"thread_ts": "1", "message_ts": "1"},
                "status": "filled",
                "filled_quantity": 10,
            }
        )
        self.assertFalse(threads.reserve_signal_for_order("1", "1"))
        self.assertTrue(self.messages.find_one({"thread_ts": "1"})["signal_completed"])
        self.assertTrue(self.manual()["sent"])
        self.assertEqual(self.sender.call_count, 2)
        self.assertFalse(threads.reserve_signal_for_order("1", "1"))
        self.assertTrue(threads.reserve_signal_for_order("2", "2"))
        self.assertFalse(threads.reserve_signal_for_order("2", "2"))

    def test_partial_fill_not_complete(self):
        self.manual()
        self.orders.insert_one(
            {
                "round": 1,
                "side": "buy",
                "slack": {"thread_ts": "1", "message_ts": "1"},
                "status": "pending",
                "filled_quantity": 3,
            }
        )
        self.assertFalse(notify._entry_filled(self.sent["s1:1-buy"], 1, {1: 100}, []))

    def test_price_or_quantity_change_requires_confirmation(self):
        preview = notify.preview_manual_signal(self.payload)
        self.strategy["rounds"][0]["quantity"] = 9
        with self.assertRaisesRegex(ValueError, "변경"):
            notify.send_manual_signal(preview)
        self.sender.assert_not_called()

    def test_ambiguous_send_not_retried(self):
        self.sender.side_effect = RuntimeError("전송 결과 불명")
        with self.assertRaises(RuntimeError):
            self.manual()
        self.assertTrue(self.manual()["duplicate"])
        self.assertEqual(self.sender.call_count, 1)

    def test_direct_trade_cycle_blocks_old_signal_and_allows_new(self):
        self.manual()
        self.strategy["fill_events"] = [
            {"cursor": ["2030-01-02", 1], "buy": True, "quantity": 10},
            {"cursor": ["2030-01-02", 2], "buy": False, "quantity": 10},
        ]
        with self.assertRaises(orders.RoundNotOrderableError):
            orders._require_current_signal("1", "a1", "test", "s1", 1)
        self.assertFalse(threads.reserve_signal_for_order("1", "1"))
        self.assertTrue(self.manual()["sent"])
        self.assertEqual(self.sender.call_count, 2)

    def test_closed_session_blocks_preview_send_and_existing_slack_order(self):
        preview = notify.preview_manual_signal(self.payload)
        with (
            patch.object(orders, "is_instrument_trading_time", return_value=False),
            patch.object(orders, "fetch_best_quote") as quote,
        ):
            with self.assertRaisesRegex(orders.BrokerApiError, "거래 시간이 아닙니다"):
                notify.preview_manual_signal(self.payload)
            with self.assertRaisesRegex(orders.BrokerApiError, "거래 시간이 아닙니다"):
                notify.send_manual_signal(preview)
            with self.assertRaisesRegex(orders.BrokerApiError, "거래 시간이 아닙니다"):
                orders.resolve_order("a1", "test", "s1", "buy", 1, fresh=False)
        self.sender.assert_not_called()
        quote.assert_not_called()
        self.assertFalse(self.sent)

    def test_manual_sell_is_single_sell_and_only_last_round(self):
        self.strategy["status"]["held_count"] = 1
        self.strategy["rounds"][0].update(
            held=True,
            is_next=False,
            fill_price=100,
            sell_limit=108,
            sell_index=108,
            profit_pct=0,
            rebuy_after_sell=True,
        )
        self.payload.update(side="sell")
        _, trigger, _ = notify._manual_signal(self.payload, fresh=False)
        self.assertEqual(trigger["action"], "sell")
        self.assertFalse(trigger["rebuy"])
        self.assertTrue(self.manual()["sent"])
        self.strategy["status"]["held_count"] = 2
        with self.assertRaises(orders.RoundNotOrderableError):
            notify.preview_manual_signal(self.payload)

    def test_different_account_or_side_can_have_separate_signal(self):
        self.manual()
        entry = self.sent["s1:1-buy"]
        self.assertIsNone(notify._matching_signal(self.sent, {**entry, "strategy_id": "s2", "account_id": "a2"}))
        self.assertIsNone(notify._matching_signal(self.sent, {**entry, "action": "sell"}))
        self.assertIsNotNone(notify._matching_signal(self.sent, {**entry, "strategy_id": "s2"}))

    def test_order_record_source_flows_to_ledger(self):
        self.manual()
        base = orders._record_base(
            "live",
            "a1",
            "test",
            "s1",
            1,
            {
                "strategy_label": "종목",
                "ticker": "ETF",
                "quantity": 10,
                "venue": "KRX",
                "country": "kor",
                "instrument_kind": "etf",
            },
            {"thread_ts": "1"},
        )
        self.assertEqual(orders._fill_source(base), "manual_slack")
        self.assertEqual(orders._fill_source({"round": 1}), "slack")
        # 회차 없는 주문은 API 주문 모달, 자동 주문 표시가 있으면 자동이다.
        self.assertEqual(orders._fill_source({"round": None}), "api")
        self.assertEqual(orders._fill_source({"round": 1, "slack": {"auto": True}}), "auto")

    def test_manual_message_uses_order_buttons_without_quote_or_order(self):
        _, trigger, _ = notify._manual_signal(self.payload, fresh=False)
        with (
            patch.object(slack, "find_duplicate_order", return_value=None),
            patch.object(slack, "account_env", return_value="live"),
            patch.object(slack, "send_thread_message", return_value="reply") as send,
            patch.object(slack, "register_signal_thread") as register,
            patch.object(slack, "get_slack_channel", return_value="test"),
            patch.object(slack, "resolve_order", side_effect=AssertionError("발송 시 주문 호가 조회 금지")),
        ):
            self.assertEqual(_REAL_SEND_ORDER_MESSAGES(self.strategy, [trigger], "root"), 1)
        self.assertEqual(send.call_args.args[2], "root")
        self.assertEqual(register.call_args.args[2][0]["request"]["side"], "buy")

    def test_manual_reply_has_no_ttl_and_close_keeps_buttons(self):
        self.manual()
        with (
            patch.object(threads, "send_slack_message_v2", return_value="reply"),
            patch.object(threads, "get_slack_channel", return_value="test"),
        ):
            _REAL_SEND_THREAD_MESSAGE("답글", thread_ts="1")
        self.assertNotIn("expires_at", self.messages.find_one({"ts": "reply"}))
        with (
            patch.object(
                slack,
                "signal_threads",
                return_value=[{"thread_ts": "1", "messages": [{"request": {"strategy_id": "s1"}}]}],
            ),
            patch.object(slack, "_update_message") as update,
        ):
            self.assertEqual(_REAL_CLOSE_THREADS([self.strategy]), 0)
        update.assert_not_called()

    def test_screen_hides_withdraw_after_order_and_signal_after_fill(self):
        self.manual()
        result = notify.mark_sent_rounds({"strategies": [deepcopy(self.strategy)]})
        signal = result["strategies"][0]["rounds"][0]["slack_signals"][0]
        self.assertTrue(signal["manual"])
        self.assertTrue(signal["can_withdraw"])
        order = {"round": 1, "side": "buy", "slack": {"thread_ts": "1", "message_ts": "1"}}
        self.orders.insert_one({**order, "status": "pending", "filled_quantity": 4})
        result = notify.mark_sent_rounds({"strategies": [deepcopy(self.strategy)]})
        self.assertFalse(result["strategies"][0]["rounds"][0]["slack_signals"][0]["can_withdraw"])
        self.orders.update_one({"slack.message_ts": "1"}, {"$set": {"status": "filled", "filled_quantity": 10}})
        result = notify.mark_sent_rounds({"strategies": [deepcopy(self.strategy)]})
        self.assertEqual(result["strategies"][0]["rounds"][0]["slack_signals"], [])

    def test_filled_message_does_not_block_other_automatic_message(self):
        self.manual()
        root = self.messages.docs[0]
        root["signal_entries"][0]["origin"] = "auto"
        self.orders.insert_one(
            {
                "round": 1,
                "side": "buy",
                "slack": {"thread_ts": "1", "message_ts": "one"},
                "status": "filled",
                "filled_quantity": 10,
            }
        )
        self.assertFalse(threads.reserve_signal_for_order("1", "one"))
        self.assertTrue(threads.reserve_signal_for_order("1", "two"))
        self.assertNotIn("signal_completed", root)

    def test_fill_sources_distinguish_manual_slack(self):
        rows = [
            {
                "ticker": "ETF",
                "order_no": str(i),
                "side": "buy",
                "filled_quantity": 1,
                "average_fill_price": 100,
                "buy_sell": "buy",
            }
            for i in range(1, 4)
        ]
        sources = {("20300102", "2"): ("manual_slack", "regular"), ("20300102", "3"): ("slack", "regular")}
        with patch("utils.strategy_trade_ledger.is_buy", return_value=True):
            trades = _day_trades(rows, "ETF", datetime(2030, 1, 2).date(), sources)
        self.assertEqual([t.source for t in trades], ["manual", "manual_slack", "slack"])


if __name__ == "__main__":
    main()
