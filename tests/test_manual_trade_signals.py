"""사용자가 요청한 수동 Slack 신호 생명주기 회귀 검증. 외부 전송·주문·DB는 사용하지 않는다."""

from copy import deepcopy
from datetime import datetime
from threading import Event, Lock, Thread
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

import requests
from pymongo.errors import DuplicateKeyError

from services import meritz_api_service as meritz
from services.broker_api_service import BrokerApiError, OrderUncertainError
from utils import slack_mock_orders as slack
from utils import strategy_trade_notify as notify
from utils import strategy_trade_orders as orders
from utils import strategy_trade_threads as threads
from utils.strategy_trade_ledger import _day_trades

_REAL_SEND_ORDER_MESSAGES = slack.send_signal_order_messages
_REAL_LOAD_SENT_LOG = notify._load_sent_log
_REAL_SAVE_SENT_LOG = notify._save_sent_log
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
            if key == "$or":
                if not any(self.matches(doc, item) for item in expected):
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
                    if op == "$gte" and (value is None or value < target):
                        return False
                    if op == "$gt" and (value is None or value <= target):
                        return False
                    if op == "$lt" and (value is None or value >= target):
                        return False
                    if op == "$in" and value not in target:
                        return False
            elif value != expected:
                return False
        return True

    def find(self, query, projection=None):
        with self.lock:
            return deepcopy([d for d in self.docs if self.matches(d, query)])

    def find_one(self, query, projection=None):
        result = self.find(query)
        return result[0] if result else None

    def insert_one(self, doc):
        with self.lock:
            if "_id" in doc and any(d.get("_id") == doc["_id"] for d in self.docs):
                raise DuplicateKeyError("중복")
            from bson import ObjectId

            doc.setdefault("_id", ObjectId())
            self.docs.append(deepcopy(doc))
            return SimpleNamespace(inserted_id=doc["_id"])

    def delete_one(self, query):
        with self.lock:
            index = next((i for i, d in enumerate(self.docs) if self.matches(d, query)), None)
            if index is None:
                return SimpleNamespace(deleted_count=0)
            del self.docs[index]
            return SimpleNamespace(deleted_count=1)

    def update_one(self, query, update, upsert=False):
        with self.lock:
            for doc in self.docs:
                if self.matches(doc, query):
                    self.apply_update(doc, update)
                    return SimpleNamespace(matched_count=1)
            if upsert:
                doc = deepcopy(query)
                self.apply_update(doc, update)
                self.docs.append(doc)
        return SimpleNamespace(matched_count=0)

    @staticmethod
    def apply_update(doc, update):
        for operator, changes in update.items():
            for key, value in changes.items():
                node = doc
                parts = key.split(".")
                for part in parts[:-1]:
                    node = node.setdefault(part, {})
                if operator == "$set":
                    node[parts[-1]] = deepcopy(value)
                elif operator == "$unset":
                    node.pop(parts[-1], None)


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
                    "strategies": {"s1": {"account_id": "a1", "pool": "test", "ticker": "ETF"}},
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
        if isinstance(value, notify._SentLog):
            for key in value.saved:
                if key not in value:
                    self.sent.pop(key, None)
            self.sent.update(deepcopy(value))
            value.saved = deepcopy(dict(value))
        else:
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

        with locking.signal_lock("a1", "ETF"):
            # 대기 한도를 넘긴 시각을 바로 돌려줘 실제로 기다리지 않는다.
            with patch.object(locking, "monotonic", side_effect=[0, locking._LOCK_WAIT_SECONDS + 1]):
                with self.assertRaisesRegex(locking.SignalBusyError, "신호 작업"):
                    self.manual()
        self.sender.assert_not_called()

    def test_slow_slack_for_one_ticker_does_not_block_other_ticker_or_lose_records(self):
        entered, release = Event(), Event()
        failures = []
        original = self.sender.side_effect

        def slow_send(*args, **kwargs):
            if not entered.is_set():
                entered.set()
                if not release.wait(5):
                    raise AssertionError("대기 시간 초과")
            return original(*args, **kwargs)

        def worker():
            try:
                notify.notify_strategy_trade()
            except Exception as exc:
                failures.append(exc)

        self.sender.side_effect = slow_send
        with (
            patch("utils.db_manager.get_db_connection", return_value={"system_config": self.config}),
            patch.object(notify, "_load_sent_log", _REAL_LOAD_SENT_LOG),
            patch.object(notify, "_save_sent_log", _REAL_SAVE_SENT_LOG),
        ):
            batch = Thread(target=worker)
            batch.start()
            try:
                self.assertTrue(entered.wait(5))
                self.strategy = {**deepcopy(self.strategy), "strategy_id": "s2", "ticker": "ETF2"}
                self.strategy["rounds"][0]["ticker"] = "ETF2"
                notify.load_settings.return_value["strategies"]["s2"] = {
                    "account_id": "a1",
                    "pool": "test",
                    "ticker": "ETF2",
                }
                self.payload["strategy_id"] = "s2"
                self.assertTrue(self.manual()["sent"])
            finally:
                release.set()
                batch.join(5)
            self.assertFalse(batch.is_alive())
            self.assertFalse(failures)
            sent = _REAL_LOAD_SENT_LOG()
            self.assertEqual(set(sent), {"s1:1-buy", "s2:1-buy"})
            self.assertTrue(all(entry["signal_ts"] for entry in sent.values()))
        self.assertEqual(self.sender.call_count, 2)

    def test_batch_skips_busy_ticker_and_continues_with_other_ticker(self):
        from utils import strategy_trade_signal_lock as locking

        second = {**deepcopy(self.strategy), "strategy_id": "s2", "ticker": "ETF2"}
        second["rounds"][0]["ticker"] = "ETF2"
        notify.load_settings.return_value["strategies"]["s2"] = {"account_id": "a1", "pool": "test", "ticker": "ETF2"}
        with (
            locking.signal_lock("a1", "ETF"),
            patch.object(
                notify, "load_views_where", return_value={"strategies": [self.strategy, second], "errors": {}}
            ),
        ):
            result = notify.notify_strategy_trade()
        self.assertTrue(result["sent"])
        self.assertIn("다음 배치", result["reason"])
        self.assertEqual([entry["strategy_id"] for entry in result["triggers"]], ["s2"])
        self.assertEqual(self.sender.call_count, 1)
        previous_second = self.sent["s2:1-buy"]["signal_ts"]
        with patch.object(
            notify, "load_views_where", return_value={"strategies": [self.strategy, second], "errors": {}}
        ):
            self.assertTrue(notify.notify_strategy_trade()["sent"])
        self.assertEqual(set(self.sent), {"s1:1-buy", "s2:1-buy"})
        self.assertEqual(self.sent["s2:1-buy"]["signal_ts"], previous_second)
        self.assertEqual(self.sender.call_count, 2)

    def test_slow_withdrawal_does_not_block_other_ticker_or_delete_its_signal(self):
        entered, release = Event(), Event()
        failures = []

        def slow_withdrawal(*args):
            entered.set()
            if not release.wait(5):
                raise AssertionError("대기 시간 초과")

        def worker():
            try:
                threads.withdraw_signal_by_key("s1:1-buy")
            except Exception as exc:
                failures.append(exc)

        with (
            patch("utils.db_manager.get_db_connection", return_value={"system_config": self.config}),
            patch.object(notify, "_load_sent_log", _REAL_LOAD_SENT_LOG),
            patch.object(notify, "_save_sent_log", _REAL_SAVE_SENT_LOG),
            patch.object(threads, "resume_withdrawal", side_effect=slow_withdrawal),
        ):
            self.manual()
            withdrawal = Thread(target=worker)
            withdrawal.start()
            try:
                self.assertTrue(entered.wait(5))
                self.strategy = {**deepcopy(self.strategy), "strategy_id": "s2", "ticker": "ETF2"}
                self.strategy["rounds"][0]["ticker"] = "ETF2"
                notify.load_settings.return_value["strategies"]["s2"] = {
                    "account_id": "a1",
                    "pool": "test",
                    "ticker": "ETF2",
                }
                self.payload["strategy_id"] = "s2"
                self.assertTrue(self.manual()["sent"])
            finally:
                release.set()
                withdrawal.join(5)
            self.assertFalse(withdrawal.is_alive())
            self.assertFalse(failures)
            self.assertEqual(set(_REAL_LOAD_SENT_LOG()), {"s2:1-buy"})
            self.assertFalse(threads.reserve_signal_for_order("1", "1"))

    def test_atomic_log_save_preserves_other_ticker_changes_and_deletions(self):
        with patch("utils.db_manager.get_db_connection", return_value={"system_config": self.config}):
            first, second = notify._SentLog({}), notify._SentLog({})
            first["s1:1-buy"] = {"ticker": "ETF"}
            second["s2:1-buy"] = {"ticker": "ETF2"}
            _REAL_SAVE_SENT_LOG(first)
            _REAL_SAVE_SENT_LOG(second)
            deleting, updating = _REAL_LOAD_SENT_LOG(), _REAL_LOAD_SENT_LOG()
            del deleting["s1:1-buy"]
            updating["s2:1-buy"]["signal_ts"] = "new"
            _REAL_SAVE_SENT_LOG(updating)
            _REAL_SAVE_SENT_LOG(deleting)
            self.assertEqual(dict(_REAL_LOAD_SENT_LOG()), {"s2:1-buy": {"ticker": "ETF2", "signal_ts": "new"}})

    def test_concurrent_tickers_cannot_exceed_account_order_limit(self):
        from services.broker_api_service import BrokerApiError

        entered, release = Event(), Event()
        failures = []
        base = {
            "account_id": "a1",
            "env": "live",
            "ticker": "ETF",
            "quantity": 1,
            "strategy_label": "종목",
            "venue": "KRX",
        }

        def slow_broker(*args):
            entered.set()
            if not release.wait(5):
                raise AssertionError("대기 시간 초과")
            return {"order_no": "1", "message": ""}

        def worker():
            try:
                orders.submit_order_record(base, "buy", 100)
            except Exception as exc:
                failures.append(exc)

        with (
            patch.object(
                orders,
                "load_account_docs",
                return_value=[
                    {
                        "account_id": "a1",
                        "api_order_enabled": True,
                        "api_order_max_count": 1,
                        "api_order_max_amount": 1000,
                    }
                ],
            ),
            patch.object(orders, "account_country", return_value="kor"),
            patch.object(orders, "krw_rate", return_value=1),
            patch.object(orders, "_order_venue", return_value="KRX"),
            patch.object(orders, "serialize_order", side_effect=lambda value: value),
            patch.object(orders, "place_limit_order", side_effect=slow_broker) as broker,
        ):
            first = Thread(target=worker)
            first.start()
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaisesRegex(BrokerApiError, "주문 건수 한도"):
                    orders.submit_order_record({**base, "ticker": "ETF2"}, "buy", 100)
                self.assertEqual(broker.call_count, 1)
            finally:
                release.set()
                first.join(5)
        self.assertFalse(first.is_alive())
        self.assertFalse(failures)
        self.assertEqual(len(self.orders.docs), 1)

    def test_live_reservation_renews_and_is_not_stolen_after_created_at_is_old(self):
        from datetime import UTC, timedelta

        from utils import strategy_trade_signal_lock as locking

        renewed = Event()
        update = self.config.update_one

        def observe_update(*args, **kwargs):
            result = update(*args, **kwargs)
            renewed.set()
            return result

        with (
            patch.object(locking, "_LOCK_RENEW_SECONDS", 0.01),
            patch.object(self.config, "update_one", side_effect=observe_update),
            locking.signal_lock("a1", "ETF"),
        ):
            key = locking._signal_key("a1", "ETF")
            update({"_id": key}, {"$set": {"created_at": datetime.now(UTC) - timedelta(seconds=121)}})
            self.assertTrue(renewed.wait(2))
            with self.assertRaises(locking.SignalBusyError):
                with locking.batch_signal_lock("a1", "ETF"):
                    self.fail("실행 중인 예약을 빼앗았습니다.")
        self.assertIsNone(self.config.find_one({"_id": key}))

    def test_next_day_price_return_and_settings_keep_manual(self):
        self.manual()
        self.strategy["index"]["close"] = 200
        self.strategy["rounds"][0]["buy_reached"] = False
        tomorrow = datetime(2030, 1, 2, 20, tzinfo=notify.KST)
        with patch.object(notify, "_session_closed", return_value=True):
            self.assertEqual(
                notify._settle_unused_signals([self.strategy], self.sent, tomorrow, {"s1"}, ("a1", "ETF")), (0, 0)
            )
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

    def test_order_finished_after_ledger_snapshot_does_not_resend_signal(self):
        notify.notify_strategy_trade()
        signal_ts = self.sent["s1:1-buy"]["signal_ts"]
        # 원장 계산 뒤 체결된 주문 — 원장은 아직 미보유라 같은 회차 트리거가 다시 나온다.
        self.orders.insert_one(
            {
                "strategy_id": "s1",
                "round": 1,
                "side": "buy",
                "slack": {"thread_ts": signal_ts, "message_ts": signal_ts},
                "status": "filled",
                "filled_quantity": 10,
                "updated_at": "2999-01-01T00:00:00+09:00",
            }
        )
        notify.notify_strategy_trade()
        self.assertEqual(self.sender.call_count, 1)

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


class MeritzOrderTransportTest(TestCase):
    """주문 전 조회 복구와 주문 전송의 중복 방지를 외부 호출 없이 검증한다."""

    def setUp(self):
        self.response = Mock(status_code=200)
        self.response.json.return_value = {"rsp_cd": "5766", "data": []}
        patches = {
            "token": patch.object(meritz, "_access_token", return_value="secret-token"),
            "mac": patch.object(meritz, "_mac_address", return_value="secret-mac"),
            "refresh": patch.object(meritz, "_issue_token", return_value="refreshed-token"),
            "get": patch.object(meritz.requests, "get", return_value=self.response),
            "post": patch.object(meritz.requests, "post", side_effect=AssertionError("실제 주문 금지")),
            "sleep": patch.object(meritz.time, "sleep"),
            "interval": patch.object(meritz, "_MIN_INTERVAL_SECONDS", 0),
            "logger": patch.object(meritz, "logger"),
            "db": patch("utils.db_manager.get_db_connection", side_effect=AssertionError("실제 DB 금지")),
        }
        self.mocks = {}
        for name, item in patches.items():
            self.mocks[name] = item.start()
            self.addCleanup(item.stop)

    def diagnostics(self):
        return "\n".join(call.args[0] % call.args[1:] for call in self.mocks["logger"].method_calls)

    def test_inquiry_recovers_after_connection_reset_and_timeout(self):
        self.mocks["get"].side_effect = [
            requests.ConnectionError(ConnectionResetError(54, "secret-token")),
            requests.ReadTimeout("secret-mac"),
            self.response,
        ]
        result = meritz._get("a1", meritz.HOLDINGS_PATH, {"private": "secret-query"})
        self.assertEqual(result["rsp_cd"], "5766")
        self.assertEqual(self.mocks["get"].call_count, 3)
        self.mocks["post"].assert_not_called()
        self.mocks["refresh"].assert_not_called()
        self.assertEqual([call.args[0] for call in self.mocks["sleep"].call_args_list], [0.5, 1.0])
        diagnostic = self.diagnostics()
        for expected in (
            meritz.HOLDINGS_PATH,
            "account=a1",
            "ConnectionResetError(errno=54)",
            "ReadTimeout",
            "outcome=recovered",
            "attempts=3",
            "elapsed_ms=",
            "request_id=",
        ):
            self.assertIn(expected, diagnostic)
        for secret in ("secret-token", "secret-mac", "secret-query", "authorization"):
            self.assertNotIn(secret, diagnostic)

    def test_exhausted_inquiry_never_submits_order(self):
        self.mocks["get"].side_effect = requests.ConnectionError(ConnectionResetError(54, "private-message"))
        with (
            patch.object(orders, "account_env", return_value="live"),
            patch.object(
                orders, "resolve_order", side_effect=lambda *args, **kwargs: meritz._get("a1", meritz.HOLDINGS_PATH, {})
            ),
            patch.object(orders, "submit_order_record") as submit,
        ):
            with self.assertRaises(orders.OrderPrecheckError) as caught:
                orders.place_order("a1", "test", "s1", "sell", 5, {"thread_ts": "root"})
        self.assertIn("주문은 전송하지 않았습니다", str(caught.exception))
        self.assertIn(meritz.HOLDINGS_PATH, str(caught.exception))
        self.assertIn("추적 ID", str(caught.exception))
        self.assertNotIn("private-message", str(caught.exception))
        self.assertEqual(self.mocks["get"].call_count, 3)
        submit.assert_not_called()
        self.mocks["post"].assert_not_called()
        self.assertIn("outcome=failed", self.diagnostics())

    def test_recovered_precheck_submits_exactly_once(self):
        self.response.json.return_value.update({"price": 100, "quantity": 4, "signal_price": 120})
        self.mocks["get"].side_effect = [requests.ReadTimeout(), self.response]
        with (
            patch.object(orders, "account_env", return_value="live"),
            patch.object(
                orders,
                "resolve_order",
                side_effect=lambda *args, **kwargs: meritz._get("a1", meritz.ORDER_HISTORY_PATH, {}),
            ),
            patch.object(orders, "_require_no_duplicate"),
            patch.object(orders, "_require_order_allowed"),
            patch.object(orders, "_record_base", return_value={}),
            patch.object(orders, "submit_order_record", return_value={"status": "pending"}) as submit,
        ):
            result = orders.place_order("a1", "test", "s1", "sell", 5, {"thread_ts": "root"})
        self.assertEqual(result["status"], "pending")
        self.assertEqual(self.mocks["get"].call_count, 2)
        submit.assert_called_once()

    def test_order_modify_cancel_posts_are_never_retried_on_network_errors(self):
        for path in (meritz.ORDER_PATH["buy"], meritz.ORDER_PATH["sell"], meritz.MODIFY_PATH, meritz.CANCEL_PATH):
            for failure in (requests.ConnectionError("secret-body"), requests.ReadTimeout("secret-body")):
                with self.subTest(path=path, error=type(failure).__name__):
                    self.mocks["post"].reset_mock()
                    self.mocks["post"].side_effect = failure
                    with self.assertRaises(OrderUncertainError) as caught:
                        meritz._request(
                            "a1", "POST", path, success=meritz._ORDER_CODES, body={"private": "secret-body"}
                        )
                    self.mocks["post"].assert_called_once()
                    self.assertIn("접수 여부", str(caught.exception))
                    self.assertNotIn("secret-body", str(caught.exception))
        self.mocks["get"].assert_not_called()
        self.mocks["refresh"].assert_not_called()
        self.mocks["sleep"].assert_not_called()

    def test_slack_reopens_only_known_unsent_order_not_uncertain_submission(self):
        target = {
            "request": {
                "env": "live",
                "account_id": "a1",
                "ticker": "ETF",
                "strategy_id": "s1",
                "pool": "test",
                "side": "sell",
                "round": 5,
                "label": "종목",
                "quantity": 4,
                "rebuy": False,
            },
            "channel": "test",
            "message_ts": "reply",
            "thread_ts": "root",
        }
        for failure, safe_to_retry in (
            (orders.OrderPrecheckError("주문 전 조회 실패. 주문은 전송하지 않았습니다."), True),
            (OrderUncertainError("접수 여부 불명"), False),
        ):
            with (
                self.subTest(failure=type(failure).__name__),
                patch.object(slack, "place_order", side_effect=failure),
                patch.object(slack, "_update_message"),
                patch.object(slack, "_note_blocks", return_value=[]),
                patch.object(slack, "_open_blocks", return_value=[]) as reopen,
                patch.object(slack, "send_thread_message") as send,
                patch.object(slack, "_release_reservation") as release,
                patch.object(slack, "logger"),
            ):
                self.assertIsNone(slack._place(target))
                if safe_to_retry:
                    release.assert_called_once_with(target)
                    reopen.assert_called_once()
                    self.assertIn("주문은 전송하지 않았습니다", send.call_args.args[0])
                else:
                    release.assert_not_called()
                    reopen.assert_not_called()
                    self.assertIn("접수 여부", send.call_args.args[0])
        self.mocks["post"].assert_not_called()

    def test_permanent_inquiry_errors_are_not_retried(self):
        for error in (requests.exceptions.SSLError(), requests.exceptions.InvalidURL(), requests.RequestException()):
            with self.subTest(error=type(error).__name__):
                self.mocks["get"].reset_mock()
                self.mocks["get"].side_effect = error
                with self.assertRaises(BrokerApiError):
                    meritz._get("a1", meritz.HOLDINGS_PATH, {})
                self.mocks["get"].assert_called_once()
        self.mocks["sleep"].assert_not_called()

    def test_api_rejection_is_not_retried(self):
        self.response.json.return_value = {"rsp_cd": "denied", "rsp_msg": "조회 거부"}
        with self.assertRaises(BrokerApiError):
            meritz._get("a1", meritz.HOLDINGS_PATH, {})
        self.mocks["get"].assert_called_once()
        self.mocks["refresh"].assert_not_called()
        self.mocks["sleep"].assert_not_called()

    def test_explicit_token_rejection_refreshes_only_once(self):
        self.response.status_code = 401
        self.response.json.return_value = {"rsp_cd": meritz._TOKEN_EXPIRED_CODE}
        with self.assertRaises(meritz.MeritzTokenExpiredError):
            meritz._get("a1", meritz.HOLDINGS_PATH, {})
        self.assertEqual(self.mocks["get"].call_count, 2)
        self.mocks["refresh"].assert_called_once_with("a1")
        self.mocks["sleep"].assert_not_called()
        self.assertIn("stage=authentication", self.diagnostics())

    def test_token_refresh_does_not_reset_network_retry_budget(self):
        token_error = Mock(status_code=401)
        token_error.json.return_value = {"rsp_cd": meritz._TOKEN_EXPIRED_CODE}
        self.mocks["get"].side_effect = [
            requests.ReadTimeout(),
            token_error,
            requests.ReadTimeout(),
            requests.ReadTimeout(),
        ]
        with self.assertRaises(BrokerApiError):
            meritz._get("a1", meritz.HOLDINGS_PATH, {})
        self.assertEqual(self.mocks["get"].call_count, 4)
        self.mocks["refresh"].assert_called_once()
        self.assertEqual(len(self.mocks["sleep"].call_args_list), 2)


if __name__ == "__main__":
    main()
