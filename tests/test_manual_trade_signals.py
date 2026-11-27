"""사고팔기 수동 주문과 신호 생명주기 회귀 검증. 외부 전송·주문·DB는 사용하지 않는다."""

import base64
import json
import time
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import date, datetime, timedelta
from io import StringIO
from threading import Event, Lock, Thread
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

import requests
from pymongo.errors import DuplicateKeyError

from scripts import slack_mock_order_notify as order_job
from services import meritz_api_service as meritz
from services.broker_api_service import BrokerApiError, OrderUncertainError
from utils import slack_mock_orders as slack
from utils import strategy_trade_ledger as ledger
from utils import strategy_trade_notify as notify
from utils import strategy_trade_orders as orders
from utils import strategy_trade_service as trade_service
from utils import strategy_trade_signal_lock as locking
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


class ManualOrdersTest(TestCase):
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
        preview = notify.preview_manual_order(self.payload)
        return notify.submit_manual_order(preview)

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

    def test_expired_manual_signal_is_hidden_and_retired_without_fill_confirmation(self):
        self.manual()
        self.orders.insert_one(
            {
                "strategy_id": "s1",
                "round": 1,
                "side": "buy",
                "status": "expired",
                "filled_quantity": 0,
                "slack": {"thread_ts": "1", "message_ts": "1"},
            }
        )
        self.assertEqual(
            notify.mark_sent_rounds({"strategies": [deepcopy(self.strategy)]})["strategies"][0]["rounds"][0][
                "slack_signals"
            ],
            [],
        )
        # 장 시작 전에도 이미 만료된 신호는 정리한다. 새 주문·체결 확인 메시지는 보내지 않는다.
        with patch.object(notify, "_can_notify", return_value=False):
            notify.notify_strategy_trade()
        self.assertFalse(self.sent)
        self.sender.assert_called_once()
        root = self.messages.find_one({"thread_ts": "1", "ts": "1"})
        self.assertTrue(root["signal_expired"])
        self.assertNotIn("signal_completed", root)
        self.assertEqual(len(root["signal_entries"]), 1)
        self.assertEqual(self.orders.docs[0]["status"], "expired")
        self.assertFalse(threads.reserve_signal_for_order("1", "1"))
        with self.assertRaises(orders.RoundNotOrderableError):
            orders._require_current_signal("1", "a1", "test", "s1", 1)

    def test_expired_signal_allows_fresh_manual_signal_before_batch_cleanup(self):
        self.manual()
        self.orders.insert_one(
            {
                "strategy_id": "s1",
                "round": 1,
                "side": "buy",
                "status": "expired",
                "filled_quantity": 0,
                "slack": {"thread_ts": "1", "message_ts": "1"},
            }
        )
        self.assertFalse(threads.reserve_signal_for_order("1", "1"))
        self.assertTrue(self.manual()["sent"])
        self.assertEqual(self.sent["s1:1-buy"]["signal_ts"], "2")
        self.assertTrue(self.messages.find_one({"thread_ts": "1"})["signal_expired"])
        self.assertTrue(threads.reserve_signal_for_order("2", "2"))
        self.assertFalse(threads.reserve_signal_for_order("1", "1"))

    def test_partially_filled_expired_sell_requires_current_remaining_quantity(self):
        self.strategy["status"]["held_count"] = 1
        self.strategy["rounds"][0].update(
            held=True, is_next=False, fill_price=100, sell_limit=108, sell_index=108, profit_pct=0
        )
        self.payload["side"] = "sell"
        self.manual()
        self.orders.insert_one(
            {
                "strategy_id": "s1",
                "round": 1,
                "side": "sell",
                "status": "expired",
                "filled_quantity": 4,
                "slack": {"thread_ts": "1", "message_ts": "1"},
            }
        )
        old_preview = notify.preview_manual_order(self.payload)
        self.strategy["rounds"][0]["quantity"] = 6
        with self.assertRaisesRegex(ValueError, "변경"):
            notify.submit_manual_order(old_preview)
        self.assertTrue(self.manual()["sent"])
        self.assertEqual(self.sent["s1:1-sell"]["quantity"], 6)
        self.assertEqual(self.sent["s1:1-sell"]["signal_ts"], "2")
        self.assertFalse(threads.reserve_signal_for_order("1", "1"))
        self.assertEqual(self.orders.docs[0]["filled_quantity"], 4)

    def test_inflight_pair_and_unsubmitted_manual_signals_are_not_retired(self):
        self.manual()
        self.assertEqual(notify.retire_expired_signals(self.sent, ("a1", "ETF")), 0)
        self.sent["s1:1-buy"]["rebuy"] = True
        self.orders.insert_one(
            {
                "strategy_id": "s1",
                "round": 1,
                "side": "sell",
                "status": "expired",
                "filled_quantity": 0,
                "slack": {"thread_ts": "1", "message_ts": "1"},
            }
        )
        self.orders.insert_one(
            {
                "strategy_id": "s1",
                "round": 1,
                "side": "buy",
                "status": "submitting",
                "filled_quantity": 0,
                "slack": {"thread_ts": "1", "message_ts": "1"},
            }
        )
        self.assertEqual(notify.retire_expired_signals(self.sent, ("a1", "ETF")), 0)
        self.assertIn("s1:1-buy", self.sent)

    def test_expiring_one_message_keeps_other_message_in_shared_thread_open(self):
        self.manual()
        self.messages.docs[0]["signal_entries"][0]["origin"] = "auto"
        other = {**deepcopy(self.sent["s1:1-buy"]), "round": 2, "action": "sell", "origin": "auto"}
        self.sent["s1:2-sell"] = other
        self.orders.insert_one(
            {
                "strategy_id": "s1",
                "round": 1,
                "side": "buy",
                "status": "expired",
                "filled_quantity": 0,
                "slack": {"thread_ts": "1", "message_ts": "expired"},
            }
        )
        self.assertEqual(notify.retire_expired_signals(self.sent, ("a1", "ETF")), 1)
        self.assertEqual(set(self.sent), {"s1:2-sell"})
        self.assertNotIn("signal_expired", self.messages.find_one({"thread_ts": "1"}))
        self.assertFalse(threads.reserve_signal_for_order("1", "expired"))
        self.assertTrue(threads.reserve_signal_for_order("1", "other"))

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
        preview = notify.preview_manual_order(self.payload)
        self.strategy["rounds"][0]["quantity"] = 9
        with self.assertRaisesRegex(ValueError, "변경"):
            notify.submit_manual_order(preview)
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
        preview = notify.preview_manual_order(self.payload)
        with (
            patch.object(orders, "is_instrument_trading_time", return_value=False),
            patch.object(orders, "fetch_best_quote") as quote,
        ):
            with self.assertRaisesRegex(orders.BrokerApiError, "거래 시간이 아닙니다"):
                notify.preview_manual_order(self.payload)
            with self.assertRaisesRegex(orders.BrokerApiError, "거래 시간이 아닙니다"):
                notify.submit_manual_order(preview)
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
        _, trigger, _ = notify._manual_order(self.payload, fresh=False)
        self.assertEqual(trigger["action"], "sell")
        self.assertFalse(trigger["rebuy"])
        self.assertTrue(self.manual()["sent"])
        self.strategy["status"]["held_count"] = 2
        with self.assertRaises(orders.RoundNotOrderableError):
            notify.preview_manual_order(self.payload)

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
        self.assertEqual(orders._fill_source(base), "manual_order")
        self.assertEqual(orders._fill_source({"round": 1}), "approved")
        # 회차 없는 주문은 API 주문 모달, 자동 주문 표시가 있으면 자동이다.
        self.assertEqual(orders._fill_source({"round": None}), "api")
        self.assertEqual(orders._fill_source({"round": 1, "slack": {"auto": True}}), "auto")

    def test_manual_message_submits_order_at_once_as_manual(self):
        _, trigger, _ = notify._manual_order(self.payload, fresh=False)
        with (
            patch.object(slack, "find_duplicate_order", return_value=None),
            patch.object(slack, "account_env", return_value="live"),
            patch.object(slack, "send_thread_message", return_value="reply") as send,
            patch.object(slack, "register_signal_thread") as register,
            patch.object(slack, "get_slack_channel", return_value="test"),
            patch.object(slack, "resolve_order", side_effect=AssertionError("발송 시 주문 호가 조회 금지")),
            patch.object(slack, "_submit_order") as submit,
        ):
            self.assertEqual(_REAL_SEND_ORDER_MESSAGES(self.strategy, [trigger], "root"), 1)
        self.assertEqual(send.call_args.args[2], "root")
        self.assertEqual(register.call_args.args[2][0]["request"]["side"], "buy")
        # 화면에서 확인한 수동 주문은 버튼 확인과 같은 경로로 바로 내고, 자동 주문 표시는 붙이지 않는다.
        target = submit.call_args.args[0]
        self.assertEqual((target["thread_ts"], target["message_ts"], target["kind"]), ("root", "reply", "order"))
        self.assertFalse(target["request"]["auto"])

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

    def test_fill_sources_distinguish_manual_order(self):
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
        sources = {("20300102", "2"): ("manual_order", "regular"), ("20300102", "3"): ("approved", "regular")}
        with patch("utils.strategy_trade_ledger.is_buy", return_value=True):
            trades = _day_trades(rows, "ETF", datetime(2030, 1, 2).date(), sources)
        self.assertEqual([t.source for t in trades], ["manual", "manual_order", "approved"])


class TradeLedgerRecoveryTest(TestCase):
    """회차 복구·설정 저장·주문 차단이 실제 보유를 추정값으로 바꾸지 않는지 검증한다."""

    def setUp(self):
        self.today = date(2026, 11, 24)
        self.cache = MemoryCollection()
        self.rows = {}
        timeline = [
            ("2026-10-21", True, 37, 16750),
            ("2026-10-22", True, 39, 16750),
            ("2026-10-27", True, 74, 15980),
            ("2026-11-09", False, 50, 16060),
            ("2026-11-12", True, 64, 15625),
            ("2026-11-13", False, 64, 15980),
            ("2026-11-18", True, 36, 15740),
            ("2026-11-20", False, 23, 15610),
        ]
        for index, (day, buy, quantity, price) in enumerate(timeline):
            self.rows[date.fromisoformat(day)] = [
                {
                    "ticker": "ETF",
                    "order_no": str(index + 1),
                    "side_label": "매수" if buy else "매도",
                    "filled_quantity": quantity,
                    "average_fill_price": price,
                }
            ]
        day = date(2026, 10, 21)
        while day < self.today:
            if day.weekday() < 5:
                self.cache.insert_one({"_id": f"a1:{day:%Y%m%d}", "rows": self.rows.get(day, [])})
            day += timedelta(days=1)
        self.sources = {
            ("20261113", "6"): (ledger.SOURCE_AUTO, ledger.REGULAR_SELL),
            ("20261120", "8"): (ledger.SOURCE_API, ledger.REGULAR_SELL),
        }
        self.history = ledger._History("a1", self.today, [], self.sources)
        self.collection = Mock()
        self.collection.find_one.return_value = {}
        self.current = {
            "s1": {
                "account_id": "a1",
                "ticker": "ETF",
                "pool": "test",
                "name": "종목",
                "trigger_pct": 5,
                "round_size": 10,
                "rounds": 10,
                "auto_order": False,
            }
        }
        patches = [
            patch("utils.db_manager.get_db_connection", side_effect=AssertionError("실제 DB 금지")),
            patch.object(ledger, "account_country", return_value="kor"),
            patch.object(ledger, "_cache", return_value=self.cache),
            patch.object(ledger, "_past_fills", side_effect=self.recent_fills),
            patch.object(ledger, "list_day_orders", side_effect=AssertionError("과거 추가 API 조회 금지")),
            patch.object(
                trade_service, "_trade_account_docs", return_value=[{"account_id": "a1", "country_code": "kor"}]
            ),
            patch.object(trade_service, "load_settings", return_value={"strategies": self.current}),
            patch.object(trade_service, "_eligible_pool_kinds", return_value=[]),
            patch.object(trade_service, "_ticker_options", return_value={}),
            patch.object(trade_service, "_settings_collection", return_value=self.collection),
            patch.object(trade_service, "_withdraw_changed_signals"),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def recent_fills(self, account, day):
        if day < self.today - timedelta(days=ledger.LOOKBACK_DAYS):
            raise AssertionError("최근 기간 이전은 저장 캐시만 사용해야 합니다.")
        return self.rows.get(day, [])

    def save(self, changes):
        row = {**self.current["s1"], "strategy_id": "s1", **changes}
        trade_service._save_account_trade("a1", True, [row], [])

    def test_amount_and_nonreducing_settings_save_without_unblocking_orders(self):
        for changes in ({"round_size": 50}, {"trigger_pct": 8}, {"auto_order": True}, {"rounds": 12}, {}):
            with self.subTest(changes=changes), patch.object(trade_service, "read_ledgers") as read:
                self.collection.update_one.reset_mock()
                self.save(changes)
                read.assert_not_called()
                self.collection.update_one.assert_called_once()
                saved = self.collection.update_one.call_args.args[1]["$set"]["strategies.s1"]
                self.assertEqual(saved["round_size"], changes.get("round_size", 10))
        with (
            patch.object(orders, "account_env", return_value="live"),
            patch.object(orders, "load_strategy_view", return_value={"error": "과거 체결 이력 부족"}),
            patch.object(orders, "submit_order_record") as submit,
        ):
            with self.assertRaises(orders.OrderPrecheckError):
                orders.place_order("a1", "test", "s1", "sell", 1, {"thread_ts": "root"})
        submit.assert_not_called()

    def test_reducing_rounds_requires_known_ledger_and_sufficient_rounds(self):
        with patch.object(trade_service, "read_ledgers", return_value={"ETF": ledger.LedgerError("이력 부족")}):
            with self.assertRaises(ValueError):
                self.save({"rounds": 5})
        self.collection.update_one.assert_not_called()
        known = SimpleNamespace(lots=[ledger.Lot(1, 100, "2026-10-21", ledger.SOURCE_MANUAL)] * 3)
        with patch.object(trade_service, "read_ledgers", return_value={"ETF": known}):
            with self.assertRaises(ValueError):
                self.save({"rounds": 2})
            self.collection.update_one.assert_not_called()
            self.save({"rounds": 5})
        self.collection.update_one.assert_called_once()

    def test_cached_recovery_matches_broker_quantity_and_preserves_old_prices(self):
        trades, closing = ledger._from_window(self.history, "ETF", 113)
        lots = ledger._replay(closing, [])
        self.assertEqual(sum(lot.shares for lot in lots), 113)
        self.assertEqual([(lot.shares, lot.price) for lot in lots], [(37, 16750), (63, 15980), (13, 15740)])
        self.assertEqual(trades[0].day, "2026-10-21")
        later = ledger._History("a1", date(2026, 12, 31), [], self.sources)
        _, carried = ledger._from_anchor(
            {"as_of": "2026-11-23", "lots": [lot.__dict__ for lot in closing]}, later, "ETF", 113
        )
        self.assertEqual(carried, closing)

    def test_cached_gap_does_not_bridge_to_older_trades_or_save_anchor(self):
        self.cache.delete_one({"_id": "a1:20261023"})
        with self.assertRaises(ledger.LedgerError) as caught:
            ledger._from_window(self.history, "ETF", 113)
        self.assertIn("2026-10-23", str(caught.exception))
        self.assertIn("76주", str(caught.exception))
        empty = Mock()
        empty.find.return_value = []
        frozen = datetime(2026, 11, 24, 10, tzinfo=ledger.KST)
        with (
            patch.object(ledger, "datetime", wraps=datetime) as clock,
            patch.object(ledger, "_consistent_snapshot", return_value=({"ETF": (113, 16000)}, [])),
            patch.object(orders, "order_sources", return_value=self.sources),
            patch.object(ledger, "_anchors", return_value=empty),
            patch.object(ledger, "_snapshots", return_value=empty),
            patch.object(ledger, "_replace_ledger_document") as write,
        ):
            clock.now.return_value = frozen
            result = ledger.refresh_ledgers("a1", ["ETF"])
        self.assertIsInstance(result["ETF"], ledger.LedgerError)
        self.assertEqual(write.call_count, 1)
        self.assertIn("error", write.call_args.args[1])
        self.assertNotIn("lots", write.call_args.args[1])

    def test_cached_quantity_mismatch_is_not_accepted_as_new_basis(self):
        with self.assertRaises(ledger.LedgerError):
            ledger._from_window(self.history, "ETF", 112)

    def test_sell_target_bought_before_window_is_recovered_from_original_trade(self):
        self.rows[date(2026, 11, 9)][0]["filled_quantity"] = 39
        self.sources[("20261109", "4")] = (ledger.SOURCE_AUTO, "2026-10-22:2")
        _, closing = ledger._from_window(self.history, "ETF", 124)
        self.assertEqual([(lot.shares, lot.price) for lot in closing], [(37, 16750), (74, 15980), (13, 15740)])

    def test_intraday_flat_point_excludes_prior_unknown_holdings(self):
        row = {
            "ticker": "ETF",
            "order_no": "0",
            "side_label": "매도",
            "filled_quantity": 50,
            "average_fill_price": 17000,
        }
        self.cache.docs[0]["rows"].insert(0, row)
        trades, closing = ledger._from_window(self.history, "ETF", 113)
        self.assertTrue(trades[0].buy)
        self.assertEqual(sum(lot.shares for lot in closing), 113)

    def test_invalid_remaining_buy_price_is_not_replaced_with_average(self):
        self.cache.docs[0]["rows"][0]["average_fill_price"] = 0
        with self.assertRaises(ledger.LedgerError):
            ledger._from_window(self.history, "ETF", 113)


class MeritzOrderTransportTest(TestCase):
    """주문 전 조회 복구와 주문 전송의 중복 방지를 외부 호출 없이 검증한다."""

    def setUp(self):
        self.response = Mock(status_code=200)
        self.response.json.return_value = {"rsp_cd": "5766", "data": []}
        patches = {
            "token": patch.object(meritz, "_access_token", return_value="secret-token"),
            "token_db": patch.object(meritz, "_token_collection", return_value=Mock(find_one=Mock(return_value={}))),
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
        self.assertEqual(self.mocks["get"].call_count, 4)
        self.mocks["refresh"].assert_called_once_with("a1", "secret-token")
        self.assertEqual([call.args[0] for call in self.mocks["sleep"].call_args_list], [1.0, 2.0])
        self.assertIn("stage=authentication", self.diagnostics())

    def test_refreshed_token_recovers_after_delayed_get_without_reissuing(self):
        expired = Mock(status_code=401)
        expired.json.return_value = {"rsp_cd": meritz._TOKEN_EXPIRED_CODE}
        self.mocks["get"].side_effect = [expired, expired, self.response]
        self.assertEqual(meritz._get("a1", meritz.HOLDINGS_PATH, {})["rsp_cd"], "5766")
        self.assertEqual(self.mocks["get"].call_count, 3)
        self.mocks["refresh"].assert_called_once_with("a1", "secret-token")
        self.mocks["sleep"].assert_called_once_with(1.0)
        self.assertIn("outcome=recovered", self.diagnostics())
        self.mocks["post"].assert_not_called()

    def test_post_auth_rejection_does_not_enter_get_delay_retry_loop(self):
        self.response.status_code = 401
        self.response.json.return_value = {"rsp_cd": meritz._TOKEN_EXPIRED_CODE}
        self.mocks["post"].side_effect = None
        self.mocks["post"].return_value = self.response
        with self.assertRaises(BrokerApiError):
            meritz._request("a1", "POST", meritz.ORDER_PATH["buy"], success=meritz._ORDER_CODES, body={})
        self.assertEqual(self.mocks["post"].call_count, 2)
        self.mocks["refresh"].assert_called_once()
        self.mocks["sleep"].assert_not_called()

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


class MeritzTokenRefreshTest(TestCase):
    """공유 토큰의 중복 발급·만료 연장과 다른 계좌의 간섭을 막는지 검증한다."""

    def setUp(self):
        self.config = MemoryCollection()
        self.now = time.time()
        self.response = Mock(ok=True, status_code=200)
        self.response.json.return_value = {"access_token": "new-private-token", "expires_in": 43200}
        self.post = Mock(return_value=self.response)
        self.log = Mock()
        patches = [
            patch.object(meritz, "_token_collection", return_value=self.config),
            patch.object(locking, "get_db_connection", return_value={"system_config": self.config}),
            patch.object(meritz, "require_env_keys", return_value=("private-key", "private-secret")),
            patch.object(meritz.requests, "post", self.post),
            patch.object(meritz.requests, "get", side_effect=AssertionError("실제 조회 금지")),
            patch.object(meritz, "logger", self.log),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def put(self, token, expiry, issued):
        self.config.insert_one({"_id": "meritz_token:a1", "token": token, "exp": expiry, "issued_at": issued})

    def jwt(self, expiry):
        claims = json.dumps({"exp": expiry, "iat": self.now - 20, "private": "private-claims"}).encode()
        payload = base64.urlsafe_b64encode(claims).decode().rstrip("=")
        return f"header.{payload}.signature"

    def test_concurrent_workers_issue_one_token(self):
        entered, contended, release = Event(), Event(), Event()
        results, failures = [], []
        insert = self.config.insert_one

        def watched_insert(doc):
            try:
                return insert(doc)
            except DuplicateKeyError:
                contended.set()
                raise

        def slow_issue(*args, **kwargs):
            entered.set()
            if not release.wait(3):
                raise AssertionError("갱신 대기 시간 초과")
            return self.response

        def worker():
            try:
                results.append(meritz._access_token("a1"))
            except Exception as exc:
                failures.append(exc)

        self.post.side_effect = slow_issue
        with patch.object(self.config, "insert_one", side_effect=watched_insert):
            first, second = Thread(target=worker), Thread(target=worker)
            first.start()
            try:
                self.assertTrue(entered.wait(2))
                second.start()
                self.assertTrue(contended.wait(2))
            finally:
                release.set()
                first.join(3)
                if second.ident is not None:
                    second.join(3)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertFalse(failures)
        self.assertEqual(results, ["new-private-token", "new-private-token"])
        self.post.assert_called_once()

    def test_rejected_old_token_reuses_current_token_and_recent_same_token(self):
        self.put("current-private-token", self.now + 1000, self.now)
        self.assertEqual(meritz._issue_token("a1", "old-private-token"), "current-private-token")
        self.assertEqual(meritz._issue_token("a1", "current-private-token"), "current-private-token")
        self.post.assert_not_called()

    def test_another_account_refresh_is_not_blocked(self):
        with locking.broker_token_lock("a1"):
            self.assertEqual(meritz._access_token("a2"), "new-private-token")
        self.post.assert_called_once()

    def test_refresh_contention_defers_without_issuing_or_changing_token(self):
        with locking.broker_token_lock("a1"):
            with patch.object(locking, "monotonic", side_effect=[0, locking._LOCK_WAIT_SECONDS + 1]):
                with self.assertRaises(meritz.MeritzTokenRefreshBusyError):
                    meritz._issue_token("a1", None)
        self.post.assert_not_called()

    def test_jwt_expiry_caps_reported_lifetime_and_logs_only_fingerprints(self):
        expiry = int(self.now + 1800)
        token = self.jwt(expiry)
        self.response.json.return_value["access_token"] = token
        self.assertEqual(meritz._access_token("a1"), token)
        self.assertEqual(self.config.find_one({"_id": "meritz_token:a1"})["exp"], expiry)
        diagnostic = "\n".join(call.args[0] % call.args[1:] for call in self.log.method_calls)
        for secret in (token, "private-claims", "private-key", "private-secret"):
            self.assertNotIn(secret, diagnostic)
        self.assertIn("token_exp=", diagnostic)
        self.assertIn("token_fp=", diagnostic)

    def test_same_token_response_does_not_extend_existing_expiry(self):
        expiry = self.now + 1000
        self.put("same-private-token", expiry, self.now - 60)
        self.response.json.return_value["access_token"] = "same-private-token"
        meritz._issue_token("a1", "same-private-token")
        self.assertEqual(self.config.find_one({"_id": "meritz_token:a1"})["exp"], expiry)
        self.post.assert_called_once()

    def test_returned_expired_jwt_is_not_saved_as_fresh(self):
        self.response.json.return_value["access_token"] = self.jwt(self.now - 1)
        with self.assertRaises(meritz.MeritzTokenExpiredError):
            meritz._access_token("a1")
        self.assertIsNone(self.config.find_one({"_id": "meritz_token:a1"}))

    def test_cached_jwt_actual_expiry_overrides_incorrect_local_lifetime(self):
        self.put(self.jwt(self.now - 1), self.now + 43200, self.now - 60)
        self.assertEqual(meritz._access_token("a1"), "new-private-token")
        self.post.assert_called_once()

    def test_same_short_token_is_used_then_deferred_until_real_expiry(self):
        clock = [self.now]
        expiry = self.now + 50
        token = self.jwt(expiry)
        self.put(token, expiry, self.now - 43150)
        self.response.json.return_value["access_token"] = token
        with patch.object(meritz.time, "time", side_effect=lambda: clock[0]):
            self.assertEqual(meritz._access_token("a1"), token)
            saved = self.config.find_one({"_id": "meritz_token:a1"})
            self.assertEqual(saved["issued_at"], self.now - 43150)
            self.assertEqual(saved["exp"], expiry)
            clock[0] += 20
            with self.assertRaises(meritz.MeritzTokenRefreshPendingError) as pending:
                meritz._access_token("a1")
            self.assertEqual(pending.exception.retry_at, expiry + 1)
            self.assertEqual(self.post.call_count, 1)
            clock[0] = expiry + 2
            self.response.json.return_value["access_token"] = self.jwt(clock[0] + 43200)
            meritz._access_token("a1")
            self.assertEqual(self.post.call_count, 2)

    def test_pending_refresh_prevents_order_submission_and_reissuance(self):
        token = self.jwt(self.now + 5)
        self.put(token, self.now + 5, self.now - 43195)
        self.config.update_one({"_id": "meritz_token:a1"}, {"$set": {"refresh_not_before": self.now + 6}})
        with patch.object(meritz, "_send_request") as send:
            with self.assertRaises(meritz.MeritzTokenRefreshPendingError):
                meritz._request("a1", "POST", meritz.ORDER_PATH["buy"], success=meritz._ORDER_CODES, body={})
        send.assert_not_called()
        self.post.assert_not_called()

    def test_other_workers_new_token_is_reused_during_refresh_cooldown(self):
        self.put("new-private-token", self.now + 1000, self.now)
        self.config.update_one({"_id": "meritz_token:a1"}, {"$set": {"refresh_not_before": self.now + 5}})
        self.assertEqual(meritz._issue_token("a1", "old-private-token"), "new-private-token")
        self.post.assert_not_called()

    def test_pending_authentication_is_not_changed_to_round_error(self):
        pending = meritz.MeritzTokenRefreshPendingError(self.now + 10)
        with (
            patch("services.broker_trade_service.require_trade_account"),
            patch.object(trade_service, "refresh_ledgers", side_effect=pending),
        ):
            with self.assertRaises(meritz.MeritzTokenRefreshPendingError):
                trade_service._account_ledgers({"a1": ["ETF"]}, fresh=True)


class OrderNotificationIsolationTest(TestCase):
    def test_account_failure_does_not_change_its_orders_or_stop_next_account(self):
        records = {
            account: [
                {
                    "_id": account,
                    "account_id": account,
                    "country": "kor",
                    "order_no": account,
                    "created_at": orders._now().isoformat(),
                    "status": "pending",
                }
            ]
            for account in ("a1", "a2")
        }
        store = Mock()
        store.distinct.return_value = ["a1", "a2"]
        store.find.side_effect = lambda query: records[query["account_id"]]

        def broker(account, day):
            if account == "a1":
                raise meritz.MeritzTokenExpiredError("token expired")
            return [
                {
                    "order_date": day,
                    "order_no": "a2",
                    "status": "filled",
                    "filled_quantity": 1,
                    "average_fill_price": 100,
                }
            ]

        with (
            patch.object(orders, "_orders", return_value=store),
            patch.object(orders, "list_order_states", side_effect=broker),
            patch.object(orders, "_broker_status", side_effect=lambda record, state: state),
        ):
            failures = orders.refresh_slack_order_accounts()
        self.assertEqual(set(failures), {"a1"})
        store.update_one.assert_called_once()
        self.assertEqual(store.update_one.call_args.args[0], {"_id": "a2"})
        self.assertEqual(store.update_one.call_args.args[1]["$set"]["status"], "filled")

    def test_successful_accounts_keep_result_notifications(self):
        failures = {"a1": meritz.MeritzTokenExpiredError("expired")}
        records = [{"_id": account, "account_id": account} for account in ("a1", "a2")]
        with (
            patch.object(slack, "refresh_slack_order_accounts", return_value=failures),
            patch.object(slack, "finished_slack_orders", return_value=records),
            patch.object(slack, "_target_of", side_effect=lambda record: record),
            patch.object(slack, "serialize_order", side_effect=lambda record: record),
            patch.object(slack, "_finish") as finish,
            patch.object(slack, "mark_slack_notified") as notified,
        ):
            sent, pending = slack.notify_finished_orders()
        self.assertEqual(sent, 1)
        self.assertEqual(set(pending), {"a1"})
        finish.assert_called_once_with(records[1], records[1])
        notified.assert_called_once_with("a2")

    def test_failed_accounts_are_excluded_from_thread_sync(self):
        threads_to_sync = [
            {"thread_ts": account, "messages": [{"request": {"account_id": account}}]} for account in ("a1", "a2")
        ]
        with (
            patch.object(slack, "signal_threads", return_value=threads_to_sync),
            patch.object(slack, "sync_signal_thread", return_value=orders.MESSAGE_DONE) as sync,
        ):
            self.assertEqual(slack.sync_signal_threads({"a1"}), 1)
        sync.assert_called_once_with("a2")

    def test_other_accounts_still_receive_close_warning(self):
        records = [{"_id": account, "account_id": account} for account in ("a1", "a2")]
        order = {
            "id": "a2",
            "env": "live",
            "side": "buy",
            "quantity": 2,
            "filled_quantity": 0,
            "price": 100,
            "country": "kor",
            "pair_id": None,
        }
        with (
            patch.object(slack, "close_warning_slack_orders", return_value=records),
            patch.object(slack, "serialize_order", return_value=order),
            patch.object(slack, "_request_of", return_value={}),
            patch.object(slack, "_target_of", return_value={"thread_ts": "a2"}),
            patch.object(slack, "close_warning_text", return_value="마감"),
            patch.object(slack, "_cancel_blocks", return_value=[]),
            patch.object(slack, "_note", return_value={}),
            patch.object(slack, "_update_message"),
            patch.object(slack, "send_thread_message") as send,
            patch.object(slack, "mark_slack_close_warned") as mark,
        ):
            self.assertEqual(slack.notify_close_approaching({"a1"}), 1)
        send.assert_called_once()
        self.assertEqual(send.call_args.kwargs["thread_ts"], "a2")
        mark.assert_called_once_with("a2")

    def test_batch_continues_warning_and_sync_and_reports_non_token_errors(self):
        for failure, exit_code in ((meritz.MeritzTokenExpiredError("expired"), 0), (BrokerApiError("other"), 1)):
            with (
                self.subTest(failure=type(failure).__name__),
                patch.object(order_job, "load_env_if_present"),
                patch.object(order_job, "notify_finished_orders", return_value=(1, {"a1": failure})),
                patch.object(order_job, "notify_close_approaching", return_value=2) as warning,
                patch.object(order_job, "sync_signal_threads", return_value=3) as sync,
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(order_job.main(), exit_code)
                warning.assert_called_once_with({"a1"})
                sync.assert_called_once_with({"a1"})


if __name__ == "__main__":
    main()


class TradeHistoryRoundPlacementTest(TestCase):
    """이력의 매수·매도 연결과 원장 회차가 같은 행을 가리키는지 검증한다."""

    def attach(self, trades, initial=()):
        recorded = []
        lots = ledger._replay(list(initial), trades, recorded)
        snapshot = ledger.Ledger(
            quantity=sum(lot.shares for lot in lots),
            lots=tuple(lots),
            trades=tuple(recorded),
            manual_orders=(),
            computed_at="2026-11-25T00:25:00+09:00",
            country="us",
            market_date="2026-11-24",
            average_price=0,
        )
        regular = [lot for lot in lots if lot.arbitrary_id is None]
        positions = trade_service.ladder.ladder_positions([lot.price for lot in regular], 10)
        rows = [
            {"round": number, "position": position, "held": True, "fill_price": lot.price}
            for number, (position, lot) in enumerate(zip(positions, regular, strict=True), start=1)
        ]
        rows.extend(
            {"round": number, "position": (positions[-1] if positions else 0) + number - len(regular), "held": False}
            for number in range(len(regular) + 1, 11)
        )
        view = {
            "ticker": "SOXL",
            "rounds": rows,
            "config": {"trigger_pct": 10},
            "instrument_kind": "etf",
            "country": "us",
            "index": {"close": 32},
        }
        with patch.object(trade_service, "_day_change_pcts", return_value={}):
            trade_service._attach_trade_history(view, snapshot, {}, [])
        return view, snapshot

    @staticmethod
    def buy(price, shares=5, day="2026-11-24"):
        return ledger.Trade(day, True, shares, price, ledger.SOURCE_AUTO, None)

    @staticmethod
    def sell(price, shares, day="2026-11-24"):
        return ledger.Trade(day, False, shares, price, ledger.SOURCE_AUTO, None, ledger.REGULAR_SELL)

    @staticmethod
    def positions_for(view, side, pair):
        return [
            position
            for position, days in view["trades_by_position"].items()
            for badges in days.values()
            for badge in badges
            if badge["side"] == side and pair in badge["pairs"]
        ]

    def test_soxl_buy_15_and_16_stay_on_round_7_and_8(self):
        past = [
            event for _ in range(8) for event in (self.buy(100, day="2026-11-02"), self.sell(110, 5, day="2026-11-02"))
        ]
        trades = past + [self.buy(price) for price in (68.62, 63.89, 55.97, 53.72, 46.59, 44.32, 39.535, 35.33)]
        view, snapshot = self.attach(trades)
        self.assertEqual(self.positions_for(view, "buy", 15), [7])
        self.assertEqual(self.positions_for(view, "buy", 16), [8])
        self.assertEqual(snapshot.quantity, 40)
        self.assertAlmostEqual(view["history_summary"]["realized"], 400)

    def test_partial_multi_round_sell_and_rebuy_follow_their_buy_rounds(self):
        trades = [self.buy(price) for price in (100, 95, 90)]
        trades += [self.sell(105, 6), self.buy(89)]
        view, snapshot = self.attach(trades)
        self.assertEqual(self.positions_for(view, "buy", 2), [2])
        self.assertEqual(self.positions_for(view, "buy", 3), [3])
        self.assertEqual(self.positions_for(view, "sell", 2), [2])
        self.assertEqual(self.positions_for(view, "sell", 3), [3])
        self.assertEqual(self.positions_for(view, "buy", 4), [3])
        self.assertEqual(snapshot.quantity, 14)
        self.assertAlmostEqual(view["history_summary"]["realized"], 85)

    def test_round_with_price_gap_uses_its_display_position(self):
        view, _ = self.attach([self.buy(price) for price in (100, 89, 60)])
        position = next(row["position"] for row in view["rounds"] if row["round"] == 3)
        self.assertEqual(position, 6)
        self.assertEqual(self.positions_for(view, "buy", 3), [position])

    def test_current_holdings_link_even_when_cycle_starts_before_history(self):
        initial = [ledger.Lot(5, price, "2026-09-01", ledger.SOURCE_MANUAL) for price in (100, 96)]
        view, _ = self.attach([self.buy(90)], initial)
        self.assertEqual(self.positions_for(view, "buy", 1), [3])

    def test_old_cycle_and_arbitrary_trades_keep_price_positions(self):
        trades = [self.buy(200), self.buy(180), self.sell(190, 10), self.buy(100), self.buy(95), self.buy(105)]
        view, _ = self.attach(trades)
        self.assertEqual(self.positions_for(view, "buy", 2), [trade_service.ladder.ladder_position(180, 100, 10)])
        self.assertEqual(self.positions_for(view, "buy", 4), [2])
        self.assertEqual(
            self.positions_for(view, "arbitrary_buy", 5), [trade_service.ladder.ladder_position(105, 100, 10)]
        )

    def test_paired_orders_use_us_order_date_after_korean_midnight(self):
        trades = [self.sell(100, 5), self.buy(99)]
        records = [
            {
                "country": "us",
                "created_at": "2026-11-25T00:05:00+09:00",
                "order_no": str(index + 1),
                "side": "buy" if trade.buy else "sell",
                "filled_quantity": 5,
                "average_fill_price": trade.price,
                "pair_id": "pair",
                "round": 7,
            }
            for index, trade in enumerate(trades)
        ]
        self.assertEqual(trade_service._paired_trade_rounds(trades, records), {0: 7, 1: 7})


class TokenRefreshBatchRecoveryTest(TestCase):
    """토큰 경계 대기를 계좌별로 처리하고 재시도 소진 시에만 장애로 확정한다."""

    def run_sync(self, due_times, recover):
        from utils import broker_balance_sync as sync

        clock, calls, attempted = [1000.0], [], set()
        accounts = [{"account_id": key, "broker_api": {"provider": "MERITZ"}} for key in due_times]

        def attempt(account_id):
            calls.append((account_id, clock[0]))
            if account_id not in attempted or not recover:
                attempted.add(account_id)
                raise meritz.MeritzTokenRefreshPendingError(max(clock[0] + 1, due_times[account_id]))
            return {"account_id": account_id, "ok": True}

        def sleep(seconds):
            self.assertLessEqual(seconds, 30)
            clock[0] += seconds

        with (
            patch("utils.account_settings_store.load_live_account_docs", return_value=accounts),
            patch.object(sync, "_sync_account", side_effect=attempt),
            patch.object(sync, "_load_failing", return_value=False),
            patch.object(sync, "_save_failing") as save,
            patch.object(sync, "_notify", return_value=True) as notify_slack,
            patch.object(sync, "epoch_time", side_effect=lambda: clock[0]),
            patch.object(sync, "sleep", side_effect=sleep),
            patch.object(sync, "_MERITZ_BALANCE_OPEN", datetime.min.time()),
            patch.object(sync, "_MERITZ_BALANCE_CLOSE", datetime.max.time()),
        ):
            result = sync.sync_all()
        return result, calls, notify_slack, save

    def test_accounts_retry_independently_and_send_one_recovery_summary(self):
        result, calls, notices, save = self.run_sync({"a1": 1050, "a2": 1003}, recover=True)
        self.assertTrue(all(account["ok"] for account in result["accounts"]))
        self.assertEqual(calls, [("a1", 1000), ("a2", 1000), ("a2", 1003), ("a1", 1050)])
        notices.assert_called_once()
        save.assert_not_called()

    def test_persistent_authentication_failure_has_a_bounded_retry_budget(self):
        result, calls, notices, save = self.run_sync({"a1": 1001}, recover=False)
        self.assertFalse(result["accounts"][0]["ok"])
        self.assertEqual(len(calls), 4)
        notices.assert_called_once()
        save.assert_called_once()
