"""화면과 백테스트가 **갈라지지 않는지** 지키는 회귀 테스트.

전략 화면이 보여 주는 보유·지시가 백테스트와 다르면 백테스트 숫자는 아무 의미가 없다.
「화면에서는 이걸 사라는데 백테스트는 안 샀다」가 되는 순간 성과를 믿을 수 없기 때문이다.

그래서 화면은 판정을 **다시 하지 않고** 백테스트 엔진을 돌린 마지막 상태를 읽어야 한다.
아래 테스트는 그 관계를 못 박는다 — 어느 한쪽만 고치면 여기서 먼저 실패한다.

DB·가격 캐시가 필요하므로, 못 읽는 환경에서는 건너뛴다(테스트가 환경 문제로 붉어지지 않게).
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

import pandas as pd

from utils.mix_sleeve import SleeveSpec, slot_state
from utils.portfolio_backtest import current_positions, run_backtest
from utils.strategy_mix_service import _simulate_mix

MOMENTUM_POOL = "us_stock"
NEW_HIGH_POOL = "us_stock"
MIX_ACCOUNT = "us_test"


def _load_env() -> None:
    from utils.env import load_env_if_present

    load_env_if_present()


def _skip_if_unavailable(error: Exception) -> None:
    """환경 문제만 건너뛴다 — 나머지는 그대로 터뜨린다.

    예전에는 모든 예외를 SkipTest 로 바꿨다. 그러면 회귀로 코드가 터져도 초록불이 뜬다.
    DB·네트워크·파일이 없는 경우만 걸러내고, 그 밖의 예외는 실패로 드러나야 한다.
    """
    unavailable: tuple[type[BaseException], ...] = (ConnectionError, TimeoutError, OSError)
    try:
        from pymongo.errors import PyMongoError

        unavailable = (*unavailable, PyMongoError)
    except ImportError:
        pass
    # 전략 시작일 = 오늘인데 그날 봉이 아직 없는 상태 — 코드가 아니라 데이터 도래의 문제라
    # 회귀가 아니다. 첫 거래일 봉이 생기면(다음 실행) 원래 검증이 그대로 돈다.
    if isinstance(error, RuntimeError) and "이후의 가격 데이터가 아직 없습니다" in str(error):
        raise unittest.SkipTest(f"전략 시작 전(봉 미도래)이라 건너뜁니다: {error}")
    if not isinstance(error, unavailable):
        raise error
    raise unittest.SkipTest(f"가격 캐시·DB 를 읽을 수 없어 건너뜁니다: {type(error).__name__}: {error}")


def _positions_of(holdings: list[dict[str, Any]]) -> dict[str, str]:
    """보유 목록을 {티커: 편입일} 로 — 비교에 쓰는 최소 형태."""
    return {str(row["ticker"]): str(row["entry_date"]) for row in holdings}


def _expected_mix_targets(ctx: dict[str, Any]) -> set[str]:
    """합성 목표의 기대값 — 각 전략 운용 현황(합성이 실제로 읽는 그 페이로드)에서 만든다.

    확정·장중 어느 경우든 유효하다: 보유에서 매도(체결 예정·잠정 예정 포함)를 빼고
    진입 예정(빈 자리만큼)을 더한 '다음 시가 이후의 보유'다.
    """
    from utils.mix_sleeve import PORTFOLIO, current_state

    expected: set[str] = set()
    for spec in ctx["slots"]:
        positions = current_state(spec)
        if spec.strategy == PORTFOLIO:
            expected |= {row["ticker"] for row in positions["open_positions"] if row["sleeve_weight_pct"] > 0}
            continue
        held = positions["holdings"]
        exiting = sum(1 for row in held if row.get("status") == "sell")
        free = max(int(positions["top_n"]) - (len(held) - exiting), 0)
        expected |= {str(row["ticker"]) for row in held if row.get("status") != "sell"}
        expected |= {str(row["ticker"]) for row in positions["planned_entries"][:free]}
    return expected


class MomentumScreenMatchesBacktest(unittest.TestCase):
    """모멘텀 운용 현황의 보유 = 백테스트를 현재까지 돌린 마지막 상태."""

    def test_holdings_come_from_the_backtest(self) -> None:
        _load_env()
        from utils import momentum_backtest
        from utils.momentum_service import load_settings

        try:
            settings = load_settings(MOMENTUM_POOL)
            context = momentum_backtest.load_context(settings)
            simulated = momentum_backtest.run_backtest(12, settings, context, start_date=settings["start_date"])
            screen = momentum_backtest.current_positions(settings)
        except Exception as error:  # noqa: BLE001 - 환경 문제와 회귀를 구분한다
            _skip_if_unavailable(error)

        self.assertEqual(
            _positions_of(screen["holdings"]),
            _positions_of(simulated["open_positions"]),
            "모멘텀 화면의 보유가 백테스트 마지막 상태와 다릅니다 — 화면이 판정을 다시 하고 있습니다.",
        )
        self.assertEqual(screen["as_of"], simulated["as_of"])
        # 장중 표시 가격이 바뀌어도 합성의 기준 가격·비중은 엔진과 같아야 한다.
        engine = {row["ticker"]: row for row in simulated["open_positions"]}
        for row in screen["target_holdings"]:
            self.assertEqual(row["price"], engine[row["ticker"]]["price"])
            self.assertEqual(row["sleeve_weight_pct"], engine[row["ticker"]]["sleeve_weight_pct"])


class NewHighScreenMatchesBacktest(unittest.TestCase):
    """신고가 운용 현황의 보유 = 백테스트를 현재까지 돌린 마지막 상태."""

    def test_holdings_come_from_the_backtest(self) -> None:
        _load_env()
        from utils import new_high_backtest
        from utils.new_high_service import load_settings

        try:
            from utils.new_high_service import available_pools

            pools = available_pools()
            if NEW_HIGH_POOL not in pools:
                if not pools:
                    raise unittest.SkipTest("신고가 전략을 켠 종목풀이 없습니다.")
                # 기본 풀이 전략을 껐으면 사용 중인 첫 풀로 검증한다 — 검증 대상은 관계지 풀이 아니다.
                settings = load_settings(pools[0])
            else:
                settings = load_settings(NEW_HIGH_POOL)
            context = new_high_backtest.load_context(settings)
            simulated = new_high_backtest.run_backtest(12, settings, context, start_date=settings["start_date"])
            screen = new_high_backtest.current_positions(settings)
        except Exception as error:  # noqa: BLE001
            _skip_if_unavailable(error)

        self.assertEqual(
            _positions_of(screen["holdings"]),
            _positions_of(simulated["open_positions"]),
            "신고가 화면의 보유가 백테스트 마지막 상태와 다릅니다 — 화면이 판정을 다시 하고 있습니다.",
        )
        self.assertEqual(screen["as_of"], simulated["as_of"])
        # 장중 표시 가격이 바뀌어도 합성의 기준 가격·비중은 엔진과 같아야 한다.
        engine = {row["ticker"]: row for row in simulated["open_positions"]}
        for row in screen["target_holdings"]:
            self.assertEqual(row["price"], engine[row["ticker"]]["price"])
            self.assertEqual(row["sleeve_weight_pct"], engine[row["ticker"]]["sleeve_weight_pct"])


class MixScreenMatchesSleeveBacktests(unittest.TestCase):
    """합성 화면의 목표 종목 = 각 슬리브 백테스트가 내는 **다음 시가 이후의 보유**.

    보유에서 매도 예정을 빼고 진입 예정을 더한 것이다 — 합성 목표는 '내일 이후 들고 있어야
    할 목록'이기 때문이다. 판정은 전부 엔진이 하고 합성은 읽기만 해야 한다.
    """

    def test_target_tickers_come_from_sleeve_backtests(self) -> None:
        _load_env()
        from utils.strategy_mix_service import _resolve_mix_account, mix_positions

        try:
            ctx = _resolve_mix_account(MIX_ACCOUNT)
            screen = mix_positions(MIX_ACCOUNT)
            expected = _expected_mix_targets(ctx)
        except Exception as error:  # noqa: BLE001
            _skip_if_unavailable(error)

        # 목표 비중이 0 인 행은 '팔아야 할 계좌 보유분'이라 슬리브 목표가 아니다.
        targets = {
            str(row["ticker"])
            for row in screen["holdings"]
            if not row.get("is_cash") and float(row.get("weight_pct") or 0) > 0
        }
        self.assertEqual(
            targets,
            expected,
            "합성 화면의 목표 종목이 슬리브 백테스트의 보유와 다릅니다 — 합성이 판정을 다시 하고 있습니다.",
        )


class IntradayScreenMixConsistencyTest(unittest.TestCase):
    """장중 고정 입력 — 실시간 시세를 흉내 내(일부는 시가 미상) 화면·합성이 **같은 잠정
    실행**을 읽는지 검증한다(AGENTS.md §10-6, 3단계 일치 검증).

    가격 수준은 인위적이어도 된다 — 여기서 지키는 것은 값이 아니라 '합성이 판정을 다시
    하지 않고 전략 운용 현황(잠정 실행 반영본)을 그대로 소비한다'는 관계다.
    """

    def test_live_screen_and_mix_share_the_same_run(self) -> None:
        _load_env()
        import utils.momentum_backtest as momentum_backtest
        import utils.new_high_backtest as new_high_backtest
        from utils.strategy_mix_service import _SHARES_CACHE, _resolve_mix_account, mix_positions

        def fake_quotes(pool: str, tickers: list[str], cached_last: pd.Timestamp) -> dict[str, Any]:
            # 모의 세션은 (캐시 다음 날, 오늘) 중 늦은 날 — 실제 장중(오늘 세션)과 같은 의미라,
            # 전략 시작일이 오늘이어도 잠정 실행이 '시작 전'으로 빠지지 않는다.
            session = str(max(cached_last + pd.Timedelta(days=1), pd.Timestamp.today().normalize()).date())
            by_ticker = {}
            for index, ticker in enumerate(sorted(set(tickers))):
                by_ticker[ticker] = {
                    "price": 50.0,
                    "high": 51.0,
                    # 절반은 시가 미상(국내 ETF 흉내) — 체결 예정 경로까지 함께 태운다.
                    "open": None if index % 2 else 50.0,
                    "change_pct": -1.0,
                }
            return {"live": True, "pre_market": False, "traded_at": session, "by_ticker": by_ticker}

        from utils.portfolio_backtest import _POSITIONS_CACHE as _PORTFOLIO_CACHE

        caches = (
            momentum_backtest._POSITIONS_CACHE,
            new_high_backtest._POSITIONS_CACHE,
            _PORTFOLIO_CACHE,
            _SHARES_CACHE,
        )
        try:
            for cache in caches:
                cache.invalidate()
            with (
                patch.object(momentum_backtest, "_live_quotes", fake_quotes),
                patch.object(new_high_backtest, "_live_quotes", fake_quotes),
                patch("utils.slot_positions._live_quotes", fake_quotes),
            ):
                from utils.mix_sleeve import PORTFOLIO, current_state

                ctx = _resolve_mix_account(MIX_ACCOUNT)
                mix = mix_positions(MIX_ACCOUNT)
                expected = _expected_mix_targets(ctx)
                slot_positions = {spec.key: current_state(spec) for spec in ctx["slots"] if spec.strategy != PORTFOLIO}
        except Exception as error:  # noqa: BLE001
            _skip_if_unavailable(error)
        finally:
            # 가짜 실시간이 든 결과를 다른 테스트·다음 계산이 읽지 않게 비운다.
            for cache in caches:
                cache.invalidate()

        targets = {
            str(row["ticker"])
            for row in mix["holdings"]
            if not row.get("is_cash") and float(row.get("weight_pct") or 0) > 0
        }
        self.assertEqual(targets, expected, "장중 합성 목표가 전략 운용 현황(잠정 실행)과 다릅니다.")
        for key, positions in slot_positions.items():
            self.assertTrue(positions["live"], f"{key} 슬리브가 장중 실행을 쓰지 않았습니다.")
            for row in positions["planned_entries"]:
                self.assertIsNotNone(row.get("sleeve_weight_pct"), f"{key} 진입 예정 비중 누락: {row['ticker']}")
            for row in positions["holdings"]:
                if row.get("fill_date"):
                    # 체결 예정 매도(확정)는 잠정 예상 표시가 아니어야 한다.
                    self.assertEqual(row.get("status"), "sell")
                    self.assertFalse(row.get("is_exit_forecast"))


class PortfolioMixStateTest(unittest.TestCase):
    def test_screen_and_mix_match_portfolio_engine(self):
        cases = [
            # 일반 리밸런싱과 의도한 현금을 같은 엔진 상태로 전달한다.
            ({"A": [100, 200, 220], "B": [100, 100, 100]}, [40, 40], 20, (0, 0)),
            # 매수만 밴드를 벗어나도 화면·합성이 차입한 목표를 받으면 안 된다.
            ({"A": [100, 90, 90], "B": [100, 320 / 3, 320 / 3], "C": [100, 320 / 3, 320 / 3]}, [40, 30, 30], 0, (0, 0)),
            # 매도 비용이 있는 경우에도 실제 체결 후 비중을 공유한다.
            ({"A": [100, 200, 220], "B": [100, 100, 100]}, [40, 40], 20, (0.1, 0.2)),
        ]
        for prices, weights, cash, slippage in cases:
            with self.subTest(prices=prices, cash=cash, slippage=slippage):
                self._assert_shared_state(prices, weights, cash, slippage)

    def _assert_shared_state(self, prices, weights, cash, slippage):
        # 케이스마다 슬리피지(설정 밖 외부 조회)만 바뀌므로 결과 캐시를 비운다 —
        # 캐시 키는 설정·시작일 기준이라 이전 케이스의 실행이 재사용된다.
        from utils.portfolio_backtest import _POSITIONS_CACHE

        _POSITIONS_CACHE.invalidate()
        dates = ["2026-01-02", "2026-02-02", "2026-02-03"]
        index = pd.to_datetime(dates)
        frame = pd.DataFrame(prices, index=index)
        settings = {
            "pool": "test",
            "start_date": dates[0],
            "rebalance": "monthly",
            "band_pct": 3,
            "cash_weight_pct": cash,
            "weights": [{"ticker": t, "weight_pct": w} for t, w in zip(prices, weights, strict=True)],
        }
        with (
            patch("utils.portfolio_backtest.validate_settings", side_effect=lambda s: s),
            patch("utils.portfolio_backtest._load_close_frame", return_value=frame),
            patch("utils.portfolio_backtest.get_pool_slippage", return_value=slippage),
            patch("utils.benchmark_curve.load_benchmark_frame", return_value=pd.DataFrame({"Close": 100}, index=index)),
            patch("utils.benchmark_curve.benchmark_growth", side_effect=lambda pool, ix: pd.Series(1.0, index=ix)),
            patch("utils.portfolio_backtest.benchmark_info", return_value={"name": "기준"}),
            patch(
                "utils.portfolio_backtest._overlay_live_last_bar",
                side_effect=lambda pool, close, benchmark: (close, benchmark),
            ),
        ):
            result = run_backtest(12, settings, start_date=dates[0])
            screen = current_positions(settings)
        self.assertEqual(screen["open_positions"], result["open_positions"])
        self.assertEqual(screen["sleeve_cash_weight_pct"], result["sleeve_cash_weight_pct"])
        spec = SleeveSpec("a", "portfolio", "test", settings)
        with (
            patch("utils.mix_sleeve.current_state", return_value=screen),
            patch("utils.portfolio_service.universe_metrics", return_value=[]),
            patch("utils.settings_loader.get_ticker_type_settings", return_value={"currency": "USD"}),
        ):
            state = slot_state(spec)
        for source, target in zip(result["open_positions"], state.targets, strict=True):
            self.assertEqual(source["sleeve_weight_pct"], target["drift_pct"])
            self.assertEqual(source["price"], target["price"])
            self.assertGreaterEqual(target["drift_pct"], 0)
        self.assertGreaterEqual(result["sleeve_cash_weight_pct"], 0)
        self.assertAlmostEqual(sum(t["drift_pct"] for t in state.targets) + result["sleeve_cash_weight_pct"], 100)
        if cash == 0:
            self.assertFalse([trade for trade in result["trades"] if trade["reason"] == "리밸런싱"])
            self.assertEqual([round(t["drift_pct"], 6) for t in state.targets], [36, 32, 32])


class MixRebalanceMatchesBacktest(unittest.TestCase):
    def test_first_operating_day_uses_saved_mix_weights(self):
        slots = [SleeveSpec("a", "momentum", "us_stock", {}), SleeveSpec("b", "portfolio", "us_etf", {})]
        ctx = {"account_id": "test", "slots": slots}
        results = {
            "a": {"daily": [{"date": "2026-09-08", "strategy_pct": 0, "cash_weight_pct": 100}]},
            "b": {"daily": [{"date": "2026-09-08", "strategy_pct": 0, "cash_weight_pct": 0}]},
        }
        with (
            patch(
                "utils.strategy_mix_service.mix_weights_for_account",
                return_value={"a_pct": 40, "b_pct": 30, "cash_pct": 30},
            ),
            patch("utils.pool_settings_store.get_pool_slippage", return_value=(0.1, 0.2)),
        ):
            state = _simulate_mix(ctx, results, through_date=None)
        self.assertEqual(state["curve"].to_dict(), {"2026-09-08": 1.0})
        self.assertEqual(state["values"], {"a": 0.4, "b": 0.3})
        self.assertEqual(state["cash"], 0.3)

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


class SlotEngineProvisionalBarTest(unittest.TestCase):
    """슬롯 엔진의 잠정 마지막 봉 모드 — 고정 입력으로 §10-6 체결 규칙을 검증한다.

    어제 확정된 주문이 오늘 시가에 체결되는 날, 시가를 안 주는 종목은 체결가를 지어내지
    않고 '오늘 체결 예정'이 된다(매도는 보유 유지 + fill_date, 매수는 자리만 차지).
    잠정일의 마지막 판정은 체결 예정을 재판정하지 않고 남은 자리만 잠정 후보로 채운다.
    """

    def test_pending_orders_and_provisional_judgment(self):
        from core.strategy.slot_backtest import run_slot_backtest

        dates = pd.to_datetime(["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04"])
        tickers = ["T1", "T2", "T3", "T4"]

        def frame(rows: dict[str, list[float]]) -> pd.DataFrame:
            return pd.DataFrame(rows, index=dates)[tickers]

        nan = float("nan")
        # T1: D1 진입 신호 → D2 시가 100 체결, D3 청산 신호 → D4 시가 없음(체결 예정 매도).
        # T2: D3 진입 신호 → D4 시가 없음(체결 예정 매수, 우선순위 최상 — 자리를 먼저 차지).
        # T3: D3 진입 신호 → D4 시가 41 체결(오늘 진입). T4: D4(잠정) 진입 신호 → 내일 예정.
        close = frame(
            {"T1": [100, 100, 100, 95], "T2": [100, 100, 100, nan], "T3": [40, 40, 40, 45], "T4": [10, 10, 10, 12]}
        )
        opens = frame(
            {"T1": [100, 100, 100, nan], "T2": [100, 100, 100, nan], "T3": [40, 40, 40, 41], "T4": [10, 10, 10, nan]}
        )
        panel = {"close": close, "open": opens, "high": close, "value": close}
        entry = frame({"T1": [1, 0, 0, 0], "T2": [0, 0, 1, 0], "T3": [0, 0, 1, 0], "T4": [0, 0, 0, 1]}).astype(bool)
        exit_signal = frame({"T1": [0, 0, 1, 1], "T2": [0] * 4, "T3": [0] * 4, "T4": [0] * 4}).astype(bool)
        priority = frame({"T1": [1.0] * 4, "T2": [9.0] * 4, "T3": [5.0] * 4, "T4": [3.0] * 4})

        # 엔진은 외부 조회가 없다 — 시장 파라미터를 값으로 그대로 넘긴다(패치 불필요).
        result = run_slot_backtest(
            months=1,
            panel=panel,
            entry=entry,
            exit_signal=exit_signal,
            priority=priority,
            slots=3,
            name_by={t: t for t in tickers},
            industry_by={},
            exit_reason="이탈",
            buy_slippage=0.0,
            sell_slippage=0.0,
            initial_capital=3000.0,
            entry_blocked=lambda day: False,
            adr_at=lambda day: None,
            benchmark_growth=lambda index: pd.Series(1.0, index=index),
            benchmark_name="BM",
            start_date="2026-09-01",
            provisional_last_bar=True,
        )

        # 체결 예정 매도 — 보유 유지, fill_date 부착, 보유일은 신호가 난 어제까지, 표시는 확정 종가.
        self.assertEqual(result["pending_exits"], ["T1"])
        by_ticker = {row["ticker"]: row for row in result["open_positions"]}
        self.assertEqual(by_ticker["T1"]["fill_date"], "2026-09-04")
        self.assertEqual(by_ticker["T1"]["days"], 2)
        self.assertEqual(by_ticker["T1"]["price"], 95.0)  # 실시간 현재가는 표시하되 체결로 쓰지 않는다
        self.assertEqual(result["exited_today"], [])  # 체결가를 지어내 이탈 표로 보내지 않는다
        # 체결 예정 매수 — 우선순위대로 자리를 차지하고 슬롯 1칸 비중을 싣는다.
        self.assertEqual(result["pending_entries"], [{"ticker": "T2", "sleeve_weight_pct": 100.0 / 3}])
        # 시가를 아는 진입은 오늘 체결 — 슬롯 몫(현금 3,000 중 1,000)으로 41에 24주.
        self.assertTrue(by_ticker["T3"]["is_new"])
        self.assertEqual(by_ticker["T3"]["entry_date"], "2026-09-04")
        self.assertEqual(by_ticker["T3"]["entry_price"], 41.0)
        # 잠정일 판정 — 체결 예정 둘이 자리를 차지해 남은 한 자리만 잠정 후보(T4)로 채운다.
        self.assertEqual(result["planned_exits"], [])
        self.assertEqual(result["planned_entries"], ["T4"])
        self.assertGreater(result["planned_entry_weights"]["T4"], 0.0)
        # 확정 미체결 매수와 잠정 진입이 같은 현금을 중복 사용하지 않아야 한다.
        remaining_holdings = sum(
            row["sleeve_weight_pct"]
            for row in result["open_positions"]
            if row["ticker"] not in set(result["pending_exits"]) | set(result["planned_exits"])
        )
        target_stock = (
            remaining_holdings
            + sum(row["sleeve_weight_pct"] for row in result["pending_entries"])
            + sum(result["planned_entry_weights"].values())
        )
        self.assertLessEqual(target_stock, 100.0)

        # 잠정일이 일별 곡선에 포함된다(실시간 마지막 봉 평가).
        self.assertEqual(result["daily"][-1]["date"], "2026-09-04")

        # 엔진의 확정 주문 날짜가 합성 이벤트·액션까지 보존되는지 함께 검증한다.
        from functools import partial

        from core.strategy.intraday import mark_engine_statuses
        from core.strategy.mix.actions import build_action_groups
        from utils.mix_sleeve import _slot_state_from_positions

        # 날짜 전달 검증용 계좌: 충분한 현금, 초과 허용 없음. 목표는 아래 엔진 결과로만 만든다.
        build_action_groups = partial(
            build_action_groups,
            excess_holding_allowance=0,
            cash_balance=1_000_000,
            total_assets=0,
            fixed_asset_value=0,
        )

        held = [dict(row) for row in result["open_positions"]]
        mark_engine_statuses(held, result["planned_exits"])
        raw = {
            "holdings": held,
            "planned_entries": [
                {**row, "price": 100.0, "fill_date": result["as_of"]} for row in result["pending_entries"]
            ]
            + [
                {"ticker": t, "price": 12.0, "sleeve_weight_pct": result["planned_entry_weights"][t]}
                for t in result["planned_entries"]
            ],
            "live": True,
            "as_of": result["as_of"],
            "currency": "USD",
        }
        state = _slot_state_from_positions(SleeveSpec("a", "momentum", "test", {}), raw, 3)
        actions = {
            "slots": {
                "a": {
                    "label": "A",
                    "live": state.live,
                    "sells": state.sells,
                    "entries": state.entries,
                    "exit_forecast": [],
                }
            }
        }
        account_rows = [
            {
                "ticker": "T1",
                "price": 95,
                "held_quantity": 1,
                "trade_quantity": -1,
                "target_quantity": 0,
                "weight_pct": 0,
                "is_sell_all": True,
            },
            {
                "ticker": "T2",
                "price": 100,
                "held_quantity": 0,
                "trade_quantity": 1,
                "target_quantity": 1,
                "weight_pct": 33,
            },
            {
                "ticker": "T4",
                "price": 12,
                "held_quantity": 0,
                "trade_quantity": 1,
                "target_quantity": 1,
                "weight_pct": 33,
            },
        ]
        groups = build_action_groups(
            account_rows,
            actions,
            "2026-09-07",
            currency="USD",
            target_schedule={
                "2026-09-04": {"quantities": {"T2": 1}, "weights": {"T2": 33}},
                "2026-09-07": {"quantities": {"T2": 1, "T4": 1}, "weights": {"T2": 33, "T4": 33}},
            },
        )
        items = {item["ticker"]: item for group in groups for item in group["items"]}
        for ticker in ("T1", "T2"):
            self.assertEqual(items[ticker]["date"], result["as_of"])
            self.assertNotIn("예상", items[ticker]["title"] + items[ticker]["text"])
        self.assertEqual(items["T4"]["date"], "2026-09-07")
        self.assertIn("예상", items["T4"]["title"])
        # 같은 종목의 날짜별 목표를 슬리브 배분에서 산출한다. 실제 보유는 차이 계산에만 쓴다.
        from core.strategy.mix.targets import dated_target_shares, sleeve_target_shares

        for first_plan, second_plan, held_qty, expected_trades in [
            ("buy", "buy", 0, [2, 3]),
            ("sell", "sell", 5, [-2, -3]),
            ("buy", "sell", 3, [2, -3]),
            ("sell", "buy", 2, [-2, 3]),
        ]:
            with self.subTest(first=first_plan, second=second_plan):
                targets = {
                    key: [
                        {
                            "ticker": "SAME",
                            "price": 10.0,
                            "drift_pct": 100.0,
                            "plan": plan,
                            "is_exiting": plan == "sell",
                            "fill_date": date,
                        }
                    ]
                    for key, plan, date in [("a", first_plan, "2026-09-04"), ("b", second_plan, None)]
                }
                budgets = {"a": 20.0, "b": 30.0}
                schedule = dated_target_shares(targets, budgets, 1.0, 50.0, "2026-09-07")
                final = sleeve_target_shares(targets, budgets, 1.0)
                self.assertEqual(schedule["2026-09-07"]["quantities"], final)
                event_slots = {}
                for key, plan in [("a", first_plan), ("b", second_plan)]:
                    event_slots[key] = {
                        "label": key,
                        "live": True,
                        "exit_forecast": [],
                        "entries": [{"ticker": "SAME", "fill_date": targets[key][0]["fill_date"]}]
                        if plan == "buy"
                        else [],
                        "sells": [{"ticker": "SAME", "fill_date": targets[key][0]["fill_date"], "reason": "이탈"}]
                        if plan == "sell"
                        else [],
                    }
                rows = [
                    {
                        "ticker": "SAME",
                        "price": 10.0,
                        "held_quantity": held_qty,
                        "target_quantity": final.get("SAME", 0),
                        "weight_pct": 0,
                    }
                ]
                groups = build_action_groups(
                    rows, {"slots": event_slots}, "2026-09-07", target_schedule=schedule, currency="USD"
                )
                orders = [item for group in groups for item in group["items"]]
                self.assertEqual(
                    [item["quantity"] * (1 if item["side"] == "buy" else -1) for item in orders], expected_trades
                )
                self.assertEqual([item["date"] for item in orders], ["2026-09-04", "2026-09-07"])
                self.assertEqual(len({item["key"] for item in orders}), 2)
                self.assertNotIn("예상", orders[0]["text"])
                self.assertIn("예상", orders[1]["text"])
                self.assertEqual(held_qty + sum(expected_trades), final.get("SAME", 0))
                if first_plan == second_plan == "buy":
                    complete = [{**rows[0], "held_quantity": final["SAME"]}]
                    self.assertEqual(
                        build_action_groups(
                            complete, {"slots": event_slots}, "2026-09-07", target_schedule=schedule, currency="USD"
                        ),
                        [],
                    )

                # 앞선 목표를 이미 맞췄다면 다음 날짜 주문만 남고 키·수량도 유지된다.
                rows[0]["held_quantity"] = schedule["2026-09-04"]["quantities"].get("SAME", 0)
                remaining = build_action_groups(
                    rows, {"slots": event_slots}, "2026-09-07", target_schedule=schedule, currency="USD"
                )
                remaining_items = [item for group in remaining for item in group["items"]]
                self.assertEqual(len(remaining_items), 1)
                self.assertEqual(remaining_items[0]["key"], orders[1]["key"])
                self.assertEqual(remaining_items[0]["quantity"], orders[1]["quantity"])

        # 미래 진입 종목을 미리 보유해도 다른 종목의 오늘 주문 때문에 왕복매매를 만들지 않는다.
        future_actions = {
            "slots": {
                "a": {
                    "label": "A",
                    "live": True,
                    "sells": [],
                    "exit_forecast": [],
                    "entries": [{"ticker": "TODAY", "fill_date": "2026-09-08"}, {"ticker": "139260"}],
                }
            }
        }
        future_schedule = {
            "2026-09-08": {"quantities": {"TODAY": 1}, "weights": {"TODAY": 10}},
            "2026-09-09": {"quantities": {"TODAY": 1, "139260": 34}, "weights": {"TODAY": 10, "139260": 10}},
        }
        for already_held, difference in [(34, 0), (32, 2), (36, -2), (0, 34)]:
            with self.subTest(already_held=already_held):
                rows = [
                    {"ticker": "TODAY", "price": 10, "held_quantity": 0, "target_quantity": 1},
                    {"ticker": "139260", "price": 100, "held_quantity": already_held, "target_quantity": 34},
                ]
                groups = build_action_groups(
                    rows, future_actions, "2026-09-09", target_schedule=future_schedule, currency="KRW"
                )
                orders = [item for group in groups for item in group["items"] if item["ticker"] == "139260"]
                if difference == 0:
                    self.assertEqual(orders, [])
                else:
                    self.assertEqual(len(orders), 1)
                    self.assertEqual(orders[0]["date"], "2026-09-09")
                    self.assertEqual(orders[0]["quantity"], abs(difference))
                    self.assertEqual(orders[0]["side"], "buy" if difference > 0 else "sell")

        # 같은 날짜의 반대 방향은 목표 합산에서 상계하며, 순변화 0이면 액션이 없다.
        same_day = {
            "a": [{"ticker": "SAME", "price": 10, "drift_pct": 100, "plan": "buy", "is_exiting": False}],
            "b": [{"ticker": "SAME", "price": 10, "drift_pct": 100, "plan": "sell", "is_exiting": True}],
        }
        schedule = dated_target_shares(same_day, {"a": 20, "b": 20}, 1, 40, "2026-09-07")
        self.assertEqual(
            build_action_groups(
                [{"ticker": "SAME", "price": 10, "held_quantity": 2, "target_quantity": 2}],
                {"slots": {}},
                "2026-09-07",
                target_schedule=schedule,
                currency="USD",
            ),
            [],
        )
        # 슬리브별로 반올림하지 않고 날짜마다 중복 종목을 합친 뒤 기존 정수 배분을 쓴다.
        for key, target in same_day.items():
            target[0].update(price=7, plan="buy", is_exiting=False, fill_date="2026-09-04" if key == "a" else None)
        schedule = dated_target_shares(same_day, {"a": 20, "b": 30}, 1, 50, "2026-09-07")
        self.assertEqual(schedule["2026-09-04"]["quantities"], {"SAME": 2})
        self.assertEqual(schedule["2026-09-07"]["quantities"], {"SAME": 7})

        # 초과 보유 허용은 화면 액션의 예외이며 엔진에서 받은 목표·표의 목표는 바꾸지 않는다.
        from copy import deepcopy

        original_schedule = deepcopy(schedule)
        rows = [{"ticker": "SAME", "price": 7, "held_quantity": 9, "target_quantity": 7}]
        original_rows = deepcopy(rows)
        groups = build_action_groups(
            rows,
            {"slots": {}},
            "2026-09-04",
            target_schedule=schedule,
            excess_holding_allowance=14,
            currency="USD",
        )
        self.assertEqual(schedule, original_schedule)
        self.assertEqual(rows, original_rows)
        orders = [item for group in groups for item in group["items"]]
        # 첫날 목표 2주+허용 2주로 5주 매도, 다음 목표 7주에는 부족한 3주 전부 매수.
        self.assertEqual([(item["side"], item["quantity"]) for item in orders], [("sell", 5), ("buy", 3)])

    def test_mark_engine_statuses_labels_without_judging(self):
        from core.strategy.intraday import mark_engine_statuses

        rows = [{"ticker": "PEND", "fill_date": "2026-09-04"}, {"ticker": "PLAN"}, {"ticker": "KEEP"}]
        mark_engine_statuses(rows, ["PLAN"])
        # 체결 예정 매도는 확정 'sell'(예상 아님), 잠정 매도 예정은 예상 표시, 나머지는 유지.
        self.assertEqual((rows[0]["status"], rows[0]["is_exit_forecast"]), ("sell", False))
        self.assertEqual(
            (rows[1]["status"], rows[1]["exit_reason"], rows[1]["is_exit_forecast"]), ("sell", "이탈", True)
        )
        self.assertEqual(
            (rows[2]["status"], rows[2]["exit_reason"], rows[2]["is_exit_forecast"]), ("hold", None, False)
        )


if __name__ == "__main__":
    unittest.main()
