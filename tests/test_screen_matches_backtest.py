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
    if not isinstance(error, unavailable):
        raise error
    raise unittest.SkipTest(f"가격 캐시·DB 를 읽을 수 없어 건너뜁니다: {type(error).__name__}: {error}")


def _positions_of(holdings: list[dict[str, Any]]) -> dict[str, str]:
    """보유 목록을 {티커: 편입일} 로 — 비교에 쓰는 최소 형태."""
    return {str(row["ticker"]): str(row["entry_date"]) for row in holdings}


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


class NewHighScreenMatchesBacktest(unittest.TestCase):
    """신고가 운용 현황의 보유 = 백테스트를 현재까지 돌린 마지막 상태."""

    def test_holdings_come_from_the_backtest(self) -> None:
        _load_env()
        from utils import new_high_backtest
        from utils.new_high_service import load_settings

        try:
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


class MixScreenMatchesSleeveBacktests(unittest.TestCase):
    """합성 화면의 목표 종목 = 각 슬리브 백테스트가 내는 **다음 시가 이후의 보유**.

    보유에서 매도 예정을 빼고 진입 예정을 더한 것이다 — 합성 목표는 '내일 이후 들고 있어야
    할 목록'이기 때문이다. 판정은 전부 엔진이 하고 합성은 읽기만 해야 한다.
    """

    def test_target_tickers_come_from_sleeve_backtests(self) -> None:
        _load_env()
        from utils.mix_sleeve import PORTFOLIO, load_context, run_backtest
        from utils.strategy_mix_service import _resolve_mix_account, mix_positions

        try:
            ctx = _resolve_mix_account(MIX_ACCOUNT)
            expected: set[str] = set()
            for spec in ctx["slots"]:
                if spec.strategy == PORTFOLIO:
                    result = run_backtest(spec, 12, start_date=spec.settings["start_date"])
                    expected |= {row["ticker"] for row in result["open_positions"] if row["sleeve_weight_pct"] > 0}
                    continue
                result = run_backtest(spec, 12, load_context(spec), start_date=spec.settings["start_date"])
                held = {str(row["ticker"]) for row in result["open_positions"]}
                expected |= (held - set(result["planned_exits"])) | set(result["planned_entries"])
            screen = mix_positions(MIX_ACCOUNT)
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


class PortfolioMixStateTest(unittest.TestCase):
    def test_screen_and_mix_match_portfolio_engine(self):
        dates = ["2026-01-02", "2026-02-02", "2026-02-03"]
        a_prices, rebalance, band, cash = [100, 200, 220], "monthly", 3, 20
        index = pd.to_datetime(dates)
        frame = pd.DataFrame({"A": a_prices, "B": [100.0] * len(dates)}, index=index)
        settings = {
            "pool": "test",
            "start_date": dates[0],
            "rebalance": rebalance,
            "band_pct": band,
            "cash_weight_pct": cash,
            "weights": [{"ticker": t, "weight_pct": (100 - cash) / 2} for t in ("A", "B")],
        }
        with (
            patch("utils.portfolio_backtest.validate_settings", side_effect=lambda s: s),
            patch("utils.portfolio_backtest._load_close_frame", return_value=frame),
            patch("utils.portfolio_backtest.get_pool_slippage", return_value=(0, 0)),
            patch("utils.benchmark_curve.load_benchmark_frame", return_value=pd.DataFrame({"Close": 100}, index=index)),
            patch("utils.benchmark_curve.benchmark_growth", side_effect=lambda pool, ix: pd.Series(1.0, index=ix)),
            patch("utils.portfolio_backtest.benchmark_info", return_value={"name": "기준"}),
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
        self.assertAlmostEqual(sum(t["drift_pct"] for t in state.targets) + result["sleeve_cash_weight_pct"], 100)


class MixRebalanceMatchesBacktest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
