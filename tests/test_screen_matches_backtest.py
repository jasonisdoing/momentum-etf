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
                    # 포트폴리오는 판정이 없다 — 저장된 비중이 곧 목표다.
                    expected |= {str(row["ticker"]).strip() for row in (spec.settings.get("weights") or [])}
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
        self._assert_targets_are_consistent(screen)

    def _assert_targets_are_consistent(self, screen: dict[str, Any]) -> None:
        """목표 주수·금액·비중이 서로 맞는지 — 종목만 같고 수량이 틀리는 회귀를 잡는다.

        실제로 이 세 가지가 어긋난 적이 있다. 두 슬리브가 같은 종목을 담을 때 목표 주수가
        합산되지 않고 덮어써졌고(보유 228주에서 115주를 팔라는 지시가 났다), 고정 자산이
        비중 합계에서 빠져 목표가 27%p 낮게 보였다. 둘 다 티커 집합은 같아 위 비교를 통과했다.
        """
        total = float(screen["account"]["total_assets"])
        self.assertGreater(total, 0)

        weight_sum = 0.0
        for row in screen["holdings"]:
            weight = row.get("actual_weight_pct")
            if weight is not None:
                weight_sum += float(weight)
            quantity, amount, price = row.get("target_quantity"), row.get("target_amount"), row.get("price")
            if quantity is None or not price:
                continue
            # 목표 금액은 **주문할 금액**이라 목표 주수와 1주 값의 곱이어야 한다.
            self.assertAlmostEqual(
                float(amount) / total * 100.0,
                float(weight),
                places=2,
                msg=f"{row['ticker']} 의 목표 금액과 목표 비중이 다릅니다 — 서로 다른 기준으로 계산됐습니다.",
            )
        # 현금은 종목 행이 아니라 요약에 있다 — 빼면 합이 100 이 안 된다.
        weight_sum += float(screen["summary"]["actual_cash_pct"])
        self.assertAlmostEqual(
            weight_sum,
            100.0,
            places=1,
            msg="목표 비중의 합이 100% 가 아닙니다 — 표에서 빠진 몫이 있습니다(고정 자산·현금 행 확인).",
        )


if __name__ == "__main__":
    unittest.main()
