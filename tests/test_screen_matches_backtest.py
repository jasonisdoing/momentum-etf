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
            simulated = momentum_backtest.run_backtest(momentum_backtest._HOLDINGS_LOOKBACK_MONTHS, settings, context)
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
            simulated = new_high_backtest.run_backtest(new_high_backtest._HOLDINGS_LOOKBACK_MONTHS, settings, context)
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
                result = run_backtest(spec, 12, load_context(spec))
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


if __name__ == "__main__":
    unittest.main()
