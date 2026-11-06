"""백테스트 시작 자본 — 슬롯 엔진이 "1주"를 셀 기준 금액으로 쓴다.

목표 금액을 정수 주수로 나누던 배분(`allocate_integer_shares`)은 합성이 고정 기준금액
회수·채우기로 바뀌며 폐기했다(2026-09) — 판정은 `core/strategy/mix/capital_policy.py` 한 곳이다.
"""

from __future__ import annotations

from config import BACKTEST_INITIAL_CAPITAL


def backtest_initial_capital(pool: str) -> float:
    """종목풀의 백테스트 시작 자본(그 시장 통화). 통화를 모르면 에러 — 임의 기본값을 쓰지 않는다.

    백테스트가 정수 주수로 돌려면 "1주"를 셀 기준 금액이 있어야 한다. 계좌 잔고가 아니라
    통화별 상수를 쓰는 이유는 `config.BACKTEST_INITIAL_CAPITAL` 주석에 적어 두었다.
    """
    from utils.cash_model import currency_for_country
    from utils.settings_loader import get_ticker_type_settings

    country = str((get_ticker_type_settings(pool) or {}).get("country_code") or "").strip().lower()
    currency = currency_for_country(country)
    if currency not in BACKTEST_INITIAL_CAPITAL:
        raise ValueError(f"'{pool}'({currency}) 의 백테스트 시작 자본이 config 에 없습니다.")
    return float(BACKTEST_INITIAL_CAPITAL[currency])
