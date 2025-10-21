"""포트폴리오 추천 및 백테스트에서 공통으로 사용하는 헬퍼 함수들."""

from typing import Dict, Set, Any


def get_held_categories_excluding_sells(
    items: list,
    *,
    get_category_func,
    get_state_func,
    get_ticker_func=None,
    holdings: Set[str] = None,
) -> Set[str]:
    """매도 예정 종목을 제외한 보유 카테고리 집합을 반환합니다.

    추천과 백테스트 모두에서 사용되는 공통 로직입니다.

    Args:
        items: 종목 리스트 (dict 또는 state 객체)
        get_category_func: 카테고리를 추출하는 함수 (item -> str)
        get_state_func: 상태를 추출하는 함수 (item -> str)
        get_ticker_func: 티커를 추출하는 함수 (item -> str), 옵션
        holdings: 보유 종목 티커 집합, 옵션

    Returns:
        매도 예정이 아닌 보유 종목의 카테고리 집합
    """
    sell_states = {"SELL_TREND", "SELL_REPLACE", "CUT_STOPLOSS", "SELL_RSI"}
    held_categories = set()

    for item in items:
        state = get_state_func(item)

        # 매도 예정 종목은 제외
        if state in sell_states:
            continue

        # HOLD 또는 HOLD_CORE 상태이거나, 보유 중인 종목만 포함
        is_held = False
        if state in {"HOLD", "HOLD_CORE"}:
            is_held = True
        elif holdings and get_ticker_func:
            ticker = get_ticker_func(item)
            is_held = ticker in holdings

        if is_held or state in {"BUY", "BUY_REPLACE"}:
            category = get_category_func(item)
            if category and category != "TBD":
                held_categories.add(category)

    return held_categories


def should_exclude_from_category_count(state: str) -> bool:
    """카테고리 카운트에서 제외해야 하는 상태인지 확인합니다.

    Args:
        state: 종목 상태

    Returns:
        True if 매도 예정 종목 (카운트에서 제외), False otherwise
    """
    sell_states = {"SELL_TREND", "SELL_REPLACE", "CUT_STOPLOSS", "SELL_RSI"}
    return state in sell_states


def get_sell_states() -> Set[str]:
    """매도 상태 집합을 반환합니다.

    Returns:
        매도 상태 문자열 집합
    """
    return {"SELL_TREND", "SELL_REPLACE", "CUT_STOPLOSS", "SELL_RSI"}


def get_hold_states() -> Set[str]:
    """보유 상태 집합을 반환합니다 (매도 예정 포함).

    물리적으로 보유 중인 종목의 상태를 반환합니다.
    매도 신호가 발생했더라도 아직 매도 체결 전이면 보유 중으로 간주합니다.

    Returns:
        보유 상태 문자열 집합
    """
    return {"HOLD", "HOLD_CORE", "SELL_TREND", "SELL_REPLACE", "CUT_STOPLOSS", "SELL_RSI"}


def count_current_holdings(items: list, *, get_state_func=None) -> int:
    """현재 물리적으로 보유 중인 종목 수를 계산합니다.

    매도 예정 종목도 아직 체결 전이면 보유 중으로 카운트합니다.

    Args:
        items: 종목 리스트 (dict 또는 state 객체)
        get_state_func: 상태를 추출하는 함수 (item -> str), 없으면 item["state"] 또는 item.get("state") 사용

    Returns:
        현재 보유 중인 종목 수
    """
    hold_states = get_hold_states()

    if get_state_func:
        return sum(1 for item in items if get_state_func(item) in hold_states)
    else:
        # dict 형태의 item인 경우
        return sum(1 for item in items if isinstance(item, dict) and str(item.get("state", "")).upper() in hold_states)
