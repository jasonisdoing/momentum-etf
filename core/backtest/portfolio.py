"""포트폴리오 추천 및 백테스트에서 공통으로 사용하는 헬퍼 함수들."""


def get_hold_states() -> set[str]:
    """보유 상태 집합을 반환합니다 (매도 예정 포함).

    물리적으로 보유 중인 종목의 상태를 반환합니다.
    매도 신호가 발생했더라도 아직 매도 체결 전이면 보유 중으로 간주합니다.

    Returns:
        보유 상태 문자열 집합
    """
    return {
        "HOLD",
    }


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


def validate_universe_size(universe_size: int, account_id: str = "") -> None:
    """백테스트 대상 종목 수를 검증합니다."""
    if universe_size <= 0:
        if account_id:
            raise ValueError(f"'{account_id}' 계정의 백테스트 대상 종목 수는 0보다 커야 합니다.")
        raise ValueError("백테스트 대상 종목 수는 0보다 커야 합니다.")
