"""포트폴리오 추천 및 백테스트에서 공통으로 사용하는 헬퍼 함수들."""

from utils.logger import get_app_logger

logger = get_app_logger()


def get_sell_states() -> set[str]:
    """매도 상태 집합을 반환합니다.

    Returns:
        매도 상태 문자열 집합
    """
    return {"SELL_TREND", "SELL_REPLACE"}


def get_hold_states() -> set[str]:
    """보유 상태 집합을 반환합니다 (매도 예정 포함).

    물리적으로 보유 중인 종목의 상태를 반환합니다.
    매도 신호가 발생했더라도 아직 매도 체결 전이면 보유 중으로 간주합니다.

    Returns:
        보유 상태 문자열 집합
    """
    return {
        "HOLD",
        "SELL_TREND",
        "SELL_REPLACE",
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


def calculate_buy_budget(
    cash: float,
    current_holdings_value: float,
    top_n: int,
) -> float:
    """총자산/TOPN 기준으로 균등 비중 매수 예산을 계산합니다.

    Args:
        cash: 현재 보유 현금
        current_holdings_value: 현재 보유 자산 가치
        top_n: 목표 포트폴리오 종목 수

    Returns:
        매수 예산 (총자산 / TOPN, 단 보유 현금 한도를 넘지 않음)
    """
    if cash <= 0 or top_n <= 0:
        return 0.0

    total_equity = cash + max(current_holdings_value, 0.0)
    if total_equity <= 0:
        return 0.0

    target_value = total_equity / top_n
    if target_value <= 0:
        return 0.0

    return min(target_value, cash)


def calculate_held_count(position_state: dict) -> int:
    """현재 보유 중인 종목 수 계산 (백테스트용)

    Args:
        position_state: 포지션 상태 딕셔너리

    Returns:
        보유 중인 종목 수
    """
    return sum(1 for pos in position_state.values() if pos.get("shares", 0) > 0)


def validate_bucket_topn(topn: int, account_id: str = "") -> None:
    """포트폴리오 최대 보유 종목 수 검증

    Args:
        topn: 최대 보유 종목 수
        account_id: 계정 ID (선택)

    Raises:
        ValueError: topn이 0 이하인 경우
    """
    if topn <= 0:
        if account_id:
            raise ValueError(f"'{account_id}' 계정의 BUCKET_TOPN은 0보다 커야 합니다.")
        else:
            raise ValueError("BUCKET_TOPN은 0보다 커야 합니다.")
