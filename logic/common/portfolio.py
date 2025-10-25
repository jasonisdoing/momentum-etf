"""포트폴리오 추천 및 백테스트에서 공통으로 사용하는 헬퍼 함수들."""

from typing import Dict, Set, Any, List
from utils.logger import get_app_logger

logger = get_app_logger()


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


def validate_core_holdings(
    core_holdings_tickers: Set[str],
    universe_tickers_set: Set[str],
    account_id: str = "",
) -> Set[str]:
    """핵심 보유 종목 유효성 검증

    Args:
        core_holdings_tickers: 핵심 보유 종목 티커 집합
        universe_tickers_set: Universe 티커 집합
        account_id: 계좌 ID (로깅용)

    Returns:
        유효한 핵심 보유 종목 티커 집합
    """
    invalid_core_tickers = core_holdings_tickers - universe_tickers_set
    if invalid_core_tickers:
        account_prefix = f"[{account_id.upper()}] " if account_id else ""
        logger.warning(f"{account_prefix}CORE_HOLDINGS에 Universe에 없는 종목이 포함됨: {invalid_core_tickers}")

    valid_core_holdings = core_holdings_tickers & universe_tickers_set
    if valid_core_holdings:
        account_prefix = f"[{account_id.upper()}] " if account_id else "[백테스트] "
        logger.info(f"{account_prefix}핵심 보유 종목 (TOPN 포함): {sorted(valid_core_holdings)}")

    return valid_core_holdings


def check_buy_candidate_filters(
    ticker: str,
    category: str,
    held_categories: Set[str],
    sell_rsi_categories_today: Set[str],
    rsi_score: float,
    rsi_sell_threshold: float,
) -> tuple[bool, str]:
    """매수 후보 필터링 체크

    Args:
        ticker: 종목 티커
        category: 종목 카테고리
        held_categories: 현재 보유 카테고리 집합
        sell_rsi_categories_today: 오늘 RSI 매도한 카테고리 집합
        rsi_score: RSI 점수
        rsi_sell_threshold: RSI 매도 임계값

    Returns:
        (통과 여부, 차단 사유)
    """
    # 카테고리 중복 체크
    if category and category != "TBD" and category in held_categories:
        return False, f"카테고리 중복 ({category})"

    # SELL_RSI로 매도한 카테고리는 같은 날 매수 금지
    if category and category != "TBD" and category in sell_rsi_categories_today:
        return False, f"RSI 과매수 매도 카테고리 ({category})"

    # RSI 과매수 종목 매수 차단
    if rsi_score <= rsi_sell_threshold:
        return False, f"RSI 과매수 (RSI점수: {rsi_score:.1f})"

    return True, ""


def calculate_buy_budget(
    cash: float,
    current_holdings_value: float,
    top_n: int,
    risk_off_effective: bool = False,
    risk_off_equity_ratio: float = 1.0,
) -> float:
    """매수 예산 계산

    Args:
        cash: 현금
        current_holdings_value: 현재 보유 자산 가치
        top_n: 최대 보유 종목 수
        risk_off_effective: 리스크 오프 모드 활성화 여부
        risk_off_equity_ratio: 리스크 오프 목표 주식 비중 (0~1)

    Returns:
        매수 예산
    """
    equity_base = cash + current_holdings_value
    min_val = 1.0 / (top_n * 2.0) * equity_base
    max_val = 1.0 / top_n * equity_base
    budget = min(max_val, cash)

    if budget <= 0 or budget < min_val:
        return 0.0

    # 리스크 오프 모드에서 비중 제한
    if risk_off_effective:
        total_equity_now = cash + current_holdings_value
        target_holdings_limit = total_equity_now * risk_off_equity_ratio
        remaining_capacity = max(0.0, target_holdings_limit - current_holdings_value)
        if remaining_capacity <= 0:
            return 0.0
        budget = min(budget, remaining_capacity)

    return budget


def calculate_held_categories(
    position_state: Dict,
    ticker_to_category: Dict[str, str],
) -> Set[str]:
    """현재 보유 중인 카테고리 집합 계산

    Args:
        position_state: 포지션 상태 (백테스트용)
        ticker_to_category: 티커 -> 카테고리 매핑

    Returns:
        보유 중인 카테고리 집합
    """
    held_categories = set()
    for ticker, state in position_state.items():
        if state.get("shares", 0) > 0:
            category = ticker_to_category.get(ticker)
            if category and category != "TBD":
                held_categories.add(category)
    return held_categories


def track_sell_rsi_categories(
    decisions: List[Dict],
    etf_meta: Dict[str, Any],
    rsi_sell_threshold: float,
) -> Set[str]:
    """SELL_RSI로 매도하는 카테고리 추적

    Args:
        decisions: 의사결정 리스트
        etf_meta: ETF 메타 정보
        rsi_sell_threshold: RSI 매도 임계값

    Returns:
        SELL_RSI로 매도하는 카테고리 집합
    """
    sell_rsi_categories = set()

    for d in decisions:
        # 1. 이미 SELL_RSI 상태인 경우
        if d.get("state") == "SELL_RSI":
            category = etf_meta.get(d["tkr"], {}).get("category")
            if category and category != "TBD":
                sell_rsi_categories.add(category)
        # 2. 보유 중이지만 RSI 과매수 경고가 있는 경우 (매도 전 예방)
        elif d.get("state") in {"HOLD", "HOLD_CORE"} and d.get("rsi_score", 100.0) <= rsi_sell_threshold:
            category = etf_meta.get(d["tkr"], {}).get("category")
            if category and category != "TBD":
                sell_rsi_categories.add(category)

    return sell_rsi_categories


def calculate_held_count(position_state: Dict) -> int:
    """현재 보유 중인 종목 수 계산 (백테스트용)

    Args:
        position_state: 포지션 상태 딕셔너리

    Returns:
        보유 중인 종목 수
    """
    return sum(1 for pos in position_state.values() if pos.get("shares", 0) > 0)


def calculate_held_categories_from_holdings(
    holdings: Dict[str, Any],
    etf_meta: Dict[str, Any],
) -> Set[str]:
    """보유 종목의 카테고리 집합 계산 (추천용)

    Args:
        holdings: 보유 종목 딕셔너리
        etf_meta: ETF 메타 정보

    Returns:
        보유 중인 카테고리 집합
    """
    held_categories = set()
    for tkr in holdings.keys():
        category = etf_meta.get(tkr, {}).get("category")
        if category and category != "TBD":
            held_categories.add(category)
    return held_categories


def validate_portfolio_topn(topn: int, account_id: str = "") -> None:
    """포트폴리오 최대 보유 종목 수 검증

    Args:
        topn: 최대 보유 종목 수
        account_id: 계정 ID (선택)

    Raises:
        ValueError: topn이 0 이하인 경우
    """
    if topn <= 0:
        if account_id:
            raise ValueError(f"'{account_id}' 계정의 PORTFOLIO_TOPN은 0보다 커야 합니다.")
        else:
            raise ValueError("PORTFOLIO_TOPN은 0보다 커야 합니다.")
