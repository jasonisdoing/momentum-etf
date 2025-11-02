"""포트폴리오 추천 및 백테스트에서 공통으로 사용하는 헬퍼 함수들."""

from typing import Dict, Set, Any, List, Iterable, Tuple, Optional

import pandas as pd

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
        # logger.info(f"{account_prefix}핵심 보유 종목 (TOPN 포함): {sorted(valid_core_holdings)}")

    return valid_core_holdings


def check_buy_candidate_filters(
    category: str,
    held_categories: Set[str],
    sell_rsi_categories_today: Set[str],
    rsi_score: float,
    rsi_sell_threshold: float,
) -> tuple[bool, str]:
    """매수 후보 필터링 체크

    Args:
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
    if rsi_score >= rsi_sell_threshold:
        return False, f"RSI 과매수 (RSI점수: {rsi_score:.1f})"

    return True, ""


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


def build_weekly_rebalance_cache(
    trading_days: Iterable[Any],
) -> Dict[Tuple[int, int], Optional[pd.Timestamp]]:
    """주간 리밸런싱을 위한 (연도, 주) -> 마지막 거래일 매핑을 생성합니다.

    Args:
        trading_days: 거래일 iterable (문자열, datetime, pandas Timestamp 등 허용)

    Returns:
        (iso_year, iso_week) -> 마지막 거래일(Timestamp) 딕셔너리
    """
    cache: Dict[Tuple[int, int], Optional[pd.Timestamp]] = {}
    if trading_days is None:
        return cache

    if isinstance(trading_days, (pd.Series, pd.Index)):
        if len(trading_days) == 0:
            return cache
        iterable: Iterable[Any] = trading_days
    else:
        # list(...) to support generators/iterators and allow length check without ambiguity
        materialized = list(trading_days)
        if len(materialized) == 0:
            return cache
        iterable = materialized

    for raw_dt in iterable:
        if raw_dt is None:
            continue

        try:
            dt = pd.Timestamp(raw_dt)
        except Exception:
            continue

        if pd.isna(dt):
            continue

        normalized = dt.normalize()
        iso_year, iso_week, _ = normalized.isocalendar()
        key = (int(iso_year), int(iso_week))

        existing = cache.get(key)
        if existing is None or normalized > existing:
            cache[key] = normalized

    return cache


def calculate_held_categories(
    position_state: Dict,
    ticker_to_category: Dict[str, str],
    core_holdings: Set[str] = None,
) -> Set[str]:
    """현재 보유 중인 카테고리 집합 계산 (고정 종목 포함)

    Args:
        position_state: 포지션 상태 (백테스트용)
        ticker_to_category: 티커 -> 카테고리 매핑
        core_holdings: 고정 종목 티커 집합 (선택)

    Returns:
        보유 중인 카테고리 집합 (고정 종목 카테고리 포함)
    """
    held_categories = set()

    # 실제 보유 종목의 카테고리
    for ticker, state in position_state.items():
        if state.get("shares", 0) > 0:
            category = ticker_to_category.get(ticker)
            if category and category != "TBD":
                held_categories.add(category)

    # 고정 종목의 카테고리도 추가 (미보유 시에도 카테고리 차단)
    if core_holdings:
        for ticker in core_holdings:
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
        elif d.get("state") in {"HOLD", "HOLD_CORE"} and d.get("rsi_score", 0.0) >= rsi_sell_threshold:
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
    core_holdings: Set[str] = None,
) -> Set[str]:
    """보유 종목의 카테고리 집합 계산 (추천용, 고정 종목 포함)

    Args:
        holdings: 보유 종목 딕셔너리
        etf_meta: ETF 메타 정보
        core_holdings: 고정 종목 티커 집합 (선택)

    Returns:
        보유 중인 카테고리 집합 (고정 종목 카테고리 포함)
    """
    held_categories = set()

    # 실제 보유 종목의 카테고리
    for tkr in holdings.keys():
        category = etf_meta.get(tkr, {}).get("category")
        if category and category != "TBD":
            held_categories.add(category)

    # 고정 종목의 카테고리도 추가 (미보유 시에도 카테고리 차단)
    if core_holdings:
        for tkr in core_holdings:
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
