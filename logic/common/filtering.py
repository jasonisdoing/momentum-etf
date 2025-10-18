"""추천과 백테스트에서 공통으로 사용하는 필터링 로직."""

from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable

from utils.logger import get_app_logger

logger = get_app_logger()


def _resolve_category(
    ticker: str,
    etf_meta: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[str], str]:
    """주어진 티커의 카테고리를 반환합니다.

    반환값은 (표시용 카테고리, 내부 키) 튜플입니다. 카테고리가 없거나 'TBD'라면
    내부 키는 티커 기반의 고유 값으로 대체하여 카테고리 충돌을 방지합니다.
    """

    raw_category = None
    meta = etf_meta.get(ticker)
    if isinstance(meta, dict):
        raw_category = meta.get("category")

    if raw_category is not None:
        raw_category = str(raw_category).strip()

    if raw_category and raw_category.upper() != "TBD":
        return raw_category, raw_category

    # 카테고리가 없거나 'TBD'라면, 티커 기반의 내부 키를 만들어 카테고리 중복을 방지합니다.
    internal_key = f"__i_{ticker.upper()}"
    return None, internal_key


def select_candidates_by_category(
    candidates: Iterable[Dict[str, Any]],
    etf_meta: Dict[str, Dict[str, Any]],
    *,
    held_categories: Optional[Iterable[str]] = None,
    max_count: Optional[int] = None,
    skip_held_categories: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Tuple[Dict[str, Any], str]]]:
    """카테고리 중복을 허용하지 않는 매수 후보 목록을 선택합니다.

    Args:
        candidates: 최소한 ``tkr`` 와 ``score`` 키를 가진 후보 딕셔너리 iterable.
        etf_meta: 티커별 메타데이터(카테고리 등) 딕셔너리.
        held_categories: 이미 보유 중인 카테고리 목록. ``skip_held_categories`` 가
            True일 때만 사용됩니다.
        max_count: 선택할 최대 후보 수. None이면 제한 없음.
        skip_held_categories: True일 경우, 현재 보유 중인 카테고리를 가진 후보를
            제외합니다. False이면 카테고리 보유 여부와 무관하게 최고 점수 후보를
            반환합니다.

    Returns:
        (선택된_후보_리스트, 제외된_후보와_사유_리스트)
    """
    # 항상 튜플을 반환하기 위한 기본값
    selected_candidates = []
    rejected = []

    try:
        # 보유 중인 카테고리 집합 생성
        held_set = set()
        if skip_held_categories and held_categories:
            held_set = {str(cat).strip().upper() for cat in held_categories if cat and str(cat).strip().upper() != "TBD"}

        # 후보 처리
        best_per_category = {}
        for cand in candidates:
            if not isinstance(cand, dict):
                continue

            ticker = str(cand.get("tkr") or cand.get("ticker") or "").strip()
            if not ticker:
                continue

            # 점수 추출 (composite_score 우선, 없으면 score 사용)
            composite_score_raw = cand.get("composite_score")
            score_raw = cand.get("score")
            try:
                composite_score_val = float(composite_score_raw) if composite_score_raw is not None else float("-inf")
                score_val = float(score_raw) if score_raw is not None else float("-inf")
                # composite_score가 있으면 우선 사용, 없으면 score 사용
                final_score_val = composite_score_val if composite_score_val > float("-inf") else score_val
            except (TypeError, ValueError):
                final_score_val = float("-inf")

            # 카테고리 확인
            category, internal_key = _resolve_category(ticker, etf_meta)
            existing = best_per_category.get(internal_key)

            if existing is None:
                # 새로운 카테고리인 경우
                if category not in held_set:
                    best_per_category[internal_key] = {
                        "cand": cand,
                        "score": final_score_val,
                        "category": category,
                    }
                elif skip_held_categories:
                    rejected.append((cand, "category_held"))
            else:
                # 기존 카테고리와 비교 (종합 점수 우선)
                if final_score_val > existing["score"]:
                    if existing["category"] not in held_set:
                        rejected.append((existing["cand"], "better_candidate"))
                        best_per_category[internal_key] = {
                            "cand": cand,
                            "score": final_score_val,
                            "category": category,
                        }
                    else:
                        rejected.append((cand, "category_held"))
                else:
                    rejected.append((cand, "better_candidate"))

        # 최종 선택된 후보 목록 생성
        selected_candidates = [item["cand"] for item in best_per_category.values() if item["category"] not in held_set or not skip_held_categories]

        # 점수 순으로 정렬
        selected_candidates.sort(key=lambda x: x.get("score", float("-inf")), reverse=True)

        # 최대 개수 제한
        if max_count is not None and max_count > 0:
            selected_candidates = selected_candidates[:max_count]

        # 제외된 후보 중에서 선택되지 않은 후보들만 남김
        rejected = [(cand, reason) for cand, reason in rejected if cand not in selected_candidates]

        return selected_candidates, rejected

    except Exception as e:
        logger.exception("select_candidates_by_category 실행 중 오류: %s", e)
        return [], []


def sort_decisions_by_order_and_score(decisions: List[Dict[str, Any]]) -> None:
    """DECISION_CONFIG의 order 순으로 정렬하고, 같은 order 내에서는 composite_score > score 역순으로 정렬합니다.

    백테스트와 추천에서 공통으로 사용되는 정렬 함수입니다.
    """
    from strategies.maps.constants import DECISION_CONFIG

    def sort_key(item_dict):
        state = item_dict["state"]
        composite_score = item_dict.get("composite_score", 0.0)
        score = item_dict.get("score", 0.0)
        ticker = item_dict.get("ticker") or item_dict.get("tkr", "")
        order = DECISION_CONFIG.get(state, {}).get("order", 99)
        # composite_score 우선, 그 다음 MAPS score, 마지막으로 ticker
        return (order, -composite_score, -score, ticker)

    decisions.sort(key=sort_key)


def filter_category_duplicates(
    items: List[Dict[str, Any]],
    *,
    category_key_getter: Callable[[str], Optional[str]],
) -> List[Dict[str, Any]]:
    """카테고리별 최고 점수 1개만 남기고 필터링합니다.

    교체 매매(SELL_REPLACE/BUY_REPLACE)는 2개 모두 표시합니다.

    Args:
        items: 필터링할 항목 리스트
        category_key_getter: 카테고리를 정규화하는 함수

    Returns:
        필터링된 항목 리스트
    """
    from logic.common import should_exclude_from_category_count

    filtered_results = []
    category_best_map = {}  # 카테고리별 최고 점수 종목 추적
    replacement_tickers = set()  # 교체 매매 관련 티커
    held_categories = set()  # 이미 보유/매수 중인 카테고리

    # 1단계: 교체 매매 티커 수집 및 보유 카테고리 수집
    for item in items:
        state = item.get("state", "")
        category = item.get("category", "")
        category_key = category_key_getter(category)

        if state in {"SELL_REPLACE", "BUY_REPLACE"}:
            ticker = item.get("ticker") or item.get("tkr")
            if ticker:
                replacement_tickers.add(ticker)

        # HOLD, BUY 상태의 카테고리 수집 (매도 예정 종목 제외)
        if not should_exclude_from_category_count(state) and state in {"HOLD", "BUY", "BUY_REPLACE"}:
            if category_key and category_key != "TBD":
                held_categories.add(category_key)

    # 2단계: 필터링
    for item in items:
        ticker = item.get("ticker") or item.get("tkr")
        category = item.get("category", "")
        state = item.get("state", "")
        # composite_score 우선, 없으면 score 사용
        composite_score = item.get("composite_score") or item.get("score", 0.0)

        # 교체 매매 관련 종목은 무조건 포함
        if ticker in replacement_tickers:
            filtered_results.append(item)
            continue

        # HOLD, BUY, SELL 상태는 무조건 포함
        if state in {"HOLD", "BUY", "BUY_REPLACE", "SELL_TREND", "SELL_REPLACE", "CUT_STOPLOSS", "SELL_RSI_OVERBOUGHT"}:
            filtered_results.append(item)
            # 매도 예정 종목은 category_best_map에 포함하지 않음 (WAIT 종목이 표시될 수 있도록)
            if not should_exclude_from_category_count(state):
                # HOLD, BUY 상태만 category_best_map에 추가
                category_key = category_key_getter(category)
                if category_key and category_key != "TBD":
                    # 기존 WAIT 종목보다 우선
                    if category_key in category_best_map:
                        existing_item = category_best_map[category_key]
                        if existing_item in filtered_results:
                            filtered_results.remove(existing_item)
                    category_best_map[category_key] = item
            continue

        # WAIT 상태: 카테고리별 최고 점수만 포함
        if state == "WAIT":
            category_key = category_key_getter(category)
            if not category_key or category == "TBD":
                # 카테고리가 없는 경우 모두 포함
                filtered_results.append(item)
            else:
                # 이미 보유 중인 카테고리는 제외 (단, 매도 예정 종목은 held_categories에서 이미 제거됨)
                if category_key in held_categories:
                    continue

                # 카테고리별 최고 점수 체크
                if category_key not in category_best_map:
                    category_best_map[category_key] = item
                    filtered_results.append(item)
                else:
                    # 이미 더 높은 점수가 있으면 제외
                    existing_item = category_best_map[category_key]
                    existing_score = existing_item.get("composite_score") or existing_item.get("score", 0.0)
                    if composite_score > existing_score:
                        # 기존 항목 제거하고 새 항목 추가
                        filtered_results.remove(category_best_map[category_key])
                        category_best_map[category_key] = item
                        filtered_results.append(item)
        else:
            # 기타 상태는 모두 포함
            filtered_results.append(item)

    return filtered_results


__all__ = [
    "select_candidates_by_category",
    "sort_decisions_by_order_and_score",
    "filter_category_duplicates",
]
