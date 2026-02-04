"""추천과 백테스트에서 공통으로 사용하는 필터링 로직."""

from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

import config
from logic.backtest.portfolio import is_category_exception
from utils.logger import get_app_logger

logger = get_app_logger()


def _resolve_category(
    ticker: str,
    etf_meta: dict[str, dict[str, Any]],
) -> tuple[str | None, str]:
    """주어진 티커의 카테고리를 반환합니다.

    반환값은 (표시용 카테고리, 내부 키) 튜플입니다. 카테고리가 없거나 CATEGORY_EXCEPTIONS 라면
    내부 키는 티커 기반의 고유 값으로 대체하여 카테고리 충돌을 방지합니다.
    """

    raw_category = None
    meta = etf_meta.get(ticker)
    if isinstance(meta, dict):
        raw_category = meta.get("category")

    if raw_category is not None:
        raw_category = str(raw_category).strip()

    if raw_category and not is_category_exception(raw_category):
        return raw_category, raw_category

    # 카테고리가 없거나 예외 카테고리라면, 티커 기반의 내부 키를 만들어 카테고리 중복을 방지합니다.
    # 원본 카테고리명은 유지하여 is_category_exception 체크가 작동하도록 합니다.
    internal_key = f"__i_{ticker.upper()}"
    return raw_category, internal_key


def select_candidates_by_category(
    candidates: Iterable[dict[str, Any]],
    etf_meta: dict[str, dict[str, Any]],
    *,
    held_categories: Iterable[str] | None = None,
    max_count: int | None = None,
    skip_held_categories: bool = True,
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], str]]]:
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
            held_set = {
                str(cat).strip().upper()
                for cat in held_categories
                if cat and not is_category_exception(str(cat).strip())
            }

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
        selected_candidates = [
            item["cand"]
            for item in best_per_category.values()
            if item["category"] not in held_set or not skip_held_categories
        ]

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


def sort_decisions_by_order_and_score(decisions: list[dict[str, Any]]) -> None:
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
    items: list[dict[str, Any]],
    *,
    category_key_getter: Callable[[str], str | None],
) -> list[dict[str, Any]]:
    """카테고리별 최고 점수 1개만 남기고 필터링합니다.

    교체 매매(SELL_REPLACE/BUY_REPLACE)는 2개 모두 표시합니다.

    Args:
        items: 필터링할 항목 리스트
        category_key_getter: 카테고리를 정규화하는 함수

    Returns:
        필터링된 항목 리스트
    """
    from logic.backtest.portfolio import should_exclude_from_category_count

    category_best_map = {}  # 카테고리별 최고 점수 종목 추적
    replacement_tickers = set()  # 교체 매매 관련 티커
    held_categories = set()  # 이미 보유/매수 중인 카테고리
    wait_candidates_by_category: dict[str, list[tuple[float, int, dict[str, Any]]]] = defaultdict(list)
    max_per_category = getattr(config, "MAX_PER_CATEGORY", 1) or 1

    def _resolve_item_score(entry: dict[str, Any]) -> float:
        composite = entry.get("composite_score")
        score = entry.get("score")
        try:
            if composite is not None:
                return float(composite)
            if score is not None:
                return float(score)
            return float("-inf")
        except (TypeError, ValueError):
            return float("-inf")

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
            if category_key and not is_category_exception(category_key):
                held_categories.add(category_key)

    # 2단계: 필터링
    # 순서 보장을 위해 (original_index, item) 튜플로 저장
    filtered_entries: list[tuple[int, dict[str, Any]]] = []

    for order_idx, item in enumerate(items):
        ticker = item.get("ticker") or item.get("tkr")
        category = item.get("category", "")
        state = item.get("state", "")

        # 교체 매매 관련 종목은 무조건 포함
        if ticker in replacement_tickers:
            filtered_entries.append((order_idx, item))
            continue

        # HOLD, SELL 상태는 무조건 포함 (BUY는 제외하여 카테고리 체크 수행)
        if state in {
            "HOLD",
            "BUY_REPLACE",
            "SELL_TREND",
            "SELL_REPLACE",
            "CUT_STOPLOSS",
            "SELL_RSI",
            "SOLD",
        }:
            filtered_entries.append((order_idx, item))
            # 매도 예정 종목은 category_best_map에 포함하지 않음 (WAIT 종목이 표시될 수 있도록)
            if not should_exclude_from_category_count(state):
                # HOLD, BUY_REPLACE 상태만 category_best_map에 추가
                category_key = category_key_getter(category)
                if category_key and not is_category_exception(category_key):
                    # 기존 WAIT 종목만 제거 (HOLD/BUY 종목은 유지)
                    if category_key in category_best_map:
                        existing_entry = category_best_map[category_key]  # (idx, item)
                        existing_item = existing_entry[1]
                        existing_state = existing_item.get("state", "")
                        # WAIT 종목만 제거
                        if existing_state == "WAIT" and existing_entry in filtered_entries:
                            filtered_entries.remove(existing_entry)
                    category_best_map[category_key] = (order_idx, item)
            continue

        # BUY 상태: 카테고리 중복 체크 수행
        if state == "BUY":
            category_key = category_key_getter(category)
            # 카테고리가 없거나 예외 카테고리면 포함
            if not category_key or is_category_exception(category_key):
                filtered_entries.append((order_idx, item))
                continue
            # BUY는 항상 표시 (held_categories 체크 제거)
            filtered_entries.append((order_idx, item))
            held_categories.add(category_key)
            category_best_map[category_key] = (order_idx, item)
            continue

        category_key = category_key_getter(category)

        # 카테고리가 없거나 예외 카테고리는 그대로 표시
        if not category_key or is_category_exception(category_key):
            filtered_entries.append((order_idx, item))
            continue

        # 이미 보유 중인 카테고리는 숨김 (카테고리 중복 노출 대신)
        if category_key in held_categories:
            continue

        score_val = _resolve_item_score(item)
        wait_candidates_by_category[category_key].append((score_val, order_idx, item))

    selected_wait_items: list[tuple[float, int, dict[str, Any]]] = []
    for category_key, entries in wait_candidates_by_category.items():
        entries.sort(key=lambda entry: (-entry[0], entry[1]))
        limit = max(1, int(max_per_category))
        selected_wait_items.extend(entries[:limit])

    # 선택된 대기 항목 추가 (점수, 인덱스, 아이템) -> (인덱스, 아이템)
    filtered_entries.extend([(entry[1], entry[2]) for entry in selected_wait_items])

    # 원래 순서(인덱스)대로 정렬
    filtered_entries.sort(key=lambda x: x[0])

    return [entry[1] for entry in filtered_entries]


__all__ = [
    "select_candidates_by_category",
    "sort_decisions_by_order_and_score",
    "filter_category_duplicates",
]
