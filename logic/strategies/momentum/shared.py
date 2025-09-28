"""Momentum 전략 모듈에서 공통으로 사용하는 유틸리티."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


SIGNAL_TABLE_HEADERS: Sequence[str] = [
    "#",
    "티커",
    "종목명",
    "카테고리",
    "상태",
    "매수일자",
    "보유일",
    "현재가",
    "일간수익률",
    "보유수량",
    "금액",
    "누적수익률",
    "비중",
    "고점대비",
    "점수",
    "지속",
    "문구",
]


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

    held_set: Set[str] = set()
    if skip_held_categories and held_categories:
        for cat in held_categories:
            if not cat:
                continue
            cat_str = str(cat).strip()
            if cat_str and cat_str.upper() != "TBD":
                held_set.add(cat_str)

    best_per_category: Dict[str, Dict[str, Any]] = {}
    for cand in candidates:
        ticker = str(cand.get("tkr") or cand.get("ticker") or "").strip()
        if not ticker:
            continue

        score_raw = cand.get("score")
        try:
            score_val = float(score_raw)
        except (TypeError, ValueError):
            score_val = float("-inf")

        category, internal_key = _resolve_category(ticker, etf_meta)
        existing = best_per_category.get(internal_key)
        if existing is None:
            best_per_category[internal_key] = dict(cand, _category=category, _score=score_val)
        else:
            existing_score = existing.get("_score", float("-inf"))
            if score_val > existing_score:
                best_per_category[internal_key] = dict(cand, _category=category, _score=score_val)

    sorted_candidates = sorted(
        best_per_category.values(), key=lambda c: c.get("_score", float("-inf")), reverse=True
    )

    selected: List[Dict[str, Any]] = []
    rejected: List[Tuple[Dict[str, Any], str]] = []
    used_categories: Set[str] = set(held_set)

    for cand in sorted_candidates:
        category = cand.get("_category")
        score_val = cand.get("_score", float("-inf"))

        if skip_held_categories and category and category in used_categories:
            rejected.append((cand, "category_held"))
            continue

        if max_count is not None and len(selected) >= max_count:
            rejected.append((cand, "slot_limit"))
            continue

        # 선택된 후보는 내부용 키를 제거하고 반환합니다.
        selected_cand = dict(cand)
        selected_cand.pop("_category", None)
        selected_cand.pop("_score", None)
        selected.append(selected_cand)

        if category:
            used_categories.add(category)

    return selected, rejected
