"""Momentum 전략 모듈에서 공통으로 사용하는 유틸리티."""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SIGNAL_TABLE_HEADERS: Sequence[str] = [
    "#",
    "티커",
    "종목명",
    "카테고리",
    "상태",
    "보유일",
    "현재가",
    "일간(%)",
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
                if cat and str(cat).strip().upper() != "TBD"
            }

        # 후보 처리
        best_per_category = {}
        for cand in candidates:
            if not isinstance(cand, dict):
                continue

            ticker = str(cand.get("tkr") or cand.get("ticker") or "").strip()
            if not ticker:
                continue

            # 점수 추출
            score_raw = cand.get("score")
            try:
                score_val = float(score_raw) if score_raw is not None else float("-inf")
            except (TypeError, ValueError):
                score_val = float("-inf")

            # 카테고리 확인
            category, internal_key = _resolve_category(ticker, etf_meta)
            existing = best_per_category.get(internal_key)

            if existing is None:
                # 새로운 카테고리인 경우
                if category not in held_set:
                    best_per_category[internal_key] = {
                        "cand": cand,
                        "score": score_val,
                        "category": category,
                    }
                elif skip_held_categories:
                    rejected.append((cand, "category_held"))
            else:
                # 기존 카테고리와 비교
                if score_val > existing["score"]:
                    if existing["category"] not in held_set:
                        rejected.append((existing["cand"], "better_candidate"))
                        best_per_category[internal_key] = {
                            "cand": cand,
                            "score": score_val,
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
        print(f"select_candidates_by_category 실행 중 오류: {str(e)}")
        import traceback

        traceback.print_exc()
        return [], []


from .constants import DECISION_CONFIG


def sort_decisions_by_order_and_score(decisions: List[Dict[str, Any]]) -> None:
    """DECISION_CONFIG의 order 순으로 정렬하고, 같은 order 내에서는 score 역순으로 정렬합니다.

    백테스트와 추천에서 공통으로 사용되는 정렬 함수입니다.
    """

    def sort_key(item_dict):
        state = item_dict["state"]
        score = item_dict.get("score", 0.0)
        ticker = item_dict.get("ticker") or item_dict.get("tkr", "")
        order = DECISION_CONFIG.get(state, {}).get("order", 99)
        return (order, -score, ticker)

    decisions.sort(key=sort_key)
