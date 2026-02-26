"""추천과 백테스트에서 공통으로 사용하는 필터링 로직."""

from collections.abc import Iterable
from typing import Any

from utils.logger import get_app_logger

logger = get_app_logger()


def select_candidates(
    candidates: Iterable[dict[str, Any]],
    *,
    max_count: int | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], str]]]:
    """점수 기반으로 매수 후보 목록을 선택합니다.

    Args:
        candidates: 최소한 ``tkr`` 와 ``score`` 키를 가진 후보 딕셔너리 iterable.
        max_count: 선택할 최대 후보 수. None이면 제한 없음.

    Returns:
        (선택된_후보_리스트, 제외된_후보와_사유_리스트)
    """
    selected_candidates = []
    rejected = []

    try:
        all_candidates = []
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
                final_score_val = composite_score_val if composite_score_val > float("-inf") else score_val
            except (TypeError, ValueError):
                final_score_val = float("-inf")

            all_candidates.append({"cand": cand, "score": final_score_val})

        # 점수 내림차순 정렬
        all_candidates.sort(key=lambda x: x["score"], reverse=True)

        effective_max = max_count if max_count is not None and max_count > 0 else len(all_candidates)
        selected_candidates = [item["cand"] for item in all_candidates[:effective_max]]
        rejected = [(item["cand"], "over_limit") for item in all_candidates[effective_max:]]

        return selected_candidates, rejected

    except Exception as e:
        logger.exception("select_candidates 실행 중 오류: %s", e)
        return [], []


def sort_decisions_by_order_and_score(decisions: list[dict[str, Any]]) -> None:
    """BACKTEST_STATUS_LIST의 order 순으로 정렬하고, 같은 order 내에서는 composite_score > score 역순으로 정렬합니다.

    백테스트와 추천에서 공통으로 사용되는 정렬 함수입니다.
    """
    from strategies.maps.constants import BACKTEST_STATUS_LIST

    def sort_key(item_dict):
        state = item_dict["state"]
        composite_score = item_dict.get("composite_score", 0.0)
        score = item_dict.get("score", 0.0)
        ticker = item_dict.get("ticker") or item_dict.get("tkr", "")
        order = BACKTEST_STATUS_LIST.get(state, {}).get("order", 99)
        # composite_score 우선, 그 다음 MAPS score, 마지막으로 ticker
        return (order, -composite_score, -score, ticker)

    decisions.sort(key=sort_key)


__all__ = [
    "select_candidates",
    "sort_decisions_by_order_and_score",
]
