"""스코어 비례 비중 계산 모듈.

모멘텀 점수에 비례해 종목별 목표 비중을 산출하고,
가드레일(MIN/MAX)과 리밸런싱 버퍼를 적용합니다.
"""

from __future__ import annotations


def calculate_score_weights(
    scores: dict[str, float],
    *,
    min_weight: float = 0.10,
    max_weight: float = 0.30,
) -> dict[str, float]:
    """스코어에 비례하는 목표 비중을 계산합니다.

    1. 음수 점수는 0으로 치환
    2. 점수 비례로 비중 산출
    3. MIN/MAX 가드레일 적용 후 정규화(합계=1.0)

    Args:
        scores: {ticker: 모멘텀 점수} 딕셔너리
        min_weight: 종목당 최소 비중
        max_weight: 종목당 최대 비중

    Returns:
        {ticker: 목표 비중} 딕셔너리, 합계 1.0

    Raises:
        ValueError: 유효한 종목이 없는 경우
    """
    if not scores:
        raise ValueError("비중 계산에 필요한 종목 점수가 없습니다.")

    n = len(scores)
    if n == 0:
        raise ValueError("비중 계산에 필요한 종목 점수가 없습니다.")

    # 가드레일 유효성 검사
    if min_weight * n > 1.0:
        # min_weight이 너무 크면 균등 배분으로 자동 조정
        min_weight = 1.0 / n
    if max_weight < min_weight:
        max_weight = min_weight

    # 1단계: 음수 점수는 0으로 치환
    clamped: dict[str, float] = {ticker: max(score, 0.0) for ticker, score in scores.items()}

    total_score = sum(clamped.values())

    # 모든 점수가 0이면 균등 배분
    if total_score <= 0:
        equal_weight = 1.0 / n
        return {ticker: equal_weight for ticker in scores}

    # 2단계: 점수 비례 비중 계산
    raw_weights: dict[str, float] = {ticker: score / total_score for ticker, score in clamped.items()}

    # 3단계: 가드레일 적용 (반복적 정규화)
    weights = _apply_guardrails(raw_weights, min_weight, max_weight)

    return weights


def _apply_guardrails(
    raw_weights: dict[str, float],
    min_weight: float,
    max_weight: float,
    max_iterations: int = 10,
) -> dict[str, float]:
    """MIN/MAX 가드레일을 적용하고 합계를 1.0으로 정규화합니다.

    한쪽에서 잘라낸 잔여분을 다른 종목에 비례 배분하는 과정을
    수렴할 때까지 반복합니다.
    """
    weights = dict(raw_weights)

    for _ in range(max_iterations):
        clamped_tickers: set[str] = set()
        clamped_total = 0.0

        # 상한/하한 초과 종목 식별 및 클램핑
        for ticker, w in weights.items():
            if w <= min_weight:
                weights[ticker] = min_weight
                clamped_tickers.add(ticker)
                clamped_total += min_weight
            elif w >= max_weight:
                weights[ticker] = max_weight
                clamped_tickers.add(ticker)
                clamped_total += max_weight

        # 클램핑되지 않은 종목들
        free_tickers = [t for t in weights if t not in clamped_tickers]

        if not free_tickers:
            # 모든 종목이 클램핑됨 → 마지막 종목에 잔여분 보정
            break

        # 남은 비중을 자유 종목에 비례 배분
        remaining = 1.0 - clamped_total
        if remaining <= 0:
            # 클램핑된 값의 합계만으로 1.0 초과 → 균등 배분 fallback
            equal = 1.0 / len(weights)
            return {t: equal for t in weights}

        free_total = sum(weights[t] for t in free_tickers)
        if free_total <= 0:
            # 자유 종목 비중 합이 0 → 균등 배분
            per_free = remaining / len(free_tickers)
            for t in free_tickers:
                weights[t] = per_free
        else:
            scale = remaining / free_total
            for t in free_tickers:
                weights[t] *= scale

        # 수렴 확인: 모든 종목이 범위 내인지 체크
        all_within = all(min_weight - 1e-9 <= weights[t] <= max_weight + 1e-9 for t in weights)
        if all_within:
            break

    # 최종 정규화 (부동소수점 오차 보정)
    total = sum(weights.values())
    if total > 0 and abs(total - 1.0) > 1e-9:
        for t in weights:
            weights[t] /= total

    return weights


def should_rebalance(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    buffer: float = 0.02,
) -> dict[str, bool]:
    """종목별로 리밸런싱이 필요한지 판단합니다.

    |현재비중 - 목표비중| > buffer 인 종목만 True를 반환합니다.

    Args:
        current_weights: {ticker: 현재 비중}
        target_weights: {ticker: 목표 비중}
        buffer: 리밸런싱 버퍼 (기본 2%)

    Returns:
        {ticker: 리밸런싱 필요 여부}
    """
    result: dict[str, bool] = {}
    all_tickers = set(current_weights) | set(target_weights)

    for ticker in all_tickers:
        current = current_weights.get(ticker, 0.0)
        target = target_weights.get(ticker, 0.0)
        result[ticker] = abs(current - target) > buffer

    return result
