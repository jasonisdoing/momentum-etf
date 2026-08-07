"""비중 관련 보조 유틸리티 모듈.

현재 순위 화면에서는 직접 사용하지 않지만,
실보유 자산 보조 계산에 재사용할 수 있는 공통 함수를 둡니다.
"""

from __future__ import annotations


def calculate_score_weights(
    scores: dict[str, float],
    *,
    min_weight: float = 0.10,
    max_weight: float = 0.30,
) -> dict[str, float]:
    """점수 기반 비중 계산용 보조 함수입니다.

    1. 음수 점수는 0으로 치환
    2. 점수 비례로 비중 산출
    3. MIN/MAX 가드레일 적용 후 정규화(합계=1.0)

    Args:
        scores: {대상 ID: 점수} 딕셔너리
        min_weight: 대상당 최소 비중
        max_weight: 대상당 최대 비중

    Returns:
        {대상 ID: 목표 비중} 딕셔너리, 합계 1.0

    Raises:
        ValueError: 유효한 대상이 없는 경우
    """
    if not scores:
        raise ValueError("비중 계산에 필요한 종목 점수가 없습니다.")

    n = len(scores)
    if n == 0:
        raise ValueError("비중 계산에 필요한 종목 점수가 없습니다.")

    # 가드레일 유효성 검사
    if min_weight <= 0:
        raise ValueError("최소 비중은 0보다 커야 합니다.")
    if max_weight <= 0:
        raise ValueError("최대 비중은 0보다 커야 합니다.")
    if max_weight < min_weight:
        raise ValueError("최대 비중은 최소 비중보다 크거나 같아야 합니다.")
    if min_weight * n > 1.0:
        raise ValueError(
            f"최소 비중 {min_weight:.2%}는 대상 수 {n}개와 양립할 수 없습니다. 설정을 낮추거나 대상 수를 줄이세요."
        )
    if max_weight * n < 1.0:
        raise ValueError(
            f"최대 비중 {max_weight:.2%}는 대상 수 {n}개를 합쳐도 100%를 채울 수 없습니다. 설정을 높이세요."
        )

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


def calculate_ranked_score_weights(
    scores: dict[str, float],
    *,
    min_weight: float = 0.10,
    max_weight: float = 0.30,
) -> dict[str, float]:
    """점수 순위를 비중으로 변환한다.

    모든 종목에 최소 비중을 먼저 부여하고, 남는 비중은 점수 순위가 높을수록 많이 배분한다.
    점수가 음수여도 제외하지 않고 "덜 나쁜" 종목이 더 높은 순위 점수를 받는다.
    """
    if not scores:
        raise ValueError("비중 계산에 필요한 종목 점수가 없습니다.")

    n = len(scores)
    if min_weight <= 0:
        raise ValueError("최소 비중은 0보다 커야 합니다.")
    if max_weight <= 0:
        raise ValueError("최대 비중은 0보다 커야 합니다.")
    if max_weight < min_weight:
        raise ValueError("최대 비중은 최소 비중보다 크거나 같아야 합니다.")
    if min_weight * n > 1.0:
        raise ValueError(
            f"최소 비중 {min_weight:.2%}는 대상 수 {n}개와 양립할 수 없습니다. 설정을 낮추거나 대상 수를 줄이세요."
        )
    if max_weight * n < 1.0:
        raise ValueError(
            f"최대 비중 {max_weight:.2%}는 대상 수 {n}개를 합쳐도 100%를 채울 수 없습니다. 설정을 높이세요."
        )

    ordered = sorted(scores.items(), key=lambda item: (-float(item[1]), item[0]))
    rank_points = {ticker: float(n - index) for index, (ticker, _score) in enumerate(ordered)}
    return _apply_rank_points_guardrails(rank_points, min_weight, max_weight)


def calculate_ranked_score_weights_with_cash(
    scores: dict[str, float],
    *,
    defensive_tickers: set[str],
    min_weight: float = 0.10,
    max_weight: float = 0.30,
    cash_max_weight: float = 0.40,
    forced_cash_weight: float | None = None,
    cash_ticker: str = "__CASH__",
) -> dict[str, float]:
    """방어 대상 종목을 최소 비중에 가깝게 두고 남는 비중 일부를 현금으로 배분한다.

    방어 대상은 추세가 음수인 종목처럼 추가 비중을 주기 꺼려지는 대상을 뜻한다.
    현금 비중은 `방어 대상 비율 × 현금 최대 비중`을 기본값으로 사용한다.
    `forced_cash_weight`가 있으면 벤치마크 레짐 같은 상위 정책이 지정한 현금 비중을 우선한다.
    """
    if not scores:
        raise ValueError("비중 계산에 필요한 종목 점수가 없습니다.")

    n = len(scores)
    if min_weight < 0:
        raise ValueError("최소 비중은 0 이상이어야 합니다.")
    if max_weight <= 0:
        raise ValueError("최대 비중은 0보다 커야 합니다.")
    if max_weight < min_weight:
        raise ValueError("최대 비중은 최소 비중보다 크거나 같아야 합니다.")
    if cash_max_weight < 0:
        raise ValueError("현금 최대 비중은 0 이상이어야 합니다.")
    if min_weight * n > 1.0:
        raise ValueError(
            f"최소 비중 {min_weight:.2%}는 대상 수 {n}개와 양립할 수 없습니다. 설정을 낮추거나 대상 수를 줄이세요."
        )
    if (max_weight * n) + cash_max_weight < 1.0:
        raise ValueError(
            f"최대 비중 {max_weight:.2%}와 현금 최대 비중 {cash_max_weight:.2%}로는 100%를 채울 수 없습니다."
        )

    tickers = list(scores.keys())
    defensive = {ticker for ticker in defensive_tickers if ticker in scores}
    cash_capacity = min(float(cash_max_weight), max(0.0, 1.0 - (len(tickers) * min_weight)))
    if forced_cash_weight is None:
        cash_weight = cash_capacity * (len(defensive) / n) if defensive else 0.0
    else:
        cash_weight = min(max(0.0, float(forced_cash_weight)), cash_capacity)

    weights = {ticker: float(min_weight) for ticker in tickers}
    remaining = max(0.0, 1.0 - cash_weight - (len(tickers) * min_weight))
    capacity = {ticker: float(max_weight - min_weight) for ticker in tickers}
    ordered = sorted(scores.items(), key=lambda item: (-float(item[1]), item[0]))
    rank_points = {ticker: float(n - index) for index, (ticker, _score) in enumerate(ordered)}

    # 방어 대상이 아닌 종목에 추가 비중을 우선 배분한다.
    growth_tickers = [ticker for ticker in tickers if ticker not in defensive]
    primary_targets = growth_tickers if growth_tickers else tickers
    remaining = _add_rank_weight(weights, capacity, rank_points, set(primary_targets), remaining)

    # 성장 대상 상한 때문에 남는 비중은 현금 최대치까지 먼저 흡수한다.
    if remaining > 1e-12 and cash_weight < cash_capacity:
        add_to_cash = min(remaining, cash_capacity - cash_weight)
        cash_weight += add_to_cash
        remaining -= add_to_cash

    # 그래도 남으면 상한이 남은 종목 전체에 덜 나쁜 순서로 배분한다.
    if remaining > 1e-12:
        remaining = _add_rank_weight(weights, capacity, rank_points, set(tickers), remaining)

    if remaining > 1e-9:
        raise ValueError("비중 계산 결과 100%를 채우지 못했습니다. 최대 비중 또는 현금 최대 비중을 높이세요.")

    if cash_weight > 1e-12:
        weights[cash_ticker] = cash_weight
    return weights


def _add_rank_weight(
    weights: dict[str, float],
    capacity: dict[str, float],
    rank_points: dict[str, float],
    targets: set[str],
    remaining: float,
) -> float:
    """남은 비중을 대상 종목에 순위 포인트 비례로 추가하고 잔여 비중을 반환한다."""
    active = {ticker for ticker in targets if capacity.get(ticker, 0.0) > 1e-12}
    while remaining > 1e-12 and active:
        active_total = sum(rank_points[ticker] for ticker in active)
        if active_total <= 1e-12:
            break

        progressed = False
        for ticker in list(active):
            desired = remaining * (rank_points[ticker] / active_total)
            addable = min(desired, capacity[ticker])
            if addable > 1e-12:
                weights[ticker] += addable
                capacity[ticker] -= addable
                remaining -= addable
                progressed = True
            if capacity[ticker] <= 1e-12:
                active.remove(ticker)
        if not progressed:
            break
    return remaining


def _apply_rank_points_guardrails(
    rank_points: dict[str, float],
    min_weight: float,
    max_weight: float,
) -> dict[str, float]:
    """최소 비중을 먼저 배분한 뒤 남은 비중을 순위 포인트대로 나눈다."""
    tickers = list(rank_points.keys())
    weights = {ticker: float(min_weight) for ticker in tickers}
    remaining = 1.0 - (len(tickers) * min_weight)
    capacity = {ticker: float(max_weight - min_weight) for ticker in tickers}
    active = {ticker for ticker in tickers if capacity[ticker] > 1e-12}

    while remaining > 1e-12 and active:
        active_total = sum(rank_points[ticker] for ticker in active)
        if active_total <= 1e-12:
            break

        progressed = False
        for ticker in list(active):
            desired = remaining * (rank_points[ticker] / active_total)
            addable = min(desired, capacity[ticker])
            if addable > 1e-12:
                weights[ticker] += addable
                capacity[ticker] -= addable
                remaining -= addable
                progressed = True
            if capacity[ticker] <= 1e-12:
                active.remove(ticker)
        if not progressed:
            break

    if remaining > 1e-9:
        candidates = [ticker for ticker in tickers if weights[ticker] < max_weight - 1e-12]
        if candidates:
            per_ticker = remaining / len(candidates)
            for ticker in candidates:
                addable = min(per_ticker, max_weight - weights[ticker])
                weights[ticker] += addable
                remaining -= addable

    return weights


def _apply_guardrails(
    raw_weights: dict[str, float],
    min_weight: float,
    max_weight: float,
) -> dict[str, float]:
    """MIN/MAX 가드레일을 적용하고 합계를 1.0으로 정규화합니다.

    기본적으로는 모든 종목에 최소 비중을 먼저 할당한 뒤,
    남는 비중만 점수 비례로 추가 배분합니다.
    상한에 도달한 종목은 제외하고 반복 배분하여 합계 1.0을 맞춥니다.
    """
    tickers = list(raw_weights.keys())
    weights = {ticker: float(min_weight) for ticker in tickers}
    remaining = 1.0 - (len(tickers) * min_weight)
    if remaining <= 1e-12:
        return weights

    capacity = {ticker: float(max_weight - min_weight) for ticker in tickers}
    active = {ticker for ticker, cap in capacity.items() if cap > 1e-12}
    extras = {ticker: max(float(raw_weights.get(ticker, 0.0)) - min_weight, 0.0) for ticker in tickers}

    while remaining > 1e-12 and active:
        active_total = sum(extras[ticker] for ticker in active)
        if active_total <= 1e-12:
            equal_share = remaining / len(active)
            progressed = False
            for ticker in list(active):
                addable = min(equal_share, capacity[ticker])
                if addable > 0:
                    weights[ticker] += addable
                    capacity[ticker] -= addable
                    remaining -= addable
                    progressed = True
                if capacity[ticker] <= 1e-12:
                    active.remove(ticker)
            if not progressed:
                break
            continue

        progressed = False
        for ticker in list(active):
            desired = remaining * (extras[ticker] / active_total)
            addable = min(desired, capacity[ticker])
            if addable > 0:
                weights[ticker] += addable
                capacity[ticker] -= addable
                remaining -= addable
                progressed = True
            if capacity[ticker] <= 1e-12:
                active.remove(ticker)
        if not progressed:
            break

    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9:
        deficit = 1.0 - total
        candidates = [ticker for ticker in tickers if weights[ticker] < max_weight - 1e-12]
        if candidates and deficit > 0:
            per_ticker = deficit / len(candidates)
            for ticker in candidates:
                weights[ticker] += min(per_ticker, max_weight - weights[ticker])

    return weights


def should_rebalance(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    buffer: float = 0.02,
) -> dict[str, bool]:
    """종목별 비중 차이가 버퍼를 넘는지 판단합니다.

    |현재비중 - 목표비중| > buffer 인 종목만 True를 반환합니다.

    Args:
        current_weights: {ticker: 현재 비중}
        target_weights: {ticker: 목표 비중}
        buffer: 비중 차이 허용 버퍼 (기본 2%)

    Returns:
        {ticker: 비중 조정 필요 여부}
    """
    result: dict[str, bool] = {}
    all_tickers = set(current_weights) | set(target_weights)

    for ticker in all_tickers:
        current = current_weights.get(ticker, 0.0)
        target = target_weights.get(ticker, 0.0)
        result[ticker] = abs(current - target) > buffer

    return result
