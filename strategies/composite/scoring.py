"""종합 점수 계산 함수."""

from __future__ import annotations


def calculate_composite_score(
    maps_score: float,
    rsi_score: float,
    method: str = "rsi_adjusted",
    config: dict | None = None,
) -> float:
    """
    MAPS와 RSI 점수를 결합하여 종합 점수를 계산합니다.

    Args:
        maps_score: MAPS 점수 (0~100)
        rsi_score: RSI 점수 (0~100)
        method: 계산 방식
            - "weighted_average": 가중 평균 방식
            - "rsi_adjusted": RSI를 MAPS 조정 팩터로 사용
        config: 계산 설정 딕셔너리
            - enabled: 종합 점수 계산 활성화 여부 (기본값: True)
            - maps_weight: MAPS 가중치 (weighted_average용, 기본값: 0.7)
            - rsi_weight: RSI 가중치 (weighted_average용, 기본값: 0.3)
            - rsi_multiplier_min: RSI 최소 배율 (rsi_adjusted용, 기본값: 0.8)
            - rsi_multiplier_max: RSI 최대 배율 (rsi_adjusted용, 기본값: 1.2)

    Returns:
        float: 종합 점수 (0~100+)

    Examples:
        >>> # 가중 평균 방식
        >>> calculate_composite_score(80, 60, method="weighted_average")
        74.0  # (80 * 0.7) + (60 * 0.3)

        >>> # RSI 조정 방식
        >>> calculate_composite_score(80, 100, method="rsi_adjusted")
        96.0  # 80 * 1.2 (RSI=100이면 1.2배 증폭)

        >>> # RSI 조정 방식 (과매수)
        >>> calculate_composite_score(80, 0, method="rsi_adjusted")
        64.0  # 80 * 0.8 (RSI=0이면 0.8배 감소)
    """
    if config is None:
        config = {}

    # 종합 점수 계산 비활성화 시 MAPS 점수 반환
    if not config.get("enabled", True):
        return maps_score

    # 입력값 검증
    maps_score = max(0.0, float(maps_score))
    rsi_score = max(0.0, min(100.0, float(rsi_score)))

    if method == "weighted_average":
        return _calculate_weighted_average(maps_score, rsi_score, config)
    elif method == "rsi_adjusted":
        return _calculate_rsi_adjusted(maps_score, rsi_score, config)
    else:
        # 알 수 없는 방식이면 MAPS 점수 반환
        return maps_score


def _calculate_weighted_average(
    maps_score: float,
    rsi_score: float,
    config: dict,
) -> float:
    """
    가중 평균 방식으로 종합 점수를 계산합니다.

    Args:
        maps_score: MAPS 점수 (0~100)
        rsi_score: RSI 점수 (0~100)
        config: 설정 딕셔너리

    Returns:
        float: 종합 점수 (0~100)

    Notes:
        composite = (MAPS * maps_weight) + (RSI * rsi_weight)
        가중치 합이 1.0이 아니면 자동으로 정규화됩니다.
    """
    maps_weight = float(config.get("maps_weight", 0.7))
    rsi_weight = float(config.get("rsi_weight", 0.3))

    # 가중치 정규화 (합이 1.0이 되도록)
    total_weight = maps_weight + rsi_weight
    if total_weight > 0:
        maps_weight = maps_weight / total_weight
        rsi_weight = rsi_weight / total_weight

    composite = (maps_score * maps_weight) + (rsi_score * rsi_weight)
    return round(composite, 2)


def _calculate_rsi_adjusted(
    maps_score: float,
    rsi_score: float,
    config: dict,
) -> float:
    """
    RSI를 MAPS 조정 팩터로 사용하여 종합 점수를 계산합니다.

    Args:
        maps_score: MAPS 점수 (0~100)
        rsi_score: RSI 점수 (0~100)
        config: 설정 딕셔너리

    Returns:
        float: 종합 점수 (0~120 범위, MAPS * multiplier)

    Notes:
        RSI 점수를 multiplier로 변환:
        - RSI = 0 (과매수) → multiplier_min (기본 0.8)
        - RSI = 100 (과매도) → multiplier_max (기본 1.2)
        - 선형 보간으로 중간값 계산

        composite = MAPS * multiplier

        예시:
        - MAPS=80, RSI=100 → 80 * 1.2 = 96 (과매도, 매수 기회 강조)
        - MAPS=80, RSI=50 → 80 * 1.0 = 80 (중립)
        - MAPS=80, RSI=0 → 80 * 0.8 = 64 (과매수, 위험 신호)
    """
    multiplier_min = float(config.get("rsi_multiplier_min", 0.8))
    multiplier_max = float(config.get("rsi_multiplier_max", 1.2))

    # RSI 점수(0~100)를 multiplier로 선형 변환
    # rsi_score = 0 → multiplier_min
    # rsi_score = 100 → multiplier_max
    rsi_multiplier = multiplier_min + (rsi_score / 100.0) * (multiplier_max - multiplier_min)

    composite = maps_score * rsi_multiplier
    return round(composite, 2)


__all__ = [
    "calculate_composite_score",
]
