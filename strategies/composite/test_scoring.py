"""종합 점수 계산 함수 테스트."""

from .scoring import calculate_composite_score


def test_weighted_average():
    """가중 평균 방식 테스트."""
    config = {
        "enabled": True,
        "method": "weighted_average",
        "maps_weight": 0.7,
        "rsi_weight": 0.3,
    }

    # MAPS=80, RSI=60 → (80*0.7) + (60*0.3) = 56 + 18 = 74
    result = calculate_composite_score(80, 60, method="weighted_average", config=config)
    assert result == 74.0, f"Expected 74.0, got {result}"

    # MAPS=100, RSI=0 → (100*0.7) + (0*0.3) = 70
    result = calculate_composite_score(100, 0, method="weighted_average", config=config)
    assert result == 70.0, f"Expected 70.0, got {result}"

    print("✅ weighted_average 테스트 통과")


def test_rsi_adjusted():
    """RSI 조정 방식 테스트."""
    config = {
        "enabled": True,
        "method": "rsi_adjusted",
        "rsi_multiplier_min": 0.8,
        "rsi_multiplier_max": 1.2,
    }

    # MAPS=80, RSI=100 → 80 * 1.2 = 96 (과매도, 매수 기회)
    result = calculate_composite_score(80, 100, method="rsi_adjusted", config=config)
    assert result == 96.0, f"Expected 96.0, got {result}"

    # MAPS=80, RSI=50 → 80 * 1.0 = 80 (중립)
    result = calculate_composite_score(80, 50, method="rsi_adjusted", config=config)
    assert result == 80.0, f"Expected 80.0, got {result}"

    # MAPS=80, RSI=0 → 80 * 0.8 = 64 (과매수, 위험)
    result = calculate_composite_score(80, 0, method="rsi_adjusted", config=config)
    assert result == 64.0, f"Expected 64.0, got {result}"

    print("✅ rsi_adjusted 테스트 통과")


def test_disabled():
    """종합 점수 비활성화 테스트."""
    config = {
        "enabled": False,
    }

    # 비활성화 시 MAPS 점수 그대로 반환
    result = calculate_composite_score(80, 60, method="weighted_average", config=config)
    assert result == 80.0, f"Expected 80.0, got {result}"

    print("✅ disabled 테스트 통과")


if __name__ == "__main__":
    test_weighted_average()
    test_rsi_adjusted()
    test_disabled()
    print("\n🎉 모든 테스트 통과!")
