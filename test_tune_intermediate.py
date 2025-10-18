"""튜닝 중간 저장 기능 테스트 스크립트."""

from pathlib import Path
import numpy as np

from logic.tune.runner import run_account_tuning
from utils.logger import get_app_logger

RESULTS_DIR = Path(__file__).resolve().parent / "data" / "results"

# 작은 탐색 공간으로 빠른 테스트
TUNING_CONFIG = {
    "aus": {
        "MA_RANGE": np.arange(20, 55, 5),
        "PORTFOLIO_TOPN": np.arange(3, 9, 1),
        "REPLACE_SCORE_THRESHOLD": np.arange(0, 5.5, 0.5),
        "OVERBOUGHT_SELL_THRESHOLD": np.arange(5, 25, 5),
    },
    "k1": {
        "MA_RANGE": [15, 20],  # 2개만
        "PORTFOLIO_TOPN": [6, 7],  # 2개만
        "REPLACE_SCORE_THRESHOLD": [1.0, 1.5],  # 2개만
        "OVERBOUGHT_SELL_THRESHOLD": [10, 15],  # 2개만
        "COOLDOWN_DAYS": [1, 2],  # 2개만
    },
}


def main() -> None:
    logger = get_app_logger()

    logger.info("=" * 60)
    logger.info("튜닝 중간 저장 기능 테스트 시작")
    logger.info("탐색 공간: 2×2×2×2×2 = 32개 조합 (COOLDOWN_DAYS 추가)")
    logger.info("=" * 60)

    # months_range를 None으로 설정하면 설정 파일의 여러 기간을 테스트
    # 이 경우 각 기간마다 중간 저장이 발생
    output = run_account_tuning(
        "k1",
        output_path=None,
        results_dir=RESULTS_DIR,
        tuning_config=TUNING_CONFIG,
        months_range=None,  # 설정 파일의 모든 기간 테스트
    )

    if output:
        logger.info("✅ 튜닝 완료: %s", output)
        logger.info("중간 저장 파일을 확인하세요.")
    else:
        logger.error("❌ 튜닝 실패")


if __name__ == "__main__":
    main()
