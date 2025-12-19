"""계정별 전략 파라미터 튜닝 실행 스크립트."""

from __future__ import annotations

import sys
from pathlib import Path

from logic.tune.runner import run_account_tuning
from utils.account_registry import get_account_settings, get_strategy_rules
from utils.data_loader import MissingPriceDataError
from utils.logger import get_app_logger

# 튜닝·최적화 작업이 공유하는 계정별 파라미터 탐색 설정
TUNING_CONFIG: dict[str, dict] = {
    "kor_kr": {
        # 1. 포트폴리오: 8개 확정
        "PORTFOLIO_TOPN": [8],
        # 2. 이동평균: 170일 주변 점검
        "MA_RANGE": [160, 165, 170, 175, 180],
        "MA_TYPE": ["SMA"],
        # 3. 교체 점수: 0점 확정 (혹은 1점 비교)
        "REPLACE_SCORE_THRESHOLD": [0, 1],
        # 4. 손절: 12~16% 넓은 범위 유지보수
        "STOP_LOSS_PCT": [12, 13, 14, 15, 16],
        # 5. 나머지 고정
        "OVERBOUGHT_SELL_THRESHOLD": [86],
        "TRAILING_STOP_PCT": [0],
        "COOLDOWN_DAYS": [3],
        "CORE_HOLDINGS": [],
        "OPTIMIZATION_METRIC": "CAGR",
    },
    "kor_us": {
        # Top 8의 안정성이 확인되었으므로 8개 중심 (혹은 5개와 비교)
        "PORTFOLIO_TOPN": [8],
        # 60일 최적값 주변 점검
        "MA_RANGE": [50, 55, 60, 65, 70],
        "MA_TYPE": ["EMA"],
        # 교체 점수 2~3점 확인
        "REPLACE_SCORE_THRESHOLD": [2, 3],
        # 손절 8~10% 확인
        "STOP_LOSS_PCT": [8, 9, 10],
        # RSI 86 고정
        "OVERBOUGHT_SELL_THRESHOLD": [86],
        "COOLDOWN_DAYS": [3],
        "CORE_HOLDINGS": [],
        "OPTIMIZATION_METRIC": "CAGR",
    },
    "us": {
        # 1. 포트폴리오: 8개 고정
        "PORTFOLIO_TOPN": [8],
        # 2. 이동평균: 70일 주변 점검
        "MA_RANGE": [65, 70, 75],
        "MA_TYPE": ["SMA"],
        # 3. 교체 점수: 1~2점
        "REPLACE_SCORE_THRESHOLD": [1, 2],
        # 4. 손절: 10~13% 사이 점검
        "STOP_LOSS_PCT": [10, 11, 12, 13],
        # 5. RSI: 83~85 점검 (84가 좋았으므로)
        "OVERBOUGHT_SELL_THRESHOLD": [83, 84, 85],
        # 6. 나머지 고정
        "TRAILING_STOP_PCT": [0],
        "COOLDOWN_DAYS": [2, 3],
        "CORE_HOLDINGS": [],
        "OPTIMIZATION_METRIC": "CAGR",
    },
}


# "kor_kr": {
#     # 1. 포트폴리오: 8개로 고정 (사용자 요청 반영)
#     "PORTFOLIO_TOPN": [8],

#     # 2. 이동평균: 중기(60)부터 초장기(200)까지 전체 탐색
#     "MA_RANGE": [60, 90, 120, 150, 180, 200],

#     # 3. 이평선 타입 모두 비교
#     "MA_TYPE": ["SMA", "EMA", "WMA", "DEMA", "TEMA", "HMA"],

#     # 4. 교체 점수: 0~3점 전체 탐색 (적극 교체 vs 진득 보유)
#     "REPLACE_SCORE_THRESHOLD": [0, 1, 2, 3],

#     # 5. 손절: 5~15% 넓은 범위 확인
#     "STOP_LOSS_PCT": [5, 10, 15],

#     # 6. 과매수: 80~90 넓은 범위 확인
#     "OVERBOUGHT_SELL_THRESHOLD": [80, 85, 90],

#     # 7. 쿨다운: 1~3일 확인
#     "COOLDOWN_DAYS": [1, 2, 3],

#     "TRAILING_STOP_PCT": [0], # 트레일링 스탑은 보통 0이 우세하므로 고정 (원하시면 [0, 5] 추가)
#     "CORE_HOLDINGS": [],
#     "OPTIMIZATION_METRIC": "CAGR",
# },


RESULTS_DIR = Path(__file__).resolve().parent / "zaccounts"


def main() -> None:
    logger = get_app_logger()

    if len(sys.argv) < 2:
        print("Usage: python tune.py <account_id>")
        raise SystemExit(1)

    account_id = sys.argv[1].strip().lower()

    try:
        get_account_settings(account_id)
        get_strategy_rules(account_id)
    except Exception as exc:  # pragma: no cover - 잘못된 입력 방어 전용 처리
        raise SystemExit(f"계정 설정을 로드하는 중 오류가 발생했습니다: {exc}")

    try:
        output = run_account_tuning(
            account_id,
            output_path=None,
            results_dir=RESULTS_DIR,
            tuning_config=TUNING_CONFIG,
        )
    except MissingPriceDataError as exc:
        logger.error(str(exc))
        raise SystemExit(1)
    if output is None:
        logger.error("튜닝이 실패하여 결과를 저장하지 않습니다.")


if __name__ == "__main__":
    main()
