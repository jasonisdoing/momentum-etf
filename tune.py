"""계정별 전략 파라미터 튜닝 실행 스크립트."""

from __future__ import annotations

import sys
from pathlib import Path

from logic.tune.runner import run_account_tuning
from utils.account_registry import get_account_settings, get_strategy_rules
from utils.data_loader import MissingPriceDataError
from utils.logger import get_app_logger

# =========================================================
# 계좌별 성격 맞춤형 설정
# =========================================================
ACCOUNT_TUNING_CONFIG = {
    # 🇰🇷 국내 ETF: 삼각편대 (수익 160% + MDD -14%)
    "kor_kr": {
        "PORTFOLIO_TOPN": [3],
        "REPLACE_SCORE_THRESHOLD": [0],
    },
    # 🇺🇸 국내상장 미국 ETF: 6개 분산 (수익 90% + MDD -15%)
    "kor_us": {
        "PORTFOLIO_TOPN": [6],
        "REPLACE_SCORE_THRESHOLD": [0],
    },
    # 🇺🇸 미국 직투: 테스트 중
    "us": {
        "PORTFOLIO_TOPN": [5],
        "REPLACE_SCORE_THRESHOLD": [0, 1, 2, 3],
    },
}

# =========================================================
# 공통 설정 (나머지 변수 통제)
# =========================================================
COMMON_TUNING_CONFIG = {
    # 1. 이동평균: 유일한 변수 (50~100일 사이에서 최적값 찾기)
    "MA_RANGE": [50, 60, 70, 80, 90, 100],
    "MA_TYPE": ["SMA", "EMA"],
    # 2. 손절: 10% 고정 (리스크 관리)
    "STOP_LOSS_PCT": [10],
    # 3. 나머지 고정
    "OVERBOUGHT_SELL_THRESHOLD": [82, 84, 86, 88, 90],
    "COOLDOWN_DAYS": [1, 2, 3],
    # 4. 목표: 수익률 극대화
    "OPTIMIZATION_METRIC": "CAGR",  # CAGR, SHARPE, SDR 중 선택
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

    # 공통 설정과 계정별 설정을 조합
    merged_config = COMMON_TUNING_CONFIG.copy()
    account_config = ACCOUNT_TUNING_CONFIG.get(account_id, {})
    merged_config.update(account_config)

    try:
        output = run_account_tuning(
            account_id,
            output_path=None,
            results_dir=RESULTS_DIR,
            tuning_config={account_id: merged_config},
        )
    except MissingPriceDataError as exc:
        logger.error(str(exc))
        raise SystemExit(1)
    if output is None:
        logger.error("튜닝이 실패하여 결과를 저장하지 않습니다.")


if __name__ == "__main__":
    main()
