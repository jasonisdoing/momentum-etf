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
    "kor1": {
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
    "kor2": {
        "PORTFOLIO_TOPN": [10],
        # 1. MA_RANGE: 160과 180 주변을 5단위로 정밀 타격
        "MA_RANGE": [155, 160, 165, 170, 175, 180, 185],
        # 2. MA_TYPE: SMA로 확정 (탐색 비용 절감)
        "MA_TYPE": ["SMA"],
        # 3. 교체 점수: 1점이 우세했으므로, 더 공격적인 0점과 보수적인 2점까지 비교
        "REPLACE_SCORE_THRESHOLD": [0, 1, 2],
        # 4. 손절: 6%가 좋았으므로 5~7% 구간 정밀 확인
        "STOP_LOSS_PCT": [5, 6, 7],
        # 5. RSI: 86으로 고정 (이미 검증됨)
        "OVERBOUGHT_SELL_THRESHOLD": [86],
        # 7. 쿨다운: 3일로 고정 (이미 검증됨)
        "COOLDOWN_DAYS": [3],
        "CORE_HOLDINGS": [],
        "OPTIMIZATION_METRIC": "CAGR",  # "CAGR", "Sharpe", "SDR" 중 선택
    },
    "kor10": {
        "PORTFOLIO_TOPN": [6],
        "MA_RANGE": [20, 25, 30, 35, 40, 45, 50],  # 범위가 넓어지면 과최적화 위험 증가
        "MA_TYPE": ["EMA"],
        "REPLACE_SCORE_THRESHOLD": [0, 1, 2, 3, 4, 5],
        "STOP_LOSS_PCT": [6],
        "OVERBOUGHT_SELL_THRESHOLD": [85, 86, 87, 88, 89, 90, 91, 92, 93],
        "COOLDOWN_DAYS": [0, 1, 2, 3],
        "CORE_HOLDINGS": [],
        "OPTIMIZATION_METRIC": "CAGR",  # "CAGR", "Sharpe", "SDR" 중 선택
    },
    "us1": {
        "PORTFOLIO_TOPN": [8],
        "MA_RANGE": [20, 25, 30, 35, 40, 45, 50],  # 범위가 넓어지면 과최적화 위험 증가
        # "MA_TYPE": ["SMA", "EMA"],
        "MA_TYPE": ["SMA"],
        "REPLACE_SCORE_THRESHOLD": [1, 2, 3, 4, 5],
        "STOP_LOSS_PCT": [5, 6, 7, 8, 9, 10],
        "OVERBOUGHT_SELL_THRESHOLD": [84, 86, 88, 90],
        "COOLDOWN_DAYS": [2, 3, 4],
        "CORE_HOLDINGS": [],
        "OPTIMIZATION_METRIC": "CAGR",  # "CAGR", "Sharpe", "SDR" 중 선택ㅉ
    },
}


# TUNING_CONFIG: dict[str, dict] = {
#     "k1": {
#         "PORTFOLIO_TOPN": [10],
#         "MA_RANGE": [25, 30, 35, 40, 45, 50],  # 범위가 넓어지면 과최적화 위험 증가
#         "MA_TYPE": ["EMA"],
#         "REPLACE_SCORE_THRESHOLD": [2, 3],
#         "STOP_LOSS_PCT": [5, 6, 7, 8, 9, 10],
#         "OVERBOUGHT_SELL_THRESHOLD": [86],
#         "COOLDOWN_DAYS": [2],
#         "CORE_HOLDINGS": [],
#         "OPTIMIZATION_METRIC": "CAGR",  # "CAGR", "Sharpe", "SDR" 중 선택
#     }
# }

# TUNING_CONFIG: dict[str, dict] = {
#     "k1": {
#         "PORTFOLIO_TOPN": [8],
#         "MA_RANGE": [25, 30, 35, 40, 45, 50],  # 범위가 넓어지면 과최적화 위험 증가
#         # "MA_TYPE": ["EMA"],
#         "MA_TYPE": ["SMA", "EMA", "WMA", "DEMA", "TEMA", "HMA"],
#         "REPLACE_SCORE_THRESHOLD": [1, 2, 3, 4, 5],
#         "STOP_LOSS_PCT": [5, 6, 7, 8, 9, 10],
#         "OVERBOUGHT_SELL_THRESHOLD": [83, 84, 85, 86, 87, 88, 89, 90],
#         "COOLDOWN_DAYS": [0, 1, 2, 3],
#         "CORE_HOLDINGS": [],
#         "OPTIMIZATION_METRIC": "CAGR",  # "CAGR", "Sharpe", "SDR" 중 선택
#     }
# }


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
