"""
Walk-Forward Analysis 결과 검증 스크립트

특정 테스트 시점의 결과를 재현하여 검증합니다.
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # 프로젝트 루트를 sys.path에 추가
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

import pandas as pd
from utils.logger import get_app_logger

logger = get_app_logger()


def verify_single_result(
    account_id: str,
    test_start: str,  # "2024-11-21"
    test_end: str,  # "2024-12-21"
    ma_period: int,
    portfolio_topn: int,
    rsi_threshold: int,
):
    """
    특정 테스트 시점의 결과를 재현합니다.

    Args:
        account_id: 계정 ID (예: "k1")
        test_start: 테스트 시작일
        test_end: 테스트 종료일
        ma_period: MA 기간
        portfolio_topn: 포트폴리오 종목 수
        rsi_threshold: RSI 임계값
    """
    # 여기서 import
    from logic.backtest.account_runner import run_account_backtest
    from strategies.maps.rules import StrategyRules

    logger.info("=" * 80)
    logger.info("결과 검증 시작")
    logger.info(f"계정: {account_id}")
    logger.info(f"테스트 기간: {test_start} ~ {test_end}")
    logger.info(f"파라미터: MA={ma_period}, TOPN={portfolio_topn}, RSI={rsi_threshold}")
    logger.info("=" * 80)

    # 전략 설정
    strategy_rules = StrategyRules.from_values(
        ma_period=ma_period,
        portfolio_topn=portfolio_topn,
        replace_threshold=1,
        ma_type="SMA",
        core_holdings=[],
    )

    # 백테스트 실행
    result = run_account_backtest(
        account_id=account_id,
        override_settings={
            "start_date": pd.Timestamp(test_start),
            "end_date": pd.Timestamp(test_end),
            "strategy_overrides": {
                "RSI_SELL_THRESHOLD": rsi_threshold,
                "COOLDOWN_DAYS": 1,
            },
        },
        strategy_override=strategy_rules,
        quiet=False,
    )

    if not result or not result.summary:
        logger.error("백테스트 실패!")
        return None

    # 결과 출력
    summary = result.summary
    logger.info("=" * 80)
    logger.info("검증 결과")
    logger.info("=" * 80)
    logger.info(f"기간수익률(%): {summary.get('period_return', 0):.2f}%")
    logger.info(f"CAGR(%): {summary.get('cagr', 0):.2f}%")
    logger.info(f"MDD(%): {summary.get('mdd', 0):.2f}%")
    logger.info(f"Sharpe: {summary.get('sharpe', 0):.2f}")
    logger.info(f"SDR (Sharpe/MDD): {summary.get('sharpe_to_mdd', 0):.3f}")
    logger.info("=" * 80)

    return summary


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법: python scripts/verify_lookback_result.py <account_id>")
        print("예시: python scripts/verify_lookback_result.py k1")
        sys.exit(1)

    account_id = sys.argv[1]

    # 상세 로그에서 확인한 최근 시점 (2025-09-21 ~ 2025-10-21)
    # 참조 6개월: MA=55, TOPN=6, RSI=10
    # 예상 결과: 수익률 +17.18%, Sharpe 5.65, MDD 5.03%

    logger.info(f"계정 {account_id}의 최근 시점 검증을 시작합니다...")

    verify_single_result(
        account_id=account_id,
        test_start="2025-09-21",
        test_end="2025-10-21",
        ma_period=55,
        portfolio_topn=6,
        rsi_threshold=10,
    )
