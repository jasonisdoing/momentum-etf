"""
포트폴리오 단위 백테스팅 및 관리.
핵심 로직은 logics/ 폴더 아래의 개별 전략 파일로 이전되었습니다.
예: logics/jason.py
"""
import pandas as pd
from typing import Optional, List, Tuple, Dict


def run_portfolio_backtest(
    pairs: List[Tuple[str, str]],
    months_range: Optional[List[int]] = None,
    initial_capital: float = 100_000_000.0,
    core_start_date: Optional[pd.Timestamp] = None,
    top_n: int = 10,
    initial_positions: Optional[dict] = None,
) -> Dict[str, pd.DataFrame]:
    """
    공유 현금 Top-N 포트폴리오를 시뮬레이션합니다.
    이 함수는 logics 폴더의 전략 파일로 이전되었습니다.
    test.py에서 --strategy 플래그를 사용하여 전략을 지정하세요.
    """
    raise NotImplementedError("핵심 로직은 logics/{strategy_name}.py 파일로 이전되었습니다. test.py를 통해 실행하세요.")