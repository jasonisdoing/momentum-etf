"""공통 시그널 계산 및 포지션 선택 로직."""

import pandas as pd


def compute_signals(prices: pd.Series, settings: dict) -> pd.DataFrame:
    """가격 시계열로 추세/변동성/드로다운 신호를 계산합니다."""
    df = pd.DataFrame(index=prices.index)
    df["close"] = prices

    # 변동성 필터 제거: 변동성은 0으로 두고 사용하지 않음
    df["vol"] = 0.0

    peak = prices.cummax()
    df["drawdown"] = prices / peak - 1.0
    return df.dropna()


def pick_target(row, prev_target: str, settings: dict, *, offense_set=None, select_offense=None) -> str:
    """
    신호 행과 이전 타깃을 받아 매수 대상 티커를 결정합니다 (이중 임계값 적용).

    - drawdown_buy_cutoff (예: 1.0 -> -1.0%): 이보다 높으면(회복되면) 공격 자산 매수
    - drawdown_sell_cutoff (예: 2.0 -> -2.0%): 이보다 낮으면(악화되면) 공격 자산 매도

    동적 공격 선택:
    - offense_set: 공격 후보 티커 집합 (없으면 단일 offense_ticker)
    - select_offense: 방어→공격 진입 시 후보 중 1개를 고르는 콜백(날짜 인자).
      진입 후에는 매도 신호가 날 때까지 같은 공격 자산(prev_target)을 유지한다.
    """
    buy_cut = -settings["drawdown_buy_cutoff"] / 100
    sell_cut = -settings["drawdown_sell_cutoff"] / 100

    offense = settings["offense_ticker"]
    defense = settings["defense_ticker"]
    offenses = offense_set or {offense}

    current_dd = row["drawdown"]

    if prev_target in offenses:
        # 공격 자산 보유 중: 매도 기준보다 더 떨어지면 방어 전환, 아니면 같은 자산 유지
        if current_dd < sell_cut:
            return defense
        return prev_target
    else:
        # 방어 자산 보유 중: 매수 기준보다 더 오르면 공격 전환 (진입 시점에 후보 선택)
        if current_dd > buy_cut:
            if select_offense is not None:
                return select_offense(row.name)
            return offense
        return defense
