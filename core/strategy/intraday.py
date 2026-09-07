"""장중 공통 계산 — 모멘텀·신고가 운용 현황이 공유한다.

장중의 판정·체결·수량은 엔진(`core.strategy.slot_backtest`)의 잠정 마지막 봉 모드가
한다(AGENTS.md §10-6). 이 모듈에는 그 실행의 **입력**(실시간 값을 마지막 봉으로 붙인
프레임)과 **결과 표기**(보유 행 상태)를 만드는 얇은 공통 계산만 둔다.
외부 조회는 하지 않는다 — 필요한 값은 전부 인자로 받는다.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


def effective_close_frame(frame: pd.DataFrame, live_values: dict[str, float], session_ts: pd.Timestamp) -> pd.DataFrame:
    """실시간 값을 **오늘 마지막 봉**으로 붙인 (날짜 × 티커) 프레임 — 장중 실행의 입력이다.

    종가 프레임에 현재가를, 고가 프레임에 실시간 고가를, 시가 프레임에 오늘 시가를 붙이는
    식으로 쓴다. 실시간이 없는 종목은 오늘 칸이 NaN 으로 남아 판정 불가로 처리된다 —
    값을 추정하지 않는다(AGENTS.md §10-6).
    """
    effective = frame.copy()
    effective.loc[session_ts] = pd.Series(live_values).reindex(effective.columns)
    return effective


def mark_engine_statuses(
    holdings: list[dict[str, Any]], planned_exits: Iterable[str], *, exit_reason: str = "이탈"
) -> None:
    """엔진 잠정 실행의 보유 행에 화면 상태를 붙인다 — 판정은 하지 않고 표기만 한다.

    체결 예정 매도(``fill_date`` 있음)는 확정 'sell'(예상 아님), 잠정 매도 예정
    (``planned_exits``)은 'sell' + 예상 표시(``is_exit_forecast``), 나머지는 'hold'.
    """
    planned = set(planned_exits)
    for held in holdings:
        if held.get("fill_date"):
            held["status"] = "sell"
            held["exit_reason"] = exit_reason
            held["is_exit_forecast"] = False
            continue
        hit = held["ticker"] in planned
        held["status"] = "sell" if hit else "hold"
        held["exit_reason"] = exit_reason if hit else None
        held["is_exit_forecast"] = hit
