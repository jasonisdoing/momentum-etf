"""전략 사고팔기 슬랙 알림 배치 (무인자 실행 — 스케줄러용).

거래시간 중 10분마다 실행돼, 매수·매도 지정가에 현재가가 닿은 회차가 있을 때만
슬랙을 보낸다. 신호가 없으면 아무것도 보내지 않고 로그만 남긴다.
같은 ``회차-동작`` 은 하루 1회만 발송한다(중복 방지).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from utils.env import load_env_if_present
from utils.strategy_trade_notify import notify_strategy_trade


def main() -> int:
    load_env_if_present()
    result = notify_strategy_trade()
    signals = ", ".join(
        f"{t['round']}호 {t['ticker']} {'매도' if t['action'] == 'sell' else '매수'}" for t in result["triggers"]
    )
    print(
        f"[strategy_trade_notify] sent={result['sent']} "
        f"reason={result['reason']}" + (f" signals=[{signals}]" if signals else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
