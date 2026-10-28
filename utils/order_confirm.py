"""슬랙으로 요청한 주문의 체결 확인 — 전략 사고팔기·합성 알림 공용.

알림이 주문을 요청하면 그 주문을 **대기 이력**으로 남기고, 다음 배치마다 실제 보유를 읽어
체결됐으면 확인 한 줄을 보내고 이력에서 지운다. 체결 판정 기준만 알림마다 다르다
(사고팔기는 보유 회차 수, 합성은 종목별 보유 수량) — 판정 함수를 넘겨받는다.

확인은 **신호 유무와 무관하게** 매번 검사해야 한다 — 주문이 체결되면 그 지시가 바로
사라지므로, 신호가 있을 때만 검사하면 확인은 영영 나가지 못한다.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def build_confirm_line(label: str, action: str, detail: str) -> str:
    """확인 한 줄 — 조치가 이미 끝난 뒤라 `@channel` 은 붙이지 않는다."""
    word = "매도 확인" if action == "sell" else "매수 확인"
    return f"✅ *{label} {word}* · {detail}"


def pop_filled(
    pending: dict[str, dict[str, Any]],
    is_filled: Callable[[dict[str, Any]], bool],
) -> list[dict[str, Any]]:
    """대기 이력 중 체결된 것을 ``pending`` 에서 빼 돌려준다."""
    filled_keys = [key for key, entry in pending.items() if is_filled(entry)]
    return [pending.pop(key) for key in filled_keys]
