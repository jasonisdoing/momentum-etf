"""leverage SMA 크로스 추천 배치 (무인자 래퍼).

server_scheduler/run_batch 는 `python <script.py>` 형태(무인자)로만 실행하므로,
한국·미국 두 시장의 추천 상태를 확정 저장하고(장 마감 직후 슬랙 알림 켜져 있으면 발송) 얇게 감싼다.

- 각 시장은 자기 거래 달력 기준으로 '장 마감 직후'일 때만 슬랙을 1회 보낸다(중복 방지).
- 따라서 이 배치는 두 시장의 마감 시각을 커버하도록 여러 번(크론) 호출돼도 안전하다.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leverage.config_store import sma_cross_profile  # noqa: E402
from utils.leverage_sma_service import persist_sma_cross_state  # noqa: E402


def main() -> None:
    for market in ("kor", "us"):
        profile = sma_cross_profile(market)
        try:
            result = persist_sma_cross_state(profile)
            print(
                f"[{profile}] status={result.get('market_status')} "
                f"saved={result.get('state_saved')} slack_sent={result.get('slack_sent')}"
            )
        except Exception as exc:  # 한 시장 실패가 다른 시장을 막지 않게
            print(f"[{profile}] 실패: {exc}")


if __name__ == "__main__":
    main()
