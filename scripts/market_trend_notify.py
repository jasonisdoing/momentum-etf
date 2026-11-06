"""시장 추세 알림 배치 — 코스피~미국달러의 현재 추세를 나열하고 어제와 달라지면 @channel 로 알린다.

사용법:
    python scripts/market_trend_notify.py            # 발송(성공 시 기준 갱신)
    python scripts/market_trend_notify.py --dry-run  # 메시지만 출력(발송·저장 없음)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.env import load_env_if_present  # noqa: E402
from utils.market_trend_notify import notify_market_trend  # noqa: E402


def main() -> None:
    load_env_if_present()
    dry_run = "--dry-run" in sys.argv[1:]
    result = notify_market_trend(dry_run=dry_run)
    if dry_run:
        print(result["message"])
    print(f"[market_trend_notify] sent={result['sent']} changed={result['changed']}")


if __name__ == "__main__":
    main()
