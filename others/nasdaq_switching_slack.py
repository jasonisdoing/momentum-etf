"""
나스닥 스위칭 전략 추천 결과를 Slack으로 전송하는 스크립트.
"""

import os
import sys
from datetime import datetime

# 프로젝트 루트 경로 추가
# 프로젝트 루트 경로 추가 (others/.. -> root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from others.nasdaq_switching import get_recommendation
from utils.logger import get_app_logger

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
except ImportError:
    WebClient = None
    SlackApiError = None

logger = get_app_logger()

# 대상 슬랙 채널 ID
TARGET_CHANNEL_ID = "C0A0X2LTS3X"


def run_nasdaq_switching_notification() -> None:
    """나스닥 스위칭 전략 추천을 생성하고 슬랙으로 전송합니다."""
    logger.info("[NASDAQ_SWITCH] 추천 생성 및 알림 시작")

    try:
        report = get_recommendation()
    except Exception as e:
        logger.error(f"[NASDAQ_SWITCH] 추천 생성 실패: {e}", exc_info=True)
        return

    # 메시지 포맷팅
    lines = report.get("status_lines", [])
    table_lines = report.get("table_lines", [])
    as_of = report.get("as_of", "N/A")
    target = report.get("target", "N/A")

    # 슬랙 메시지 본문 구성
    message_blocks = []

    # 1. 헤더
    message_blocks.append({"type": "header", "text": {"type": "plain_text", "text": "🇺🇸 나스닥 스위칭 전략 추천", "emoji": True}})

    # 2. 요약 정보
    summary_text = f"*기준일*: {as_of}\n*최종 타깃*: *{target}*"
    message_blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": summary_text}})

    # 3. 상세 테이블 (코드 블록)
    if table_lines:
        table_text = "\n".join(table_lines)
        message_blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"{table_text}"}})

    # 4. 상태 요약
    if lines:
        status_text = "\n".join(lines)
        message_blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": status_text}]})

    # 5. 채널 알림 (맨 아래 혹은 맨 위, 여기서는 맨 아래에 추가하거나 텍스트에 포함)
    # 사용자가 "항상 channel 를 언급" 원함.
    message_blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "<!channel>"}})

    # 슬랙 전송
    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        logger.warning("[NASDAQ_SWITCH] SLACK_BOT_TOKEN이 설정되지 않아 전송을 건너뜁니다.")
        return

    if not WebClient:
        logger.warning("[NASDAQ_SWITCH] slack_sdk가 설치되지 않아 전송을 건너뜁니다.")
        return

    client = WebClient(token=token)

    try:
        client.chat_postMessage(channel=TARGET_CHANNEL_ID, text=f"<!channel> 나스닥 스위칭 전략 추천 ({as_of})", blocks=message_blocks)
        logger.info(f"[NASDAQ_SWITCH] Slack 알림 전송 완료 (channel={TARGET_CHANNEL_ID})")
    except SlackApiError as e:
        logger.error(f"[NASDAQ_SWITCH] Slack 전송 실패: {e.response['error']}", exc_info=True)
    except Exception as e:
        logger.error(f"[NASDAQ_SWITCH] 알 수 없는 오류: {e}", exc_info=True)


if __name__ == "__main__":
    # 테스트 실행
    from utils.env import load_env_if_present

    load_env_if_present()
    run_nasdaq_switching_notification()
