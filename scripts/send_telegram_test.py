"""
Send a simple Telegram test message using DB common settings.

Usage:
  python scripts/send_telegram_test.py
  python scripts/send_telegram_test.py --text "Hello from MomentumPilot"
"""

import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.env import load_env_if_present
from utils.notify import send_telegram_message, get_last_error


def main():
    p = argparse.ArgumentParser(description="Send a Telegram test message using DB common settings")
    p.add_argument("--text", type=str, default=None, help="Custom message text")
    args = p.parse_args()

    load_env_if_present()
    text = args.text or f"[테스트] MomentumPilot 핑 메시지 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"
    ok = send_telegram_message(text)
    if ok:
        print("[OK] Telegram message sent.")
    else:
        err = get_last_error()
        print(f"[ERROR] Failed to send message: {err or 'unknown error'}")


if __name__ == "__main__":
    main()

