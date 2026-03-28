from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.ai_summary import generate_ai_summary_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI용 요약 TSV를 생성한다.")
    parser.add_argument("--account", required=True, help="대상 account_id")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    account_id = str(args.account or "").strip().lower()
    if not account_id:
        print(json.dumps({"error": "account_id가 필요합니다."}, ensure_ascii=False))
        return 1

    try:
        payload = generate_ai_summary_payload(account_id)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1

    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
