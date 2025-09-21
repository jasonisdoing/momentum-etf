import argparse
import json
import os
import sys
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db_manager import get_account_settings
from utils.env import load_env_if_present


def json_datetime_serializer(obj):
    """datetime 객체를 JSON 직렬화 가능하도록 처리합니다."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def main():
    """지정된 계좌의 설정을 DB에서 직접 조회하여 출력합니다."""
    parser = argparse.ArgumentParser(description="지정된 계좌의 설정을 DB에서 확인합니다.")
    parser.add_argument("account", help="설정을 확인할 계좌 코드 (예: m1, a1, b1)")
    args = parser.parse_args()

    load_env_if_present()

    print(f"'{args.account}' 계좌의 설정을 DB에서 조회합니다...")
    settings = get_account_settings(args.account)

    if settings:
        print("\n[조회 결과]")
        # JSON 형식으로 예쁘게 출력
        print(json.dumps(settings, indent=2, ensure_ascii=False, default=json_datetime_serializer))

        portfolio_topn = settings.get("portfolio_topn")
        if portfolio_topn is not None:
            print(f"\n>>> PORTFOLIO_TOPN 값: {portfolio_topn}")
        else:
            print("\n>>> PORTFOLIO_TOPN 값이 설정되지 않았습니다.")

    else:
        print(f"\n'{args.account}' 계좌에 대한 설정을 찾을 수 없습니다.")


if __name__ == "__main__":
    main()
