"""
전역 설정 파일

이 파일에서는 데이터베이스 연결 등 인프라 관련 설정을 정의합니다.
"""

import os

from dotenv import load_dotenv

# .env 파일이 있다면 로드합니다.
load_dotenv()
# --- 웹앱 UI 및 마스터 데이터 관련 설정 ---

APP_DATE_TIME = "2025-09-29-17"

APP_TYPE = os.environ.get("APP_TYPE", f"APP-{APP_DATE_TIME}")
