"""
전역 설정 파일

이 파일에서는 데이터베이스 연결 등 인프라 관련 설정을 정의합니다.
"""

# --- 데이터베이스 및 인프라 설정 ---
# MongoDB 연결 문자열. 보안을 위해 환경 변수 사용을 권장합니다.
# 예: "mongodb://user:password@host:port/"
MONGO_DB_CONNECTION_STRING = "mongodb+srv://jasonisdoing:bdqSPwnQ3H5mxN8V@cluster.m3jtdwa.mongodb.net/?retryWrites=true&w=majority&appName=Cluster"
MONGO_DB_NAME = "momentum_pilot_db"

# --- 웹앱 비밀번호 (선택 사항) ---
# WEBAPP_PASSWORD = "your_password"

