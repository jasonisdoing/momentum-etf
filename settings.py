"""
전역 설정 파일

이 파일에서는 데이터베이스 연결 등 인프라 관련 설정을 정의합니다.
"""

# --- 웹앱 UI 및 마스터 데이터 관련 설정 ---

# 벤치마크 티커 매핑 (국가별)
BENCHMARK_TICKERS = {
    "kor": "379800",  # KODEX 미국S&P500 ETF
    "aus": "IVV.AX",  # iShares S&P 500 (AUD)
}
