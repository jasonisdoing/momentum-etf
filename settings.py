import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# Global settings for tfs tools

# 로그 출력 여부
# SHOW_LOGS = True
SHOW_LOGS = False

# 백테스트 기본값
INITIAL_CAPITAL = 100_000_000  # 1억

# 실행용: 과거 12개월 전부터 현재(0개월)까지
MONTHS_RANGE = [8, 0]
