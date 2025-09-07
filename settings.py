import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# Global settings for tfs tools

# 로그 출력 여부
# SHOW_LOGS = True
SHOW_LOGS = False

# 백테스트 기본값
INITIAL_CAPITAL = 100_000_000  # 1억

# 백테스트 기간 설정. ['YYYY-MM-DD', 'YYYY-MM-DD'] 형식으로 지정합니다.
TEST_DATE_RANGE = ["2025-01-01", "2025-09-05"]


