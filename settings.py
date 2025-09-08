import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")


# 백테스트 기본값
INITIAL_CAPITAL = 100_000_000  # 1억

# 백테스트 기간 설정. ['YYYY-MM-DD', 'YYYY-MM-DD'] 형식으로 지정합니다.
TEST_DATE_RANGE = ["2025-01-01", "2025-09-07"]

# --- 포트폴리오 및 리스크 관리 설정 ---
PORTFOLIO_TOPN = 10  # 포트폴리오 최대 보유 종목 수
MIN_POSITION_PCT = 0.10  # 최소 포지션 비중
MAX_POSITION_PCT = 0.99  # 최대 포지션 비중
ENABLE_MAX_POSITION_TRIM = True  # 최대 비중 초과 시 부분매도 기능 활성화
COOLDOWN_DAYS = 0  # 매수/매도 후 반대 거래 금지 기간
HOLDING_STOP_LOSS_PCT = -10.0  # 보유 종목 공통 손절매 비율 (%)
