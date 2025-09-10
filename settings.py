import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# 웹앱에서 이용하는 전략
KOR_STRATEGY = "donchian"
AUS_STRATEGY = "donchian"

KOR_INITIAL_CAPITAL = 180_000_000  # 1억 8천만원
AUS_INITIAL_CAPITAL = 71_000_000  # 7100만원

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

# --- 웹앱 보안 설정 ---
# 웹앱에 접근하기 위한 비밀번호.
# 비워두면 ('') 비밀번호 없이 바로 접근 가능합니다.
# 실제 배포 시에는 이 값을 비워두고, Streamlit Cloud나 Render의 Secrets 기능에
# 'WEBAPP_PASSWORD'라는 이름으로 비밀번호를 설정하는 것을 권장합니다.
WEBAPP_PASSWORD = "jason"
