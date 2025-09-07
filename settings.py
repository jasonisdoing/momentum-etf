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

# --- 티커 선택 설정 ---
# 'STATIC': data/tickers.txt 파일의 고정된 종목 사용
# 'DYNAMIC_WEEKLY': 매주 월요일, 시가총액 상위 Pool 내에서 주간 수익률 상위 종목으로 유니버스 교체
TICKER_UNIVERSE_MODE = 'DYNAMIC_WEEKLY'

# 동적 유니버스 모드에서 사용할 시가총액 상위 종목 수 (데이터 사전 로딩용)
DYNAMIC_UNIVERSE_POOL_SIZE = 100
EXCLUDE_KEWORDS = ['레버리지', '인버스']
TOP_PERFORMERS_COUNT = 10

# --- 포트폴리오 공통 설정 ---

# 포트폴리오 Top-N 모드 설정
# 전체 자본(INITIAL_CAPITAL)을 신호가 좋은 상위 N개 종목에 투자합니다.
# 0이면 비활성(개별 종목 고정 자본 방식)
PORTFOLIO_TOPN =10

# 포지션 최소/최대 비중 (포트폴리오 모드)
MIN_POSITION_PCT = 0.10
MAX_POSITION_PCT = 0.99

# 최대 비중 초과 시 부분매도 허용 여부 (포트폴리오 모드)
# True이면 MAX_POSITION_PCT를 초과한 보유분을 당일 부분매도로 상한까지만 줄입니다.
ENABLE_MAX_POSITION_TRIM = True

# --- 리스크 관리 공통 설정 ---
# 쿨다운(거래일 수) — 매수/매도 발생 후 해당 일수 동안 반대 방향 거래 금지
COOLDOWN_DAYS = 0

# 보유 손절 임계값(%) — 보유 수익률이 이 값 이하로 떨어지면 즉시 전량 매도 (비활성화하려면 None)
HOLDING_STOP_LOSS_PCT = -10.0
