import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# Global settings for tfs tools

# 로그 출력 여부
# SHOW_LOGS = True
SHOW_LOGS = False

# 백테스트 기본값
INITIAL_CAPITAL = 100_000_000  # 1억

# 데이터 조회 구간(달 단위, 과거 기준 오프셋)
# 튠용: 과거 36개월 전부터 12개월 전까지
TUNE_MONTHS_RANGE = [12,0]
# 실행용: 과거 12개월 전부터 현재(0개월)까지
MONTHS_RANGE = [12, 0]


# 매도 조건 임계값(%) — 1~3주 수익률 합계가 이 값보다 작으면 전량 매도
# 기본값: -3.0
SELL_SUM_THRESHOLD = -3.0

# 매수 조건 임계값(%) — 1~3주 수익률 합계가 이 값보다 크면 매수 램프 시작/유지
# 기본값: +3.0
BUY_SUM_THRESHOLD = 3.0

# 보유 손절 임계값(%) — 보유 수익률이 이 값 이하로 떨어지면 즉시 전량 매도 (비활성화하려면 None)
HOLDING_STOP_LOSS_PCT = -3.0

# 쿨다운(거래일 수) — 매수/매도 발생 후 해당 일수 동안 반대 방향 거래 금지
# 기본값: 0 (금지 안함)
COOLDOWN_DAYS = 0

# (제거됨) 매수 MA 필터 관련 설정 제거

# 슈퍼트렌드(로그 표시용) 기본 파라미터
# ATR 기간과 배수(멀티플라이어)를 조정할 수 있습니다.
ST_ATR_PERIOD = 14
ST_ATR_MULTIPLIER = 3.0

# 급락 시 매도 금지 규칙
# 하루 등락률이 이 값(%) 이하로 떨어지면, 당일 포함 다음 N거래일 동안 매도 금지
BIG_DROP_PCT = -10.0
BIG_DROP_SELL_BLOCK_DAYS = 5

# 포트폴리오 Top-N 모드 설정
# 전체 자본(INITIAL_CAPITAL)을 신호가 좋은 상위 N개 종목에 투자합니다.
# 0이면 비활성(기존: 종목별 고정 자본 방식)
PORTFOLIO_TOPN = 20

# 포지션 최대 비중(포트폴리오 모드)
# 한 종목이 포트폴리오에서 차지할 수 있는 최대 비중(0.0~1.0)
# 예: 0.1 = 10% (기본값)
MIN_POSITION_PCT = 0.10
MAX_POSITION_PCT = 0.15

# 최대 비중 초과 시 부분매도 허용 여부(포트폴리오 모드)
# True이면 MAX_POSITION_PCT를 초과한 보유분을 당일 부분매도로 상한까지만 줄입니다.
ENABLE_MAX_POSITION_TRIM = True
