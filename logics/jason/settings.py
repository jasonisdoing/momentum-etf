# 쿨다운(거래일 수) — 매수/매도 발생 후 해당 일수 동안 반대 방향 거래 금지
# 기본값: 0 (금지 안함)
COOLDOWN_DAYS = 0

# 급락 시 매도 금지 규칙
# 하루 등락률이 이 값(%) 이하로 떨어지면, 당일 포함 다음 N거래일 동안 매도 금지
BIG_DROP_PCT = -10.0
BIG_DROP_SELL_BLOCK_DAYS = 5

# 포트폴리오 Top-N 모드 설정
# 전체 자본(INITIAL_CAPITAL)을 신호가 좋은 상위 N개 종목에 투자합니다.
# 0이면 비활성(기존: 종목별 고정 자본 방식)
PORTFOLIO_TOPN = 10

# 포지션 최대 비중(포트폴리오 모드)
# 한 종목이 포트폴리오에서 차지할 수 있는 최대 비중(0.0~1.0)
# 예: 0.1 = 10% (기본값)
MIN_POSITION_PCT = 0.10
MAX_POSITION_PCT = 0.20

# 최대 비중 초과 시 부분매도 허용 여부(포트폴리오 모드)
# True이면 MAX_POSITION_PCT를 초과한 보유분을 당일 부분매도로 상한까지만 줄입니다.
ENABLE_MAX_POSITION_TRIM = True
