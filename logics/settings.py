# 'seykota' 전략 전용 설정

# Ed Seykota Trend Following Strategy
SEYKOTA_FAST_MA = 50       # 단기 이동평균 기간 (일)
SEYKOTA_SLOW_MA = 150      # 장기 이동평균 기간 (일)
SEYKOTA_STOP_LOSS_PCT = -10.0  # 보유 손절 임계값 (%)

# 포트폴리오 및 리스크 관리 설정
COOLDOWN_DAYS = 0              # 매수/매도 후 반대 방향 거래 금지 기간 (거래일)
MIN_POSITION_PCT = 0.10        # 포지션 최소 비중
MAX_POSITION_PCT = 0.20        # 포지션 최대 비중
ENABLE_MAX_POSITION_TRIM = True # 최대 비중 초과 시 부분매도 허용 여부