# 'donchian' 전략 전용 설정

# 이 파일에서는 'donchian' 전략에만 해당하는 고유 파라미터를 정의합니다.

# --- 전략 고유 파라미터 ---
# Richard Donchian-style trend-following
# 가격이 이동평균선 위에 있으면 매수, 아래에 있으면 매도합니다.
DONCHIAN_MA_PERIOD = 15
DONCHIAN_ENTRY_DELAY_DAYS = 0  # 이동평균선 돌파 후 진입 대기일. 0이면 즉시 진입.
