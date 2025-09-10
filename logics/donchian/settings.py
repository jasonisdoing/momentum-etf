# 'donchian' 전략 전용 설정

# 이 파일에서는 'donchian' 전략에만 해당하는 고유 파라미터를 정의합니다.

# --- 전략 고유 파라미터 ---
# Richard Donchian-style trend-following
# 가격이 이동평균선 위에 있으면 매수, 아래에 있으면 매도합니다.
DONCHIAN_MA_PERIOD_FOR_ETF = 15
DONCHIAN_MA_PERIOD_FOR_STOCK = 75

# 포트폴리오가 가득 찼을 때(PORTFOLIO_TOPN), 새로운 강한 신호가 발생한 종목이
# 기존 보유 종목 중 가장 점수가 낮은 종목보다 강할 경우 교체 매매를 실행할지 여부.
# True: 교체 매매 실행
# False: 신규 매수 안 함
DONCHIAN_REPLACE_WEAKER_STOCK = True

# 교체 매매 시, 새로운 종목의 점수가 기존 보유 종목의 점수보다
# 이 값(ATR 단위) 이상 높을 때만 교체를 실행합니다. (과도한 교체 방지)
DONCHIAN_REPLACE_SCORE_THRESHOLD = 1.5

# 하루에 교체 매매를 실행할 최대 종목 수를 제한합니다.
DONCHIAN_MAX_REPLACEMENTS_PER_DAY = 5

# 점수 정규화를 위한 ATR(Average True Range) 기간.
# 이격도를 ATR로 나누어 변동성을 고려한 점수를 계산합니다.
DONCHIAN_ATR_PERIOD_FOR_NORMALIZATION = 14
