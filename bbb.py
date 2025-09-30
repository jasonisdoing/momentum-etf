import pandas as pd
from utils.indicators import calculate_moving_average_signals

# 실제 데이터로 재현
data_before = {"2025-09-27": 158109000.0, "2025-09-28": 162864000.0, "2025-09-29": 162941000.0}

data_after = {
    "2025-09-27": 158109000.0,
    "2025-09-28": 162864000.0,
    "2025-09-29": 162941000.0,
    "2025-09-30": 162914000.0,
}

data_1month = {
    "2025-08-30": 150787000.0,
    "2025-08-31": 152030000.0,
    "2025-09-01": 154600000.0,
    "2025-09-02": 155588000.0,
    "2025-09-03": 153649000.0,
    "2025-09-04": 155001000.0,
    "2025-09-05": 155222000.0,
    "2025-09-06": 155114000.0,
    "2025-09-07": 156400000.0,
    "2025-09-08": 155688000.0,
    "2025-09-09": 157999000.0,
    "2025-09-10": 159143000.0,
    "2025-09-11": 160402000.0,
    "2025-09-12": 160216000.0,
    "2025-09-13": 160140000.0,
    "2025-09-14": 159899000.0,
    "2025-09-15": 160782000.0,
    "2025-09-16": 161303000.0,
    "2025-09-17": 163158000.0,
    "2025-09-18": 161986000.0,
    "2025-09-19": 162364000.0,
    "2025-09-20": 161704000.0,
    "2025-09-21": 160513000.0,
    "2025-09-22": 160268000.0,
    "2025-09-23": 161032000.0,
    "2025-09-24": 159847000.0,
    "2025-09-25": 158360000.0,
    "2025-09-26": 157299000.0,
    "2025-09-27": 158109000.0,
    "2025-09-28": 162864000.0,
    "2025-09-29": 162941000.0,
    "2025-09-30": 162914000.0,
}

# 실제 시그널 생성과 동일한 방식으로 모멘텀 계산
ma_period = 2
prices_series_before = pd.Series(list(data_before.values()))
moving_avg_before, _, _ = calculate_moving_average_signals(prices_series_before, ma_period)
current_price_before = prices_series_before.iloc[-1]
current_ma_before = moving_avg_before.iloc[-1] if not moving_avg_before.empty else 0
momentum_before = (
    ((current_price_before / current_ma_before) - 1) * 100 if current_ma_before > 0 else 0
)
# print(f'3일 가격: {prices_series_before}')
# print(f'이동평균: {moving_avg_before:,.0f}')
print(f"현재가: {current_price_before:,.0f}")
print(f"모멘텀: {momentum_before:.1f}%")

print("\n=== 기준일 데이터 추가 후 ===")
ma_period = 2
prices_series_after = pd.Series(list(data_after.values()))
moving_avg_after, _, _ = calculate_moving_average_signals(prices_series_after, ma_period)
current_price_after = prices_series_after.iloc[-1]
current_ma_after = moving_avg_after.iloc[-1] if not moving_avg_after.empty else 0
momentum_after = ((current_price_after / current_ma_after) - 1) * 100 if current_ma_after > 0 else 0
# print(f'3일 가격: {prices_series_after}')
# print(f'이동평균: {moving_avg_after:,.0f}')
print(f"현재가: {current_price_after:,.0f}")
print(f"모멘텀: {momentum_after:.1f}%")


print("\n=== 1달 데이터 + 기준일 데이터 추가 후 ===")
ma_period = 2
prices_series_1month = pd.Series(list(data_1month.values()))
moving_avg_1month, _, _ = calculate_moving_average_signals(prices_series_1month, ma_period)
current_price_1month = prices_series_1month.iloc[-1]
current_ma_1month = moving_avg_1month.iloc[-1] if not moving_avg_1month.empty else 0
momentum_1month = (
    ((current_price_1month / current_ma_1month) - 1) * 100 if current_ma_1month > 0 else 0
)
# print(f'3일 가격: {prices_series_1month}')
# print(f'이동평균: {moving_avg_1month:,.0f}')
print(f"현재가: {current_price_1month:,.0f}")
print(f"모멘텀: {momentum_1month:.1f}%")
