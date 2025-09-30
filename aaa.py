import sys

sys.path.insert(0, ".")
from logic.signals.pipeline import _load_ticker_data
from utils.data_loader import fetch_bithumb_realtime_price
from utils.indicators import calculate_moving_average_signals
import pandas as pd

# 실제 시그널 생성과 동일한 전체 과정 시뮬레이션
try:
    base_date = pd.Timestamp.now().normalize()
    ma_period = 3

    print(f"기준일: {base_date}")

    # 1단계: 기존 데이터 로드
    df = _load_ticker_data("BTC", "coin", 1, base_date)
    print(f"기존 데이터 개수: {len(df) if df is not None else 0}")

    if df is not None and not df.empty:
        # 2단계: 실시간 가격 조회
        rt_price = fetch_bithumb_realtime_price("BTC")
        print(f"실시간 가격: {rt_price:,.0f}" if rt_price else "실시간 가격 조회 실패")

        if rt_price:
            # # 3단계: 기준일 데이터 추가 (시그널 생성과 동일한 방식)
            if base_date not in df.index:
                df.loc[base_date, "Close"] = rt_price
                print("기준일 데이터 추가 완료")

            # 4단계: 모멘텀 계산
            close_prices = df["Close"]
            print(close_prices)

            # close_prices = {
            #     '2025-09-27': 158109000.0,
            #     '2025-09-28': 162864000.0,
            #     '2025-09-29': 162941000.0,
            #     '2025-09-30': 162934000.0
            # }
            moving_avg, buy_signal_active, consecutive_buy_days = calculate_moving_average_signals(
                close_prices, ma_period
            )

            current_price = close_prices.iloc[-1] if not close_prices.empty else 0
            current_ma = moving_avg.iloc[-1] if not moving_avg.empty else 0

            if pd.notna(current_ma) and current_ma > 0:
                ma_score = round(((current_price / current_ma) - 1.0) * 100, 1)
                print(f"✅ 모멘텀 계산 성공: {ma_score}%")
            else:
                print("❌ 이동평균 계산 불가")
        else:
            print("❌ 실시간 가격 조회 실패")
    else:
        print("❌ 기존 데이터 로드 실패")

except Exception as e:
    print(f"오류: {e}")
    import traceback

    traceback.print_exc()
