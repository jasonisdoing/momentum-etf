import pandas as pd
import yfinance as yf

# yfinance를 통해 전체 기간 데이터 다운로드
data = yf.download("463690.KS", period="max", progress=False, auto_adjust=False)
print(data)

# yfinance가 MultiIndex 컬럼을 반환하는 경우 단일 레벨로 정리
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
    data = data.loc[:, ~data.columns.duplicated()]

# 중복된 인덱스가 있을 경우 마지막 항목만 남김
if not data.index.is_unique:
    data = data[~data.index.duplicated(keep="last")]

# 1. 상장일 업데이트
first_trading_day = data.index.min().strftime("%Y-%m-%d")
print(first_trading_day)
