import yfinance as yf

ticker = yf.Ticker("^GSPC")
data = ticker.history(period="1d")
current_price = data["Close"].iloc[-1]

print(f"S&P 500 (^GSPC) 현재 종가: {current_price}")
