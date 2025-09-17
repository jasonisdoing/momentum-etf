import glob
import json

import requests
import yfinance as yf
from pykrx.stock import get_etf_ticker_name


def update_etf_names():
    """Finds all JSON files in the data/ directory, checks for empty 'name' fields,
    and fills them using the appropriate API based on the filename.
    """
    json_files = glob.glob("data/*/*.json")

    # Prepare Bithumb coin map once
    coin_map = {}
    try:
        url = "https://api.bithumb.com/v1/market/all"
        headers = {"accept": "application/json"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        bithumb_data = response.json()
        if isinstance(bithumb_data, list):
            for coin_info in bithumb_data:
                market = coin_info.get("market")
                if market and market.startswith("KRW-"):
                    ticker = market.split("-")[1]
                    coin_map[ticker] = coin_info.get("korean_name")
    except Exception as e:
        print(f"Could not build Bithumb coin map: {e}")

    for file_path in json_files:
        if "etf.json" not in file_path:
            continue
        print(f"Processing {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            original_content = f.read()
            data = json.loads(original_content)

        changes_made = False
        for category in data:
            for ticker_info in category.get("tickers", []):
                if not ticker_info.get("name"):
                    ticker = ticker_info.get("ticker")
                    new_name = ""
                    try:
                        if "kor" in file_path:
                            new_name = get_etf_ticker_name(ticker)
                        elif "aus" in file_path:
                            yfinance_ticker = ticker.replace("ASX:", "") + ".AX"
                            stock_info = yf.Ticker(yfinance_ticker).info
                            new_name = stock_info.get("longName", stock_info.get("shortName", ""))
                        elif "coin" in file_path:
                            new_name = coin_map.get(ticker, "")

                        if new_name:
                            print(f"  Found name for {ticker}: {new_name}")
                            ticker_info["name"] = new_name
                            changes_made = True
                    except Exception as e:
                        print(f"  Could not find name for {ticker} in {file_path}: {e}")

        if changes_made:
            print(f"  Writing updated content to {file_path}")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    update_etf_names()
