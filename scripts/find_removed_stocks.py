import json
import re
import subprocess


def main():
    # 1. Load current tickers
    current_tickers = set()
    try:
        with open("zaccounts/kor_kr/stocks.json", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                current_tickers.add(item.get("ticker"))
    except Exception as e:
        print(f"Error reading stocks.json: {e}")
        return

    # 2. Get git log content
    # We use -U1000 to get full context or enough context, or just look for lines
    # actually we just want to find ANY mention of "ticker": "CODE" in the history
    cmd = ["git", "log", "-p", "--", "zaccounts/kor_kr/stocks.json"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    log_content = result.stdout

    # 3. Extract all tickers from log
    # Regex for "ticker": "XXXXXX"
    # We might match added lines (+) or removed lines (-) or context lines ( )
    # It doesn't matter, if it was in the file at some point, it will appear.
    # We just want to find tickers that are NOT in current_tickers.

    historical_tickers = set()
    matches = re.findall(r'"ticker":\s*"([^"]+)"', log_content)
    for m in matches:
        historical_tickers.add(m)

    # 4. Find removed
    removed = sorted(list(historical_tickers - current_tickers))

    print(f"Found {len(removed)} removed tickers:")
    for t in removed:
        # Try to find name in log
        name_match = re.search(r'"ticker":\s*"' + t + r'"\s*,\s*\n\s*[+\-]?\s*"name":\s*"([^"]+)"', log_content)
        name = name_match.group(1) if name_match else "Unknown"
        print(f"- {t} ({name})")


if __name__ == "__main__":
    main()
