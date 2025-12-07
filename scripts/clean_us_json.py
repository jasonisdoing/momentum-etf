import json
import os

JSON_PATH = "zsettings/stocks/us.json"


def clean_categories():
    if not os.path.exists(JSON_PATH):
        print(f"Error: {JSON_PATH} not found")
        return

    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    # dedicated categories map
    # "미국지수" -> "👑 미국지수"
    # "💰 금융" and "금융" might be duplicates too? let's check

    # Let's find the target category "👑 미국지수"
    target_cat_name = "👑 미국지수"
    target_idx = -1

    # Also look for plain "미국지수"
    source_cat_name = "미국지수"
    source_idx = -1

    for i, item in enumerate(data):
        if item["category"] == target_cat_name:
            target_idx = i
        elif item["category"] == source_cat_name:
            source_idx = i

    if target_idx != -1 and source_idx != -1:
        print(f"Merging '{source_cat_name}' into '{target_cat_name}'...")
        target_tickers = data[target_idx]["tickers"]
        source_tickers = data[source_idx]["tickers"]

        # Merge source into target, avoiding duplicates
        existing_tickers = set()
        for t in target_tickers:
            if isinstance(t, dict):
                existing_tickers.add(t["ticker"])
            else:
                existing_tickers.add(t)

        for t in source_tickers:
            ticker_val = t["ticker"] if isinstance(t, dict) else t
            if ticker_val not in existing_tickers:
                target_tickers.append(t)
                existing_tickers.add(ticker_val)
                print(f"  Moved {ticker_val}")
            else:
                print(f"  Skipped {ticker_val} (duplicate)")

        # Remove source category
        data.pop(source_idx)

        # Save
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print("Merge complete.")
    else:
        print("Categories not found for merging, skipping.")

    # Check for other duplicates like "금융" vs "💰 금융"
    # User's list had "금융" mapped to "💰 금융" in my script so it should be fine,
    # but let's just be sure.


if __name__ == "__main__":
    clean_categories()
