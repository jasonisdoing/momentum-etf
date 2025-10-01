from pymongo import MongoClient
from datetime import datetime


def check_aus_trades():
    try:
        # MongoDB 연결
        client = MongoClient("mongodb://localhost:27017/")
        db = client["trading"]

        # AUS 거래 내역 조회
        aus_trades = list(db.trades.find({"country": "aus"}).sort("date", 1))

        print(f"[DEBUG] Found {len(aus_trades)} AUS trades in database")

        for i, trade in enumerate(aus_trades[:10], 1):  # 처음 10개만 출력
            print(
                f"{i}. {trade.get('ticker')} - {trade.get('action')} {trade.get('quantity')} @ {trade.get('date')}"
            )

        if len(aus_trades) > 10:
            print(f"... and {len(aus_trades) - 10} more trades")

    except Exception as e:
        print(f"Error checking trades: {e}")


if __name__ == "__main__":
    check_aus_trades()
