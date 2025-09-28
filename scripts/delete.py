from datetime import datetime
from utils.db_manager import get_db_connection

TARGET = datetime(2024, 9, 20)
END = TARGET.replace(hour=23, minute=59, second=59, microsecond=999_999)

conn = get_db_connection()
if conn is None:
    raise SystemExit("DB connection not available")

query = {"date": {"$gte": TARGET, "$lte": END}}

for coll_name in ["signals", "daily_equities", "trades"]:
    result = conn[coll_name].delete_many(query)
    print(f"{coll_name}: deleted {result.deleted_count}")
