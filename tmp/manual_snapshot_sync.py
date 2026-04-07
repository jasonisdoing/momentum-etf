import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.snapshot_service import update_today_snapshot_all_accounts
from utils.logger import get_app_logger

logger = get_app_logger()

if __name__ == "__main__":
    logger.info("Starting manual snapshot synchronization for today...")
    try:
        result = update_today_snapshot_all_accounts()
        logger.info(f"Successfully updated today's snapshot. Total Assets: {result['total_assets']}, Accounts: {result['account_count']}")
        print(f"\n✅ Snapshot Sync Successful: {result['total_assets']:,.0f} KRW for {result['account_count']} accounts.")
    except Exception as e:
        logger.error(f"Failed to sync snapshot: {e}")
        print(f"\n❌ Snapshot Sync Failed: {e}")
        sys.exit(1)
