#!/usr/bin/env python
"""
Migration script to initialize the Snapshots-based Holding Days system.
Sets all current holdings in portfolio_master to have a legacy snapshot on 2026-03-31
with days_held_int = 0, so that today (2026-04-01) becomes Day 1.
"""

import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.account_registry import load_account_configs
from utils.portfolio_io import load_portfolio_master, save_daily_snapshot
from utils.logger import get_app_logger

logger = get_app_logger()

def migrate():
    logger.info("Starting holdings migration to snapshots...")
    
    accounts = load_account_configs()
    if not accounts:
        logger.error("No accounts found.")
        return

    # 마이그레이션 기준 날짜 설정 (어제 날짜로 스냅샷 생성)
    # 오늘이 2026-04-01일 때, 어제(3/31) 기록을 0으로 만들어야 오늘이 1이 됨.
    target_date_str = "2026-03-31"
    
    from utils.db_manager import get_db_connection
    db = get_db_connection()
    if db is None:
        logger.error("DB connection failed.")
        return

    for account in accounts:
        account_id = account["account_id"]
        logger.info(f"Processing account: {account_id}")
        
        master = load_portfolio_master(account_id)
        if not master or not master.get("holdings"):
            logger.info(f"  No holdings found for {account_id}. Skipping.")
            continue
            
        holdings = master["holdings"]
        holding_details = []
        for h in holdings:
            ticker = str(h.get("ticker", "")).strip().upper()
            if ticker:
                holding_details.append({
                    "ticker": ticker,
                    "days_held_int": 0  # 어제 기준 0일로 설정
                })
        
        # 2026-03-31 스냅샷 강제 생성 (또는 업데이트)
        # save_daily_snapshot은 내부적으로 _now_kst()를 사용하므로, 
        # 마이그레이션을 위해 직접 DB에 박거나 save_daily_snapshot을 수정해야 함.
        # 여기서는 일회성이므로 직접 DB update_one을 수행함.
        
        snapshot_doc = {
            "snapshot_date": target_date_str,
            "total_assets": master.get("base_principal", 0.0) + master.get("base_cash", 0.0),
            "total_principal": master.get("base_principal", 0.0),
            "cash_balance": master.get("base_cash", 0.0),
            "valuation_krw": 0.0, # 마이그레이션 시점의 정확한 평가는 생략
            "accounts": [
                {
                    "account_id": account_id,
                    "total_assets": master.get("base_principal", 0.0) + master.get("base_cash", 0.0),
                    "total_principal": master.get("base_principal", 0.0),
                    "cash_balance": master.get("base_cash", 0.0),
                    "valuation_krw": 0.0,
                    "holdings": holding_details
                }
            ],
            "updated_at": datetime.now()
        }
        
        # 기존 스냅샷 확인 및 업데이트
        existing = db.daily_snapshots.find_one({"snapshot_date": target_date_str})
        if existing:
            # 해당 계좌 정보만 업데이트하거나 추가
            db.daily_snapshots.update_one(
                {"snapshot_date": target_date_str, "accounts.account_id": account_id},
                {"$set": {"accounts.$.holdings": holding_details}}
            )
            # 계좌가 없었다면 push
            db.daily_snapshots.update_one(
                {"snapshot_date": target_date_str, "accounts.account_id": {"$ne": account_id}},
                {"$push": {"accounts": snapshot_doc["accounts"][0]}}
            )
        else:
            db.daily_snapshots.insert_one(snapshot_doc)
            
        logger.info(f"  Successfully migrated {len(holding_details)} holdings for {account_id}")

    logger.info("Migration completed.")

if __name__ == "__main__":
    migrate()
