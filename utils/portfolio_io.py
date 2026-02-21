import datetime
from typing import Any

import pandas as pd

from utils.db_manager import get_db_connection
from utils.logger import get_app_logger

logger = get_app_logger()


def save_portfolio_snapshot(account_id: str, snapshot_date: str, df: pd.DataFrame) -> bool:
    """Save the parsed dataframe as a snapshot to portfolio_balances collection."""
    db = get_db_connection()
    if db is None:
        logger.error("DB connection failed in save_portfolio_snapshot.")
        return False

    try:
        # Load past history to carry-over first_buy_date
        history = list(db.portfolio_balances.find({"account_id": account_id}).sort("snapshot_date", 1))

        first_buy_lookup = {}
        for snap in history:
            for item in snap.get("holdings", []):
                ticker = item.get("ticker")
                if ticker and ticker not in first_buy_lookup:
                    first_buy_lookup[ticker] = item.get("first_buy_date") or snap.get("snapshot_date")

        holdings = []
        for _, row in df.iterrows():
            ticker = str(row["티커"])
            # Fallback to 2026-02-20 if no history exists for this ticker
            first_buy_date = first_buy_lookup.get(ticker, "2026-02-20")

            # Helper to parse string back to float safely
            def to_float(val: Any) -> float:
                if pd.isna(val):
                    return 0.0
                if isinstance(val, (int, float)):
                    return float(val)
                s = str(val).replace(",", "").replace("원", "").replace("%", "").strip()
                try:
                    return float(s)
                except ValueError:
                    return 0.0

            item = {
                "ticker": ticker,
                "name": str(row["종목명"]),
                "quantity": to_float(row["수량"]),
                "average_buy_price": to_float(row["평균 매입가"]),
                "currency": str(row["환종"]),
                "current_price": to_float(row.get("현재가", 0)),
                "valuation_krw": to_float(row.get("평가금액(KRW)", 0)),
                "purchase_krw": to_float(row.get("매입금액(KRW)", 0)),
                "first_buy_date": first_buy_date,
                "bucket": int(row["bucket"]) if "bucket" in row else 1,
            }
            holdings.append(item)

        doc = {
            "account_id": account_id,
            "snapshot_date": snapshot_date,
            "total_principal": 0.0,  # Placeholder for net principal tracking
            "cash_balance": 0.0,  # Placeholder for cash inputs
            "holdings": holdings,
            "updated_at": datetime.datetime.now(),
        }

        db.portfolio_balances.update_one(
            {"account_id": account_id, "snapshot_date": snapshot_date}, {"$set": doc}, upsert=True
        )
        return True
    except Exception as e:
        logger.error(f"Error saving portfolio snapshot: {e}")
        return False


def get_latest_portfolio_snapshot(account_id: str) -> dict[str, Any] | None:
    """Retrieve the latest portfolio snapshot for the account."""
    db = get_db_connection()
    if db is None:
        return None

    cursor = db.portfolio_balances.find({"account_id": account_id}).sort("snapshot_date", -1).limit(1)
    results = list(cursor)
    if results:
        return results[0]
    return None


def load_real_holdings_with_recommendations(account_id: str) -> pd.DataFrame | None:
    """
    Load the actual portfolio holdings from portfolio_master (live)
    and merge with recommendation data.
    """
    from utils.ui import load_account_recommendations

    # 1. Fetch live holdings from master only
    snapshot = load_portfolio_master(account_id)
    if not snapshot or not snapshot.get("holdings"):
        return None

    # 2. Fetch recommendations for scores, trends, etc.
    rec_df, _, _ = load_account_recommendations(account_id)

    # 3. Build holdings dataframe from raw master data
    holdings_list = snapshot["holdings"]
    df_holdings = pd.DataFrame(holdings_list)

    import numpy as np

    # Ensure required columns exist
    for col in ["ticker", "name", "quantity", "average_buy_price", "currency", "bucket", "first_buy_date"]:
        if col not in df_holdings.columns:
            df_holdings[col] = "" if col in ("ticker", "name", "currency", "first_buy_date") else 0

    df_holdings["quantity"] = pd.to_numeric(df_holdings["quantity"], errors="coerce").fillna(0.0)
    df_holdings["average_buy_price"] = pd.to_numeric(df_holdings["average_buy_price"], errors="coerce").fillna(0.0)

    # Calculate days held
    try:
        now = pd.Timestamp.now()
        df_holdings["first_buy_date"] = pd.to_datetime(df_holdings["first_buy_date"], errors="coerce")
        df_holdings["first_buy_date"] = df_holdings["first_buy_date"].fillna(now)
        df_holdings["보유일"] = (now - df_holdings["first_buy_date"]).dt.days
    except Exception as e:
        logger.warning(f"Error calculating dates: {e}")
        df_holdings["보유일"] = 0

    # Fetch prices from price cache and exchange rates
    from utils.cache_utils import load_cached_frames_bulk
    from utils.data_loader import get_exchange_rate_series

    tickers = df_holdings["ticker"].tolist()
    cached_frames = load_cached_frames_bulk(account_id, tickers)

    def _get_current_price(row):
        ticker = str(row["ticker"]).strip().upper()
        df_cached = cached_frames.get(ticker)
        if df_cached is None or df_cached.empty:
            logger.warning(f"가격 캐시에 '{ticker}'가 없습니다. 캐시 업데이트를 실행하세요.")
            return 0.0
        return float(df_cached["Close"].iloc[-1])

    import streamlit as st

    @st.cache_data(ttl=3600, show_spinner=False)
    def _get_cached_exchange_rates() -> dict[str, float]:
        import yfinance as yf

        rates = {"USD": 0.0, "AUD": 0.0}
        today_dt = datetime.datetime.today()

        # USD/KRW
        usd_krw_series = get_exchange_rate_series(today_dt - pd.Timedelta(days=5), today_dt)
        rates["USD"] = float(usd_krw_series.iloc[-1]) if not usd_krw_series.empty else 0.0

        # AUD/KRW
        try:
            aud_krw_df = yf.download("AUDKRW=X", period="5d", progress=False, auto_adjust=True)
            rates["AUD"] = float(aud_krw_df["Close"].dropna().iloc[-1]) if not aud_krw_df.empty else 0.0
        except Exception:
            pass

        return rates

    rates = _get_cached_exchange_rates()
    usd_krw = rates["USD"]
    aud_krw = rates["AUD"]

    def _get_multiplier(currency):
        if currency == "USD":
            return usd_krw
        elif currency == "AUD":
            return aud_krw
        return 1.0

    df_holdings["현재가"] = df_holdings.apply(_get_current_price, axis=1)
    multiplier = df_holdings["currency"].apply(_get_multiplier)
    df_holdings["매입금액(KRW)"] = (df_holdings["quantity"] * df_holdings["average_buy_price"] * multiplier).astype(
        float
    )
    df_holdings["평가금액(KRW)"] = (df_holdings["quantity"] * df_holdings["현재가"] * multiplier).astype(float)

    # Rename columns to match UI
    df_holdings = df_holdings.rename(
        columns={
            "ticker": "티커",
            "name": "종목명",
            "currency": "환종",
            "quantity": "수량",
            "average_buy_price": "평균 매입가",
            "bucket": "bucket_id",
        }
    )

    # Calculate derived columns
    df_holdings["평가손익(KRW)"] = (df_holdings["평가금액(KRW)"] - df_holdings["매입금액(KRW)"]).astype(float)
    df_holdings["수익률(%)"] = np.where(
        df_holdings["매입금액(KRW)"] > 0, (df_holdings["평가손익(KRW)"] / df_holdings["매입금액(KRW)"]) * 100, 0.0
    ).astype(float)

    float_cols = ["수량", "평균 매입가", "현재가", "수익률(%)"]
    for col in float_cols:
        if col in df_holdings.columns:
            df_holdings[col] = pd.to_numeric(df_holdings[col], errors="coerce").fillna(0.0).astype(float)

    # Round KRW money columns to int
    int_cols = ["매입금액(KRW)", "평가금액(KRW)", "평가손익(KRW)"]
    for col in int_cols:
        if col in df_holdings.columns:
            df_holdings[col] = pd.to_numeric(df_holdings[col], errors="coerce").fillna(0).round(0).astype(int)

    # Fill Bucket (버킷) from bucket_id
    from config import BUCKET_MAPPING

    df_holdings["버킷"] = df_holdings["bucket_id"].apply(lambda x: BUCKET_MAPPING.get(x, f"{x}. Bucket"))

    # 4. Merge with recommendations
    if rec_df is not None and not rec_df.empty:
        # We only want to pull in indicator columns from rec_df that aren't already in df_holdings
        df_holdings["상태"] = "HOLD"

        cols_to_pull = [
            "티커",
            "일간(%)",
            "1주(%)",
            "1달(%)",
            "3달(%)",
            "6달(%)",
            "12달(%)",
            "고점대비",
            "추세(3달)",
            "점수",
            "지속",
            "문구",
        ]
        cols_to_pull = [c for c in cols_to_pull if c in rec_df.columns]

        # Left join on Ticker
        df_merged = pd.merge(df_holdings, rec_df[cols_to_pull], on="티커", how="left")

        return df_merged
    else:
        # Fallback if no recommendation data
        df_holdings["상태"] = "HOLD"
        return df_holdings


def list_portfolio_snapshots(account_id: str | None = None) -> list[dict[str, Any]]:
    """List all portfolio snapshots, optionally filtered by account_id."""
    db = get_db_connection()
    if db is None:
        return []

    query = {}
    if account_id:
        query["account_id"] = account_id

    return list(db.portfolio_balances.find(query).sort([("snapshot_date", -1), ("account_id", 1)]))


def delete_portfolio_snapshot(account_id: str, snapshot_date: str) -> bool:
    """Delete a specific portfolio snapshot from the database."""
    db = get_db_connection()
    if db is None:
        return False

    try:
        result = db.portfolio_balances.delete_one({"account_id": account_id, "snapshot_date": snapshot_date})
        return result.deleted_count > 0
    except Exception as e:
        logger.error(f"Error deleting portfolio snapshot: {e}")
        return False


def load_portfolio_master(account_id: str) -> dict[str, Any] | None:
    """Load the current live balance (master) for an account."""
    db = get_db_connection()
    if db is None:
        return None
    return db.portfolio_master.find_one({"account_id": account_id})


def save_portfolio_master(account_id: str, holdings: list[dict[str, Any]]) -> bool:
    """Save the live balance to portfolio_master."""
    db = get_db_connection()
    if db is None:
        return False

    try:
        doc = {"account_id": account_id, "holdings": holdings, "updated_at": datetime.datetime.now()}
        db.portfolio_master.update_one({"account_id": account_id}, {"$set": doc}, upsert=True)
        return True
    except Exception as e:
        logger.error(f"Error saving portfolio master: {e}")
        return False


def close_portfolio_period(target_date: str, collection_name: str = "portfolio_weekly") -> bool:
    """Take a snapshot of current master for ALL accounts and save as one document."""
    from utils.account_registry import load_account_configs

    configs = load_account_configs()
    accounts_data = []

    for c in configs:
        acc_id = c["account_id"]
        master = load_portfolio_master(acc_id)
        if master and master.get("holdings"):
            accounts_data.append(
                {
                    "account_id": acc_id,
                    "total_principal": master.get("total_principal", 0.0),
                    "cash_balance": master.get("cash_balance", 0.0),
                    "holdings": master["holdings"],
                }
            )

    if not accounts_data:
        logger.warning(f"No master balances found to close for {target_date}.")
        return False

    db = get_db_connection()
    if db is None:
        return False

    try:
        doc = {"snapshot_date": target_date, "accounts": accounts_data, "updated_at": datetime.datetime.now()}
        db[collection_name].update_one({"snapshot_date": target_date}, {"$set": doc}, upsert=True)
        return True
    except Exception as e:
        logger.error(f"Error closing period {target_date} in {collection_name}: {e}")
        return False


def list_period_snapshots(collection_name: str = "portfolio_weekly") -> list[dict[str, Any]]:
    """List snapshots from the specified period collection."""
    db = get_db_connection()
    if db is None:
        return []

    return list(db[collection_name].find({}).sort("snapshot_date", -1))


def delete_period_snapshot(collection_name: str, snapshot_date: str) -> bool:
    """Delete a specific snapshot from the specified period collection."""
    db = get_db_connection()
    if db is None:
        return False

    try:
        db[collection_name].delete_one({"snapshot_date": snapshot_date})
        return True
    except Exception as e:
        logger.error(f"Error deleting snapshot from {collection_name}: {e}")
        return False
