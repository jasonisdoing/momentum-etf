import datetime
from typing import Any

import pandas as pd
from bson import ObjectId

from utils.db_manager import get_db_connection
from utils.logger import get_app_logger

logger = get_app_logger()


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

    df_holdings["quantity"] = (
        pd.to_numeric(df_holdings["quantity"], errors="coerce").fillna(0.0).apply(np.floor).astype(int)
    )
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

    import streamlit as st

    # Initialize a warnings dict in session_state if it doesn't exist (if running in Streamlit)
    try:
        if "cache_warnings" not in st.session_state:
            st.session_state.cache_warnings = {}  # {account_id: {ticker1, ticker2, ...}}
    except Exception:
        pass

    def _get_current_price(row):
        ticker = str(row["ticker"]).strip().upper()
        df_cached = cached_frames.get(ticker)
        if df_cached is None or df_cached.empty:
            msg = f"가격 캐시에 '{ticker}'가 없습니다. 캐시 업데이트를 실행하세요."
            logger.warning(msg)
            # Add to session_state so the UI can display it
            try:
                if account_id not in st.session_state.cache_warnings:
                    st.session_state.cache_warnings[account_id] = set()
                st.session_state.cache_warnings[account_id].add(ticker)
            except Exception:
                pass
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

    float_cols = ["평균 매입가", "현재가", "수익률(%)"]
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


def load_portfolio_master(account_id: str) -> dict[str, Any] | None:
    """Load the current live balance (master) for a specific account from the consolidated document."""
    db = get_db_connection()
    if db is None:
        return None

    doc = db.portfolio_master.find_one({"master_id": "GLOBAL"})
    if not doc or "accounts" not in doc:
        return None

    for acc in doc["accounts"]:
        if acc["account_id"] == account_id:
            # Flatten for backward compatibility in functions that expect account-level dict
            return {
                "account_id": acc["account_id"],
                "total_principal": acc.get("total_principal", 0.0),
                "cash_balance": acc.get("cash_balance", 0.0),
                "holdings": acc.get("holdings", []),
                "updated_at": acc.get("updated_at"),
            }
    return None


def save_portfolio_master(
    account_id: str,
    holdings: list[dict[str, Any]],
    total_principal: float | None = None,
    cash_balance: float | None = None,
) -> bool:
    """Save/Update one account's balance within the consolidated portfolio_master document."""
    db = get_db_connection()
    if db is None:
        return False

    try:
        doc = db.portfolio_master.find_one({"master_id": "GLOBAL"})
        if not doc:
            doc = {"master_id": "GLOBAL", "accounts": []}

        accounts = doc.get("accounts", [])
        found = False

        for acc in accounts:
            if acc["account_id"] == account_id:
                if total_principal is not None:
                    acc["total_principal"] = float(total_principal)
                if cash_balance is not None:
                    acc["cash_balance"] = float(cash_balance)

                # Enforce integer quantity
                import math

                for h in holdings:
                    h["quantity"] = int(math.floor(float(h.get("quantity", 0.0))))

                acc["holdings"] = holdings
                acc["updated_at"] = datetime.datetime.now()
                found = True
                break

        if not found:
            # Enforce integer quantity
            import math

            for h in holdings:
                h["quantity"] = int(math.floor(float(h.get("quantity", 0.0))))

            accounts.append(
                {
                    "account_id": account_id,
                    "total_principal": float(total_principal or 0.0),
                    "cash_balance": float(cash_balance or 0.0),
                    "holdings": holdings,
                    "updated_at": datetime.datetime.now(),
                }
            )

        db.portfolio_master.update_one({"master_id": "GLOBAL"}, {"$set": {"accounts": accounts}}, upsert=True)
        return True
    except Exception as e:
        logger.error(f"Error saving portfolio master: {e}")
        return False


def save_daily_snapshot(
    account_id: str,
    total_assets: float,
    total_principal: float,
    cash_balance: float,
    valuation_krw: float,
) -> bool:
    """
    Save a daily snapshot.
    In the consolidated schema, 'TOTAL' values are stored in the root,
    and individual accounts in an 'accounts' array.
    """
    db = get_db_connection()
    if db is None:
        return False

    snapshot_date = datetime.datetime.now().strftime("%Y-%m-%d")

    try:
        # Find existing snapshot for today
        doc = db.daily_snapshots.find_one({"snapshot_date": snapshot_date})
        if not doc:
            doc = {
                "snapshot_date": snapshot_date,
                "total_assets": 0.0,
                "total_principal": 0.0,
                "cash_balance": 0.0,
                "valuation_krw": 0.0,
                "accounts": [],
                "updated_at": datetime.datetime.now(),
            }

        if account_id == "TOTAL":
            doc["total_assets"] = float(total_assets)
            doc["total_principal"] = float(total_principal)
            doc["cash_balance"] = float(cash_balance)
            doc["valuation_krw"] = float(valuation_krw)
        else:
            accounts = doc.get("accounts", [])
            found = False
            for acc in accounts:
                if acc["account_id"] == account_id:
                    acc["total_assets"] = float(total_assets)
                    acc["total_principal"] = float(total_principal)
                    acc["cash_balance"] = float(cash_balance)
                    acc["valuation_krw"] = float(valuation_krw)
                    found = True
                    break
            if not found:
                accounts.append(
                    {
                        "account_id": account_id,
                        "total_assets": float(total_assets),
                        "total_principal": float(total_principal),
                        "cash_balance": float(cash_balance),
                        "valuation_krw": float(valuation_krw),
                    }
                )
            doc["accounts"] = accounts

        doc["updated_at"] = datetime.datetime.now()

        db.daily_snapshots.update_one({"snapshot_date": snapshot_date}, {"$set": doc}, upsert=True)
        return True
    except Exception as e:
        logger.error(f"Error saving daily snapshot: {e}")
        return False


def get_latest_daily_snapshot(account_id: str, before_today: bool = True) -> dict[str, Any] | None:
    """Retrieve the latest daily snapshot for an account from the consolidated documents."""
    db = get_db_connection()
    if db is None:
        return None

    query = {}
    if before_today:
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        query["snapshot_date"] = {"$lt": today_str}

    try:
        cursor = db.daily_snapshots.find(query).sort("snapshot_date", -1).limit(1)
        results = list(cursor)
        if not results:
            return None

        doc = results[0]
        if account_id == "TOTAL":
            return doc

        for acc in doc.get("accounts", []):
            if acc["account_id"] == account_id:
                # Add date for context
                acc["snapshot_date"] = doc["snapshot_date"]
                return acc
        return None
    except Exception as e:
        logger.error(f"Error fetching latest daily snapshot: {e}")
        return None


def list_daily_snapshots(account_id: str | None = None) -> list[dict[str, Any]]:
    """
    List daily snapshots.
    If account_id is provided, returns account-specific data flattened.
    """
    db = get_db_connection()
    if db is None:
        return []

    try:
        all_docs = list(db.daily_snapshots.find().sort("snapshot_date", -1))

        if not account_id:
            return all_docs

        flattened = []
        for doc in all_docs:
            if account_id == "TOTAL":
                flattened.append(doc)
            else:
                for acc in doc.get("accounts", []):
                    if acc["account_id"] == account_id:
                        acc["_id"] = doc["_id"]  # Keep same ID for deletion if needed
                        acc["snapshot_date"] = doc["snapshot_date"]
                        flattened.append(acc)
        return flattened
    except Exception as e:
        logger.error(f"Error listing daily snapshots: {e}")
        return []


def delete_daily_snapshot(snapshot_id: str) -> bool:
    """Delete a daily snapshot (all accounts for that day) by its ID."""
    db = get_db_connection()
    if db is None:
        return False

    try:
        result = db.daily_snapshots.delete_one({"_id": ObjectId(snapshot_id)})
        return result.deleted_count > 0
    except Exception as e:
        logger.error(f"Error deleting daily snapshot: {e}")
        return False
