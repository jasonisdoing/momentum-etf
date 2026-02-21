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


def load_portfolio_master(account_id: str) -> dict[str, Any] | None:
    """Load the current live balance (master) for an account."""
    db = get_db_connection()
    if db is None:
        return None
    return db.portfolio_master.find_one({"account_id": account_id})


def save_portfolio_master(
    account_id: str,
    holdings: list[dict[str, Any]],
    total_principal: float | None = None,
    cash_balance: float | None = None,
) -> bool:
    """Save the live balance to portfolio_master."""
    db = get_db_connection()
    if db is None:
        return False

    try:
        # Prevent overwriting cash/principal with 0 if not provided
        existing = load_portfolio_master(account_id)

        if total_principal is None:
            total_principal = existing.get("total_principal", 0.0) if existing else 0.0
        if cash_balance is None:
            cash_balance = existing.get("cash_balance", 0.0) if existing else 0.0

        doc = {
            "account_id": account_id,
            "total_principal": float(total_principal),
            "cash_balance": float(cash_balance),
            "holdings": holdings,
            "updated_at": datetime.datetime.now(),
        }
        db.portfolio_master.update_one({"account_id": account_id}, {"$set": doc}, upsert=True)
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
    """Save a daily snapshot of account totals to MongoDB."""
    db = get_db_connection()
    if db is None:
        return False

    snapshot_date = datetime.datetime.now().strftime("%Y-%m-%d")

    doc = {
        "account_id": account_id,
        "snapshot_date": snapshot_date,
        "total_assets": float(total_assets),
        "total_principal": float(total_principal),
        "cash_balance": float(cash_balance),
        "valuation_krw": float(valuation_krw),
        "updated_at": datetime.datetime.now(),
    }

    try:
        # Use upsert to only have one record per day per account
        db.daily_snapshots.update_one(
            {"account_id": account_id, "snapshot_date": snapshot_date}, {"$set": doc}, upsert=True
        )
        return True
    except Exception as e:
        logger.error(f"Error saving daily snapshot: {e}")
        return False


def get_latest_daily_snapshot(account_id: str, before_today: bool = True) -> dict[str, Any] | None:
    """Retrieve the latest daily snapshot for an account."""
    db = get_db_connection()
    if db is None:
        return None

    query = {"account_id": account_id}
    if before_today:
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        query["snapshot_date"] = {"$lt": today_str}

    try:
        cursor = db.daily_snapshots.find(query).sort("snapshot_date", -1).limit(1)
        results = list(cursor)
        return results[0] if results else None
    except Exception as e:
        logger.error(f"Error fetching latest daily snapshot: {e}")
        return None


def list_daily_snapshots(account_id: str | None = None) -> list[dict[str, Any]]:
    """List all daily snapshots, optionally filtered by account_id."""
    db = get_db_connection()
    if db is None:
        return []

    query = {}
    if account_id:
        query["account_id"] = account_id

    try:
        return list(db.daily_snapshots.find(query).sort("snapshot_date", -1))
    except Exception as e:
        logger.error(f"Error listing daily snapshots: {e}")
        return []


def delete_daily_snapshot(snapshot_id: str) -> bool:
    """Delete a daily snapshot by its ID."""
    db = get_db_connection()
    if db is None:
        return False

    try:
        result = db.daily_snapshots.delete_one({"_id": ObjectId(snapshot_id)})
        return result.deleted_count > 0
    except Exception as e:
        logger.error(f"Error deleting daily snapshot: {e}")
        return False
