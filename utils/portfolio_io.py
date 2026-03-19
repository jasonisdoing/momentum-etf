import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from bson import ObjectId

from utils.db_manager import get_db_connection
from utils.logger import get_app_logger
from utils.settings_loader import get_account_settings

logger = get_app_logger()
KST = ZoneInfo("Asia/Seoul")


def _now_kst() -> datetime.datetime:
    """KST 기준 현재 시각을 반환한다."""
    return datetime.datetime.now(KST)


class MissingPriceCacheError(RuntimeError):
    """보유 종목의 가격 캐시가 누락된 경우 발생한다."""

    def __init__(self, account_id: str, tickers: list[str]):
        self.account_id = str(account_id or "").strip()
        self.tickers = sorted({str(ticker or "").strip().upper() for ticker in tickers if str(ticker or "").strip()})
        joined = ", ".join(self.tickers)
        super().__init__(f"[{self.account_id}] 가격 캐시 누락: {joined}")


def load_all_account_holding_tickers() -> set[str]:
    """전체 계좌의 실보유 티커 집합을 반환한다."""
    from utils.settings_loader import list_available_accounts

    held_tickers: set[str] = set()
    for account_id in list_available_accounts():
        snapshot = load_portfolio_master(account_id)
        if not snapshot:
            continue

        for holding in snapshot.get("holdings", []):
            ticker = str(holding.get("ticker") or "").strip().upper()
            if ticker:
                held_tickers.add(ticker)

    return held_tickers


def _apply_kor_realtime_overlay_to_holdings(df_holdings: pd.DataFrame) -> pd.DataFrame:
    """한국 종목 보유 테이블에 실시간 현재가/NAV/괴리율을 덮어쓴다."""
    tickers = [
        str(ticker or "").strip().upper() for ticker in df_holdings.get("ticker", []) if str(ticker or "").strip()
    ]
    if not tickers:
        return df_holdings

    try:
        from utils.data_loader import fetch_naver_etf_inav_snapshot

        realtime_data = fetch_naver_etf_inav_snapshot(tickers)
    except Exception as exc:
        logger.warning("보유 종목 실시간 오버레이 실패: %s", exc)
        return df_holdings

    if not realtime_data:
        return df_holdings

    overlaid = df_holdings.copy()
    overlaid["Nav"] = overlaid.get("Nav")
    overlaid["괴리율"] = overlaid.get("괴리율")

    for idx, row in overlaid.iterrows():
        ticker = str(row.get("ticker") or "").strip().upper()
        rt = realtime_data.get(ticker)
        if not rt:
            continue
        if rt.get("nowVal") is not None:
            overlaid.at[idx, "현재가"] = float(rt["nowVal"])
        if rt.get("changeRate") is not None:
            overlaid.at[idx, "일간(%)"] = float(rt["changeRate"])
        if rt.get("nav") is not None:
            overlaid.at[idx, "Nav"] = float(rt["nav"])
        if rt.get("deviation") is not None:
            overlaid.at[idx, "괴리율"] = float(rt["deviation"])

    return overlaid


def load_real_holdings_with_recommendations(
    account_id: str, *, strict_price_cache: bool = False
) -> pd.DataFrame | None:
    """
    Load the actual portfolio holdings from portfolio_master (live)
    and calculate display metrics directly from cached price data.
    """
    # 1. Fetch live holdings from master only
    snapshot = load_portfolio_master(account_id)
    if not snapshot or not snapshot.get("holdings"):
        return None

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
        from utils.formatters import format_trading_days

        now = pd.Timestamp.now().normalize()
        df_holdings["first_buy_date"] = (
            pd.to_datetime(df_holdings["first_buy_date"], errors="coerce").dt.tz_localize(None).dt.normalize()
        )
        df_holdings["first_buy_date"] = df_holdings["first_buy_date"].fillna(now)

        # Calculate business days
        bus_days = np.busday_count(
            df_holdings["first_buy_date"].values.astype("datetime64[D]"), now.to_datetime64().astype("datetime64[D]")
        )

        # Format the business days using format_trading_days
        df_holdings["보유일"] = [format_trading_days(max(d, 0)) for d in bus_days]
    except Exception as e:
        logger.warning(f"Error calculating dates: {e}")
        df_holdings["보유일"] = "-"

    # Fetch prices from price cache and exchange rates
    from utils.cache_utils import load_cached_frames_bulk_with_fallback
    from utils.data_loader import get_exchange_rate_series

    tickers = df_holdings["ticker"].tolist()
    cached_frames = load_cached_frames_bulk_with_fallback(account_id, tickers)
    missing_price_tickers: set[str] = set()

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
            missing_price_tickers.add(ticker)
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
        rates = {"USD": 0.0, "AUD": 0.0}
        today_dt = datetime.datetime.today()

        # USD/KRW
        usd_krw_series = get_exchange_rate_series(today_dt - pd.Timedelta(days=5), today_dt)
        if usd_krw_series.empty:
            raise RuntimeError("USD/KRW 환율 데이터를 가져오지 못했습니다.")
        rates["USD"] = float(usd_krw_series.iloc[-1])

        # AUD/KRW
        aud_krw_series = get_exchange_rate_series(today_dt - pd.Timedelta(days=5), today_dt, symbol="AUDKRW=X")
        if aud_krw_series.empty:
            raise RuntimeError("AUD/KRW 환율 데이터를 가져오지 못했습니다.")
        rates["AUD"] = float(aud_krw_series.iloc[-1])

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

    def _calc_period_return(close_series: pd.Series, days: int) -> float | None:
        try:
            series = pd.to_numeric(close_series, errors="coerce").dropna()
        except Exception:
            return None

        if series.empty:
            return None

        current = float(series.iloc[-1])
        if current <= 0:
            return None

        if len(series) > days:
            previous = float(series.iloc[-(days + 1)])
            if previous > 0:
                return (current / previous - 1.0) * 100.0

        if days == 252 and len(series) >= 240:
            previous = float(series.iloc[0])
            if previous > 0:
                return (current / previous - 1.0) * 100.0

        return None

    def _build_cached_metrics(ticker: str) -> dict[str, Any]:
        ticker_norm = str(ticker or "").strip().upper()
        cached_df = cached_frames.get(ticker_norm)
        if cached_df is None or cached_df.empty:
            return {
                "일간(%)": None,
                "1주(%)": None,
                "2주(%)": None,
                "1달(%)": None,
                "3달(%)": None,
                "6달(%)": None,
                "12달(%)": None,
                "고점대비": None,
                "추세(3달)": [],
            }

        close_col = "Close" if "Close" in cached_df.columns else "close"
        if close_col not in cached_df.columns:
            return {
                "일간(%)": None,
                "1주(%)": None,
                "2주(%)": None,
                "1달(%)": None,
                "3달(%)": None,
                "6달(%)": None,
                "12달(%)": None,
                "고점대비": None,
                "추세(3달)": [],
            }

        close_series = pd.to_numeric(cached_df[close_col], errors="coerce").dropna()
        if close_series.empty:
            return {
                "일간(%)": None,
                "1주(%)": None,
                "2주(%)": None,
                "1달(%)": None,
                "3달(%)": None,
                "6달(%)": None,
                "12달(%)": None,
                "고점대비": None,
                "추세(3달)": [],
            }

        current_price = float(close_series.iloc[-1])
        daily_pct = None
        if len(close_series) > 1:
            prev_close = float(close_series.iloc[-2])
            if prev_close > 0:
                daily_pct = ((current_price / prev_close) - 1.0) * 100.0

        max_price = float(close_series.max()) if not close_series.empty else 0.0
        drawdown = None
        if max_price > 0:
            drawdown = (current_price / max_price - 1.0) * 100.0

        return {
            "일간(%)": daily_pct,
            "1주(%)": _calc_period_return(close_series, 5),
            "2주(%)": _calc_period_return(close_series, 10),
            "1달(%)": _calc_period_return(close_series, 20),
            "3달(%)": _calc_period_return(close_series, 60),
            "6달(%)": _calc_period_return(close_series, 126),
            "12달(%)": _calc_period_return(close_series, 252),
            "고점대비": drawdown,
            "추세(3달)": close_series.iloc[-60:].astype(float).tolist(),
        }

    df_holdings["현재가"] = df_holdings.apply(_get_current_price, axis=1)
    if strict_price_cache and missing_price_tickers:
        raise MissingPriceCacheError(account_id, sorted(missing_price_tickers))

    try:
        account_settings = get_account_settings(account_id)
        account_country = str(account_settings.get("country_code") or "").strip().lower()
    except Exception:
        account_country = ""
    if account_country == "kor":
        df_holdings = _apply_kor_realtime_overlay_to_holdings(df_holdings)

    multiplier = df_holdings["currency"].apply(_get_multiplier)
    df_holdings["매입금액(KRW)"] = (df_holdings["quantity"] * df_holdings["average_buy_price"] * multiplier).astype(
        float
    )
    df_holdings["평가금액(KRW)"] = (df_holdings["quantity"] * df_holdings["현재가"] * multiplier).astype(float)

    # -----------------------------------------------------
    # Pseudo-holding logic for International Shares
    # -----------------------------------------------------
    intl_val = snapshot.get("intl_shares_value", 0.0)
    intl_change = snapshot.get("intl_shares_change", 0.0)
    # 계좌코드가 바뀌면 여기 조건도 함께 수정해야 International Shares 평가금이 합산됩니다.
    if account_id == "aus_account" and (intl_val > 0 or intl_change != 0):
        intl_princi = intl_val - intl_change

        intl_princi_krw = intl_princi * aud_krw
        intl_val_krw = intl_val * aud_krw

        # We append a row to df_holdings
        pseudo_row = {
            "ticker": "IS",
            "name": "International Shares",
            "quantity": 1,
            "average_buy_price": intl_princi,
            "currency": "AUD",
            "bucket": 3,  # "3. 시장지수"
            "first_buy_date": pd.Timestamp.now().normalize(),
            "보유일": "-",
            "현재가": intl_val,
            "매입금액(KRW)": intl_princi_krw,
            "평가금액(KRW)": intl_val_krw,
        }
        df_holdings = pd.concat([df_holdings, pd.DataFrame([pseudo_row])], ignore_index=True)

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

    metrics_rows = [_build_cached_metrics(ticker) for ticker in df_holdings["티커"].tolist()]
    metrics_df = pd.DataFrame(metrics_rows)
    for col in metrics_df.columns:
        df_holdings[col] = metrics_df[col]

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
            base_principal = acc.get("total_principal", 0.0)
            base_cash = acc.get("cash_balance", 0.0)
            cash_balance_native = acc.get("cash_balance_native")
            cash_currency = str(acc.get("cash_currency") or "").strip().upper()

            try:
                settings = get_account_settings(account_id)
                account_currency = str(settings.get("currency") or "").strip().upper()
            except Exception:
                account_currency = ""
            cash_currency = cash_currency or account_currency

            if cash_balance_native is None and cash_currency == "KRW":
                cash_balance_native = base_cash

            intl_val = acc.get("intl_shares_value", 0.0)
            intl_change = acc.get("intl_shares_change", 0.0)

            return {
                "account_id": acc["account_id"],
                "total_principal": base_principal,
                "cash_balance": base_cash,
                "cash_balance_native": cash_balance_native,
                "cash_currency": cash_currency,
                "base_principal": base_principal,
                "base_cash": base_cash,
                "intl_shares_value": intl_val,
                "intl_shares_change": intl_change,
                "holdings": acc.get("holdings", []),
                "updated_at": acc.get("updated_at"),
            }
    return None


def save_portfolio_master(
    account_id: str,
    holdings: list[dict[str, Any]],
    total_principal: float | None = None,
    cash_balance: float | None = None,
    cash_balance_native: float | None = None,
    cash_currency: str | None = None,
    intl_shares_value: float | None = None,
    intl_shares_change: float | None = None,
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
                if cash_balance_native is not None:
                    acc["cash_balance_native"] = float(cash_balance_native)
                if cash_currency is not None:
                    acc["cash_currency"] = str(cash_currency).strip().upper()
                if intl_shares_value is not None:
                    acc["intl_shares_value"] = float(intl_shares_value)
                if intl_shares_change is not None:
                    acc["intl_shares_change"] = float(intl_shares_change)

                # Enforce integer quantity
                import math

                for h in holdings:
                    h["quantity"] = int(math.floor(float(h.get("quantity", 0.0))))

                acc["holdings"] = holdings
                acc["updated_at"] = _now_kst()
                found = True
                break

        if not found:
            # Enforce integer quantity
            import math

            for h in holdings:
                h["quantity"] = int(math.floor(float(h.get("quantity", 0.0))))

            new_acc = {
                "account_id": account_id,
                "total_principal": float(total_principal or 0.0),
                "cash_balance": float(cash_balance or 0.0),
                "holdings": holdings,
                "updated_at": _now_kst(),
            }
            if cash_balance_native is not None:
                new_acc["cash_balance_native"] = float(cash_balance_native)
            if cash_currency is not None:
                new_acc["cash_currency"] = str(cash_currency).strip().upper()
            if intl_shares_value is not None:
                new_acc["intl_shares_value"] = float(intl_shares_value)
            if intl_shares_change is not None:
                new_acc["intl_shares_change"] = float(intl_shares_change)
            accounts.append(new_acc)

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
    purchase_amount: float | None = None,
) -> bool:
    """
    Save a daily snapshot.
    In the consolidated schema, 'TOTAL' values are stored in the root,
    and individual accounts in an 'accounts' array.
    """
    db = get_db_connection()
    if db is None:
        return False

    snapshot_date = _now_kst().strftime("%Y-%m-%d")

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
                "purchase_amount": 0.0,
                "accounts": [],
                "updated_at": _now_kst(),
            }

        if account_id == "TOTAL":
            doc["total_assets"] = float(total_assets)
            doc["total_principal"] = float(total_principal)
            doc["cash_balance"] = float(cash_balance)
            doc["valuation_krw"] = float(valuation_krw)
            if purchase_amount is not None:
                doc["purchase_amount"] = float(purchase_amount)
        else:
            accounts = doc.get("accounts", [])
            found = False
            for acc in accounts:
                if acc["account_id"] == account_id:
                    acc["total_assets"] = float(total_assets)
                    acc["total_principal"] = float(total_principal)
                    acc["cash_balance"] = float(cash_balance)
                    acc["valuation_krw"] = float(valuation_krw)
                    if purchase_amount is not None:
                        acc["purchase_amount"] = float(purchase_amount)
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
                        "purchase_amount": float(purchase_amount or 0.0),
                    }
                )
            doc["accounts"] = accounts

        doc["updated_at"] = _now_kst()

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
        today_str = _now_kst().strftime("%Y-%m-%d")
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
