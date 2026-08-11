import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from bson import ObjectId

from config import HIGH_WATERMARK_MONTHS
from services.price_service import get_exchange_rates, get_realtime_snapshot
from utils.asx_ticker import ensure_asx_prefix
from utils.db_manager import get_db_connection
from utils.logger import get_app_logger
from utils.settings_loader import get_account_settings

logger = get_app_logger()
KST = ZoneInfo("Asia/Seoul")


def _now_kst() -> datetime.datetime:
    """KST 기준 현재 시각을 반환한다."""
    return datetime.datetime.now(KST)


def _round_snapshot_money(value: Any) -> int:
    """스냅샷 KRW 금액을 정수로 반올림한다."""
    try:
        return int(round(float(value or 0)))
    except (TypeError, ValueError):
        return 0


def _resolve_snapshot_date() -> str:
    """자산 스냅샷의 기준일 — 일별 집계(`daily_fund_data`)와 같은 거래일을 쓴다.

    예전에는 KST 달력 날짜였다. 그러면 토요일에 스냅샷만 새 행이 생기는데 집계는
    금요일 행을 계속 갱신해서, `/assets` 의 금일 손익이 합계(집계 기준)와
    계좌별 값(스냅샷 기준)이 서로 다른 구간을 비교하게 됐다.
    """
    from utils.data_loader import resolve_active_trading_date

    return resolve_active_trading_date()


class MissingPriceCacheError(RuntimeError):
    """보유 종목의 가격 캐시가 누락된 경우 발생한다."""

    def __init__(self, ticker_type: str, tickers: list[str]):
        self.ticker_type = str(ticker_type or "").strip()
        self.tickers = sorted({str(ticker or "").strip().upper() for ticker in tickers if str(ticker or "").strip()})
        joined = ", ".join(self.tickers)
        super().__init__(f"[{self.ticker_type}] 가격 캐시 누락: {joined}")


def load_holding_accounts_by_ticker(country_code: str | None = None) -> dict[str, list[str]]:
    """실보유 티커 → 그 종목을 보유한 계좌명 목록.

    계좌 순서는 ``list_available_accounts()`` 순서를 따른다.
    country_code를 지정하면 해당 국가 계좌만 본다.
    """
    from utils.settings_loader import get_account_settings, list_available_accounts

    accounts_by_ticker: dict[str, list[str]] = {}
    for t_id in list_available_accounts():
        account_settings = get_account_settings(t_id)
        account_country = str(account_settings.get("country_code") or "").strip().lower()
        if country_code is not None and account_country != country_code.strip().lower():
            continue
        snapshot = load_portfolio_master(t_id)
        if not snapshot:
            continue

        account_name = str(account_settings.get("name") or t_id)
        for holding in snapshot.get("holdings", []):
            ticker = str(holding.get("ticker") or "").strip().upper()
            if account_country == "au" and ticker:
                ticker = ensure_asx_prefix(ticker)
            qty = float(holding.get("quantity") or 0)
            if ticker and qty > 0:
                names = accounts_by_ticker.setdefault(ticker, [])
                if account_name not in names:
                    names.append(account_name)

    return accounts_by_ticker


def load_all_holding_tickers(country_code: str | None = None) -> set[str]:
    """전체 계좌의 실보유 티커 집합을 반환한다.

    country_code를 지정하면 해당 국가 계좌의 보유 종목만 반환한다.
    """
    return set(load_holding_accounts_by_ticker(country_code))


def _apply_realtime_overlay_to_holdings(
    df_holdings: pd.DataFrame,
    country_code: str,
    realtime_data: dict[str, dict[str, float]] | None = None,
    only_tickers: set[str] | None = None,
) -> pd.DataFrame:
    """보유 종목 테이블에 실시간 현재가/NAV/괴리율 등을 덮어쓴다.

    only_tickers 를 주면 그 티커들만 해당 시장 API 로 조회·적용한다
    (한 계좌에 여러 시장 종목이 섞여 있을 때 시장별로 나눠 호출하기 위함).
    """
    tickers = [
        str(ticker or "").strip().upper() for ticker in df_holdings.get("ticker", []) if str(ticker or "").strip()
    ]
    if only_tickers is not None:
        tickers = [ticker for ticker in tickers if ticker in only_tickers]
    if not tickers:
        return df_holdings

    if realtime_data is None:
        try:
            realtime_data = get_realtime_snapshot(country_code, tickers)
        except Exception as exc:
            logger.warning("보유 종목 실시간 오버레이 실패 (%s): %s", country_code, exc)
            return df_holdings

    if not realtime_data:
        return df_holdings

    overlaid = df_holdings.copy()
    # 필요한 컬럼 보장
    for col in ["Nav", "괴리율"]:
        if col not in overlaid.columns:
            overlaid[col] = None

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


def load_real_holdings_table(
    account_id: str,
    *,
    strict_price_cache: bool = False,
    preloaded_exchange_rates: dict[str, Any] | None = None,
    preloaded_kor_realtime_snapshot: dict[str, dict[str, float]] | None = None,
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
    holdings_list = list(snapshot["holdings"])
    for index, holding in enumerate(holdings_list):
        if "sort_order" not in holding:
            holding["sort_order"] = index
    holdings_list.sort(key=lambda holding: int(holding.get("sort_order", 0)))
    df_holdings = pd.DataFrame(holdings_list)

    # 4. 동적 버킷 및 명칭 매핑: 개별 항목에 저장된 정보 대신 종목풀(stock_meta)의 최신 정보를 사용
    from utils.db_manager import get_db_connection
    from utils.settings_loader import get_account_settings

    db = get_db_connection()
    if db is not None and not df_holdings.empty:
        all_tickers = df_holdings["ticker"].unique().tolist()

        # 같은 티커가 여러 시장(stock_meta 문서 여러 개, 예: 미국·호주 IOO)일 수 있으므로 티커별로 문서를 모은다.
        docs_by_ticker: dict[str, list[dict[str, Any]]] = {}
        cursor = db.stock_meta.find(
            {"ticker": {"$in": all_tickers}, "is_deleted": {"$ne": True}},
            {"ticker": 1, "bucket": 1, "name": 1, "ticker_type": 1, "is_etf": 1},
        )
        for doc in cursor:
            docs_by_ticker.setdefault(doc["ticker"], []).append(doc)

        # 종목풀(ticker_type) → 통화 매핑 캐시. 보유의 통화로 올바른 시장 문서를 고르기 위해 사용.
        from utils.cash_model import currency_for_country
        from utils.settings_loader import get_ticker_type_settings

        _tt_currency: dict[str, str] = {}

        def _ticker_type_currency(ticker_type: object) -> str:
            key = str(ticker_type or "").strip().lower()
            if key not in _tt_currency:
                try:
                    country = str(get_ticker_type_settings(key).get("country_code") or "").strip().lower()
                except Exception:
                    country = ""
                _tt_currency[key] = currency_for_country(country)
            return _tt_currency[key]

        def _pick_meta_doc(ticker: str, currency: object) -> dict[str, Any] | None:
            docs = docs_by_ticker.get(ticker) or []
            if not docs:
                return None
            if len(docs) == 1:
                return docs[0]
            # 여러 시장 — 보유 통화와 일치하는 시장 문서를 우선(없으면 첫 문서).
            cur = str(currency or "").strip().upper()
            for doc in docs:
                if _ticker_type_currency(doc.get("ticker_type")) == cur:
                    return doc
            return docs[0]

        picked_docs = [
            _pick_meta_doc(str(row.get("ticker") or ""), row.get("currency")) for _, row in df_holdings.iterrows()
        ]

        # 데이터 업데이트 (종목풀 정보 우선 적용, 다시장은 보유 통화 기준 문서 사용)
        df_holdings["bucket"] = [(doc or {}).get("bucket", 1) for doc in picked_docs]
        df_holdings["name"] = [
            ((doc.get("name") if doc else None) or orig_name or ticker)
            for doc, orig_name, ticker in zip(
                picked_docs, list(df_holdings.get("name", [])), list(df_holdings["ticker"])
            )
        ]
        df_holdings["ticker_type"] = [(doc.get("ticker_type") if doc else "") or "" for doc in picked_docs]
        df_holdings["is_etf"] = [bool(doc.get("is_etf", False)) if doc else False for doc in picked_docs]

        # 계좌의 country_code 찾아와서 부여
        try:
            account_info = get_account_settings(account_id)
            account_country = account_info.get("country_code", "kor")
        except Exception:
            account_country = "kor"

        df_holdings["country_code"] = account_country

        # ticker_type이 없는 미등록 종목인 경우, 국가 코드를 기반으로 기본값 할당
        def _fallback_ticker_type(row):
            if row.get("ticker_type"):
                return row["ticker_type"]
            c_code = row.get("country_code", "kor")
            return "us" if c_code == "us" else "aus" if c_code == "au" else "kor"

        df_holdings["ticker_type"] = df_holdings.apply(_fallback_ticker_type, axis=1)

    import numpy as np

    # Ensure required columns exist
    for col in [
        "ticker",
        "name",
        "quantity",
        "average_buy_price",
        "currency",
        "bucket",
        "first_buy_date",
        "last_buy_date",
        "memo",
    ]:
        if col not in df_holdings.columns:
            df_holdings[col] = (
                "" if col in ("ticker", "name", "currency", "first_buy_date", "last_buy_date", "memo") else 0
            )

    df_holdings["memo"] = df_holdings["memo"].fillna("").astype(str)

    df_holdings["quantity"] = (
        pd.to_numeric(df_holdings["quantity"], errors="coerce").fillna(0.0).apply(np.floor).astype(int)
    )
    df_holdings["average_buy_price"] = pd.to_numeric(df_holdings["average_buy_price"], errors="coerce").fillna(0.0)

    # 보유일 계산은 화면/슬랙에서 사용하지 않으므로 제거됨.
    # DB 의 first_buy_date / last_buy_date 필드는 그대로 유지.

    # Fetch prices from price cache and exchange rates
    # 다시장 티커(예: 미국·호주 IOO)가 올바른 시장 가격을 쓰도록, 보유별 ticker_type 의 캐시를 우선 사용한다.
    # (전체 종목풀에서 먼저 걸리는 캐시를 쓰면 호주/미국이 뒤바뀔 수 있다.)
    from utils.cache_utils import (
        load_cached_frames_bulk_from_all_ticker_types,
        load_cached_frames_bulk_from_ticker_types,
    )

    tickers = df_holdings["ticker"].tolist()
    tickers_by_type: dict[str, list[str]] = {}
    for _, row in df_holdings.iterrows():
        ticker_type = str(row.get("ticker_type") or "").strip().lower()
        ticker_upper = str(row.get("ticker") or "").strip().upper()
        if ticker_type and ticker_upper:
            tickers_by_type.setdefault(ticker_type, []).append(ticker_upper)

    cached_frames: dict[str, pd.DataFrame] = {}
    for ticker_type, type_tickers in tickers_by_type.items():
        fetched = load_cached_frames_bulk_from_ticker_types([ticker_type], type_tickers)
        for ticker_key, frame in fetched.items():
            cached_frames.setdefault(str(ticker_key).strip().upper(), frame)

    # ticker_type 이 없거나 그 종목풀 캐시에 없던 티커는 전체 종목풀에서 폴백 조회.
    missing_for_cache = [tk for tk in tickers if str(tk).strip().upper() not in cached_frames]
    if missing_for_cache:
        fallback = load_cached_frames_bulk_from_all_ticker_types(missing_for_cache)
        for ticker_key, frame in fallback.items():
            cached_frames.setdefault(str(ticker_key).strip().upper(), frame)

    missing_price_tickers: set[str] = set()

    def _get_current_price(row):
        ticker = str(row["ticker"]).strip().upper()
        df_cached = cached_frames.get(ticker)
        if df_cached is None or df_cached.empty:
            msg = f"가격 캐시에 '{ticker}'가 없습니다. 캐시 업데이트를 실행하세요."
            logger.warning(msg)
            missing_price_tickers.add(ticker)
            return 0.0
        return float(df_cached["Close"].iloc[-1])

    rates = preloaded_exchange_rates if preloaded_exchange_rates is not None else get_exchange_rates()
    usd_krw = float((rates.get("USD") or {}).get("rate"))
    aud_krw = float((rates.get("AUD") or {}).get("rate"))

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
                "고점": None,
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
                "고점": None,
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
                "고점": None,
                "추세(3달)": [],
            }

        current_price = float(close_series.iloc[-1])
        daily_pct = None
        if len(close_series) > 1:
            prev_close = float(close_series.iloc[-2])
            if prev_close > 0:
                daily_pct = ((current_price / prev_close) - 1.0) * 100.0

        # 고점 대비(%) — 순위 화면과 같은 규칙: 최근 HIGH_WATERMARK_MONTHS(12개월) 최고가 대비.
        high_window = close_series.loc[close_series.index[-1] - pd.DateOffset(months=HIGH_WATERMARK_MONTHS) :]
        max_price = float(high_window.max()) if not high_window.empty else 0.0
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
            "고점": drawdown,
            "추세(3달)": close_series.iloc[-60:].astype(float).tolist(),
        }

    df_holdings["현재가"] = df_holdings.apply(_get_current_price, axis=1)

    # 수익률 계산 (매입 단가 대비 현재가, 소수점 1자리)
    def _calc_return_pct(row):
        buy = float(row.get("average_buy_price") or 0)
        curr = float(row.get("현재가") or 0)
        if buy > 0:
            return round(((curr / buy) - 1.0) * 100.0, 1)
        return 0.0

    df_holdings["return_pct"] = df_holdings.apply(_calc_return_pct, axis=1)

    if missing_price_tickers:
        df_holdings.attrs["missing_price_tickers"] = sorted(missing_price_tickers)
    if strict_price_cache and missing_price_tickers:
        raise MissingPriceCacheError(account_id, sorted(missing_price_tickers))

    # 실시간 오버레이 — 계좌 국가가 아니라 "보유 종목의 통화" 기준으로 시장별 API 를 나눠 호출한다.
    # kor 계좌가 미국 종목을 담아도 토스 실시간(프리장·애프터 포함)이 적용된다.
    country_by_currency = {"KRW": "kor", "USD": "us", "AUD": "au"}
    tickers_by_market: dict[str, set[str]] = {}
    for _, holding_row in df_holdings.iterrows():
        market = country_by_currency.get(str(holding_row.get("currency") or "").strip().upper())
        ticker_key = str(holding_row.get("ticker") or "").strip().upper()
        # IS 는 시장에 없는 가상 티커 — 가격·일간은 위에서 VGS 프록시로 이미 채웠다.
        if market and ticker_key and ticker_key != "IS":
            tickers_by_market.setdefault(market, set()).add(ticker_key)
    for market, market_tickers in tickers_by_market.items():
        df_holdings = _apply_realtime_overlay_to_holdings(
            df_holdings,
            country_code=market,
            realtime_data=preloaded_kor_realtime_snapshot if market == "kor" else None,
            only_tickers=market_tickers,
        )

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
    # 종목풀이 바뀌어 여기 조건도 함께 수정해야 International Shares 평가금이 합산됩니다.
    if account_id == "aus_account":
        intl_princi = intl_val - intl_change

        intl_princi_krw = intl_princi * aud_krw
        intl_val_krw = intl_val * aud_krw

        # IS 를 실제 VGS 보유처럼 표시한다 — 수량 = 평가액 ÷ VGS 가격, 일간(%) = VGS 변동률.
        # (수량 × 현재가 = 입력한 평가액 항등식이 유지되므로 평가액·손익·수익률은 변하지 않는다)
        # VGS 가격은 실시간(QuoteAPI) 우선, 실패 시 aus 캐시 종가. 둘 다 없으면 기존 방식
        # (수량 1, 현재가 = 평가액, 일간 미표시)으로 표시한다 — 임의 값을 만들지 않는다.
        is_proxy_ticker = "ASX:VGS"
        vgs_price = None
        vgs_daily_pct = None
        try:
            proxy_quote = get_realtime_snapshot("au", [is_proxy_ticker]).get(is_proxy_ticker) or {}
            if proxy_quote.get("nowVal"):
                vgs_price = float(proxy_quote["nowVal"])
                if proxy_quote.get("changeRate") is not None:
                    vgs_daily_pct = float(proxy_quote["changeRate"])
        except Exception as exc:
            logger.warning("IS 프록시(VGS) 실시간 조회 실패: %s", exc)
        if vgs_price is None:
            try:
                from utils.cache_utils import load_cached_frame

                vgs_frame = load_cached_frame("aus", is_proxy_ticker)
                closes = pd.to_numeric(vgs_frame["Close"], errors="coerce").dropna()
                if not closes.empty:
                    vgs_price = float(closes.iloc[-1])
                    if len(closes) > 1 and float(closes.iloc[-2]) > 0:
                        vgs_daily_pct = (vgs_price / float(closes.iloc[-2]) - 1.0) * 100.0
            except Exception as exc:
                logger.warning("IS 프록시(VGS) 캐시 조회 실패: %s", exc)

        if vgs_price is not None and vgs_price > 0 and intl_val > 0:
            is_quantity = intl_val / vgs_price
            is_price = vgs_price
            is_avg_price = intl_princi / is_quantity
            is_daily_pct = vgs_daily_pct
        else:
            is_quantity = 1.0
            is_price = intl_val
            is_avg_price = intl_princi
            is_daily_pct = None

        # We append a row to df_holdings
        pseudo_row = {
            "ticker": "IS",
            # 표시명은 가격 프록시(VGS)의 정식 명칭으로 통일한다 — 내부 티커·고정자산 역할은 IS 유지.
            "name": "Vanguard MSCI Index International Shares ETF",
            "quantity": is_quantity,
            "average_buy_price": is_avg_price,
            "currency": "AUD",
            "bucket": 2,  # "2. 시장지수"
            "first_buy_date": pd.Timestamp.now().normalize(),
            "현재가": is_price,
            "매입금액(KRW)": intl_princi_krw,
            "평가금액(KRW)": intl_val_krw,
            "일간(%)": is_daily_pct,
            "is_etf": False,
            "country_code": "au",
            "ticker_type": "aus",
        }
        # IS 위치: 저장된 순서(intl_shares_sort_order)가 있으면 그 자리에, 없으면 맨 뒤.
        # 실보유 sort_order 는 0..n-1 정수이므로 (위치 - 0.5) 로 끼워 넣은 뒤 재정렬한다.
        stored_is_order = snapshot.get("intl_shares_sort_order")
        pseudo_row["sort_order"] = (
            float(stored_is_order) - 0.5 if stored_is_order is not None else float(len(df_holdings))
        )
        df_holdings = pd.concat([df_holdings, pd.DataFrame([pseudo_row])], ignore_index=True)
        df_holdings["sort_order"] = pd.to_numeric(df_holdings["sort_order"], errors="coerce").fillna(0)
        df_holdings = df_holdings.sort_values("sort_order", kind="stable").reset_index(drop=True)
        df_holdings["sort_order"] = range(len(df_holdings))
        # Ensure value columns are numeric after concat
        for col in ["수량", "평균 매입가", "매입금액(KRW)", "평가금액(KRW)"]:
            if col in df_holdings.columns:
                df_holdings[col] = pd.to_numeric(df_holdings[col], errors="coerce").fillna(0)

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

    # 비중(Portfolio Weight %) 계산
    # 수량이 0인 종목을 포함한 모든 종목의 평가액 합계와 현금을 합산하여 '총 자산' 기준 비중 계산
    vals_for_sum = pd.to_numeric(df_holdings["평가금액(KRW)"], errors="coerce").fillna(0)
    cash_val = pd.to_numeric(snapshot.get("cash_balance", 0), errors="coerce") or 0
    total_assets = vals_for_sum.sum() + cash_val

    if total_assets > 0:
        df_holdings["weight_pct"] = (vals_for_sum / total_assets * 100).round(2)
    else:
        df_holdings["weight_pct"] = 0.0

    # 소수점 반올림 및 타입 변환 처리
    price_digits = 4 if account_country in ("us", "au") else 0
    percent_cols = ["수익률(%)", "일간(%)", "1주(%)", "2주(%)", "1달(%)", "3달(%)", "6달(%)", "12달(%)", "고점"]
    price_cols = ["평균 매입가", "현재가", "Nav", "괴리율"]
    int_cols = ["매입금액(KRW)", "평가금액(KRW)", "평가손익(KRW)"]

    for col in percent_cols:
        if col in df_holdings.columns:
            df_holdings[col] = pd.to_numeric(df_holdings[col], errors="coerce").round(2)

    # 가격 반올림 자리수는 계좌가 아니라 "종목 통화"별로 정한다(다통화 계좌 지원).
    # USD/AUD 4자리, KRW 0자리. 통화 컬럼(환종)이 없으면 계좌 기준(price_digits)으로 폴백.
    if "환종" in df_holdings.columns:
        row_digits = df_holdings["환종"].astype(str).str.upper().map(lambda cur: 4 if cur in ("USD", "AUD") else 0)
    else:
        row_digits = pd.Series(price_digits, index=df_holdings.index)
    for col in price_cols:
        if col in df_holdings.columns:
            numeric = pd.to_numeric(df_holdings[col], errors="coerce")
            df_holdings[col] = [
                round(float(value), int(digits)) if pd.notna(value) else value
                for value, digits in zip(numeric, row_digits)
            ]

    for col in int_cols:
        if col in df_holdings.columns:
            df_holdings[col] = pd.to_numeric(df_holdings[col], errors="coerce").fillna(0).round(0).astype(int)

    # Fill Bucket (버킷) from bucket_id
    from config import BUCKET_MAPPING

    df_holdings["버킷"] = df_holdings["bucket_id"].apply(lambda x: BUCKET_MAPPING.get(x, f"{x}. Bucket"))

    metrics_rows = [_build_cached_metrics(ticker) for ticker in df_holdings["티커"].tolist()]
    metrics_df = pd.DataFrame(metrics_rows)
    for col in metrics_df.columns:
        if col == "일간(%)" and col in df_holdings.columns:
            # 실시간 오버레이가 이미 값을 넣었을 수 있으므로, 비어있는 경우에만 캐시값으로 채움.
            # (컬럼이 없을 때 df.get() 의 빈 Series 에 fillna 하면 전부 NaN 이 되는 버그가 있었다)
            df_holdings[col] = df_holdings[col].fillna(metrics_df[col])
        else:
            df_holdings[col] = metrics_df[col]

    df_holdings["상태"] = "보유"
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
            base_principal = acc.get("total_principal", 0.0)
            base_cash = acc.get("cash_balance", 0.0)
            cash_balance_native = acc.get("cash_balance_native")
            cash_currency = str(acc.get("cash_currency") or "").strip().upper()

            try:
                account_settings = get_account_settings(account_id)
                account_currency = str(account_settings.get("currency") or "").strip().upper()
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
                "asset_helper": acc.get("asset_helper"),
                "intl_shares_sort_order": acc.get("intl_shares_sort_order"),
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
    intl_shares_sort_order: int | None = None,
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
                if intl_shares_sort_order is not None:
                    acc["intl_shares_sort_order"] = int(intl_shares_sort_order)

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
            if intl_shares_sort_order is not None:
                new_acc["intl_shares_sort_order"] = int(intl_shares_sort_order)
            accounts.append(new_acc)

        db.portfolio_master.update_one({"master_id": "GLOBAL"}, {"$set": {"accounts": accounts}}, upsert=True)
        return True
    except Exception as e:
        logger.error(f"Error saving portfolio master: {e}")
        return False


def update_account_asset_helper(
    account_id: str,
    *,
    target_ratio_by_ticker: dict[str, float],
    helper_settings: dict[str, Any],
) -> None:
    """자산 헬퍼 데이터를 portfolio_master 에 저장한다 (단일 컬렉션 원칙).

    - 종목별 목표비중: 해당 계좌 보유 항목의 ``target_ratio`` 필드로 저장.
      맵에 없는 보유 항목은 필드를 제거한다(미설정 명시 — 임의 0 보정 금지).
    - 계좌 단위 설정(weight_mode·백테스트 등): 계좌 객체의 ``asset_helper`` 필드로 저장.

    맵의 티커가 보유 목록에 없으면 에러를 낸다(fail loud — 종목 목록의 소스는 보유 목록이다).
    """
    from utils.asx_ticker import strip_asx_prefix

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결 실패 — portfolio_master 를 저장할 수 없습니다.")

    doc = db.portfolio_master.find_one({"master_id": "GLOBAL"})
    if not doc:
        raise RuntimeError("portfolio_master 문서가 없습니다.")

    accounts = doc.get("accounts", [])
    account = next((a for a in accounts if str(a.get("account_id")) == str(account_id)), None)
    if account is None:
        raise RuntimeError(f"portfolio_master 에 계좌가 없습니다: {account_id}")

    # 비교 키는 ASX: 접두사를 벗겨 통일한다(저장 표기는 보유 항목 원본을 유지).
    normalized_ratios = {strip_asx_prefix(t): float(r) for t, r in target_ratio_by_ticker.items()}
    holdings = account.get("holdings", [])
    holding_keys = {strip_asx_prefix(str(h.get("ticker") or "")) for h in holdings}
    unmatched = sorted(set(normalized_ratios) - holding_keys)
    if unmatched:
        raise RuntimeError(f"보유 목록에 없는 종목의 비중은 저장할 수 없습니다: {', '.join(unmatched)}")

    for holding in holdings:
        key = strip_asx_prefix(str(holding.get("ticker") or ""))
        if key in normalized_ratios:
            holding["target_ratio"] = normalized_ratios[key]
        else:
            holding.pop("target_ratio", None)

    account["asset_helper"] = dict(helper_settings)
    account["updated_at"] = _now_kst()
    db.portfolio_master.update_one({"master_id": "GLOBAL"}, {"$set": {"accounts": accounts}}, upsert=True)


def save_daily_snapshot(
    account_id: str,
    total_assets: float,
    total_principal: float,
    cash_balance: float,
    valuation_krw: float,
    purchase_amount: float | None = None,
    holding_details: list[dict[str, Any]] | None = None,
    cash_balance_native: float | None = None,
    cash_currency: str | None = None,
    intl_shares_value: float | None = None,
) -> bool:
    """
    Save a daily snapshot.
    In the consolidated schema, 'TOTAL' values are stored in the root,
    and individual accounts in an 'accounts' array.
    """
    db = get_db_connection()
    if db is None:
        return False

    snapshot_date = _resolve_snapshot_date()
    # 소급 갱신 여부 — 주말·휴일 실행은 직전 거래일 행을 갱신한다(자산은 미국·호주 장
    # 새벽 반영을 위해 소급이 맞다). 그러나 **원금은 달력일 귀속**이라 소급하면 안 된다:
    # 주말에 옮긴 원금이 금요일 행에 스며들면 다음 거래일의 입출금 차감(Δ원금)이 0이 되어
    # 이동액이 금일 손익으로 잡힌다. 소급 갱신에서는 기존 원금 값을 보존한다.
    is_backfill = snapshot_date < _now_kst().strftime("%Y-%m-%d")

    try:
        # Find existing snapshot for today
        doc = db.daily_snapshots.find_one({"snapshot_date": snapshot_date})
        if not doc:
            doc = {
                "snapshot_date": snapshot_date,
                "total_assets": 0,
                "total_principal": 0,
                "cash_balance": 0,
                "valuation_krw": 0,
                "purchase_amount": 0,
                "accounts": [],
                "updated_at": _now_kst(),
            }

        if account_id == "TOTAL":
            doc["total_assets"] = _round_snapshot_money(total_assets)
            if not (is_backfill and doc.get("total_principal")):
                doc["total_principal"] = _round_snapshot_money(total_principal)
            doc["cash_balance"] = _round_snapshot_money(cash_balance)
            doc["valuation_krw"] = _round_snapshot_money(valuation_krw)
            if purchase_amount is not None:
                doc["purchase_amount"] = _round_snapshot_money(purchase_amount)
            # TOTAL에는 별도 holdings를 저장하지 않음 (계좌별로 저장됨)
        else:
            accounts = doc.get("accounts", [])
            found = False
            for acc in accounts:
                if acc["account_id"] == account_id:
                    acc["total_assets"] = _round_snapshot_money(total_assets)
                    if not (is_backfill and acc.get("total_principal")):
                        acc["total_principal"] = _round_snapshot_money(total_principal)
                    acc["cash_balance"] = _round_snapshot_money(cash_balance)
                    acc["valuation_krw"] = _round_snapshot_money(valuation_krw)
                    if purchase_amount is not None:
                        acc["purchase_amount"] = _round_snapshot_money(purchase_amount)
                    if holding_details is not None:
                        acc["holdings"] = holding_details
                    if cash_balance_native is not None:
                        acc["cash_balance_native"] = float(cash_balance_native)
                    if cash_currency is not None:
                        acc["cash_currency"] = str(cash_currency).strip().upper()
                    if intl_shares_value is not None:
                        acc["intl_shares_value"] = float(intl_shares_value)
                    found = True
                    break

            if not found:
                acc_data = {
                    "account_id": account_id,
                    "total_assets": _round_snapshot_money(total_assets),
                    "total_principal": _round_snapshot_money(total_principal),
                    "cash_balance": _round_snapshot_money(cash_balance),
                    "valuation_krw": _round_snapshot_money(valuation_krw),
                    "purchase_amount": _round_snapshot_money(purchase_amount),
                }
                if holding_details is not None:
                    acc_data["holdings"] = holding_details
                if cash_balance_native is not None:
                    acc_data["cash_balance_native"] = float(cash_balance_native)
                if cash_currency is not None:
                    acc_data["cash_currency"] = str(cash_currency).strip().upper()
                if intl_shares_value is not None:
                    acc_data["intl_shares_value"] = float(intl_shares_value)
                accounts.append(acc_data)
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
