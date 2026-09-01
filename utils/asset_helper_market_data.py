"""자산 헬퍼 시장 데이터 층 — 종가 프레임·환산·수익률/MDD/실시간 맵·계좌 스냅샷.

`utils/asset_helper_service.py` 에서 분리(이동만, 로직 불변). 서비스가 이 모듈을
임포트하며, 이 모듈은 서비스에 의존하지 않는다(단방향).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from config import TRADING_DAYS_PER_MONTH
from core.strategy.metrics import period_return_pct
from utils.cache_utils import (
    get_all_ticker_type_lookup_keys,
    load_cached_close_series_bulk,
    load_cached_close_series_bulk_with_fallback,
)
from utils.logger import get_app_logger

logger = get_app_logger()


IS_PRICE_PROXY = {"IS": ("aus", "VGS")}


def _cache_lookup_ticker(ticker: str, ticker_type: str) -> str:
    """가격 캐시 조회 키 — 호주 풀은 시스템 표준 표기(ASX: 접두사)로 저장돼 있다."""
    from utils.asx_ticker import ensure_asx_prefix
    from utils.settings_loader import get_ticker_type_settings

    try:
        country = str(get_ticker_type_settings(ticker_type).get("country_code") or "").strip().lower()
    except Exception:
        country = ""
    return ensure_asx_prefix(ticker) if country == "au" else ticker


def _load_close_frame(tickers: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[str]]:
    # 종목풀별로 {캐시 조회 키: 요청 티커} 를 만든다.
    # 요청 티커는 접두사 없는 형태(화면 표준)지만 호주 캐시 키는 ASX: 가 붙어 있고,
    # IS 는 VGS 시계열로 대리하므로 조회 키와 요청 티커가 다를 수 있다.
    grouped: dict[str, dict[str, str]] = defaultdict(dict)
    for item in tickers:
        ticker = str(item.get("ticker") or "").strip().upper()
        ticker_type = str(item.get("ticker_type") or "").strip().lower()
        if not ticker or not ticker_type:
            continue
        proxy = IS_PRICE_PROXY.get(ticker)
        if proxy:
            proxy_type, proxy_ticker = proxy
            grouped[proxy_type][_cache_lookup_ticker(proxy_ticker, proxy_type)] = ticker
        else:
            grouped[ticker_type][_cache_lookup_ticker(ticker, ticker_type)] = ticker

    series_map: dict[str, pd.Series] = {}
    for ticker_type, lookup_map in grouped.items():
        fetched = load_cached_close_series_bulk_with_fallback(ticker_type, list(lookup_map))
        for fetched_ticker, series in fetched.items():
            if series is None or series.empty:
                continue
            requested = lookup_map.get(str(fetched_ticker).strip().upper())
            if not requested:
                continue
            normalized = pd.to_numeric(series, errors="coerce").dropna()
            normalized.index = pd.to_datetime(normalized.index).normalize()
            normalized = normalized[~normalized.index.duplicated(keep="last")].sort_index()
            if not normalized.empty:
                series_map[requested] = normalized

    unresolved = [
        str(item.get("ticker") or "").strip().upper()
        for item in tickers
        if str(item.get("ticker") or "").strip().upper()
        and str(item.get("ticker") or "").strip().upper() not in series_map
    ]
    for cache_key in get_all_ticker_type_lookup_keys():
        if not unresolved:
            break
        fetched = load_cached_close_series_bulk(cache_key, unresolved)
        for ticker, series in fetched.items():
            if series is None or series.empty:
                continue
            normalized = pd.to_numeric(series, errors="coerce").dropna()
            normalized.index = pd.to_datetime(normalized.index).normalize()
            normalized = normalized[~normalized.index.duplicated(keep="last")].sort_index()
            if not normalized.empty:
                series_map[str(ticker).strip().upper()] = normalized
        unresolved = [ticker for ticker in unresolved if ticker not in series_map]

    missing = [
        str(item.get("ticker") or "").strip().upper()
        for item in tickers
        if str(item.get("ticker") or "").strip().upper() not in series_map
    ]
    if not series_map:
        return pd.DataFrame(), missing

    union_index = sorted({ts for series in series_map.values() for ts in series.index})
    close_frame = pd.DataFrame(
        {ticker: series.reindex(union_index) for ticker, series in series_map.items()},
        index=pd.DatetimeIndex(union_index),
    )
    return close_frame, missing


# 국가코드 → 통화, 통화 → KRW 환율 심볼(Yahoo). 국내(kor)는 원화라 변환 없음.
_CURRENCY_BY_COUNTRY = {"kor": "KRW", "us": "USD", "au": "AUD"}
_FX_SYMBOL_BY_CURRENCY = {"USD": "KRW=X", "AUD": "AUDKRW=X"}


def _resolve_backtest_currency(ticker: str, country_code: str | None, ticker_type: str | None) -> str:
    """백테스트 통화 판정: country_code → ticker_type → 티커 형식 순으로 추정한다.

    /market-trend·resolve 와 동일한 형식 규칙(6자리 숫자=국내, .AX=호주, 그 외 알파벳=미국).
    """
    cc = str(country_code or "").strip().lower()
    if not cc and ticker_type:
        from utils.settings_loader import get_ticker_type_settings

        try:
            cc = str(get_ticker_type_settings(ticker_type).get("country_code") or "").strip().lower()
        except Exception:
            cc = ""
    if not cc:
        t = str(ticker or "").strip().upper()
        if t.isdigit() and len(t) == 6:
            cc = "kor"
        elif t.endswith(".AX"):
            cc = "au"
        else:
            cc = "us"
    return _CURRENCY_BY_COUNTRY.get(cc, "KRW")


def _convert_close_frame_to_krw(close_frame: pd.DataFrame, currency_by_ticker: dict[str, str]) -> pd.DataFrame:
    """비원화(USD/AUD) 종목의 종가를 해당 일자 환율로 곱해 원화(KRW) 시계열로 환산한다.

    이렇게 하면 백테스트가 시점별 환율(환차손익)까지 반영한다. 원화 종목은 그대로 둔다.
    """
    if close_frame.empty:
        return close_frame
    from utils.data_loader import get_exchange_rate_series

    start = close_frame.index.min()
    end = close_frame.index.max()
    fx_cache: dict[str, pd.Series] = {}
    converted = close_frame.copy()
    for ticker in list(converted.columns):
        currency = str(currency_by_ticker.get(str(ticker).strip().upper(), "KRW")).upper()
        if currency == "KRW":
            continue
        symbol = _FX_SYMBOL_BY_CURRENCY.get(currency)
        if not symbol:
            raise ValueError(f"백테스트 환율 변환을 지원하지 않는 통화입니다: {currency} ({ticker})")
        if symbol not in fx_cache:
            fx = get_exchange_rate_series(start, end, symbol=symbol, allow_partial=True)
            if fx is None or fx.empty:
                raise ValueError(f"{symbol} 환율 시계열을 불러오지 못해 백테스트 환율 변환이 불가합니다.")
            fx = fx.copy()
            fx.index = pd.to_datetime(fx.index).normalize()
            fx = fx[~fx.index.duplicated(keep="last")].sort_index()
            fx_cache[symbol] = fx
        aligned = fx_cache[symbol].reindex(converted.index).ffill().bfill()
        converted[ticker] = pd.to_numeric(converted[ticker], errors="coerce") * aligned
    return converted


def _build_return_map(close_frame: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    if close_frame.empty:
        return {}

    eval_date = close_frame.index.max()
    result: dict[str, dict[str, float | None]] = {}
    for ticker in close_frame.columns:
        series = close_frame[ticker]
        result[str(ticker)] = {
            "return_1m_pct": period_return_pct(series, 1, eval_date),
            "return_3m_pct": period_return_pct(series, 3, eval_date),
            "return_6m_pct": period_return_pct(series, 6, eval_date),
            "return_12m_pct": period_return_pct(series, 12, eval_date),
        }
    return result


def _build_mdd_map(close_frame: pd.DataFrame, window_months: int) -> dict[str, float | None]:
    """Sortino 설정과 동일한 N개월 거래일 구간의 종목별 MDD를 반환한다."""
    if close_frame.empty:
        return {}

    window = max(2, int(window_months) * int(TRADING_DAYS_PER_MONTH))
    result: dict[str, float | None] = {}
    for ticker in close_frame.columns:
        series = pd.to_numeric(close_frame[ticker], errors="coerce").dropna().iloc[-window:]
        if len(series) < 2:
            result[str(ticker)] = None
            continue
        running_max = series.cummax()
        drawdowns = (series / running_max) - 1.0
        result[str(ticker)] = round(float(drawdowns.min()) * 100.0, 2)
    return result


def _build_cached_daily_change_map(close_frame: pd.DataFrame) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for ticker in close_frame.columns:
        series = pd.to_numeric(close_frame[ticker], errors="coerce").dropna()
        if len(series) < 2:
            result[str(ticker)] = None
            continue
        current = float(series.iloc[-1])
        previous = float(series.iloc[-2])
        result[str(ticker)] = round(((current / previous) - 1.0) * 100.0, 2) if previous > 0 else None
    return result


def _extract_realtime_change_pct(snapshot: dict[str, Any]) -> float | None:
    raw_value = snapshot.get("changeRate")
    if raw_value is None:
        raw_value = snapshot.get("change_pct")
    if raw_value is None:
        return None
    try:
        return round(float(str(raw_value).replace(",", "")), 2)
    except (TypeError, ValueError):
        return None


def _extract_realtime_price(snapshot: dict[str, Any]) -> float | None:
    raw_value = snapshot.get("nowVal")
    if raw_value is None:
        raw_value = snapshot.get("price")
    if raw_value is None:
        return None
    try:
        price = float(str(raw_value).replace(",", ""))
    except (TypeError, ValueError):
        return None
    return price if price > 0 else None


def _build_cached_current_price_map(close_frame: pd.DataFrame) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for ticker in close_frame.columns:
        series = pd.to_numeric(close_frame[ticker], errors="coerce").dropna()
        if series.empty:
            result[str(ticker)] = None
            continue
        price = float(series.iloc[-1])
        result[str(ticker)] = price if price > 0 else None
    return result


def _build_current_price_map(tickers: list[dict[str, Any]], close_frame: pd.DataFrame) -> dict[str, float | None]:
    result = _build_cached_current_price_map(close_frame)
    grouped: dict[str, list[str]] = defaultdict(list)
    for item in tickers:
        ticker = str(item.get("ticker") or "").strip().upper()
        country = str(item.get("country_code") or "").strip().lower()
        if not country:
            ticker_type = str(item.get("ticker_type") or "").strip().lower()
            country = ticker_type.split("_", 1)[0] if ticker_type else ""
        if ticker and country in {"kor", "au", "us"}:
            grouped[country].append(ticker)

    if not grouped:
        return result

    from services.price_service import get_realtime_snapshot

    for country, group_tickers in grouped.items():
        try:
            snapshot_map = get_realtime_snapshot(country, group_tickers)
        except Exception as exc:
            logger.warning("실시간 현재가 조회 실패 (%s): %s", country, exc)
            continue
        for ticker, snapshot in snapshot_map.items():
            price = _extract_realtime_price(snapshot)
            if price is not None:
                result[str(ticker).strip().upper()] = price
    return result


def _build_daily_change_map(tickers: list[dict[str, Any]], close_frame: pd.DataFrame) -> dict[str, float | None]:
    result = _build_cached_daily_change_map(close_frame)
    grouped: dict[str, list[str]] = defaultdict(list)
    for item in tickers:
        ticker = str(item.get("ticker") or "").strip().upper()
        country = str(item.get("country_code") or "").strip().lower()
        if not country:
            ticker_type = str(item.get("ticker_type") or "").strip().lower()
            country = ticker_type.split("_", 1)[0] if ticker_type else ""
        if ticker and country in {"kor", "au", "us"}:
            grouped[country].append(ticker)

    if not grouped:
        return result

    from services.price_service import get_realtime_snapshot

    for country, group_tickers in grouped.items():
        try:
            snapshot_map = get_realtime_snapshot(country, group_tickers)
        except Exception as exc:
            logger.warning("실시간 등락률 조회 실패 (%s): %s", country, exc)
            continue
        for ticker, snapshot in snapshot_map.items():
            change_pct = _extract_realtime_change_pct(snapshot)
            if change_pct is not None:
                result[str(ticker).strip().upper()] = change_pct
    return result


def _normalize_kor_holding_ticker(value: Any) -> str:
    # 보유내역 티커 표시 접두사(KR:, ASX:)를 제거해 티커와 매칭 가능한 형태로 정규화한다.
    return str(value or "").strip().upper().replace("KR:", "").replace("ASX:", "")


def _load_asset_helper_account_snapshot(account_id: str) -> dict[str, Any]:
    if not account_id:
        raise ValueError("적용 계좌를 선택해주세요.")

    from utils.holdings_detail_service import load_all_holdings_detail

    payload = load_all_holdings_detail(account_id)
    summaries = [row for row in payload.get("account_summaries", []) if str(row.get("account_id") or "") == account_id]
    if not summaries:
        raise ValueError(f"적용 계좌 정보를 찾을 수 없습니다: {account_id}")

    summary = summaries[0]
    currency = str(summary.get("currency") or "KRW").strip().upper()
    # KRW 계좌는 KRW 금액 그대로, 그 외(AUD 등)는 계좌 환종 네이티브 금액으로 통일한다.
    # (가격 캐시의 현재가가 네이티브라 금액·수량 계산이 같은 통화로 맞아떨어져야 한다)
    native_mode = currency != "KRW"

    account_rows = [row for row in payload.get("rows", []) if str(row.get("account_id") or "") == account_id]
    holdings: dict[str, dict[str, Any]] = {}
    total_native_value = 0.0
    for row in account_rows:
        ticker = _normalize_kor_holding_ticker(row.get("ticker"))
        if not ticker:
            continue
        # IS는 수동으로 관리하는 고정 자산이므로 운용 자산과 성과 계산에서 제외한다.
        if ticker == "IS":
            continue
        quantity = int(float(row.get("quantity") or 0))
        valuation_krw = float(row.get("valuation_krw") or 0)
        if native_mode:
            price_native = float(row.get("current_price_num") or 0)
            value = quantity * price_native
            # KRW 환산에 쓰인 환율을 역산해 손익/매입금액도 같은 통화로 복원한다.
            row_rate = (valuation_krw / value) if value > 0 else 0.0
            pnl = float(row.get("pnl_krw_num") or 0.0) / row_rate if row_rate > 0 else 0.0
            buy_amount = float(row.get("buy_amount_krw") or 0.0) / row_rate if row_rate > 0 else 0.0
            total_native_value += value
        else:
            value = valuation_krw
            pnl = float(row.get("pnl_krw_num") or 0.0)
            buy_amount = float(row.get("buy_amount_krw") or 0.0)
        current = holdings.setdefault(
            ticker,
            {
                "current_quantity": 0,
                "current_amount_krw": 0,
                "name": str(row.get("name") or ticker),
                "pnl_krw": 0.0,
                "buy_amount_krw": 0.0,
                "return_pct": 0.0,
                "daily_change_pct": row.get("daily_change_pct"),
                "bucket": row.get("bucket"),
            },
        )
        current["current_quantity"] += quantity
        current["current_amount_krw"] += round(value, 2) if native_mode else int(round(value))
        current["pnl_krw"] += pnl
        current["buy_amount_krw"] += buy_amount
        if not current.get("name"):
            current["name"] = str(row.get("name") or ticker)
        if current.get("daily_change_pct") is None and row.get("daily_change_pct") is not None:
            current["daily_change_pct"] = row.get("daily_change_pct")

    for ticker, current in holdings.items():
        buy_amt = current["buy_amount_krw"]
        pnl = current["pnl_krw"]
        if buy_amt > 0:
            current["return_pct"] = round((pnl / buy_amt) * 100.0, 2)
        else:
            current["return_pct"] = 0.0

    if native_mode:
        # 다통화 현금 개편: 주 통화 환산 현금(cash_display_native)을 우선 사용. 없으면 레거시 native 필드.
        cash_native = summary.get("cash_display_native")
        if cash_native is None:
            cash_native = summary.get("cash_balance_native")
        if cash_native is None and float(summary.get("cash_balance_krw") or 0) > 0:
            raise ValueError(f"계좌 '{account_id}' 의 현금 원화({currency}) 잔액 정보가 없습니다.")
        cash_balance = float(cash_native or 0.0)
        account_amount = total_native_value + cash_balance
    else:
        cash_balance = float(summary.get("cash_balance_krw") or 0)
        account_amount = float(summary.get("total_assets_krw") or 0)

    return {
        "account_id": account_id,
        "account_name": summary.get("name") or account_id,
        "currency": currency,
        "account_amount_krw": round(account_amount, 2) if native_mode else int(round(account_amount)),
        "cash_balance_krw": round(cash_balance, 2) if native_mode else int(round(cash_balance)),
        "holdings": holdings,
    }


def _normalize_bucket_label(bucket_val: Any) -> str | None:
    if not bucket_val:
        return None
    val_str = str(bucket_val).strip()
    if not val_str:
        return None
    import re

    pure_name = re.sub(r"^(\d+\.\s*)+", "", val_str).strip()

    mapping = {
        "모멘텀": "1. 모멘텀",
        "시장지수": "2. 시장지수",
        "배당방어": "3. 배당방어",
        "비당방어": "3. 배당방어",
        "대체헷지": "4. 대체헷지",
        "현금": "5. 현금",
    }
    return mapping.get(pure_name, val_str)
