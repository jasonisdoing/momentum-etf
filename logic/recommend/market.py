"""Market status utilities (regime filter status string)."""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, List

import math
import pandas as pd

try:  # 선택적 의존성 로딩
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None

from utils.data_loader import fetch_ohlcv
from utils.account_registry import (
    load_account_configs,
    pick_default_account,
)
from utils.settings_loader import (
    AccountSettingsError,
    get_market_regime_settings,
    get_market_regime_aux_tickers,
    load_common_settings,
)
from utils.logger import get_app_logger

logger = get_app_logger()

_TICKER_NAME_OVERRIDES = {
    "^GSPC": "S&P500",
    "^IXIC": "NASDAQ",
    "^DJI": "Dow Jones",
}


def _resolve_display_name(ticker: str) -> Optional[str]:
    override = _TICKER_NAME_OVERRIDES.get(ticker.upper())
    if override:
        return override

    if yf is None:
        return None

    try:
        ticker_obj = yf.Ticker(ticker)
        try:
            info = ticker_obj.get_info()
        except AttributeError:
            info = getattr(ticker_obj, "info", None)
        if isinstance(info, dict):
            for key in ("shortName", "longName", "name"):
                value = info.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    except Exception:
        return None
    return None


_PREPARED_TICKERS: set[tuple[str, str]] = set()


def _prepare_regime_cache(ticker: str, country: str) -> None:
    key = (ticker.upper(), (country or "").strip().lower())
    if key in _PREPARED_TICKERS:
        return

    try:
        common_settings = load_common_settings()
        cache_start = common_settings.get("CACHE_START_DATE")
    except Exception:
        cache_start = None

    if cache_start:
        try:
            fetch_ohlcv(
                ticker,
                country=country,
                date_range=[cache_start, None],
                cache_country="common",
            )
        except Exception as exc:
            logger.warning("시장 레짐 캐시 준비 실패 (%s/%s): %s", country, ticker, exc)

    _PREPARED_TICKERS.add(key)


def _ensure_accounts_available() -> None:
    account_configs = load_account_configs()
    if not account_configs:
        raise RuntimeError("계정 설정 없음")
    pick_default_account(account_configs)


def _compute_market_regime_status(
    ticker: str,
    *,
    ma_period: int,
    country: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    required_days = int(ma_period) + 30
    required_months = max(3, (required_days // 22) + 2)

    _prepare_regime_cache(ticker, country)

    df_regime = fetch_ohlcv(
        ticker,
        country=country,
        months_range=[required_months, 0],
        cache_country="common",
    )

    if df_regime is not None and not df_regime.empty:
        try:
            df_regime.index = pd.to_datetime(df_regime.index).normalize()
            df_regime = df_regime[~df_regime.index.duplicated(keep="last")]
        except Exception:
            pass

    if df_regime is None or df_regime.empty or len(df_regime) < ma_period:
        df_regime = fetch_ohlcv(
            ticker,
            country=country,
            months_range=[required_months * 2, 0],
            cache_country="common",
        )

    if df_regime is None or df_regime.empty or len(df_regime) < ma_period:
        return None, '<span style="color:grey">시장 상태: 데이터 부족</span>'

    df_regime = df_regime.sort_index()

    price_column = None
    for candidate in ("Close", "Adj Close", "Price"):
        if candidate in df_regime.columns:
            price_column = candidate
            break
    if price_column is None:
        return None, '<span style="color:grey">시장 상태: 계산 불가</span>'

    df_regime["MA"] = df_regime[price_column].rolling(window=ma_period).mean()
    df_regime.dropna(subset=["MA"], inplace=True)

    if df_regime.empty:
        return None, '<span style="color:grey">시장 상태: 계산 불가</span>'

    current_price = df_regime[price_column].iloc[-1]
    current_ma = df_regime["MA"].iloc[-1]

    if pd.isna(current_price) or pd.isna(current_ma) or current_ma == 0:
        return None, '<span style="color:grey">시장 상태: 계산 불가</span>'

    proximity_pct = ((current_price / current_ma) - 1) * 100
    is_risk_off = bool(current_price < current_ma)
    status_label = "경고" if is_risk_off else "건강"
    status_color = "orange" if is_risk_off else "green"
    status_key = "warning" if is_risk_off else "healthy"

    is_risk_off_series = df_regime[price_column] < df_regime["MA"]

    display_name = _resolve_display_name(ticker)

    risk_periods: List[Tuple[pd.Timestamp, Optional[pd.Timestamp]]] = []
    current_start: Optional[pd.Timestamp] = None

    for dt, risk_flag in is_risk_off_series.items():
        if risk_flag:
            if current_start is None:
                current_start = dt
        else:
            if current_start is not None:
                prev_dt = df_regime.index[df_regime.index.get_loc(dt) - 1] if dt in df_regime.index else current_start
                risk_periods.append((current_start, prev_dt))
                current_start = None

    if current_start is not None:
        risk_periods.append((current_start, None))

    last_period = risk_periods[-1] if risk_periods else None
    prev_period = risk_periods[-2] if len(risk_periods) >= 2 else None
    prev_prev_period = risk_periods[-3] if len(risk_periods) >= 3 else None

    def _sanitize_period(period: Optional[Tuple[pd.Timestamp, Optional[pd.Timestamp]]]) -> Optional[Tuple[pd.Timestamp, Optional[pd.Timestamp]]]:
        if not period:
            return None
        start_dt, end_dt = period
        if start_dt is None:
            return None
        try:
            start_dt = pd.to_datetime(start_dt)
        except Exception:
            return None
        if end_dt is not None:
            try:
                end_dt = pd.to_datetime(end_dt)
            except Exception:
                end_dt = None
        return start_dt, end_dt

    last_period = _sanitize_period(last_period)
    prev_period = _sanitize_period(prev_period)
    prev_prev_period = _sanitize_period(prev_prev_period)

    risk_off_suffix = ""
    if last_period is not None:
        start_dt, end_dt = last_period
        start_str = pd.to_datetime(start_dt).strftime("%Y-%m-%d")
        end_str = "현재" if end_dt is None else pd.to_datetime(end_dt).strftime("%Y-%m-%d")
        risk_off_suffix = f" (최근 중단: {start_str} ~ {end_str})"

    html_message = f'시장: <span style="color:{status_color}">{status_label} ({proximity_pct:+.1f}%)</span>' f"{risk_off_suffix}"

    info: Dict[str, Any] = {
        "ticker": ticker,
        "ma_period": int(ma_period),
        "country": country,
        "name": display_name,
        "status": status_key,
        "status_label": status_label,
        "status_color": status_color,
        "is_risk_off": is_risk_off,
        "proximity_pct": float(proximity_pct),
        "last_risk_off_start": last_period[0] if last_period else None,
        "last_risk_off_end": last_period[1] if last_period else None,
        "last_risk_off_prev": prev_period,
        "last_risk_off_prev2": prev_prev_period,
    }

    return info, html_message


def get_market_regime_status_info(ma_period_override: Optional[int] = None) -> Tuple[Optional[Dict[str, Any]], str]:
    """메인 시장 레짐 상태를 반환합니다."""

    try:
        _ensure_accounts_available()
    except Exception as exc:
        logger.error("시장 레짐 상태 초기화에 실패했습니다: %s", exc)
        return None, '<span style="color:grey">시장 상태: 설정 파일 오류</span>'

    try:
        ticker, ma_period, country, _ = get_market_regime_settings()
    except AccountSettingsError as exc:
        logger.error("시장 레짐 공통 설정 로딩 실패: %s", exc)
        return None, '<span style="color:grey">시장 상태: 설정 파일 오류</span>'

    if ma_period_override is not None:
        try:
            override_val = int(ma_period_override)
        except (TypeError, ValueError):
            override_val = ma_period
        else:
            if override_val <= 0:
                override_val = ma_period
        ma_period = override_val

    return _compute_market_regime_status(
        ticker,
        ma_period=ma_period,
        country=country,
    )


def get_market_regime_aux_status_infos(ma_period_override: Optional[int] = None) -> List[Dict[str, Any]]:
    """보조 시장 레짐 상태 목록을 반환합니다 (유효한 항목만)."""

    try:
        ticker_main, ma_period, country, _ = get_market_regime_settings()
    except AccountSettingsError as exc:
        logger.error("시장 레짐 공통 설정 로딩 실패: %s", exc)
        return []

    if ma_period_override is not None:
        try:
            override_val = int(ma_period_override)
        except (TypeError, ValueError):
            override_val = ma_period
        else:
            if override_val <= 0:
                override_val = ma_period
        ma_period = override_val

    aux_tickers = get_market_regime_aux_tickers()
    results: List[Dict[str, Any]] = []
    for ticker in aux_tickers:
        if ticker.strip().upper() == ticker_main.strip().upper():
            continue
        info, _ = _compute_market_regime_status(
            ticker,
            ma_period=ma_period,
            country=country,
        )
        if info is not None:
            results.append(info)
    return results


def get_market_regime_status_string() -> Optional[str]:
    """메인 시장 레짐 상태를 HTML 문자열로 반환합니다."""

    _, message = get_market_regime_status_info()
    return message
