"""Market status utilities (regime filter status string).

Moved from the root signals module.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

try:  # 선택적 의존성 로딩
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None

from utils.data_loader import fetch_ohlcv, PriceDataUnavailable
from utils.account_registry import (
    load_account_configs,
    pick_default_account,
)
from utils.settings_loader import get_account_strategy_sections
from utils.logger import get_app_logger

logger = get_app_logger()


def get_market_regime_status_string() -> Optional[str]:
    """
    S&P 500 지수를 기준으로 현재 시장 레짐 상태를 계산하여 HTML 문자열로 반환합니다.
    """
    # 공통 설정 로드 (파일)
    try:
        account_configs = load_account_configs()
        if not account_configs:
            return '<span style="color:grey">시장 상태: 계정 설정 없음</span>'

        default_account = pick_default_account(account_configs)
        account_id = default_account.get("account_id")
        if not account_id:
            return '<span style="color:grey">시장 상태: 계정 식별 실패</span>'

        tuning, static = get_account_strategy_sections(account_id)
        regime_ticker = str(
            static.get("MARKET_REGIME_FILTER_TICKER")
            or tuning.get("MARKET_REGIME_FILTER_TICKER")
            or ""
        ).strip()
        regime_ma_raw = None
        if isinstance(static, dict):
            regime_ma_raw = static.get("MARKET_REGIME_FILTER_MA_PERIOD")
        if regime_ma_raw is None and isinstance(tuning, dict):
            regime_ma_raw = tuning.get("MARKET_REGIME_FILTER_MA_PERIOD")

        if not regime_ticker or regime_ma_raw is None:
            return '<span style="color:grey">시장 상태: 비활성화</span>'

        regime_ma_period = int(regime_ma_raw)
        if regime_ma_period <= 0:
            return '<span style="color:grey">시장 상태: 비활성화</span>'
    except (SystemExit, KeyError, ValueError, TypeError) as e:
        logger.error("시장 레짐 설정을 불러오는 중 문제가 발생했습니다: %s", e)
        return '<span style="color:grey">시장 상태: 설정 파일 오류</span>'

    # 데이터 로딩에 필요한 기간 계산: 레짐 MA 기간을 만족하도록 동적으로 산정
    # 거래일 기준 대략 22일/월 가정 + 여유 버퍼
    required_days = int(regime_ma_period) + 30
    required_months = max(3, (required_days // 22) + 2)

    # 데이터 조회
    try:
        df_regime = fetch_ohlcv(
            regime_ticker,
            country="kor",
            months_range=[required_months, 0],  # 지수 조회에서는 country 인자가 의미 없습니다.
        )
    except PriceDataUnavailable:
        return '<span style="color:grey">시장 상태: 데이터 부족</span>'

    # --- 인덱스 정규화 추가 ---
    if df_regime is not None and not df_regime.empty:
        try:
            df_regime.index = pd.to_datetime(df_regime.index).normalize()
            df_regime = df_regime[~df_regime.index.duplicated(keep="last")]
        except Exception:
            pass  # 정규화 실패시 원본 그대로 사용

    # 만약 데이터가 부족하면, 기간을 늘려 한 번 더 시도합니다.
    if df_regime is None or df_regime.empty or len(df_regime) < regime_ma_period:
        try:
            df_regime = fetch_ohlcv(
                regime_ticker,
                country="kor",
                months_range=[required_months * 2, 0],
            )
        except PriceDataUnavailable:
            return '<span style="color:grey">시장 상태: 데이터 부족</span>'

    if df_regime is None or df_regime.empty or len(df_regime) < regime_ma_period:
        return '<span style="color:grey">시장 상태: 데이터 부족</span>'

    # 지표 계산
    df_regime["MA"] = df_regime["Close"].rolling(window=regime_ma_period).mean()
    df_regime.dropna(subset=["MA"], inplace=True)

    # --- 최근 투자 중단 기간 찾기 ---
    risk_off_periods_str = ""
    if not df_regime.empty:
        is_risk_off_series = df_regime["Close"] < df_regime["MA"]

        # 완료된 리스크 오프 기간 수집
        completed_periods = []
        in_period = False
        start_date = None
        for i, (dt, is_off) in enumerate(is_risk_off_series.items()):
            if is_off and not in_period:
                in_period = True
                start_date = dt
            elif not is_off and in_period:
                in_period = False
                # 리스크 오프 기간의 마지막 날은 is_off가 False가 되기 바로 전날입니다.
                end_date = is_risk_off_series.index[is_risk_off_series.index.get_loc(dt) - 1]
                completed_periods.append((start_date, end_date))
                start_date = None

        if completed_periods:
            recent_periods = completed_periods[-1:]
            period_strings = [
                f"{start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')}"
                for start, end in recent_periods
            ]
            if period_strings:
                risk_off_periods_str = f" (최근 중단: {', '.join(period_strings)})"

    current_price = df_regime["Close"].iloc[-1]
    current_ma = df_regime["MA"].iloc[-1]

    if pd.notna(current_price) and pd.notna(current_ma) and current_ma > 0:
        proximity_pct = ((current_price / current_ma) - 1) * 100
        is_risk_off = current_price < current_ma

        status_text = "위험" if is_risk_off else "안전"
        color = "orange" if is_risk_off else "green"
        return (
            f'시장: <span style="color:{color}">{status_text} ({proximity_pct:+.1f}%)</span>'
            f"{risk_off_periods_str}"
        )

    return f'<span style="color:grey">시장 상태: 계산 불가</span>{risk_off_periods_str}'
