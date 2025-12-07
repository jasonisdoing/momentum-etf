"""모든 종목 목록 페이지."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from config import MARKET_SCHEDULES
from logic.common import get_buy_signal_streak
from strategies.rsi.recommend import calculate_rsi_for_ticker
from utils.data_loader import (
    fetch_naver_etf_inav_snapshot,
    fetch_ohlcv,
)
from utils.indicators import calculate_ma_score
from utils.moving_averages import calculate_moving_average
from utils.stock_list_io import get_etfs


def _format_percent(value: float) -> str:
    """퍼센트 값을 문자열로 포맷 (+1.50% 형식)."""
    if value is None:
        return "-"
    try:
        pct = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{pct:+.2f}%"


def _format_score(value: float) -> str:
    """점수 값을 문자열로 포맷 (1.5 형식)."""
    if value is None:
        return "-"
    try:
        score = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{score:.1f}"


def _calculate_return_pct(close_series: pd.Series, days: int) -> float:
    """N일 전 대비 수익률 계산."""
    if len(close_series) < days + 1:
        return 0.0
    try:
        old_price = float(close_series.iloc[-(days + 1)])
        new_price = float(close_series.iloc[-1])
        if old_price > 0:
            return ((new_price / old_price) - 1.0) * 100.0
    except (IndexError, ValueError, ZeroDivisionError):
        pass
    return 0.0


def _calculate_drawdown_from_high(close_series: pd.Series) -> float:
    """고점 대비 하락률 계산 (전체 기간 기준)."""
    if len(close_series) < 2:
        return 0.0

    price_valid = close_series.dropna()
    if price_valid.empty:
        return 0.0

    try:
        highest_price = float(price_valid.max())
        latest_price = float(price_valid.iloc[-1])
        if highest_price > 0:
            return ((latest_price / highest_price) - 1.0) * 100.0
    except (ValueError, ZeroDivisionError):
        pass
    return 0.0


def _build_all_stocks_table(country: str = "kor") -> pd.DataFrame:
    """모든 종목의 데이터를 수집하여 DataFrame으로 반환."""

    # 1. 종목 목록 로드
    etfs = get_etfs(country)
    if not etfs:
        return pd.DataFrame()

    tickers = [etf["ticker"] for etf in etfs]

    # 2. 실시간 가격/NAV 데이터 가져오기
    realtime_snapshot = fetch_naver_etf_inav_snapshot(tickers)

    # 3. 각 종목별 데이터 수집
    rows: list[dict[str, Any]] = []

    for idx, etf in enumerate(etfs, 1):
        ticker = etf["ticker"]
        name = etf.get("name", ticker)
        category = etf.get("category", "-")

        # 캐시된 과거 데이터 로드
        try:
            price_data = fetch_ohlcv(ticker, country, months_back=12)
        except Exception:
            price_data = None

        if price_data is None or price_data.empty:
            # 데이터 없는 종목은 기본값으로 표시
            rows.append(
                {
                    "#": idx,
                    "티커": ticker,
                    "종목명": name,
                    "카테고리": category,
                    "일간(%)": 0.0,
                    "현재가": "-",
                    "Nav": "-",
                    "괴리율": "-",
                    "1주(%)": 0.0,
                    "2주(%)": 0.0,
                    "1달(%)": 0.0,
                    "3달(%)": 0.0,
                    "고점대비": 0.0,
                    "추세(3달)": [],
                    "점수": 0.0,
                    "RSI": 0.0,
                    "지속": 0,
                }
            )
            continue

        close_series = price_data["Close"]

        # 실시간 가격 정보
        snapshot = realtime_snapshot.get(ticker.upper(), {})
        current_price = snapshot.get("nowVal", float(close_series.iloc[-1]) if not close_series.empty else 0.0)
        nav_price = snapshot.get("nav", 0.0)
        deviation = snapshot.get("deviation", 0.0)

        # 일간 변동률 (개장 시간 체크)
        now = datetime.now()
        market_schedule = MARKET_SCHEDULES.get(country.lower(), {})
        market_open_time = market_schedule.get("open")

        # 전일 종가 사용 (close_series.iloc[-1]이 아닌 실제 전일 종가)
        # 실시간 스냅샷이 있으면 캐시된 마지막 종가가 전일 종가
        prev_close = float(close_series.iloc[-2]) if len(close_series) >= 2 else 0.0

        if market_open_time and now.time() >= market_open_time and prev_close > 0:
            daily_pct = ((current_price / prev_close) - 1.0) * 100.0
        else:
            daily_pct = 0.0

        # 수익률 계산 (계정 페이지와 동일한 기간 사용)
        return_1w = _calculate_return_pct(close_series, 5)  # 5일
        return_2w = _calculate_return_pct(close_series, 10)  # 10일
        return_1m = _calculate_return_pct(close_series, 21)  # 21일 (1개월)
        return_3m = _calculate_return_pct(close_series, 63)  # 63일 (3개월)

        # 고점 대비 (전체 기간 기준)
        drawdown = _calculate_drawdown_from_high(close_series)

        # 추세 (3달 = 63일)
        trend_data = close_series.tail(63).tolist() if len(close_series) >= 63 else close_series.tolist()

        # 점수 계산 (MA 기반)
        ma_period = 90
        ma_type = "TEMA"
        min_buy_score = 0.0

        score_value = 0.0
        consecutive_days = 0
        rsi_score = 0.0

        if len(close_series) >= ma_period:
            try:
                moving_average = calculate_moving_average(close_series, ma_period, ma_type)
                ma_score_series = calculate_ma_score(close_series, moving_average)
                score_value = float(ma_score_series.iloc[-1]) if not ma_score_series.empty else 0.0
                consecutive_days = get_buy_signal_streak(score_value, ma_score_series, min_buy_score)
                rsi_score = calculate_rsi_for_ticker(close_series)
            except Exception:
                pass

        rows.append(
            {
                "#": idx,
                "티커": ticker,
                "종목명": name,
                "카테고리": category,
                "일간(%)": daily_pct,
                "현재가": int(current_price) if pd.notna(current_price) else None,
                "Nav": int(nav_price) if pd.notna(nav_price) else None,
                "괴리율": deviation,
                "1주(%)": return_1w,
                "2주(%)": return_2w,
                "1달(%)": return_1m,
                "3달(%)": return_3m,
                "고점대비": drawdown,
                "추세(3달)": trend_data,
                "점수": score_value,
                "RSI": rsi_score,
                "지속": consecutive_days,
            }
        )

    df = pd.DataFrame(rows)

    # 일간(%) 내림차순으로 정렬
    df = df.sort_values(by="일간(%)", ascending=False)

    # 정렬 후 순번 재부여
    df["#"] = range(1, len(df) + 1)

    return df


def _style_dataframe(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """DataFrame에 스타일 적용 (색상 및 포맷)."""

    def _color_pct(val: float | str) -> str:
        if val is None:
            return ""
        try:
            num = float(val)
        except (TypeError, ValueError):
            return ""

        if num > 0:
            return "color: red"
        if num < 0:
            return "color: blue"
        return "color: black"

    styled = df.style
    pct_columns = [
        "일간(%)",
        "1주(%)",
        "2주(%)",
        "1달(%)",
        "3달(%)",
        "고점대비",
        "괴리율",
    ]
    for col in pct_columns:
        if col in df.columns:
            styled = styled.map(_color_pct, subset=pd.IndexSlice[:, col])

    # 가격 컬럼 포맷팅 (천 단위 콤마 + 원)
    format_dict = {}
    if "현재가" in df.columns:
        format_dict["현재가"] = "{:,.0f}원"
    if "Nav" in df.columns:
        format_dict["Nav"] = "{:,.0f}원"

    if format_dict:
        styled = styled.format(format_dict)

    return styled


def render_all_stocks_page() -> None:
    """모든 종목 페이지 렌더링."""

    st.set_page_config(
        page_title="전체 종목",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.caption("모든 ETF 종목의 실시간 데이터 및 지표")

    with st.spinner("데이터 로딩 중..."):
        df = _build_all_stocks_table("kor")

    if df.empty:
        st.error("종목 데이터를 불러올 수 없습니다.")
        return

    st.caption(f"총 {len(df)}개 종목 | 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 컬럼 설정
    column_config = {
        "#": st.column_config.TextColumn("#", width=40),
        "티커": st.column_config.TextColumn("티커", width=60),
        "종목명": st.column_config.TextColumn("종목명", width="medium"),
        "카테고리": st.column_config.TextColumn("카테고리", width=100),
        "일간(%)": st.column_config.NumberColumn("일간(%)", width=70, format="%.2f%%"),
        "현재가": st.column_config.NumberColumn("현재가", width=80),
        "Nav": st.column_config.NumberColumn("Nav", width=80),
        "괴리율": st.column_config.NumberColumn("괴리율", width=70, format="%.2f%%"),
        "1주(%)": st.column_config.NumberColumn("1주(%)", width=70, format="%.2f%%"),
        "2주(%)": st.column_config.NumberColumn("2주(%)", width=70, format="%.2f%%"),
        "1달(%)": st.column_config.NumberColumn("1달(%)", width=70, format="%.2f%%"),
        "3달(%)": st.column_config.NumberColumn("3달(%)", width=70, format="%.2f%%"),
        "고점대비": st.column_config.NumberColumn("고점대비", width=80, format="%.2f%%"),
        "추세(3달)": st.column_config.LineChartColumn("추세(3달)", width=100),
        "점수": st.column_config.NumberColumn("점수", width=60, format="%.1f"),
        "RSI": st.column_config.NumberColumn("RSI", width=60, format="%.1f"),
        "지속": st.column_config.NumberColumn("지속", width=50),
    }

    # 스타일 적용
    styled_df = _style_dataframe(df)

    # 테이블 표시
    st.dataframe(
        styled_df,
        hide_index=True,
        width="stretch",
        height=600,
        column_config=column_config,
    )

    st.markdown("---")
    st.markdown(
        """
        - 본 웹사이트에서 제공되는 종목 정보 및 추천은 단순 정보 제공을 목적으로 하며, 특정 종목의 매매를 권유하는 것이 아닙니다.
        - 본 정보를 이용한 투자 판단 및 매매 결과에 대하여 웹사이트 운영자는 어떠한 책임도 지지 않습니다.
        - 투자에는 원금 손실 가능성이 있으며, 투자자는 스스로 리스크를 검토해야 합니다.
        """
    )


if __name__ == "__main__":
    render_all_stocks_page()
