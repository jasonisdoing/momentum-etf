from __future__ import annotations

from typing import Any, Tuple

import streamlit as st
import pandas as pd

from main import load_account_recommendations, render_recommendation_table
from utils.account_registry import get_icon_fallback
from utils.settings_loader import AccountSettingsError, get_account_settings
from logic.backtest.account_runner import run_account_backtest
from utils.data_loader import get_latest_trading_day, get_trading_days


_DATAFRAME_CSS = """
<style>
    .stDataFrame thead tr th {
        text-align: center;
    }
    .stDataFrame tbody tr td {
        text-align: center;
        white-space: nowrap;
    }
</style>
"""


def _normalize_code(value: Any, fallback: str) -> str:
    text = str(value or "").strip().lower()
    return text or fallback


def render_account_page(account_id: str) -> None:
    """주어진 계정 설정을 기반으로 추천 페이지를 렌더링합니다."""

    try:
        account_settings = get_account_settings(account_id)
    except AccountSettingsError as exc:  # pragma: no cover - Streamlit 오류 피드백 전용
        st.error(f"설정을 불러오지 못했습니다: {exc}")
        st.stop()

    country_code = _normalize_code(account_settings.get("country_code"), account_id)

    page_icon = account_settings.get("icon") or get_icon_fallback(country_code)
    page_title = "Momentum ETF"

    st.set_page_config(page_title=page_title, page_icon=page_icon or "📈", layout="wide")

    title_text = page_title
    if page_icon:
        title_text = f"{page_icon} {page_title}".strip()

    st.title(title_text)

    df, updated_at, loaded_country_code = load_account_recommendations(account_id)
    country_code = loaded_country_code or country_code

    if df is None:
        st.error(updated_at or "데이터를 불러오지 못했습니다.")
        return

    if updated_at:
        st.caption(f"데이터 업데이트: {updated_at}")
        strategy_cfg = account_settings.get("strategy", {}) or {}
        expected_cagr = None
        backtested_date = None
        strategy_tuning: dict[str, Any] = {}
        if isinstance(strategy_cfg, dict):
            expected_cagr = strategy_cfg.get("expected_cagr")
            backtested_date = strategy_cfg.get("backtested_date")
            tuning_cfg = strategy_cfg.get("tuning")
            if isinstance(tuning_cfg, dict):
                strategy_tuning = tuning_cfg
        if strategy_tuning:
            params_to_show = {
                "MA": strategy_tuning.get("MA_PERIOD"),
                "MA타입": strategy_tuning.get("MA_TYPE"),
                "TopN": strategy_tuning.get("PORTFOLIO_TOPN"),
                "교체점수": strategy_tuning.get("REPLACE_SCORE_THRESHOLD"),
                "과매수 지표": strategy_tuning.get("OVERBOUGHT_SELL_THRESHOLD"),
                "쿨다운 일자": strategy_tuning.get("COOLDOWN_DAYS"),
            }
            param_strs = [f"{key}: {value}" for key, value in params_to_show.items() if value is not None]
        else:
            param_strs = []

        caption_parts: list[str] = []
        if param_strs:
            param_display = ", ".join(param_strs)
            caption_parts.append(f"설정: [{param_display}]")
        else:
            caption_parts.append("설정: N/A")

        # 슬리피지 정보 추가
        from data.settings.common import BACKTEST_SLIPPAGE

        slippage_config = BACKTEST_SLIPPAGE.get(country_code, {})
        buy_slip = slippage_config.get("buy_pct")
        sell_slip = slippage_config.get("sell_pct")
        if buy_slip is not None and sell_slip is not None:
            if buy_slip == sell_slip:
                caption_parts.append(f"슬리피지: ±{buy_slip}%")
            else:
                caption_parts.append(f"슬리피지: 매수+{buy_slip}%/매도-{sell_slip}%")

        try:
            from logic.common import get_hold_states

            hold_states = get_hold_states()
            # buy_states = {"BUY", "BUY_REPLACE"}
            # sell_states = {"SELL_REPLACE", "SELL_TRIM", "SELL_TREND", "CUT_STOPLOSS"}
            current_holdings = int(df[df["상태"].isin(hold_states)].shape[0])
            # exits = int(df[df["상태"].isin(sell_states)].shape[0])
            # buys = int(df[df["상태"].isin(buy_states)].shape[0])
            # future_holdings = current_holdings - exits + buys
            target_topn = strategy_tuning.get("PORTFOLIO_TOPN") if isinstance(strategy_tuning, dict) else None
            if target_topn:
                caption_parts.append(f"보유종목 수 {current_holdings}/{target_topn}")
        except Exception:
            pass

        caption_text = ", ".join(caption_parts)
        if caption_text:
            st.caption(caption_text)
        else:
            st.caption("설정 정보를 찾을 수 없습니다.")

        if expected_cagr is not None:
            try:
                expected_val = float(expected_cagr)
            except (TypeError, ValueError):
                expected_val = None
            expected_html = (
                f"<span style='color:#d32f2f;'>예상 CAGR (연간 복리 성장률): {expected_val:+.2f}%, 백테스트 일자: {backtested_date}</span>"
            )
            st.markdown(f"<small>{expected_html}</small>", unsafe_allow_html=True)
    else:
        # updated_at이 없는 경우에 대한 폴백
        st.caption("데이터를 찾을 수 없습니다.")

    render_recommendation_table(df, country_code=country_code)
    _render_benchmark_table(account_id, account_settings, country_code)
    # st.markdown(_DATAFRAME_CSS, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        """
        - 본 웹사이트에서 제공되는 종목 정보 및 추천은 단순 정보 제공을 목적으로 하며, 특정 종목의 매매를 권유하는 것이 아닙니다.
        - 본 정보를 이용한 투자 판단 및 매매 결과에 대하여 웹사이트 운영자는 어떠한 책임도 지지 않습니다.
        - 투자에는 원금 손실 가능성이 있으며, 투자자는 스스로 리스크를 검토해야 합니다.
        """
    )


@st.cache_data(show_spinner=False)
def _cached_benchmark_data(
    account_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> Tuple[pd.DataFrame, float]:
    from utils.performance import calculate_actual_performance
    from utils.account_registry import get_account_settings

    # 계정 설정 로드
    account_settings = get_account_settings(account_id)
    country_code = account_settings.get("country_code", "kor")

    # 전략 설정에서 포트폴리오 수 가져오기
    strategy = account_settings.get("strategy", {})
    tuning = strategy.get("tuning", {})
    portfolio_topn = tuning.get("PORTFOLIO_TOPN", 5)

    # 초기 자본 가져오기
    initial_capital_raw = account_settings.get("initial_capital", 100_000_000)
    try:
        initial_capital = float(initial_capital_raw)
    except (TypeError, ValueError):
        initial_capital = 100_000_000.0

    # 실제 거래 기반 수익률 계산 시도
    actual_perf = calculate_actual_performance(
        account_id=account_id,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        portfolio_topn=portfolio_topn,
        country_code=country_code,
    )

    # 벤치마크 정보를 위해 항상 백테스트 실행
    result = run_account_backtest(
        account_id,
        quiet=True,
        override_settings={
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
        },
    )
    summary = result.summary or {}
    benchmarks = summary.get("benchmarks") or []

    rows: list[dict[str, str]] = []

    # Momentum ETF 수익률: 실제 거래 우선, 없으면 백테스트
    if actual_perf:
        account_return = actual_perf.get("cumulative_return_pct")  # 실제 거래는 기존 키 사용
    else:
        account_return = summary.get("period_return")  # 백테스트는 새 키 사용

    if account_return is not None:
        rows.append(
            {
                "티커": "-",
                "종목": "Momentum ETF",
                "누적 수익률": f"{float(account_return):+.2f}%",
            }
        )

    # 벤치마크 정보 (항상 표시)
    for entry in benchmarks:
        if not isinstance(entry, dict):
            continue
        ret = entry.get("cumulative_return_pct")
        ticker = entry.get("ticker", "-")
        name = entry.get("name") or entry.get("ticker")
        if ret is None or name is None:
            continue
        rows.append(
            {
                "티커": str(ticker),
                "종목": f"{name} (Buy & Hold)",
                "누적 수익률": f"{float(ret):+.2f}%",
            }
        )

    table_df = pd.DataFrame(rows)
    cached_at = pd.Timestamp.now(tz="Asia/Seoul")
    return table_df, cached_at.isoformat()


def _render_benchmark_table(account_id: str, settings: dict[str, Any], country_code: str) -> None:
    start_raw = settings.get("initial_date")
    if not start_raw:
        st.info("계정 설정에 시작일(initial_date)이 없어 벤치마크를 표시할 수 없습니다.")
        return

    try:
        start_date = pd.to_datetime(start_raw).normalize()
    except Exception:
        st.warning(f"시작일을 해석할 수 없습니다: {start_raw}")
        return

    try:
        end_date = get_latest_trading_day(country_code)
    except Exception as exc:
        st.warning(f"최근 거래일 정보를 불러오지 못했습니다: {exc}")
        return

    try:
        table_df, cached_iso = _cached_benchmark_data(account_id, start_date, end_date)
    except Exception as exc:
        st.warning(f"벤치마크 성과를 계산하지 못했습니다: {exc}")
        return

    if table_df.empty:
        st.info("표시할 벤치마크 수익률이 없습니다.")
        return

    trading_days = get_trading_days(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), country_code)
    day_count = len(trading_days)
    st.markdown(f"**벤치마크 누적 수익률 ({start_date.strftime('%Y년 %m월 %d일')} 이후 {day_count} 거래일)**")
    st.table(table_df)
    try:
        cached_kst = pd.to_datetime(cached_iso)
        if cached_kst.tzinfo is None or cached_kst.tzinfo.utcoffset(cached_kst) is None:
            cached_kst = cached_kst.tz_localize("UTC").tz_convert("Asia/Seoul")
        else:
            cached_kst = cached_kst.tz_convert("Asia/Seoul")
        ts_text = cached_kst.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        ts_text = str(cached_iso)
    st.caption(f"Momentum ETF 의 수익률은 기간 내 매수/보유/매도한 모든 종목의 실현·미실현 수익을 포함해서 계산합니다. 데이터 업데이트: {ts_text}")


__all__ = ["render_account_page"]
