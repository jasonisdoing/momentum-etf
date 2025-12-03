from __future__ import annotations

from typing import Any

import streamlit as st

from utils.ui import load_account_recommendations, render_recommendation_table
from utils.account_registry import get_icon_fallback, load_account_configs
from utils.settings_loader import AccountSettingsError, get_account_settings, resolve_strategy_params


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

    # 메뉴명과 동일한 이름 사용 (PORTFOLIO_TOPN 포함)
    account_configs = load_account_configs()
    account_name = None
    for config in account_configs:
        if config["account_id"] == account_id:
            account_name = config["name"]
            break

    page_title = account_name or "Momentum ETF"

    st.set_page_config(page_title=page_title, page_icon=page_icon or "📈", layout="wide")

    # 계좌 설명 표시
    account_desc = account_settings.get("desc")
    if account_desc:
        st.caption(account_desc)

    df, updated_at, loaded_country_code = load_account_recommendations(account_id)
    country_code = loaded_country_code or country_code

    if df is None:
        st.error(updated_at or "추천 데이터를 불러오지 못했습니다. 먼저 `python recommend.py <account>` 명령으로 스냅샷을 생성해 주세요.")
        return

    render_recommendation_table(df, country_code=country_code)

    if updated_at:
        st.caption(f"데이터 업데이트: {updated_at}")

        with st.expander("설정", expanded=True):
            strategy_cfg = account_settings.get("strategy", {}) or {}
            expected_cagr = None
            backtested_date = None
            strategy_tuning: dict[str, Any] = {}
            if isinstance(strategy_cfg, dict):
                expected_cagr = strategy_cfg.get("EXPECTED_CAGR")
                backtested_date = strategy_cfg.get("BACKTESTED_DATE")
                strategy_tuning = resolve_strategy_params(strategy_cfg)
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
            from config import BACKTEST_SLIPPAGE

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

    # st.markdown(_DATAFRAME_CSS, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        """
        - 본 웹사이트에서 제공되는 종목 정보 및 추천은 단순 정보 제공을 목적으로 하며, 특정 종목의 매매를 권유하는 것이 아닙니다.
        - 본 정보를 이용한 투자 판단 및 매매 결과에 대하여 웹사이트 운영자는 어떠한 책임도 지지 않습니다.
        - 투자에는 원금 손실 가능성이 있으며, 투자자는 스스로 리스크를 검토해야 합니다.
        """
    )


__all__ = ["render_account_page"]
