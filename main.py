import os
import sys
import warnings
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# pkg_resources 워닝 억제
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

try:
    import pytz
except ImportError:
    pytz = None


# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import yfinance as yf
except ImportError:
    st.error("yfinance 라이브러리가 설치되지 않았습니다. `pip install yfinance`로 설치해주세요.")
    yf = None
    st.stop()

from signals import get_market_regime_status_string
from utils.account_registry import (
    get_accounts_by_country,
    load_accounts,
)
from utils.db_manager import (
    get_available_snapshot_dates,
    get_latest_signal_report,
    get_signal_report_on_or_after,
)


# 캐시 관련 설정
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "dashboard_data.pkl"
CACHE_DURATION_MINUTES = 5  # 캐시 유효 시간 (분)


def get_cache_key(selected_date_str: str) -> str:
    """캐시 키를 생성합니다."""
    return f"dashboard_{selected_date_str}"


def load_cached_data(selected_date_str: str) -> tuple[dict, datetime] | None:
    """캐시된 데이터를 로드합니다."""
    try:
        cache_file = CACHE_DIR / f"{get_cache_key(selected_date_str)}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
                cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)

                # 캐시 유효성 검사
                if datetime.now() - cache_time < timedelta(minutes=CACHE_DURATION_MINUTES):
                    return cached_data, cache_time
    except Exception:
        pass
    return None


def save_cached_data(selected_date_str: str, data: dict) -> None:
    """데이터를 캐시에 저장합니다."""
    try:
        cache_file = CACHE_DIR / f"{get_cache_key(selected_date_str)}.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
    except Exception:
        pass


def clear_cache() -> None:
    """캐시를 삭제합니다."""
    try:
        for cache_file in CACHE_DIR.glob("dashboard_*.pkl"):
            cache_file.unlink()
    except Exception:
        pass


def load_dashboard_data(selected_date_str: str, all_accounts: list) -> dict:
    """대시보드 데이터를 로드합니다."""
    selected_date_dt = pd.to_datetime(selected_date_str)
    selected_date_py = selected_date_dt.to_pydatetime()

    account_summaries = []
    total_initial_capital_krw = 0.0
    total_current_equity_krw = 0.0
    total_daily_profit_loss_krw = 0.0
    total_eval_profit_loss_krw = 0.0
    total_cum_profit_loss_krw = 0.0
    total_cash_krw = 0.0
    total_holdings_value_krw = 0.0
    accounts_without_data: list[str] = []
    fallback_notes: list[str] = []

    for account_info in all_accounts:
        country = account_info["country"]
        account = account_info["account"]

        try:
            # 선택한 날짜에 해당하는 요약 데이터를 가져옵니다.
            report_doc = get_latest_signal_report(country, account, date=selected_date_py)
            fallback_doc = None
            if not report_doc:
                fallback_doc = get_signal_report_on_or_after(country, account, selected_date_py)
            target_doc = report_doc or fallback_doc
            if not target_doc or "summary" not in target_doc:
                accounts_without_data.append(account_info["display_name"])
                continue

            summary = target_doc.get("summary", {})
            doc_date = target_doc.get("date")
            data_date_str = (
                pd.to_datetime(doc_date).strftime("%Y-%m-%d")
                if doc_date is not None
                else selected_date_str
            )

            # --- KRW로 모든 값 변환 ---
            initial_capital_krw_local = summary.get("principal", 0.0)
            current_equity_krw_local = summary.get("total_equity", 0.0)
            daily_profit_loss_krw_local = summary.get("daily_profit_loss", 0.0)
            eval_profit_loss_krw_local = summary.get("eval_profit_loss", 0.0)
            cum_profit_loss_krw_local = summary.get("cum_profit_loss", 0.0)
            cash_krw_local = summary.get("total_cash", 0.0)
            holdings_value_krw_local = summary.get("total_holdings_value", 0.0)

            # --- Add to totals (already in KRW) ---
            total_initial_capital_krw += initial_capital_krw_local
            total_current_equity_krw += current_equity_krw_local
            total_daily_profit_loss_krw += daily_profit_loss_krw_local
            total_eval_profit_loss_krw += eval_profit_loss_krw_local
            total_cum_profit_loss_krw += cum_profit_loss_krw_local
            total_cash_krw += cash_krw_local
            total_holdings_value_krw += holdings_value_krw_local

            # --- Prepare summary for display (all in KRW) ---
            account_summaries.append(
                {
                    "display_name": account_info["display_name"],
                    "principal": initial_capital_krw_local,
                    "current_equity": current_equity_krw_local,
                    "total_cash": cash_krw_local,
                    "daily_profit_loss": daily_profit_loss_krw_local,
                    "daily_return_pct": summary.get("daily_return_pct", 0.0),
                    "eval_profit_loss": eval_profit_loss_krw_local,
                    "eval_return_pct": summary.get("eval_return_pct", 0.0),
                    "cum_profit_loss": cum_profit_loss_krw_local,
                    "cum_return_pct": summary.get("cum_return_pct", 0.0),
                    "currency": "KRW",  # Always display in KRW
                    "amt_precision": 0,  # Always display as integer KRW
                    "qty_precision": 0,
                    "order": account_info.get("order", 99),
                    "data_date": data_date_str,
                }
            )
        except Exception as e:
            st.warning(f"'{account_info['display_name']}' 계좌 정보를 불러오는 중 오류 발생: {e}")
            continue

    return {
        "account_summaries": account_summaries,
        "total_initial_capital_krw": total_initial_capital_krw,
        "total_current_equity_krw": total_current_equity_krw,
        "total_daily_profit_loss_krw": total_daily_profit_loss_krw,
        "total_eval_profit_loss_krw": total_eval_profit_loss_krw,
        "total_cum_profit_loss_krw": total_cum_profit_loss_krw,
        "total_cash_krw": total_cash_krw,
        "total_holdings_value_krw": total_holdings_value_krw,
        "accounts_without_data": accounts_without_data,
        "fallback_notes": fallback_notes,
    }


def main():
    """대시보드를 렌더링합니다."""
    st.set_page_config(page_title="Momentum ETF", page_icon="📈", layout="wide")
    # st.title("📈 대시보드")

    hide_amounts = st.toggle("금액 숨기기", key="hide_amounts")

    # st.markdown("---")

    status_html = get_market_regime_status_string()
    if status_html:
        # 페이지 우측 상단에 시장 상태를 표시합니다.
        st.markdown(f'<div style="text-align: right;">{status_html}</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');
            /* 전역 폰트: D2Coding 우선 적용, 미설치 시 폴백 */
            body, code, pre {
                font-family: 'D2Coding', 'NanumGothic Coding', 'Noto Sans KR', 'Consolas', 'Courier New', monospace !important;
            }
            .block-container {
                max-width: 100%;
                padding-top: 1rem;
                padding-left: 2rem;
                padding-right: 2rem;
            }
            /* Custom CSS to reduce sidebar width */
            [data-testid="stSidebar"] {
                width: 150px !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("계좌 정보 로딩 중..."):
        load_accounts(force_reload=False)
        all_accounts = []
        for country_code in ["kor", "aus", "coin"]:
            accounts = get_accounts_by_country(country_code)
            if accounts:
                for acc in accounts:
                    if acc.get("is_active", True):
                        all_accounts.append(acc)

    if not all_accounts:
        st.info("활성화된 계좌가 없습니다. `country_mapping.json`에 계좌를 추가하고 `is_active: true`로 설정해주세요.")
        st.stop()

    available_dates: set[str] = set()
    for account_info in all_accounts:
        account_dates = get_available_snapshot_dates(
            account_info["country"], account_info["account"]
        )
        available_dates.update(account_dates)

    if not available_dates:
        st.info("표시할 시그널 데이터가 없습니다. 먼저 시그널을 생성해주세요.")
        st.stop()

    today = pd.Timestamp.now().normalize()
    date_options = [d for d in available_dates if pd.to_datetime(d) <= today]
    date_options = sorted(date_options, reverse=True)
    selected_date_str = st.selectbox(
        "조회 날짜",
        date_options,
        index=0,
        key="dashboard_date_select",
    )

    # 캐시 정보 표시

    # 캐시에서 데이터 로드 시도
    cached_result = load_cached_data(selected_date_str)
    cache_time = None

    if cached_result:
        dashboard_data, cache_time = cached_result
        st.info(f"📊 캐시된 데이터를 사용합니다 (저장 시간: {cache_time.strftime('%Y-%m-%d %H:%M:%S')})")
    else:
        # 캐시가 없거나 만료된 경우 새로 로드
        with st.spinner("대시보드 데이터를 로딩 중..."):
            dashboard_data = load_dashboard_data(selected_date_str, all_accounts)
            save_cached_data(selected_date_str, dashboard_data)
            cache_time = datetime.now()
            st.success("✅ 최신 데이터를 로드했습니다")
    if cache_time:
        cache_age = datetime.now() - cache_time
        if cache_age.total_seconds() < 60:
            age_text = f"{int(cache_age.total_seconds())}초 전"
        elif cache_age.total_seconds() < 3600:
            age_text = f"{int(cache_age.total_seconds() // 60)}분 전"
        else:
            age_text = f"{int(cache_age.total_seconds() // 3600)}시간 전"

        st.caption(f"📊 캐시 데이터 저장 시간: {cache_time.strftime('%Y-%m-%d %H:%M:%S')} ({age_text})")
    else:
        st.caption("📊 캐시 데이터 없음")

    # st.markdown("---")

    # 데이터 추출
    account_summaries = dashboard_data["account_summaries"]
    total_initial_capital_krw = dashboard_data["total_initial_capital_krw"]
    total_current_equity_krw = dashboard_data["total_current_equity_krw"]
    total_daily_profit_loss_krw = dashboard_data["total_daily_profit_loss_krw"]
    total_eval_profit_loss_krw = dashboard_data["total_eval_profit_loss_krw"]
    total_cum_profit_loss_krw = dashboard_data["total_cum_profit_loss_krw"]
    total_cash_krw = dashboard_data["total_cash_krw"]
    total_holdings_value_krw = dashboard_data["total_holdings_value_krw"]
    accounts_without_data = dashboard_data["accounts_without_data"]
    fallback_notes = dashboard_data["fallback_notes"]

    if fallback_notes:
        fallback_msg = "<br/>".join(fallback_notes)
        st.caption(
            f"선택한 날짜에 일부 계좌의 데이터가 없어 다음 거래일 데이터를 사용했습니다.<br/>{fallback_msg}",
            unsafe_allow_html=True,
        )

    if accounts_without_data:
        missing_list = ", ".join(accounts_without_data)
        st.warning(f"다음 계좌의 시그널 데이터가 '{selected_date_str}' 이후로 존재하지 않습니다: {missing_list}")

    if not account_summaries:
        st.info(f"'{selected_date_str}' 날짜에 표시할 데이터가 없습니다.")
        st.stop()

    # Display header
    header_cols = st.columns((1.5, 1.5, 1.5, 1, 1.5, 1, 1.5, 1, 1.5, 1.5))
    header_cols[0].markdown("**계좌**")
    header_cols[1].markdown(
        "<div style='text-align: right;'><b>원금</b></div>", unsafe_allow_html=True
    )
    header_cols[2].markdown(
        "<div style='text-align: right;'><b>일간손익</b></div>", unsafe_allow_html=True
    )
    header_cols[3].markdown(
        "<div style='text-align: right;'><b>일간(%)</b></div>", unsafe_allow_html=True
    )
    header_cols[4].markdown(
        "<div style='text-align: right;'><b>평가손익</b></div>", unsafe_allow_html=True
    )
    header_cols[5].markdown(
        "<div style='text-align: right;'><b>평가(%)</b></div>", unsafe_allow_html=True
    )
    header_cols[6].markdown(
        "<div style='text-align: right;'><b>누적손익</b></div>", unsafe_allow_html=True
    )
    header_cols[7].markdown(
        "<div style='text-align: right;'><b>누적(%)</b></div>", unsafe_allow_html=True
    )
    header_cols[8].markdown(
        "<div style='text-align: right;'><b>현금</b></div>", unsafe_allow_html=True
    )
    header_cols[9].markdown(
        "<div style='text-align: right;'><b>평가금액</b></div>", unsafe_allow_html=True
    )
    st.markdown("""<hr style="margin:0.5rem 0;" />""", unsafe_allow_html=True)

    for summary in sorted(account_summaries, key=lambda x: x.get("order", 99)):
        currency_symbol = "원"  # All summaries are now in KRW
        amt_precision = summary["amt_precision"]

        cols = st.columns((1.5, 1.5, 1.5, 1, 1.5, 1, 1.5, 1, 1.5, 1.5))
        data_date = summary.get("data_date")
        display_label = summary["display_name"]

        # 계좌명을 클릭 가능한 링크로 표시
        account_code = None
        for account_info in all_accounts:
            if account_info["display_name"] == display_label:
                account_code = account_info["account"]
                break

        if account_code:
            # 같은 창에서 열리도록 HTML 링크 사용 - signal 페이지로 이동
            if data_date:
                cols[0].markdown(
                    f"<div><a href='/results?account={account_code}' target='_self' style='text-decoration: none; color: #1f77b4; font-weight: bold;'>{display_label}</a><br/><span style='color:#666;font-size:0.85em;'>기준일: {data_date}</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                cols[0].markdown(
                    f"<div><a href='/results?account={account_code}' target='_self' style='text-decoration: none; color: #1f77b4; font-weight: bold;'>{display_label}</a></div>",
                    unsafe_allow_html=True,
                )
        else:
            if data_date:
                cols[0].markdown(
                    f"<div><strong>{display_label}</strong><br/><span style='color:#666;font-size:0.85em;'>기준일: {data_date}</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                cols[0].write(display_label)

        def format_amount(value):
            return f"{value:,.{amt_precision}f} {currency_symbol}"

        def format_amount_with_sign(value):
            color = "red" if value >= 0 else "blue"
            sign = "+" if value > 0 else ""
            return f"<div style='text-align: right; color: {color};'>{sign}{value:,.{amt_precision}f} {currency_symbol}</div>"

        def format_pct(value):
            color = "red" if value > 0 else "blue" if value < 0 else "black"
            return f"<div style='text-align: right; color: {color};'>{value:+.2f}%</div>"

        if hide_amounts:
            hidden_str = "****** " + currency_symbol
            cols[1].markdown(
                f"<div style='text-align: right;'>{hidden_str}</div>", unsafe_allow_html=True
            )
            cols[2].markdown(
                f"<div style='text-align: right;'>{hidden_str}</div>", unsafe_allow_html=True
            )
            cols[4].markdown(
                f"<div style='text-align: right;'>{hidden_str}</div>", unsafe_allow_html=True
            )
            cols[6].markdown(
                f"<div style='text-align: right;'>{hidden_str}</div>", unsafe_allow_html=True
            )
            cols[8].markdown(
                f"<div style='text-align: right;'>{hidden_str}</div>", unsafe_allow_html=True
            )
            cols[9].markdown(
                f"<div style='text-align: right;'>{hidden_str}</div>", unsafe_allow_html=True
            )
        else:
            # 원금
            cols[1].markdown(
                f"<div style='text-align: right;'>{format_amount(summary['principal'])}</div>",
                unsafe_allow_html=True,
            )
            # 일간손익
            cols[2].markdown(
                format_amount_with_sign(summary["daily_profit_loss"]), unsafe_allow_html=True
            )
            # 평가손익
            cols[4].markdown(
                format_amount_with_sign(summary["eval_profit_loss"]), unsafe_allow_html=True
            )
            # 누적손익
            cols[6].markdown(
                format_amount_with_sign(summary["cum_profit_loss"]), unsafe_allow_html=True
            )
            # 현금
            cols[8].markdown(
                f"<div style='text-align: right;'>{format_amount(summary['total_cash'])}</div>",
                unsafe_allow_html=True,
            )
            # 평가금액
            cols[9].markdown(
                f"<div style='text-align: right;'>{format_amount(summary['current_equity'])}</div>",
                unsafe_allow_html=True,
            )

        # % 값들
        cols[3].markdown(format_pct(summary["daily_return_pct"]), unsafe_allow_html=True)
        cols[5].markdown(format_pct(summary["eval_return_pct"]), unsafe_allow_html=True)
        cols[7].markdown(format_pct(summary["cum_return_pct"]), unsafe_allow_html=True)

    # --- 총 자산 합계 행 추가 ---
    st.markdown("""<hr style="margin:0.5rem 0;" />""", unsafe_allow_html=True)

    # --- 총계 수익률 계산 ---
    total_prev_equity_krw = total_current_equity_krw - total_daily_profit_loss_krw
    total_daily_return_pct = (
        (total_daily_profit_loss_krw / total_prev_equity_krw) * 100
        if total_prev_equity_krw > 0
        else 0.0
    )
    total_acquisition_cost_krw = total_holdings_value_krw - total_eval_profit_loss_krw
    total_eval_return_pct = (
        (total_eval_profit_loss_krw / total_acquisition_cost_krw) * 100
        if total_acquisition_cost_krw > 0
        else 0.0
    )
    total_cum_return_pct = (
        (total_cum_profit_loss_krw / total_initial_capital_krw) * 100
        if total_initial_capital_krw > 0
        else 0.0
    )

    # --- 총계 행 렌더링 ---
    total_cols = st.columns((1.5, 1.5, 1.5, 1, 1.5, 1, 1.5, 1, 1.5, 1.5))
    total_cols[0].markdown("<b>총 자산</b>", unsafe_allow_html=True)

    def format_total_amount(value):
        return f"{value:,.0f} 원"

    def format_total_amount_with_sign(value):
        color = "red" if value >= 0 else "blue"
        sign = "+" if value > 0 else ""
        return f"<div style='text-align: right; color: {color};'><b>{sign}{value:,.0f} 원</b></div>"

    def format_total_pct(value):
        color = "red" if value > 0 else "blue" if value < 0 else "black"
        return f"<div style='text-align: right; color: {color};'><b>{value:+.2f}%</b></div>"

    if hide_amounts:
        hidden_str = "****** 원"
        for i in [1, 2, 4, 6, 8, 9]:
            total_cols[i].markdown(
                f"<div style='text-align: right;'><b>{hidden_str}</b></div>",
                unsafe_allow_html=True,
            )
    else:
        # 원금
        total_cols[1].markdown(
            f"<div style='text-align: right;'><b>{format_total_amount(total_initial_capital_krw)}</b></div>",
            unsafe_allow_html=True,
        )
        # 일간손익
        total_cols[2].markdown(
            format_total_amount_with_sign(total_daily_profit_loss_krw), unsafe_allow_html=True
        )
        # 평가손익
        total_cols[4].markdown(
            format_total_amount_with_sign(total_eval_profit_loss_krw), unsafe_allow_html=True
        )
        # 누적손익
        total_cols[6].markdown(
            format_total_amount_with_sign(total_cum_profit_loss_krw), unsafe_allow_html=True
        )
        # 현금
        total_cols[8].markdown(
            f"<div style='text-align: right;'><b>{format_total_amount(total_cash_krw)}</b></div>",
            unsafe_allow_html=True,
        )
        # 평가금액
        total_cols[9].markdown(
            f"<div style='text-align: right;'><b>{format_total_amount(total_current_equity_krw)}</b></div>",
            unsafe_allow_html=True,
        )

    # % 값들 (금액 숨기기와 무관)
    total_cols[3].markdown(format_total_pct(total_daily_return_pct), unsafe_allow_html=True)
    total_cols[5].markdown(format_total_pct(total_eval_return_pct), unsafe_allow_html=True)
    total_cols[7].markdown(format_total_pct(total_cum_return_pct), unsafe_allow_html=True)

    # 새로고침 버튼을 왼쪽 정렬
    if st.button("🔄 최신 데이터 가져오기", key="refresh_dashboard_data"):
        clear_cache()
        st.rerun()


if __name__ == "__main__":
    main()
