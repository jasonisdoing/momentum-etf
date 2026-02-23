from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth

from app_pages.account_page import render_account_page
from utils.account_registry import (
    get_icon_fallback,
    load_account_configs,
)
from utils.ui import render_recommendation_table


def _to_plain_dict(value):
    if isinstance(value, Mapping):
        return {k: _to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain_dict(v) for v in value]
    return value


def format_korean_currency(value):
    """주어진 원화 값을 '억/만원' 단위로 변환합니다."""
    if value == 0:
        return "0원"

    abs_value = abs(value)
    eok = int(abs_value // 100_000_000)
    remain = abs_value % 100_000_000
    man = int(round(remain / 10_000))

    if man == 10000:
        eok += 1
        man = 0

    parts = []
    if eok > 0:
        parts.append(f"{eok}억")
    if man > 0:
        parts.append(f"{man}만원")

    result = " ".join(parts) if parts else "0원"
    if value < 0:
        result = "-" + result
    return result


def _load_authenticator() -> stauth.Authenticate:
    raw_config = st.secrets.get("auth")
    if not raw_config:
        st.error("인증 설정(st.secrets['auth'])이 구성되지 않았습니다.")
        st.stop()

    config = _to_plain_dict(raw_config)

    credentials = config.get("credentials")
    cookie = config.get("cookie") or {}
    preauthorized = config.get("preauthorized", {})

    required_keys = {"name", "key", "expiry_days"}
    if not credentials or not cookie or not required_keys.issubset(cookie):
        st.error("인증 설정 필드가 누락되었습니다. credentials/cookie 구성을 확인하세요.")
        st.stop()

    return stauth.Authenticate(
        credentials,
        cookie.get("name"),
        cookie.get("key"),
        cookie.get("expiry_days"),
        preauthorized,
    )


def _build_account_page(page_cls: Callable[..., object], account: dict[str, Any]):
    account_id = account["account_id"]
    icon = account.get("icon") or get_icon_fallback(account.get("country_code", ""))

    def _render(account_key: str = account_id) -> None:
        render_account_page(account_key)

    return page_cls(
        _render,
        title=account["name"],
        icon=icon,
        url_path=account_id,
    )


def _build_home_page(accounts: list[dict[str, Any]]):
    def _render_home_page() -> None:
        from utils.portfolio_io import (
            get_latest_daily_snapshot,
            load_portfolio_master,
            load_real_holdings_with_recommendations,
        )

        all_holdings = []
        account_summaries = []
        global_principal = 0.0
        global_cash = 0.0
        global_change = 0.0
        global_change_pct = 0.0
        total_assets = 0.0
        total_principal = 0.0
        total_net_profit = 0.0
        total_net_profit_pct = 0.0
        total_cash = 0.0
        total_purchase = 0.0
        total_valuation = 0.0
        total_stock_profit = 0.0
        total_stock_profit_pct = 0.0

        # 데이터 로딩 (첫 로딩 시 환율/가격 조회로 시간이 걸릴 수 있음)
        visible_accounts = [a for a in accounts if a.get("settings", {}).get("show_hold", True)]
        loading_placeholder = st.empty()
        for idx, account in enumerate(visible_accounts):
            account_id = account["account_id"]
            account_name = account.get("name") or account_id.upper()
            loading_placeholder.info(f"⏳ 로딩 중... {account_name} ({idx + 1}/{len(visible_accounts)})")

            # 원금 및 현금 로드
            m_data = load_portfolio_master(account_id)
            if m_data:
                global_principal += m_data.get("total_principal", 0.0)
                global_cash += m_data.get("cash_balance", 0.0)

            df = load_real_holdings_with_recommendations(account_id)

            if df is not None and not df.empty:
                df.insert(0, "계좌", account_name)
                all_holdings.append(df)
                acc_valuation = df["평가금액(KRW)"].sum()
                acc_purchase = df["매입금액(KRW)"].sum()
            else:
                acc_valuation = 0.0
                acc_purchase = 0.0

            # 계좌별 요약용 데이터 계산
            acc_stock_profit = acc_valuation - acc_purchase
            acc_stock_profit_pct = (acc_stock_profit / acc_purchase) * 100 if acc_purchase > 0 else 0.0

            acc_principal = m_data.get("total_principal", 0.0) if m_data else 0.0
            acc_cash = m_data.get("cash_balance", 0.0) if m_data else 0.0
            acc_total_assets = acc_valuation + acc_cash
            acc_net_profit = acc_total_assets - acc_principal
            acc_net_profit_pct = (acc_net_profit / acc_principal) * 100 if acc_principal > 0 else 0.0

            # 하나라도 데이터가 있는 경우에만 요약에 추가 (또는 모든 계좌 표시)
            if acc_principal > 0 or acc_cash > 0 or acc_valuation > 0:
                account_summaries.append(
                    {
                        "계좌": account_name,
                        "총 원금": acc_principal,
                        "총 수익금": acc_net_profit,
                        "계좌 수익률": acc_net_profit_pct,
                        "총 자산": acc_total_assets,
                        "매입 금액": acc_purchase,
                        "평가 금액": acc_valuation,
                        "평가 손익": acc_stock_profit,
                        "평가 수익률": acc_stock_profit_pct,
                        "현금": acc_cash,
                    }
                )
        loading_placeholder.empty()

        if not all_holdings and not account_summaries:
            st.info("현재 모든 계좌를 통틀어 보유 중인 종목이나 자산 정보가 없습니다.")
            return

        # 합계(Total) 데이터 계산 및 추가
        if account_summaries:
            total_principal = sum(acc["총 원금"] for acc in account_summaries)
            total_net_profit = sum(acc["총 수익금"] for acc in account_summaries)
            total_net_profit_pct = (total_net_profit / total_principal) * 100 if total_principal > 0 else 0.0

            total_assets = sum(acc["총 자산"] for acc in account_summaries)
            total_purchase = sum(acc["매입 금액"] for acc in account_summaries)
            total_valuation = sum(acc["평가 금액"] for acc in account_summaries)
            total_stock_profit = sum(acc["평가 손익"] for acc in account_summaries)
            total_stock_profit_pct = (total_stock_profit / total_purchase) * 100 if total_purchase > 0 else 0.0
            total_cash = sum(acc["현금"] for acc in account_summaries)

            # Fetch previous snapshot for change calculation
            prev_global = get_latest_daily_snapshot("TOTAL", before_today=True)
            global_change = 0.0
            global_change_pct = 0.0
            if prev_global:
                prev_total = prev_global.get("total_assets", 0.0)
                if prev_total > 0:
                    global_change = total_assets - prev_total
                    global_change_pct = (global_change / prev_total) * 100

            account_summaries.append(
                {
                    "계좌": "합계",
                    "총 원금": total_principal,
                    "총 수익금": total_net_profit,
                    "계좌 수익률": total_net_profit_pct,
                    "총 자산": total_assets,
                    "매입 금액": total_purchase,
                    "평가 금액": total_valuation,
                    "평가 손익": total_stock_profit,
                    "평가 수익률": total_stock_profit_pct,
                    "현금": total_cash,
                }
            )

        combined_df = pd.concat(all_holdings, ignore_index=True) if all_holdings else pd.DataFrame()

        weight_df = None

        # 통계용 3컬럼 테이블 데이터 생성
        stat_df = pd.DataFrame(
            [
                {
                    "총 자산": f"{total_assets:,.0f}원",
                    "매입 금액": f"{total_purchase:,.0f}원",
                    "평가 금액": f"{total_valuation:,.0f}원",
                }
            ]
        )

        def get_stat_styles(df):
            style_df = pd.DataFrame("", index=df.index, columns=df.columns)
            style_df.iloc[0, 0] = (
                "background-color: #93c47d; color: black; font-weight: bold; text-align: center; padding: 8px; border: 1px solid #dee2e6;"
            )
            style_df.iloc[0, 1] = (
                "background-color: #76a5af; color: black; font-weight: bold; text-align: center; padding: 8px; border: 1px solid #dee2e6;"
            )
            style_df.iloc[0, 2] = (
                "background-color: #6fa8dc; color: black; font-weight: bold; text-align: center; padding: 8px; border: 1px solid #dee2e6;"
            )
            return style_df

        styled_stat_df = stat_df.style.apply(get_stat_styles, axis=None).hide(axis="index")

        # 2. 포트폴리오 비중 테이블 데이터 생성
        bucket_cols = ["1. 모멘텀", "2. 혁신기술", "3. 시장지수", "4. 배당방어", "5. 대체헷지"]
        bucket_totals = {}
        for col in bucket_cols:
            if not combined_df.empty and "버킷" in combined_df.columns:
                val = combined_df.loc[combined_df["버킷"] == col, "평가금액(KRW)"].sum()
            else:
                val = 0.0
            bucket_totals[col] = val

        bucket_totals["6. 현금"] = global_cash

        if total_assets > 0:
            weight_row = {}
            amount_row = {}
            for k, v in bucket_totals.items():
                weight_row[k] = f"{(v / total_assets) * 100:.2f}%"
                amount_row[k] = format_korean_currency(v)
            weight_df = pd.DataFrame([weight_row, amount_row])
        else:
            weight_df = pd.DataFrame(
                [{k: "0.00%" for k in bucket_totals.keys()}, {k: "0원" for k in bucket_totals.keys()}]
            )

        def get_weight_styles(df):
            return pd.DataFrame(
                "text-align: center; padding: 8px; border: 1px solid #dee2e6;", index=df.index, columns=df.columns
            )

        styled_weight_df = weight_df.style.apply(get_weight_styles, axis=None).hide(axis="index")

        # 3. 계좌별 요약 테이블 생성
        if account_summaries:
            # 리스트를 DataFrame으로 변환 후 전치(Transpose)
            summary_df = pd.DataFrame(account_summaries).set_index("계좌").T.reset_index()
            summary_df.columns.name = None
            summary_df = summary_df.rename(columns={"index": "계좌"})

            def style_account_summary(styler):
                # 기본 스타일
                styler.format(
                    {
                        col: (
                            lambda v: f"{v:,.0f}원"
                            if isinstance(v, (int, float)) and "수익률" not in str(styler.data.iloc[i, 0])
                            else f"{v:+.2f}%"
                        )
                        for i, col in enumerate(styler.columns)
                        if col != "계좌"
                    },
                    na_rep="",
                )

                # 행별 포맷팅 및 색상 적용
                def apply_row_styles(row):
                    styles = [""] * len(row)
                    metric_name = row["계좌"]

                    # 배경색 설정
                    if metric_name == "총 원금":
                        styles = ["background-color: #93c47d; color: black; font-weight: bold;"] * len(row)
                    elif metric_name == "총 자산":
                        styles = ["background-color: #fce5cd; color: black; font-weight: bold;"] * len(row)

                    # 글자색 설정 (수익금, 수익률 관련)
                    if "수익" in metric_name or "손익" in metric_name:
                        for i in range(1, len(row)):
                            val = row.iloc[i]
                            if isinstance(val, (int, float)):
                                if val > 0:
                                    styles[i] += " color: #e06666; font-weight: bold;"  # 빨간색
                                elif val < 0:
                                    styles[i] += " color: #3d85c6; font-weight: bold;"  # 파란색

                    # "계좌" 열은 별도 스타일 (헤더 느낌)
                    styles[0] = "background-color: #efefef; color: black; font-weight: bold;"
                    return styles

                return styler.apply(apply_row_styles, axis=1)

            # 행별로 다른 포맷 적용을 위해 수동 포맷팅 함수 정의
            def format_value(val, row_name):
                if not isinstance(val, (int, float)):
                    return val
                if "수익률" in row_name:
                    return f"{val:+.2f}%"
                return f"{val:,.0f}원"

            # 전치된 데이터이므로 각 셀에 대해 포맷팅 적용
            formatted_summary_df = summary_df.copy()
            for i, row in summary_df.iterrows():
                row_name = row["계좌"]
                for col in summary_df.columns[1:]:
                    formatted_summary_df.at[i, col] = format_value(row.at[col], row_name)

            # 위 방식은 col 루프가 필요하므로 다시 작성
            def get_styles(df_raw, df_formatted):
                style_df = pd.DataFrame("", index=df_formatted.index, columns=df_formatted.columns)
                for i, row in df_raw.iterrows():
                    metric_name = row["계좌"]
                    for col in df_raw.columns:
                        s = "padding: 8px; border: 1px solid #dee2e6;"
                        if col == "계좌":
                            s += " background-color: #cfcfcf; color: black; font-weight: bold; text-align: left;"
                        else:
                            s += " text-align: right;"
                            if metric_name == "총 원금":
                                s += " background-color: #b6d7a8; color: black; font-weight: bold;"
                            elif metric_name == "총 자산":
                                s += " background-color: #d9ead3; color: black; font-weight: bold;"

                            if "수익" in metric_name or "손익" in metric_name:
                                val = row[col]
                                if isinstance(val, (int, float)):
                                    if val > 0:
                                        s += " color: #e06666; font-weight: bold;"
                                    elif val < 0:
                                        s += " color: #3d85c6; font-weight: bold;"
                            elif metric_name == "계좌 수익률" or metric_name == "현금":
                                s += " font-weight: bold;"
                        style_df.at[i, col] = s
                return style_df

            styled_summary_df = formatted_summary_df.style.apply(
                lambda _: get_styles(summary_df, formatted_summary_df), axis=None
            ).hide(axis="index")
        else:
            styled_summary_df = None

        if "cache_warnings" in st.session_state and st.session_state.cache_warnings:
            # {account_id: {ticker_set}}
            warning_msg = "⚠️ **다음 계좌에서 일부 종목의 가격 데이터를 불러오지 못했습니다:**\n\n"

            # 계좌 ID를 이름으로 매핑하기 위한 맵 생성
            id_to_name = {acc["account_id"]: (acc.get("name") or acc["account_id"].upper()) for acc in accounts}

            for acc_id, tickers in sorted(st.session_state.cache_warnings.items()):
                target_name = id_to_name.get(acc_id, acc_id.upper())
                ticker_str = ", ".join(sorted(tickers))
                warning_msg += f"- **{target_name}**: {ticker_str}\n"

            st.warning(
                f"{warning_msg}\n"
                "현재가가 0원으로 표시될 수 있습니다. 해결을 위해 백그라운드 스크립트(`python scripts/update_price_cache.py`)를 "
                "실행하여 가격 정보를 갱신해 주시기 바랍니다."
            )
            # 한 번 보여준 후 다음 렌더링을 위해 초기화
            st.session_state.cache_warnings = {}

        if "home_active_subtab" not in st.session_state:
            st.session_state.home_active_subtab = "📊 요약"

        active_subtab = st.segmented_control(
            "홈 메뉴",
            options=["📊 요약", "📋 상세"],
            default=st.session_state.home_active_subtab,
            key="home_subtab_selector",
            label_visibility="collapsed",
        )
        if active_subtab:
            st.session_state.home_active_subtab = active_subtab
        else:
            active_subtab = st.session_state.home_active_subtab

        if active_subtab == "📊 요약":
            if total_assets > 0 or total_purchase > 0:
                # 섹션 간 간격 최소화를 위한 전역 CSS
                st.markdown(
                    """
                    <style>
                        [data-testid="stMetric"] { padding-bottom: 0px; }
                        [data-testid="stSubheader"] { margin-bottom: -15px; margin-top: 10px; }
                        div.stMarkdown { margin-bottom: -10px; }
                        .summary-table {
                            width: 100%;
                            border-collapse: collapse;
                            font-size: 14px;
                            margin-top: 5px;
                            margin-bottom: 5px;
                        }
                        .summary-table th {
                            background-color: #cfcfcf !important;
                            color: black !important;
                            font-weight: bold !important;
                            padding: 8px;
                            border: 1px solid #dee2e6;
                            text-align: center;
                        }
                        .summary-table td {
                            padding: 8px;
                            border: 1px solid #dee2e6;
                        }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                st.subheader("총 자산 요약")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric(label="총 자산 (주식+현금)", value=format_korean_currency(total_assets))
                c2.metric(label="총 투자 원금", value=format_korean_currency(total_principal))
                c3.metric(
                    label="전일 대비",
                    value=format_korean_currency(global_change),
                    delta=f"{global_change_pct:+.2f}%",
                )
                c4.metric(
                    label="총 평가손익",
                    value=format_korean_currency(total_net_profit),
                    delta=f"{total_net_profit_pct:,.2f}%",
                )
                c5.metric(label="총 현금 보유량", value=format_korean_currency(total_cash))

                if styled_summary_df is not None:
                    st.subheader("계좌별 요약")
                    table_html = styled_summary_df.to_html()
                    full_html = f'<div style="overflow-x: auto;">{table_html.replace("<table ", "<table class='summary-table' ")}</div>'
                    st.html(full_html)

                st.subheader("포트폴리오 구성 비중")
                table_weight_html = styled_weight_df.to_html()
                full_weight_html = f'<div style="width: 70%; overflow-x: auto;">{table_weight_html.replace("<table ", "<table class='summary-table' ")}</div>'
                st.html(full_weight_html)

                st.subheader("통계용")
                table_stat_html = styled_stat_df.to_html()
                full_stat_html = f'<div style="width: 50%; overflow-x: auto;">{table_stat_html.replace("<table ", "<table class='summary-table' ")}</div>'
                st.html(full_stat_html)
            else:
                st.info("평가금액 및 매입금액 데이터가 없어 요약을 표시할 수 없습니다.")

        elif active_subtab == "📋 상세":
            # 정렬: 계좌순(이름에 order가 포함됨) -> 버킷순
            if "bucket" in combined_df.columns:
                combined_df = combined_df.sort_values(["계좌", "bucket"], ascending=[True, True])
            else:
                combined_df = combined_df.sort_values(["계좌"], ascending=[True])

            # Rename target column to 평가수익률(%)
            if "수익률(%)" in combined_df.columns:
                combined_df = combined_df.rename(columns={"수익률(%)": "평가수익률(%)"})

            # render_recommendation_table 호출 (컬럼 순서 제어를 위해 visible_columns 명시)
            visible_cols = [
                "계좌",
                "환종",
                "버킷",
                "티커",
                "종목명",
                "일간(%)",
                "보유일",
                "평가수익률(%)",
                "수량",
                "평균 매입가",
                "현재가",
                "매입금액(KRW)",
                "평가금액(KRW)",
                "평가손익(KRW)",
                "추세(3달)",
            ]
            # Warnings moved to the top of the tabs

            render_recommendation_table(combined_df, grouped_by_bucket=False, visible_columns=visible_cols, height=900)

    return _render_home_page


def main() -> None:
    navigation = getattr(st, "navigation", None)
    page_cls = getattr(st, "Page", None)
    if navigation is None or page_cls is None:
        st.error("현재 설치된 Streamlit 버전이 `st.navigation`을 지원하지 않습니다.")
        st.stop()

    accounts = load_account_configs()
    if not accounts:
        st.error("사용할 수 있는 계정 설정이 없습니다. `zaccounts/account` 폴더를 확인해주세요.")
        st.stop()

    default_icon = "📈"

    st.set_page_config(
        page_title="Momentum ETF",
        page_icon=default_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Open Graph 메타 태그
    st.markdown(
        """
        <meta property="og:title" content="Momentum ETF" />
        <meta property="og:description" content="추세추종 전략 기반 ETF 투자" />
        <meta property="og:image" content="https://etf.dojason.com/static/og-image.png" />
        <meta property="og:url" content="https://etf.dojason.com/" />
        <meta property="og:type" content="website" />
        <meta property="og:site_name" content="Momentum ETF" />
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content="Momentum ETF" />
        <meta name="twitter:description" content="추세추종 전략 기반 ETF 투자" />
        <meta name="twitter:image" content="https://etf.dojason.com/static/og-image.png" />
        """,
        unsafe_allow_html=True,
    )

    # --- 1. 페이지 정의 (인증보다 먼저 수행하여 라우팅 정보 등록) ---
    from app_pages.transactions_page import build_transaction_page

    pages = [
        page_cls(
            _build_home_page(accounts),
            title="보유종목",
            icon="🏠",
            default=True,
        )
    ]
    pages.append(build_transaction_page(page_cls))
    for account in accounts:
        pages.append(_build_account_page(page_cls, account))

    # 네비게이션 객체 생성 (상단 탭 방식 — 일관된 패딩 동작)
    pg = navigation(pages, position="top")

    # --- 인증 로직 시작 ---
    authenticator = _load_authenticator()
    _, auth_status, _ = authenticator.login(location="main")

    if auth_status is False:
        st.error("이메일/사용자명 또는 비밀번호가 올바르지 않습니다.")
        st.stop()
    elif auth_status is None:
        st.warning("계속하려면 로그인하세요.")
        st.stop()

    # 로그인 성공 시 사이드바에 로그아웃 버튼 표시
    with st.sidebar:
        st.write(f"환영합니다, {st.session_state.get('name', 'User')}님!")
        authenticator.logout(button_name="로그아웃", location="sidebar")
        st.divider()
    # --- 인증 로직 끝 ---

    # 전역 CSS 주입
    from utils.ui import inject_global_css

    inject_global_css()

    # --- 3. 라우팅 실행 ---
    pg.run()


if __name__ == "__main__":
    main()
