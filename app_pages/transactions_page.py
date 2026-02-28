import io

import pandas as pd
import streamlit as st


def _last_business_day() -> str:
    """Return the most recent business day as YYYY-MM-DD."""
    today = pd.Timestamp.today().normalize()
    bdays = pd.bdate_range(end=today, periods=1)
    return bdays[-1].strftime("%Y-%m-%d")


def _normalize_ticker(ticker: str) -> str:
    """Strip exchange prefix from tickers (e.g., 'ASX:VGS' → 'VGS')."""
    t = str(ticker).strip()
    if ":" in t:
        t = t.split(":")[-1]
    return t


def render_transaction_management_page(active_tab: str | None = None):
    from utils.account_registry import load_account_configs

    configs = load_account_configs()
    account_map = {c["name"]: c["account_id"] for c in configs}
    account_id_to_country = {c["account_id"]: c["country_code"] for c in configs}

    # --- Navigation ---
    tabs = ["📊 잔고 CRUD", "📥 벌크 입력", "💵 원금/현금", "📸 스냅샷"]

    if active_tab is None:
        # Initialize session state for the selector key if not exists
        if "transaction_tab_selector" not in st.session_state:
            st.session_state.transaction_tab_selector = tabs[0]

        active_tab = st.segmented_control(
            "메뉴 선택",
            options=tabs,
            key="transaction_tab_selector",
            label_visibility="collapsed",
            selection_mode="single",
        )

        # Fallback to current state if somehow None (though single mode with default/init should prevent this)
        if not active_tab:
            active_tab = st.session_state.transaction_tab_selector

    # --- Tab Logic ---
    if active_tab == "📊 잔고 CRUD":
        _render_manage_tab(account_map, account_id_to_country)
    elif active_tab == "📥 벌크 입력":
        _render_bulk_tab(account_map, account_id_to_country)
    elif active_tab == "💵 원금/현금":
        _render_cash_tab(account_map)
    elif active_tab == "📸 스냅샷":
        _render_snapshot_tab(account_map)


def _render_manage_tab(account_map, account_id_to_country):
    from utils.portfolio_io import load_portfolio_master

    # Load ALL master data
    all_master_holdings = []
    for acc_name, acc_id in account_map.items():
        m_data = load_portfolio_master(acc_id)
        if m_data and m_data.get("holdings"):
            for h in m_data["holdings"]:
                h["account_name"] = acc_name
                h["account_id"] = acc_id
                all_master_holdings.append(h)

    master_df = (
        pd.DataFrame(all_master_holdings)
        if all_master_holdings
        else pd.DataFrame(
            columns=[
                "account_name",
                "account_id",
                "currency",
                "bucket",
                "ticker",
                "name",
                "quantity",
                "average_buy_price",
                "first_buy_date",
            ]
        )
    )

    # Top action bar
    if st.button("➕ 신규 종목 추가"):
        add_new_stock_modal(account_map, account_id_to_country)

    # Rename for display
    display_df = master_df.rename(
        columns={
            "account_name": "계좌",
            "currency": "환종",
            "bucket": "버킷번호",
            "ticker": "티커",
            "name": "종목명",
            "quantity": "수량",
            "average_buy_price": "평균 매입가",
            "first_buy_date": "최초매수일",
        }
    )

    show_cols = ["계좌", "환종", "버킷번호", "티커", "종목명", "수량", "평균 매입가", "최초매수일"]
    display_df = display_df[show_cols]

    selection = st.dataframe(
        display_df,
        width="stretch",
        key="manage_table",
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "수량": st.column_config.NumberColumn("수량", format="%d"),
            "평균 매입가": st.column_config.NumberColumn("평균 매입가", format="%.2f"),
        },
    )

    # Handle row selection → open edit modal
    if selection and selection.selection and selection.selection.rows:
        selected_idx = selection.selection.rows[0]
        if selected_idx < len(master_df):
            selected_row = master_df.iloc[selected_idx].to_dict()
            edit_stock_modal(selected_row)


def _render_bulk_tab(account_map, account_id_to_country):
    from config import BUCKET_CONFIG
    from utils.portfolio_io import load_portfolio_master, save_portfolio_master
    from utils.stock_meta_updater import fetch_stock_info

    st.subheader("텍스트 일괄 업데이트")
    st.info(
        "엑셀 또는 증권사 화면에서 복사한 잔고 텍스트를 붙여넣으세요. 파싱 후 [현재 잔고]에 일괄 반영할 수 있습니다.\n\n"
        "⚠️ **주의**: 일괄 반영 시 선택된 계좌의 **기존 종목 데이터는 모두 삭제되고 입력한 데이터로 완전히 교체(Overwrite)** 됩니다. "
        "(단, 기존과 동일한 티커의 최초 매수일은 유지됩니다.)"
    )

    bucket_name_to_id = {v["name"]: k for k, v in BUCKET_CONFIG.items()}
    pasted_data = st.text_area("여기에 데이터 붙여넣기 (TSV)", height=200, key="bulk_text_area")

    if st.button("🔍 데이터 파싱 및 확인"):
        if pasted_data.strip():
            try:
                raw_df = pd.read_csv(io.StringIO(pasted_data), sep="\t", header=None, dtype=str)
                expected_cols = ["계좌", "환종", "버킷_텍스트", "티커", "종목명", "수량", "평균 매입가"]

                if raw_df.shape[1] >= 7:
                    parsed_df = raw_df.iloc[:, :7].copy()
                    parsed_df.columns = expected_cols

                    parsed_df["수량"] = pd.to_numeric(
                        parsed_df["수량"].str.replace(r"[^0-9.-]", "", regex=True), errors="coerce"
                    ).fillna(0)
                    parsed_df["평균 매입가"] = pd.to_numeric(
                        parsed_df["평균 매입가"].str.replace(r"[^0-9.-]", "", regex=True), errors="coerce"
                    ).fillna(0)

                    errors = []
                    for acc_name in parsed_df["계좌"].unique():
                        if str(acc_name).strip() not in account_map:
                            errors.append(f"🏦 계좌 '{acc_name}'을(를) 찾을 수 없습니다.")
                    for bucket_text in parsed_df["버킷_텍스트"].unique():
                        if str(bucket_text).strip() not in bucket_name_to_id:
                            errors.append(f"🪣 버킷 '{bucket_text}'을(를) 찾을 수 없습니다.")

                    if errors:
                        for err in errors:
                            st.error(err)
                    else:
                        parsed_df["티커"] = parsed_df["티커"].apply(_normalize_ticker)
                        parsed_df["bucket"] = parsed_df["버킷_텍스트"].str.strip().map(bucket_name_to_id)
                        parsed_df["계좌코드"] = parsed_df["계좌"].str.strip().map(account_map)
                        st.session_state.bulk_parsed_df = parsed_df
                        st.success("✅ 파싱 및 검증 완료!")
                else:
                    st.error("데이터 형식이 맞지 않습니다.")
            except Exception as e:
                st.error(f"파싱 중 오류 발생: {e}")

    if "bulk_parsed_df" in st.session_state:
        st.divider()
        st.dataframe(
            st.session_state.bulk_parsed_df,
            column_config={
                "수량": st.column_config.NumberColumn("수량", format="%d"),
                "평균 매입가": st.column_config.NumberColumn("평균 매입가", format="%.2f"),
            },
        )
        if st.button("🚀 위 결과를 [현재 잔고] 마스터에 일괄 반영하기", type="primary"):
            parsed_df = st.session_state.bulk_parsed_df
            unique_accounts = parsed_df["계좌"].unique()
            success_count = 0
            for acc_name in unique_accounts:
                acc_id = account_map.get(str(acc_name).strip())
                if not acc_id:
                    continue
                acc_rows = parsed_df[parsed_df["계좌"] == acc_name]
                new_holdings = []
                country_code = account_id_to_country.get(acc_id, "kor")
                existing = load_portfolio_master(acc_id)
                fb_lookup = (
                    {item["ticker"]: item.get("first_buy_date") for item in existing["holdings"]}
                    if existing and existing.get("holdings")
                    else {}
                )
                name_lookup = (
                    {item["ticker"]: item.get("name") for item in existing["holdings"]}
                    if existing and existing.get("holdings")
                    else {}
                )

                for _, row in acc_rows.iterrows():
                    ticker = _normalize_ticker(row["티커"])
                    stock_name = name_lookup.get(ticker)
                    if not stock_name:
                        info = fetch_stock_info(ticker, country_code)
                        stock_name = info["name"] if info and info.get("name") else ticker

                    new_holdings.append(
                        {
                            "ticker": ticker,
                            "name": stock_name,
                            "quantity": float(row["수량"]),
                            "average_buy_price": float(row["평균 매입가"]),
                            "currency": str(row["환종"]),
                            "bucket": int(row["bucket"]),
                            "first_buy_date": fb_lookup.get(ticker, _last_business_day()),
                        }
                    )

                if save_portfolio_master(acc_id, new_holdings):
                    success_count += 1

            st.success(f"✅ 총 {success_count}개 계좌 업데이트 완료.")
            del st.session_state.bulk_parsed_df


def _render_cash_tab(account_map):
    from utils.portfolio_io import load_portfolio_master, save_portfolio_master

    st.subheader("계좌별 원금 및 현금 관리")
    with st.form("cash_manager_bulk_form"):
        input_values = {}
        for acc_name, acc_id in account_map.items():
            st.markdown(f"#### 🏦 {acc_name}")
            m_data = load_portfolio_master(acc_id)
            current_principal = m_data.get("total_principal", 0.0) if m_data else 0.0
            current_cash = m_data.get("cash_balance", 0.0) if m_data else 0.0
            current_holdings = m_data.get("holdings", []) if m_data else []
            c1, c2 = st.columns(2)
            with c1:
                new_principal = st.number_input(
                    f"투자 원금 ({acc_name})", value=int(current_principal), min_value=0, key=f"prin_{acc_id}"
                )
            with c2:
                new_cash = st.number_input(
                    f"보유 현금 ({acc_name})", value=int(current_cash), min_value=0, key=f"cash_{acc_id}"
                )
            input_values[acc_id] = {"holdings": current_holdings, "principal": new_principal, "cash": new_cash}
            st.divider()

        if st.form_submit_button("전체 계좌 일괄 저장하기", type="primary"):
            success_count = 0
            for acc_id, data in input_values.items():
                if save_portfolio_master(acc_id, data["holdings"], data["principal"], data["cash"]):
                    success_count += 1
            if success_count == len(input_values):
                st.success("✅ 저장 완료!")
                st.rerun()


def _render_snapshot_tab(account_map):
    from utils.portfolio_io import delete_daily_snapshot, list_daily_snapshots

    st.subheader("📸 일간 자산 스냅샷 관리")
    snapshots = list_daily_snapshots()
    if not snapshots:
        st.write("저장된 스냅샷이 없습니다.")
    else:
        snap_data = []
        for s in snapshots:
            snap_data.append(
                {
                    "ID": str(s["_id"]),
                    "날짜": s["snapshot_date"],
                    "총 자산": s.get("total_assets", 0),
                    "원금": s.get("total_principal", 0),
                    "현금": s.get("cash_balance", 0),
                    "평가액": s.get("valuation_krw", 0),
                    "계좌수": len(s.get("accounts", [])),
                }
            )
        snap_df = pd.DataFrame(snap_data)
        st.dataframe(
            snap_df,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="snapshot_table",
        )

        selection = st.session_state.get("snapshot_table")
        if selection and selection.selection and selection.selection.rows:
            selected_idx = selection.selection.rows[0]
            if selected_idx < len(snapshots):
                selected_snap = snapshots[selected_idx]
                st.divider()
                st.markdown(
                    """<style> .st-key-btn_del_snap button { background-color: #d32f2f !important; color: white !important; border: none !important; } </style>""",
                    unsafe_allow_html=True,
                )
                c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
                with c1:
                    st.markdown(f"#### 📂 {selected_snap['snapshot_date']} 계좌별 상세")
                with c2:
                    if st.button("❌ 스냅샷 삭제", type="secondary", width="stretch", key="btn_del_snap"):
                        if delete_daily_snapshot(str(selected_snap["_id"])):
                            st.session_state["snapshot_table"] = {"selection": {"rows": [], "columns": []}}
                            st.success("삭제되었습니다.")
                            st.rerun()

                acc_details = []
                for acc in selected_snap.get("accounts", []):
                    acc_name = next(
                        (name for name, aid in account_map.items() if aid == acc["account_id"]), acc["account_id"]
                    )
                    acc_details.append(
                        {
                            "계좌": acc_name,
                            "총 자산": acc.get("total_assets", 0),
                            "원금": acc.get("total_principal", 0),
                            "현금": acc.get("cash_balance", 0),
                            "평가액": acc.get("valuation_krw", 0),
                        }
                    )
                st.dataframe(pd.DataFrame(acc_details), width="stretch", hide_index=True)


@st.dialog("➕ 신규 종목 추가")
def add_new_stock_modal(account_map, account_id_to_country):
    from utils.portfolio_io import load_portfolio_master, save_portfolio_master
    from utils.stock_meta_updater import fetch_stock_info

    ss_key = "add_stock_lookup_result"
    new_acc_name = st.selectbox("계좌", options=list(account_map.keys()), key="add_acc_sel")
    new_currency = st.selectbox("환종", options=["KRW", "USD", "AUD"], key="add_currency_sel")
    c_in, c_btn = st.columns([3, 1], vertical_alignment="bottom")
    with c_in:
        new_ticker = st.text_input("티커 입력", placeholder="예: 005930", key="add_ticker_input").strip()
    with c_btn:
        do_search = st.button("🔍 조회", key="btn_add_lookup")

    if do_search:
        if new_ticker:
            target_country = account_id_to_country.get(account_map[new_acc_name], "kor")
            info = fetch_stock_info(new_ticker, target_country)
            if info and info.get("name"):
                st.session_state[ss_key] = info
            else:
                st.error("찾을 수 없습니다.")

    lookup_result = st.session_state.get(ss_key)
    if lookup_result:
        st.success(f"✅ 종목명: **{lookup_result['name']}**")
        c3, c4 = st.columns(2)
        with c3:
            new_bucket = st.number_input("버킷번호", min_value=1, max_value=5, value=1, key="add_bucket")
        with c4:
            new_qty = st.number_input("수량", min_value=0, step=1, key="add_qty")
        new_avg_price = st.number_input("평균 매입가", min_value=0.0, key="add_price")
        if st.button("✅ 목록에 추가", type="primary", key="btn_add_confirm"):
            final_ticker = _normalize_ticker(lookup_result["ticker"])
            target_acc_id = account_map[new_acc_name]
            existing_m = load_portfolio_master(target_acc_id)
            current_h = existing_m["holdings"] if existing_m and existing_m.get("holdings") else []
            found = False
            for h in current_h:
                if h["ticker"] == final_ticker:
                    t_qty = h["quantity"] + new_qty
                    if t_qty > 0:
                        h["average_buy_price"] = (
                            (h["average_buy_price"] * h["quantity"]) + (new_avg_price * new_qty)
                        ) / t_qty
                    h["quantity"] = t_qty
                    found = True
                    break
            if not found:
                current_h.append(
                    {
                        "ticker": final_ticker,
                        "name": lookup_result["name"],
                        "quantity": float(new_qty),
                        "average_buy_price": float(new_avg_price),
                        "currency": new_currency,
                        "bucket": int(new_bucket),
                        "first_buy_date": _last_business_day(),
                    }
                )
            if save_portfolio_master(target_acc_id, current_h):
                st.session_state[ss_key] = None
                st.rerun()


@st.dialog("✏️ 종목 수정")
def edit_stock_modal(row_data):
    from utils.portfolio_io import load_portfolio_master, save_portfolio_master

    ticker = str(row_data.get("ticker", ""))

    acc_id = str(row_data.get("account_id", ""))
    st.markdown(f"**{row_data.get('account_name')}** / **{ticker}** / {row_data.get('name')}")
    new_qty = st.number_input("수량", value=int(row_data.get("quantity", 0)), step=1, key="edit_qty")
    new_price = st.number_input("평균 매입가", value=row_data.get("average_buy_price", 0.0), key="edit_price")

    st.markdown(
        """<style> .st-key-btn_edit_save button { background-color: #2e7d32 !important; color: white !important; } .st-key-btn_edit_delete button { background-color: #d32f2f !important; color: white !important; } </style>""",
        unsafe_allow_html=True,
    )
    b1, b2 = st.columns(2)
    if b1.button("💾 저장", width="stretch", key="btn_edit_save"):
        m = load_portfolio_master(acc_id)
        h = m["holdings"]
        for item in h:
            if item["ticker"] == ticker:
                item["quantity"] = float(new_qty)
                item["average_buy_price"] = float(new_price)
                break
        if save_portfolio_master(acc_id, h):
            st.rerun()
        if save_portfolio_master(acc_id, h):
            st.rerun()


def build_transaction_page(page_cls, active_tab: str | None = None):
    title = active_tab if active_tab else "계좌 관리"
    # URL path에서 슬래시(/) 제거하여 Streamlit nested path 에러 방지
    clean_tab = active_tab.split()[-1].replace("/", "_") if active_tab else "main"
    url_path = f"transactions_{clean_tab}"
    return page_cls(
        lambda: render_transaction_management_page(active_tab),
        title=title,
        icon="📝",
        url_path=url_path,
    )
