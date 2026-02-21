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


def render_transaction_management_page():
    from utils.account_registry import load_account_configs
    from utils.portfolio_io import load_portfolio_master, save_portfolio_master

    configs = load_account_configs()
    account_map = {c["name"]: c["account_id"] for c in configs}
    account_id_to_country = {c["account_id"]: c["country_code"] for c in configs}

    @st.dialog("➕ 신규 종목 추가")
    def add_new_stock_modal():
        from utils.stock_meta_updater import fetch_stock_info

        ss_key = "add_stock_lookup_result"

        new_acc_name = st.selectbox("계좌", options=list(account_map.keys()), key="add_acc_sel")
        new_currency = st.selectbox("환종", options=["KRW", "USD", "AUD"], key="add_currency_sel")

        # Step 1: 티커 입력 + 조회 버튼
        c_in, c_btn = st.columns([3, 1], vertical_alignment="bottom")
        with c_in:
            new_ticker = st.text_input("티커 입력", placeholder="예: 005930", key="add_ticker_input").strip()
        with c_btn:
            do_search = st.button("🔍 조회", key="btn_add_lookup")

        if do_search:
            if not new_ticker:
                st.error("티커를 입력하세요.")
                st.session_state[ss_key] = None
            else:
                target_acc_id = account_map[new_acc_name]
                target_country = account_id_to_country.get(target_acc_id, "kor")
                with st.spinner("종목 정보 조회 중..."):
                    info = fetch_stock_info(new_ticker, target_country)
                if info and info.get("name"):
                    st.session_state[ss_key] = info
                else:
                    st.error(f"'{new_ticker}'에 해당하는 종목을 찾을 수 없습니다.")
                    st.session_state[ss_key] = None

        # Step 2: 조회 결과가 있으면 나머지 입력 필드 표시
        lookup_result = st.session_state.get(ss_key)
        if lookup_result:
            st.success(f"✅ 종목명: **{lookup_result['name']}**")

            c3, c4 = st.columns(2)
            with c3:
                new_bucket = st.number_input("버킷번호", min_value=1, max_value=5, value=1, key="add_bucket")
            with c4:
                new_qty = st.number_input("수량", min_value=0.0, format="%.2f", step=0.01, key="add_qty")

            if new_currency == "KRW":
                price_format, price_step = "%d", 1
            else:
                price_format, price_step = "%.4f", 0.0001

            new_avg_price = st.number_input(
                "평균 매입가", min_value=0.0, format=price_format, step=price_step, key="add_price"
            )

            if st.button("✅ 목록에 추가", type="primary", key="btn_add_confirm"):
                final_ticker = _normalize_ticker(lookup_result["ticker"])
                final_name = lookup_result["name"]
                target_acc_id = account_map[new_acc_name]

                existing_m = load_portfolio_master(target_acc_id)
                current_h = existing_m["holdings"] if existing_m and existing_m.get("holdings") else []

                found = False
                for h in current_h:
                    if h["ticker"] == final_ticker:
                        total_qty = h["quantity"] + new_qty
                        if total_qty > 0:
                            h["average_buy_price"] = (
                                (h["average_buy_price"] * h["quantity"]) + (new_avg_price * new_qty)
                            ) / total_qty
                        h["quantity"] = total_qty
                        found = True
                        break

                if not found:
                    current_h.append(
                        {
                            "ticker": final_ticker,
                            "name": final_name,
                            "quantity": float(new_qty),
                            "average_buy_price": float(new_avg_price),
                            "currency": new_currency,
                            "bucket": int(new_bucket),
                            "first_buy_date": _last_business_day(),
                        }
                    )

                if save_portfolio_master(target_acc_id, current_h):
                    st.session_state[ss_key] = None
                    st.success(f"{final_name} 추가 완료!")
                    st.rerun()
                else:
                    st.error("저장 실패")

    tab_manage, tab_bulk, tab_cash = st.tabs(["📊 잔고 관리 (CRUD)", "📥 잔고 벌크 입력", "💵 원금 및 현금 관리"])

    # --- Tab 1: 잔고 관리 (Unified CRUD) ---
    with tab_manage:
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

        @st.dialog("✏️ 종목 수정")
        def edit_stock_modal(row_data):
            currency = str(row_data.get("currency", "KRW"))
            ticker = str(row_data.get("ticker", ""))
            name = str(row_data.get("name", ""))
            acc_id = str(row_data.get("account_id", ""))
            acc_name = str(row_data.get("account_name", ""))

            st.markdown(f"**{acc_name}** / **{ticker}** / {name}")

            # Currency-specific formatting
            if currency == "KRW":
                qty_val = int(row_data.get("quantity", 0))
                price_val = int(row_data.get("average_buy_price", 0))
                qty_format, qty_step = "%d", 1
                price_format, price_step = "%d", 1
            elif currency == "USD":
                qty_val = int(row_data.get("quantity", 0))
                price_val = float(row_data.get("average_buy_price", 0))
                qty_format, qty_step = "%d", 1
                price_format, price_step = "%.4f", 0.0001
            else:  # AUD
                qty_val = float(row_data.get("quantity", 0))
                price_val = float(row_data.get("average_buy_price", 0))
                qty_format, qty_step = "%.4f", 0.0001
                price_format, price_step = "%.4f", 0.0001

            c1, c2 = st.columns(2)
            with c1:
                new_qty = st.number_input(
                    "수량",
                    value=qty_val,
                    min_value=0 if isinstance(qty_val, int) else 0.0,
                    format=qty_format,
                    step=qty_step,
                    key="edit_qty",
                )
            with c2:
                new_price = st.number_input(
                    "평균 매입가",
                    value=price_val,
                    min_value=0 if isinstance(price_val, int) else 0.0,
                    format=price_format,
                    step=price_step,
                    key="edit_price",
                )

            # Buttons: 저장(green) + 삭제(red) side by side, full width
            st.markdown(
                """<style>
                .st-key-btn_edit_save button {
                    background-color: #2e7d32 !important; color: white !important; border: none !important;
                }
                .st-key-btn_edit_delete button {
                    background-color: #d32f2f !important; color: white !important; border: none !important;
                }
            </style>""",
                unsafe_allow_html=True,
            )

            btn_save_col, btn_del_col = st.columns(2)
            with btn_save_col:
                save_clicked = st.button("💾 저장", width="stretch", key="btn_edit_save")
            with btn_del_col:
                delete_clicked = st.button("🗑️ 삭제", width="stretch", key="btn_edit_delete")

            if save_clicked:
                existing_m = load_portfolio_master(acc_id)
                current_h = existing_m["holdings"] if existing_m and existing_m.get("holdings") else []

                for h in current_h:
                    if h["ticker"] == ticker:
                        h["quantity"] = float(new_qty)
                        h["average_buy_price"] = float(new_price)
                        break

                if save_portfolio_master(acc_id, current_h):
                    st.success("저장 완료!")
                    st.rerun()
                else:
                    st.error("저장 실패")

            if delete_clicked:
                existing_m = load_portfolio_master(acc_id)
                current_h = existing_m["holdings"] if existing_m and existing_m.get("holdings") else []
                updated_h = [h for h in current_h if h["ticker"] != ticker]

                if save_portfolio_master(acc_id, updated_h):
                    st.success(f"{ticker} 삭제 완료!")
                    st.rerun()
                else:
                    st.error("삭제 실패")

        # Top action bar
        if st.button("➕ 신규 종목 추가"):
            add_new_stock_modal()

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
        )

        # Handle row selection → open edit modal
        if selection and selection.selection and selection.selection.rows:
            selected_idx = selection.selection.rows[0]
            if selected_idx < len(master_df):
                selected_row = master_df.iloc[selected_idx].to_dict()
                edit_stock_modal(selected_row)

    # --- Tab 2: 잔고 벌크 입력 ---
    with tab_bulk:
        st.subheader("텍스트 일괄 업데이트")
        st.info(
            "엑셀 또는 증권사 화면에서 복사한 잔고 텍스트를 붙여넣으세요. 파싱 후 [현재 잔고]에 일괄 반영할 수 있습니다.\n\n"
            "⚠️ **주의**: 일괄 반영 시 선택된 계좌의 **기존 종목 데이터는 모두 삭제되고 입력한 데이터로 완전히 교체(Overwrite)** 됩니다. "
            "(단, 기존과 동일한 티커의 최초 매수일은 유지됩니다.)"
        )

        from config import BUCKET_CONFIG

        # Build mapping: bucket_name → bucket_id (e.g., "1. 모멘텀" → 1)
        bucket_name_to_id = {v["name"]: k for k, v in BUCKET_CONFIG.items()}

        pasted_data = st.text_area("여기에 데이터 붙여넣기 (TSV)", height=200)

        if st.button("🔍 데이터 파싱 및 확인"):
            if pasted_data.strip():
                try:
                    raw_df = pd.read_csv(io.StringIO(pasted_data), sep="\t", header=None, dtype=str)
                    expected_cols = ["계좌", "환종", "버킷_텍스트", "티커", "종목명", "수량", "평균 매입가"]

                    if raw_df.shape[1] >= 7:
                        parsed_df = raw_df.iloc[:, :7].copy()
                        parsed_df.columns = expected_cols

                        # Data processing
                        parsed_df["수량"] = pd.to_numeric(
                            parsed_df["수량"].str.replace(r"[^0-9.-]", "", regex=True), errors="coerce"
                        ).fillna(0)
                        parsed_df["평균 매입가"] = pd.to_numeric(
                            parsed_df["평균 매입가"].str.replace(r"[^0-9.-]", "", regex=True), errors="coerce"
                        ).fillna(0)

                        # --- Strict Validation ---
                        errors = []

                        # 1. Account name validation
                        invalid_accounts = []
                        for acc_name in parsed_df["계좌"].unique():
                            acc_name_stripped = str(acc_name).strip()
                            if acc_name_stripped not in account_map:
                                invalid_accounts.append(acc_name_stripped)
                        if invalid_accounts:
                            valid_names = ", ".join(account_map.keys())
                            for name in invalid_accounts:
                                errors.append(f"🏦 계좌 '{name}'을(를) 찾을 수 없습니다.")
                            errors.append(f"   등록된 계좌: [{valid_names}]")

                        # 2. Bucket name validation
                        invalid_buckets = []
                        for bucket_text in parsed_df["버킷_텍스트"].unique():
                            bucket_text_stripped = str(bucket_text).strip()
                            if bucket_text_stripped not in bucket_name_to_id:
                                invalid_buckets.append(bucket_text_stripped)
                        if invalid_buckets:
                            valid_buckets = ", ".join(bucket_name_to_id.keys())
                            for name in invalid_buckets:
                                errors.append(f"🪣 버킷 '{name}'을(를) 찾을 수 없습니다.")
                            errors.append(f"   등록된 버킷: [{valid_buckets}]")

                        if errors:
                            st.warning("⚠️ 데이터 검증 실패! 아래 문제를 수정 후 다시 파싱해 주세요.")
                            for err in errors:
                                st.error(err)
                        else:
                            # Map bucket text to ID and account name to code
                            parsed_df["티커"] = parsed_df["티커"].apply(_normalize_ticker)
                            parsed_df["bucket"] = parsed_df["버킷_텍스트"].str.strip().map(bucket_name_to_id)
                            parsed_df["계좌코드"] = parsed_df["계좌"].str.strip().map(account_map)
                            st.session_state.bulk_parsed_df = parsed_df
                            st.success("✅ 파싱 및 검증 완료! 아래 결과를 확인하고 하단의 저장 버튼을 누르세요.")
                    else:
                        st.error("데이터 형식이 맞지 않습니다. 최소 7개 컬럼이 필요합니다.")
                except Exception as e:
                    st.error(f"파싱 중 오류 발생: {e}")

        if "bulk_parsed_df" in st.session_state:
            st.divider()
            st.dataframe(st.session_state.bulk_parsed_df)
            if st.button("🚀 위 결과를 [현재 잔고] 마스터에 일괄 반영하기", type="primary"):
                from utils.stock_meta_updater import fetch_stock_info

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

                    # Load existing to carry-over first_buy_date
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
                        # Auto-fetch name from library (ignore user-provided name)
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

                st.success(f"✅ 총 {success_count}개 계좌의 [현재 잔고]가 업데이트되었습니다.")
                del st.session_state.bulk_parsed_df

    # --- Tab 3: 원금 및 현금 관리 ---
    with tab_cash:
        st.subheader("계좌별 원금 및 현금 관리")
        st.info("이곳에서 입력한 투자 원금과 현금 잔고는 홈 화면의 '총 자산 요약' 및 '진짜 수익률' 계산에 반영됩니다.")

        with st.form("cash_manager_bulk_form"):
            st.write("각 계좌별 투자 원금과 보유 현금을 설정하세요.")

            # Dictionary to track input values per account
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
                        f"투자 원금 ({acc_name})",
                        value=int(current_principal),
                        min_value=0,
                        step=100000,
                        format="%d",
                        key=f"prin_{acc_id}",
                    )
                with c2:
                    new_cash = st.number_input(
                        f"보유 현금 ({acc_name})",
                        value=int(current_cash),
                        min_value=0,
                        step=100000,
                        format="%d",
                        key=f"cash_{acc_id}",
                    )

                input_values[acc_id] = {"holdings": current_holdings, "principal": new_principal, "cash": new_cash}
                st.divider()

            submitted = st.form_submit_button("전체 계좌 일괄 저장하기", type="primary", width="stretch")
            if submitted:
                success_count = 0
                for acc_id, data in input_values.items():
                    if save_portfolio_master(acc_id, data["holdings"], data["principal"], data["cash"]):
                        success_count += 1
                if success_count == len(input_values):
                    st.success(f"✅ 총 {success_count}개 계좌의 원금 및 현금 정보가 성공적으로 저장되었습니다!")
                else:
                    st.warning(f"⚠️ {success_count}/{len(input_values)}개 계좌만 저장되었습니다. 로그를 확인해 주세요.")


def build_transaction_page(page_cls):
    return page_cls(
        render_transaction_management_page,
        title="계좌 관리",
        icon="📝",
        url_path="transactions",
    )
