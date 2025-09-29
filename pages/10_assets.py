import os
import sys
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# pkg_resources 워닝 억제
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

try:
    import pytz
except ImportError:
    pytz = None
try:
    from pykrx import stock as _stock
except Exception:
    _stock = None

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.account_registry import (
    get_account_file_settings,
    get_accounts_by_country,
    load_accounts,
    get_account_info,
)
from utils.data_loader import resolve_security_name
from utils.db_manager import (
    delete_trade_by_id,
    get_all_daily_equities,
    get_all_trades,
    get_available_snapshot_dates,
    get_portfolio_snapshot,
    infer_trade_from_state_change,
    save_daily_equity,
    save_trade,
    update_trade_by_id,
)
from utils.transaction_manager import (
    delete_transaction_by_id,
    get_all_transactions,
    save_transaction,
    update_transaction_by_id,
)


COUNTRY_CODE_MAP = {"kor": "한국", "aus": "호주", "coin": "가상화폐"}


def _display_feedback_messages(key_prefix: str):
    """
    세션 상태에서 피드백 메시지를 확인하고 토스트로 표시합니다.
    주로 다이얼로그가 닫힌 후 피드백을 주기 위해 사용됩니다.
    """
    keys_to_check = [
        f"buy_message_{key_prefix}",
        f"sell_message_{key_prefix}",
    ]
    for key in keys_to_check:
        if key in st.session_state:
            # 메시지를 즉시 pop하여 중복 표시를 방지합니다.
            message = st.session_state.pop(key)
            if isinstance(message, tuple) and len(message) == 2:
                msg_type, msg_text = message
                if msg_type == "success":
                    st.toast(msg_text, icon="✅")
                elif msg_type == "error":
                    st.toast(msg_text, icon="🚨")
                elif msg_type == "warning":
                    st.toast(msg_text, icon="⚠️")


def _prepare_account_entries(
    country_code: str, accounts: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for entry in accounts or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("is_active", True):
            entries.append(entry)
    if not entries:
        entries.append(
            {
                "account": None,
                "country": country_code,
                "display_name": COUNTRY_CODE_MAP.get(country_code, country_code.upper()),
            }
        )
    return entries


def _account_label(entry: Dict[str, Any]) -> str:
    label = entry.get("display_name") or entry.get("account")
    if not label:
        return "계좌"
    return str(label)


def _account_prefix(country_code: str, account_code: Optional[str]) -> str:
    return f"{country_code}_{account_code or 'default'}"


def render_transaction_tab(
    country_code: str,
    account_code: str,
    account_prefix: str,
    transaction_type: str,
    currency: str,
    amt_precision: int,
    qty_precision: int,
):
    """현금인출 또는 자본추가 탭의 UI를 렌더링합니다."""
    is_injection = transaction_type == "capital_injection"
    title = "자본추가" if is_injection else "현금인출"
    currency_str = f" ({currency})"

    all_txs = get_all_transactions(country_code, account_code, transaction_type)

    if not all_txs:
        st.info(f"{title} 내역이 없습니다.")
    else:
        df_txs = pd.DataFrame(all_txs)
        df_txs["선택"] = False

        cols_to_show = ["선택", "date", "amount", "note", "updated_at", "id"]
        df_display = df_txs.reindex(columns=cols_to_show).copy()

        df_display["date"] = pd.to_datetime(df_display["date"], errors="coerce").dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        df_display["updated_at"] = pd.to_datetime(
            df_display["updated_at"], errors="coerce"
        ).dt.strftime("%Y-%m-%d %H:%M:%S")

        edited_df = st.data_editor(
            df_display,
            key=f"{transaction_type}_editor_{account_prefix}",
            hide_index=True,
            width="stretch",
            column_config={
                "선택": st.column_config.CheckboxColumn("선택", required=True),
                "id": None,
                "date": st.column_config.TextColumn("일시", disabled=True),
                "updated_at": st.column_config.TextColumn("수정일시", disabled=True),
                "amount": st.column_config.NumberColumn(
                    "금액", format=f"%.{amt_precision}f" if amt_precision > 0 else "%d"
                ),
                "note": st.column_config.TextColumn("비고", width="large"),
            },
            disabled=["date", "updated_at"],
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("선택 항목 수정 저장", key=f"update_{transaction_type}_btn_{account_prefix}"):
                editor_state = st.session_state[f"{transaction_type}_editor_{account_prefix}"]
                edited_rows = editor_state.get("edited_rows", {})
                if not edited_rows:
                    st.warning("수정된 내용이 없습니다.")
                else:
                    with st.spinner(f"{len(edited_rows)}개 내역을 수정하는 중..."):
                        updated_count = 0
                        for row_index, changes in edited_rows.items():
                            tx_id = df_display.iloc[row_index]["id"]
                            update_data = {
                                k: v for k, v in changes.items() if k in ["amount", "note"]
                            }
                            if update_data and update_transaction_by_id(tx_id, update_data):
                                updated_count += 1
                        st.success(f"{updated_count}개 내역을 성공적으로 수정했습니다.")
                        st.rerun()
        with col2:
            if st.button(
                "선택 항목 삭제", key=f"delete_{transaction_type}_btn_{account_prefix}", type="primary"
            ):
                txs_to_delete = edited_df[edited_df["선택"]]
                if not txs_to_delete.empty:
                    with st.spinner(f"{len(txs_to_delete)}개 내역을 삭제하는 중..."):
                        deleted_count = 0
                        for tx_id in txs_to_delete["id"]:
                            if delete_transaction_by_id(tx_id):
                                deleted_count += 1
                        st.success(f"{deleted_count}개 내역을 성공적으로 삭제했습니다.")
                        st.rerun()
                else:
                    st.warning("삭제할 내역을 선택해주세요.")

    with st.expander(f"신규 {title} 등록"):
        with st.form(f"{transaction_type}_form_{account_prefix}", clear_on_submit=True):
            tx_date = st.date_input(
                "날짜", value="today", key=f"tx_date_{account_prefix}_{transaction_type}"
            )

            # Determine min_value and step for amount based on amt_precision
            amount_min_value = 0.0
            amount_step = 1.0  # Default step for floats

            if amt_precision == 0:
                amount_min_value = 0
                amount_step = 1

            tx_amount = st.number_input(
                f"{title} 금액{currency_str}",
                min_value=amount_min_value,
                step=amount_step,
                format=f"%.{amt_precision}f" if amt_precision > 0 else "%d",
            )
            tx_note = st.text_input("비고")
            tx_submitted = st.form_submit_button(f"{title} 내역 저장")

            if tx_submitted:
                if tx_amount > 0:
                    # Combine date and time (time is set to midnight)
                    combined_datetime = datetime(tx_date.year, tx_date.month, tx_date.day, 0, 0, 0)

                    tx_data = {
                        "country": country_code,
                        "account": account_code,
                        "date": combined_datetime,
                        "type": transaction_type,
                        "amount": float(tx_amount),
                        "note": tx_note,
                    }
                    if save_transaction(tx_data):
                        st.success(f"{title} 내역이 성공적으로 저장되었습니다.")
                    else:
                        st.error("저장에 실패했습니다. 콘솔 로그를 확인해주세요.")
                else:
                    st.warning("금액을 올바르게 입력해주세요.")
                st.rerun()


def render_assets_dashboard(
    country_code: str,
    account_entry: Dict[str, Any],
    prefetched_trading_days: Optional[List[pd.Timestamp]] = None,
):
    """지정된 계좌의 평가금액/거래 내역을 렌더링합니다."""
    account_code = account_entry.get("account")
    account_prefix = _account_prefix(country_code, account_code)

    if not account_code:
        st.info("활성 계좌가 없습니다. 계좌를 등록한 후 이용해주세요.")
        return

    try:
        account_settings = get_account_file_settings(account_code)
        account_info = get_account_info(account_code)
    except SystemExit as e:
        st.error(str(e))
        st.stop()

    currency = account_info.get("currency", "KRW")
    amt_precision = account_info.get("amt_precision", 0)
    qty_precision = account_info.get("qty_precision", 0)
    currency_str = f" ({currency})"

    _display_feedback_messages(account_prefix)

    sub_tab_equity_history, sub_tab_trades, sub_tab_withdrawal, sub_tab_injection = st.tabs(
        ["평가금액", "트레이드", "현금인출", "자본추가"]
    )

    with sub_tab_equity_history:
        initial_date = account_settings.get("initial_date") or (
            datetime.now() - pd.DateOffset(months=3)
        )
        start_date_str = initial_date.strftime("%Y-%m-%d")

        raw_dates = get_available_snapshot_dates(country_code, account=account_code)
        sorted_dates = sorted(set(raw_dates), reverse=True)

        end_dt_candidates = [pd.Timestamp.now()]
        if sorted_dates:
            try:
                latest_snapshot_dt = pd.to_datetime(sorted_dates[0])
                end_dt_candidates.append(latest_snapshot_dt)
            except (ValueError, TypeError):
                pass

        final_end_dt = max(end_dt_candidates)
        end_date_str = final_end_dt.strftime("%Y-%m-%d")

        with st.spinner("거래일 및 평가금액 데이터를 불러오는 중..."):
            if prefetched_trading_days is not None:
                all_trading_days = prefetched_trading_days
            else:
                # prefetched_trading_days가 없으면 직접 거래일을 조회합니다.
                from utils.data_loader import get_trading_days

                all_trading_days = get_trading_days(start_date_str, end_date_str, country_code)
            trading_day_set = set()
            if not all_trading_days:
                if country_code == "kor":
                    st.warning("거래일을 조회할 수 없습니다.")
            else:
                trading_day_set = {pd.to_datetime(day).normalize() for day in all_trading_days}

            start_dt_obj = pd.to_datetime(start_date_str).to_pydatetime()
            end_dt_obj = pd.to_datetime(end_date_str).to_pydatetime()
            existing_equities = get_all_daily_equities(
                country_code, account_code, start_dt_obj, end_dt_obj
            )
            equity_data_map = {pd.to_datetime(e["date"]).normalize(): e for e in existing_equities}

            db_day_set = set(equity_data_map.keys())
            combined_days = sorted(trading_day_set.union(db_day_set), reverse=True)
            all_trading_days = combined_days

            data_for_editor = []
            for trade_date in all_trading_days:
                existing_data = equity_data_map.get(trade_date, {})
                row = {
                    "date": trade_date,
                    "total_equity": existing_data.get("total_equity", 0.0),
                    "updated_at": existing_data.get("updated_at"),
                    "updated_by": existing_data.get("updated_by"),
                }
                if country_code == "aus":
                    is_data = existing_data.get("international_shares", {})
                    row["is_value"] = is_data.get("value", 0.0)
                    row["is_change_pct"] = is_data.get("change_pct", 0.0)
                data_for_editor.append(row)

            df_to_edit = pd.DataFrame(data_for_editor)

            column_config = {
                "date": st.column_config.DateColumn("일자", format="YYYY-MM-DD", disabled=True),
                "total_equity": st.column_config.NumberColumn(
                    f"총 평가금액{currency_str}",
                    format=f"%.{amt_precision}f" if amt_precision > 0 else "%d",
                    required=True,
                ),
                "updated_at": st.column_config.DatetimeColumn(
                    "변경일시", format="YYYY-MM-DD HH:mm:ss", disabled=True
                ),
                "updated_by": st.column_config.TextColumn("변경자", disabled=True),
            }
            if country_code == "aus":
                column_config["is_value"] = st.column_config.NumberColumn(
                    f"해외주식 평가액{currency_str}", format=f"%.{amt_precision}f"
                )
                column_config["is_change_pct"] = st.column_config.NumberColumn(
                    "해외주식 수익률(%)", format="%.2f", help="수익률(%)만 입력합니다. 예: 5.5"
                )

            # 컬럼 순서 정의: 'updated_at'을 가장 오른쪽으로 이동
            columns_to_display = ["date", "total_equity", "updated_by"]
            if country_code == "aus":
                columns_to_display.extend(["is_value", "is_change_pct"])
            columns_to_display.append("updated_at")

            st.info("총 평가금액을 수정한 후 아래 '저장하기' 버튼을 눌러주세요.")

            edited_df = st.data_editor(
                df_to_edit[columns_to_display],
                key=f"equity_editor_{account_prefix}",
                width="stretch",
                hide_index=True,
                column_config=column_config,
            )

            if st.button("평가금액 저장하기", key=f"save_all_equities_{account_prefix}"):
                with st.spinner("변경된 평가금액을 저장하는 중..."):
                    editor_state = st.session_state[f"equity_editor_{account_prefix}"]
                    edited_rows = editor_state.get("edited_rows", {})

                    saved_count = 0
                    for row_index, changes in edited_rows.items():
                        original_row = df_to_edit.iloc[row_index]
                        date_to_save = original_row["date"].to_pydatetime()
                        equity_to_save = changes.get("total_equity", original_row["total_equity"])
                        is_data_to_save = None
                        if country_code == "aus":
                            is_data_to_save = {
                                "value": changes.get("is_value", original_row.get("is_value")),
                                "change_pct": changes.get(
                                    "is_change_pct", original_row.get("is_change_pct")
                                ),
                            }

                        if save_daily_equity(
                            country_code,
                            account_code,
                            date_to_save,
                            equity_to_save,
                            is_data_to_save,
                            updated_by="사용자",
                        ):
                            saved_count += 1
                        if saved_count > 0:
                            st.success(f"{saved_count}개 날짜의 평가금액을 업데이트했습니다.")
                            st.rerun()
                        else:
                            st.info("변경된 내용이 없어 저장하지 않았습니다.")

    with sub_tab_trades:
        all_trades = get_all_trades(country_code, account_code)
        if not all_trades:
            st.info("거래 내역이 없습니다.")
        else:
            df_trades = pd.DataFrame(all_trades)
            if "name" in df_trades.columns:
                missing_name_mask = df_trades["name"].isna() | (
                    df_trades["name"].astype(str).str.strip() == ""
                )
                if missing_name_mask.any():
                    df_trades.loc[missing_name_mask, "name"] = df_trades.loc[
                        missing_name_mask, "ticker"
                    ].apply(lambda t: resolve_security_name(country_code, str(t)))
            if country_code == "coin" and "ticker" in df_trades.columns:
                unique_tickers = sorted({str(t).upper() for t in df_trades["ticker"].dropna()})
                options = ["ALL"] + unique_tickers
                selected = st.selectbox(
                    "티커 필터", options, key=f"coin_trades_filter_{account_prefix}"
                )
                if selected != "ALL":
                    df_trades = df_trades[df_trades["ticker"].str.upper() == selected]

            df_trades["선택"] = False

            cols_to_show = [
                "선택",
                "date",
                "action",
                "ticker",
                "name",
                "shares",
                "price",
                "note",
                "updated_at",
                "id",
            ]
            df_display = df_trades.reindex(columns=cols_to_show).copy()

            df_display["date"] = pd.to_datetime(df_display["date"], errors="coerce").dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            df_display["updated_at"] = (
                pd.to_datetime(df_display["updated_at"], errors="coerce").dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                if "updated_at" in df_display.columns
                else "-"
            )

            edited_df = st.data_editor(
                df_display,
                key=f"trades_editor_{account_prefix}",
                hide_index=True,
                width="stretch",
                column_config={
                    "선택": st.column_config.CheckboxColumn("선택", required=True),
                    "id": None,
                    "date": st.column_config.TextColumn("거래시간", disabled=True),
                    "updated_at": st.column_config.TextColumn("수정일시", disabled=True),
                    "action": st.column_config.TextColumn("종류", disabled=True),
                    "ticker": st.column_config.TextColumn("티커", disabled=True),
                    "name": st.column_config.TextColumn("종목명", width="medium", disabled=True),
                    "shares": st.column_config.NumberColumn(
                        "수량",
                        format=(
                            "%.8f"
                            if country_code == "coin"
                            else ("%.4f" if country_code == "aus" else "%.0f")
                        ),
                    ),
                    "price": st.column_config.NumberColumn(
                        "가격", format=f"%.{amt_precision}f" if amt_precision > 0 else "%d"
                    ),
                    "note": st.column_config.TextColumn("비고", width="large"),
                },
                disabled=["date", "action", "ticker", "name", "updated_at"],
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("선택 항목 수정 저장", key=f"update_trade_btn_{account_prefix}"):
                    editor_state = st.session_state[f"trades_editor_{account_prefix}"]
                    edited_rows = editor_state.get("edited_rows", {})
                    if not edited_rows:
                        st.warning("수정된 내용이 없습니다.")
                    else:
                        with st.spinner(f"{len(edited_rows)}개 거래를 수정하는 중..."):
                            updated_count = 0
                            for row_index, changes in edited_rows.items():
                                trade_id = df_display.iloc[row_index]["id"]
                                update_data = {
                                    k: v
                                    for k, v in changes.items()
                                    if k in ["shares", "price", "note"]
                                }
                                if update_data and update_trade_by_id(trade_id, update_data):
                                    updated_count += 1
                            st.success(f"{updated_count}개 거래를 성공적으로 수정했습니다.")
                            st.rerun()
            with col2:
                if st.button("선택 항목 삭제", key=f"delete_trade_btn_{account_prefix}", type="primary"):
                    trades_to_delete = edited_df[edited_df["선택"]]
                    if not trades_to_delete.empty:
                        with st.spinner(f"{len(trades_to_delete)}개 거래를 삭제하는 중..."):
                            deleted_count = 0
                            for trade_id in trades_to_delete["id"]:
                                if delete_trade_by_id(trade_id):
                                    deleted_count += 1
                            st.success(f"{deleted_count}개 거래를 성공적으로 삭제했습니다.")
                            st.rerun()
                    else:
                        st.warning("삭제할 거래를 선택해주세요.")

        if country_code != "coin":
            st.markdown("---")
            with st.expander("최종 잔액 입력"):
                with st.form(f"balance_input_form_{account_prefix}", clear_on_submit=True):
                    ticker_input = st.text_input("종목코드 (티커)")
                    shares_format_str = "%.4f" if country_code == "aus" else "%d"

                    # Determine min_value and step for shares based on country_code
                    if country_code == "kor":
                        shares_min_value = 0
                        shares_step = 1
                    elif country_code == "aus":
                        shares_min_value = 0.0
                        shares_step = 0.0001

                    final_shares = st.number_input(
                        "최종 보유 수량",
                        min_value=shares_min_value,
                        step=shares_step,
                        format=shares_format_str,
                    )

                    # amt_precision에 따라 min_value와 step의 타입을 동적으로 설정
                    if amt_precision == 0:
                        price_min_value = 0
                        price_step = 1
                    else:
                        price_min_value = 0.0
                        price_step = 1.0

                    final_avg_price = st.number_input(
                        f"최종 평균 단가{currency_str}",
                        min_value=price_min_value,
                        step=price_step,
                        format=f"%.{amt_precision}f" if amt_precision > 0 else "%d",
                    )
                    balance_submitted = st.form_submit_button("거래 생성/저장")

                    if balance_submitted:
                        message_key = f"buy_message_{account_prefix}"
                        ticker = ticker_input.strip().upper()

                        if not ticker:
                            st.session_state[message_key] = (
                                "error",
                                "종목코드를 입력해주세요.",
                            )
                            st.rerun()

                        # 1. 기존 보유 현황 조회
                        q_old, avg_old = 0.0, 0.0
                        snapshot = get_portfolio_snapshot(country_code, account=account_code)
                        if snapshot and snapshot.get("holdings"):
                            for h in snapshot["holdings"]:
                                if h.get("ticker") == ticker:
                                    q_old = float(h.get("shares", 0.0))
                                    avg_old = float(h.get("avg_cost", 0.0))
                                    break

                        # 2. 거래 추론
                        q_new, avg_new = float(final_shares), float(final_avg_price)
                        inferred_trade, message = infer_trade_from_state_change(
                            q_old, avg_old, q_new, avg_new
                        )

                        if message:
                            st.session_state[message_key] = ("warning", message)
                            st.rerun()

                        trade_data = None
                        if inferred_trade:
                            # 추론된 거래가 있는 경우
                            trade_data = inferred_trade
                        elif q_old == 0 and q_new > 0:
                            # 신규 매수
                            trade_data = {
                                "action": "BUY",
                                "shares": q_new,
                                "price": avg_new,
                            }
                        else:
                            st.session_state[message_key] = (
                                "error",
                                "거래를 추론할 수 없습니다. 입력값을 확인해주세요.",
                            )
                            st.rerun()

                        if trade_data:
                            trade_time = datetime.now()
                            if pytz:
                                try:
                                    korea_tz = pytz.timezone("Asia/Seoul")
                                    trade_time = datetime.now(korea_tz).replace(tzinfo=None)
                                except pytz.UnknownTimeZoneError:
                                    pass

                            etf_name = resolve_security_name(country_code, ticker)

                            db_payload = {
                                "country": country_code,
                                "account": account_code,
                                "date": trade_time,
                                "ticker": ticker,
                                "name": etf_name,
                                "action": trade_data["action"],
                                "shares": float(trade_data["shares"]),
                                "price": float(trade_data["price"]),
                                "note": "사용자 입력",
                            }

                            if save_trade(db_payload):
                                action_str = "매수" if trade_data["action"] == "BUY" else "매도"
                                st.session_state[message_key] = (
                                    "success",
                                    f"'{etf_name}' {action_str} 거래가 성공적으로 저장되었습니다.",
                                )
                            else:
                                st.session_state[message_key] = (
                                    "error",
                                    "거래 저장에 실패했습니다. 콘솔 로그를 확인해주세요.",
                                )
                        st.rerun()

            with st.expander("보유 종목 매도 (SELL)"):
                snapshot_dates = get_available_snapshot_dates(country_code, account=account_code)
                latest_date_str = snapshot_dates[0] if snapshot_dates else None
                holdings_data = []
                if latest_date_str:
                    snapshot = get_portfolio_snapshot(
                        country_code, date_str=latest_date_str, account=account_code
                    )
                    if snapshot and snapshot.get("holdings"):
                        holdings_data = [
                            h for h in snapshot.get("holdings", []) if h.get("shares", 0.0) > 0
                        ]

                if not holdings_data:
                    st.info("매도할 보유 종목이 없습니다.")
                else:
                    for item in holdings_data:
                        if not item.get("name") and item.get("ticker"):
                            item["name"] = resolve_security_name(
                                country_code, str(item.get("ticker"))
                            )

                    with st.form(f"sell_form_{account_prefix}"):
                        df_holdings = pd.DataFrame(holdings_data)
                        df_holdings["선택"] = False
                        df_display = df_holdings[["선택", "name", "ticker", "shares"]].copy()
                        df_display.rename(
                            columns={"name": "종목명", "ticker": "티커", "shares": "보유수량"},
                            inplace=True,
                        )

                        edited_sell_df = st.data_editor(
                            df_display,
                            hide_index=True,
                            width="stretch",
                            disabled=["종목명", "티커", "보유수량"],
                            column_config={
                                "선택": st.column_config.CheckboxColumn("선택", required=True),
                                "보유수량": st.column_config.NumberColumn(format="%.8f"),
                            },
                        )
                        sell_submitted = st.form_submit_button("선택 종목 매도")

                        if sell_submitted:
                            message_key = f"sell_message_{account_prefix}"
                            selected_rows = edited_sell_df[edited_sell_df["선택"]]

                            if selected_rows.empty:
                                st.session_state[message_key] = ("warning", "매도할 종목을 선택해주세요.")
                            else:
                                trade_time = datetime.now()
                                if pytz:
                                    try:
                                        korea_tz = pytz.timezone("Asia/Seoul")
                                        trade_time = datetime.now(korea_tz).replace(tzinfo=None)
                                    except pytz.UnknownTimeZoneError:
                                        pass

                                success_count = 0
                                original_indices = selected_rows.index
                                rows_to_sell = df_holdings.loc[original_indices]

                                for _, row in rows_to_sell.iterrows():
                                    trade_data = {
                                        "country": country_code,
                                        "account": account_code,
                                        "date": trade_time,
                                        "ticker": row["ticker"],
                                        "name": row["name"],
                                        "action": "SELL",
                                        "shares": row["shares"],
                                        "note": "",
                                    }
                                    if save_trade(trade_data):
                                        success_count += 1

                                if success_count == len(rows_to_sell):
                                    st.session_state[message_key] = (
                                        "success",
                                        f"{success_count}개 종목의 매도 거래가 성공적으로 저장되었습니다.",
                                    )
                                else:
                                    st.session_state[message_key] = ("error", "일부 거래 저장에 실패했습니다.")
                            st.rerun()

    with sub_tab_withdrawal:
        render_transaction_tab(
            country_code,
            account_code,
            account_prefix,
            "cash_withdrawal",
            currency,
            amt_precision,
            qty_precision,
        )

    with sub_tab_injection:
        render_transaction_tab(
            country_code,
            account_code,
            account_prefix,
            "capital_injection",
            currency,
            amt_precision,
            qty_precision,
        )


def main():
    """자산 관리 페이지를 렌더링합니다."""
    st.title("🗂️ 자산 관리 (Assets)")

    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');
            body {
                font-family: 'Noto Sans KR', sans-serif;
            }
            .block-container {
                max-width: 100%;
                padding-top: 1rem;
                padding-left: 2rem;
                padding-right: 2rem;
            }
        </style>
    """,
        unsafe_allow_html=True,
    )
    with st.spinner("계좌 정보 로딩 중..."):
        load_accounts(force_reload=False)
        account_map = {
            "kor": get_accounts_by_country("kor"),
            "aus": get_accounts_by_country("aus"),
            "coin": get_accounts_by_country("coin"),
        }

    tab_kor, tab_aus, tab_coin = st.tabs(["한국", "호주", "코인"])

    for country_code, tab in [("kor", tab_kor), ("aus", tab_aus), ("coin", tab_coin)]:
        with tab:
            account_entries = _prepare_account_entries(country_code, account_map.get(country_code))
            if len(account_entries) == 1 and account_entries[0].get("account") is None:
                render_assets_dashboard(country_code, account_entries[0])
            else:
                account_labels = [_account_label(entry) for entry in account_entries]
                account_tabs = st.tabs(account_labels)
                for account_tab, entry in zip(account_tabs, account_entries):
                    with account_tab:
                        render_assets_dashboard(country_code, entry)


if __name__ == "__main__":
    main()
