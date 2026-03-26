from __future__ import annotations

import subprocess
from io import StringIO
from typing import Any

import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as components_html

from services.price_service import get_exchange_rates, get_realtime_snapshot
from utils.account_notes import load_account_note, save_account_note
from utils.account_registry import load_account_configs
from utils.db_manager import get_db_connection
from utils.portfolio_io import (
    get_latest_daily_snapshot,
    load_portfolio_master,
    load_real_holdings_table,
)
from utils.rankings import build_account_rankings, get_account_rank_defaults
from utils.stock_list_io import get_etfs
from utils.ui import format_relative_time

CURRENT_ACCOUNT_STATE_KEY = "current_account_id"


def _run_background(command: list[str], success_message: str) -> None:
    try:
        subprocess.Popen(command)
        st.success(success_message)
    except Exception as exc:
        st.error(f"⚠️ 실행 시작 오류: {exc}")


def _build_empty_rank_header() -> str:
    return "보유\t버킷\t티커\t종목명\t현재가\t점수\t일간(%)\t1주(%)\t2주(%)\t1달(%)\t3달(%)\t6달(%)\t12달(%)\t고점대비\tRSI\t지속"


def _format_note_saved_at(value: Any) -> str | None:
    if value is None:
        return None

    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts = ts.tz_convert("Asia/Seoul").tz_localize(None)

    ampm = "오전" if ts.hour < 12 else "오후"
    hour12 = ts.hour % 12 or 12
    absolute_text = f"{ts.year}년 {ts.month}월 {ts.day}일 {ampm} {hour12}:{ts.minute:02d}분"
    relative_text = format_relative_time(ts)
    if relative_text:
        return f"{absolute_text} {relative_text}"
    return absolute_text


def _render_summary_note_styles(*, readonly: bool = False) -> None:
    if readonly:
        st.markdown(
            """
            <style>
            textarea[aria-label="계좌 메모 (AI 분석 시 상단에 포함됨)"] {
                background: #f1f3f5;
                border: 1px solid #cfd4da;
                color: #111111;
                border-radius: 16px;
                padding: 18px 20px;
                box-shadow: none;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        """
        <style>
        textarea[aria-label="계좌 메모 (AI 분석 시 상단에 포함됨)"] {
            background: #fff1a8;
            border: 1px solid #d8b93f;
            color: #3a3522;
            border-radius: 16px;
            padding: 18px 20px;
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.32);
        }

        textarea[aria-label="계좌 메모 (AI 분석 시 상단에 포함됨)"]::placeholder {
            color: #9a8634;
            opacity: 1;
        }

        textarea[aria-label="계좌 메모 (AI 분석 시 상단에 포함됨)"]:focus {
            border-color: #d0ae2f;
            box-shadow: 0 0 0 1px #d0ae2f;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_summary_beforeunload_warning(*, enabled: bool) -> None:
    script = """
    <script>
    const enabled = %s;
    const handler = function(event) {
        event.preventDefault();
        event.returnValue = "";
        return "";
    };
    window.parent.onbeforeunload = enabled ? handler : null;
    </script>
    """ % ("true" if enabled else "false")
    components_html(script, height=0, width=0)


def _load_account_options() -> tuple[list[dict[str, Any]], list[str], dict[str, str]]:
    accounts = load_account_configs()
    account_ids = [acc["account_id"] for acc in accounts]
    account_label_map = {
        acc["account_id"]: f"{acc.get('name') or acc['account_id']} ({acc['account_id']})" for acc in accounts
    }
    return accounts, account_ids, account_label_map


def _resolve_target_account_id(account_ids: list[str]) -> str:
    current_id = str(st.session_state.get(CURRENT_ACCOUNT_STATE_KEY) or "").strip()
    if current_id not in account_ids:
        current_id = account_ids[0]
        st.session_state[CURRENT_ACCOUNT_STATE_KEY] = current_id

    try:
        if "account" in st.query_params:
            del st.query_params["account"]
    except Exception:
        pass

    return current_id


def _compute_readonly_note_height(content: str) -> int:
    line_count = max(len(str(content or "").splitlines()) or 1, 1)
    return max(250, min(1100, 24 + (line_count * 24)))


def _format_summary_price(value: Any, *, country_code: str) -> str:
    if value is None or value == "":
        return ""
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return str(value)

    if country_code == "au":
        return f"A${amount:,.2f}"
    if country_code == "kor":
        return f"{int(round(amount)):,}원"
    return str(value)


def _format_summary_percent(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{amount:.2f}%"


def _format_summary_krw(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{int(round(amount)):,}원"


def _format_summary_export_df(df: pd.DataFrame, *, country_code: str, kind: str) -> pd.DataFrame:
    formatted = df.copy()
    price_columns = ["현재가"]
    if kind == "holdings":
        price_columns.append("평균 매입가")

    percent_columns = [column for column in formatted.columns if "(%)" in column]
    krw_columns = [
        column for column in ("매입금액(KRW)", "평가금액(KRW)", "평가손익(KRW)") if column in formatted.columns
    ]

    for column in price_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value: _format_summary_price(value, country_code=country_code)
            )

    for column in percent_columns:
        formatted[column] = formatted[column].map(_format_summary_percent)

    for column in krw_columns:
        formatted[column] = formatted[column].map(_format_summary_krw)

    return formatted.fillna("")


def _collect_kor_realtime_snapshot(
    accounts: list[dict[str, Any]],
    *,
    status_placeholder: st.delta_generator.DeltaGenerator,
    warnings_list: list[str],
) -> dict[str, dict[str, float]]:
    kor_tickers: set[str] = set()
    for account in accounts:
        country_code = str(account.get("country_code") or "").strip().lower()
        if country_code != "kor":
            continue
        account_id = str(account["account_id"])
        for item in get_etfs(account_id):
            ticker = str(item.get("ticker") or "").strip().upper()
            if ticker:
                kor_tickers.add(ticker)

    if not kor_tickers:
        return {}

    status_placeholder.info(f"실시간 시세 조회 중: 한국 ETF {len(kor_tickers)}개")
    try:
        snapshot = get_realtime_snapshot("kor", sorted(kor_tickers))
    except Exception as exc:
        warnings_list.append(f"네이버 실시간 조회 실패로 캐시 기준으로 진행했습니다: {exc}")
        return {}

    if not snapshot:
        warnings_list.append("네이버 실시간 조회에 실패해 한국 계좌는 캐시 기준으로 진행했습니다.")
    return snapshot


def _load_holdings_map(
    accounts: list[dict[str, Any]],
    *,
    warnings_list: list[str],
) -> dict[str, set[str]]:
    db = get_db_connection()
    if db is None:
        warnings_list.append("MongoDB 실보유 조회에 실패해 보유 컬럼은 공백 기준으로 진행했습니다.")
        return {}

    try:
        doc = db.portfolio_master.find_one({"master_id": "GLOBAL"})
    except Exception as exc:
        warnings_list.append(f"MongoDB 실보유 조회에 실패해 보유 컬럼은 공백 기준으로 진행했습니다: {exc}")
        return {}

    account_docs = doc.get("accounts") if isinstance(doc, dict) else None
    if not isinstance(account_docs, list):
        return {}

    account_ids = {str(account["account_id"]) for account in accounts}
    holdings_map: dict[str, set[str]] = {account_id: set() for account_id in account_ids}
    for account_doc in account_docs:
        if not isinstance(account_doc, dict):
            continue
        account_id = str(account_doc.get("account_id") or "").strip().lower()
        if account_id not in holdings_map:
            continue
        holdings = account_doc.get("holdings")
        if not isinstance(holdings, list):
            continue
        holdings_map[account_id] = {
            str(item.get("ticker") or "").strip().upper()
            for item in holdings
            if isinstance(item, dict) and str(item.get("ticker") or "").strip()
        }
    return holdings_map


def _build_manual_rank_extract_tsv(
    *,
    progress_bar: st.delta_generator.DeltaGenerator,
    status_placeholder: st.delta_generator.DeltaGenerator,
    target_account_id: str | None = None,
    memo_content: str = "",
) -> tuple[str, list[str]]:
    column_order_rank = [
        "보유",
        "버킷",
        "티커",
        "종목명",
        "현재가",
        "점수",
        "일간(%)",
        "1주(%)",
        "2주(%)",
        "1달(%)",
        "3달(%)",
        "6달(%)",
        "12달(%)",
        "고점대비",
        "RSI",
        "지속",
    ]
    column_order_holdings = [
        "버킷",
        "티커",
        "종목명",
        "현재가",
        "수량",
        "평균 매입가",
        "매입금액(KRW)",
        "평가금액(KRW)",
        "평가손익(KRW)",
        "수익률(%)",
        "비중(%)",
        "보유일",
    ]

    accounts = load_account_configs()
    if target_account_id:
        accounts = [acc for acc in accounts if str(acc["account_id"]).lower() == target_account_id.lower()]

    warnings_list: list[str] = []
    rates = get_exchange_rates()
    kor_snapshot = _collect_kor_realtime_snapshot(
        accounts, status_placeholder=status_placeholder, warnings_list=warnings_list
    )
    holdings_map = _load_holdings_map(accounts, warnings_list=warnings_list)
    total_accounts = len(accounts)
    sections: list[str] = []

    for index, account in enumerate(accounts, start=1):
        account_id = str(account["account_id"])
        account_name = str(account.get("name") or account_id)
        country_code = str(account.get("country_code") or "").strip().lower()
        status_placeholder.info(f"데이터 추출 중: {account_name} ({index}/{total_accounts})")

        # 1. 순위 데이터 (Rankings)
        ma_type, ma_months = get_account_rank_defaults(account_id)
        account_snapshot = None
        if country_code == "kor":
            account_tickers = {
                str(item.get("ticker") or "").strip().upper()
                for item in get_etfs(account_id)
                if str(item.get("ticker") or "").strip()
            }
            account_snapshot = {ticker: kor_snapshot[ticker] for ticker in account_tickers if ticker in kor_snapshot}

        df_rank = build_account_rankings(
            account_id,
            ma_type=ma_type,
            ma_months=ma_months,
            realtime_snapshot_override=account_snapshot,
            held_tickers_override=holdings_map.get(account_id, set()),
        )

        rank_title = f"[{account_name}] 순위 - {ma_type} {ma_months}개월"
        rank_text = ""
        if df_rank.empty:
            rank_text = f"{rank_title}\n{_build_empty_rank_header()}"
        else:
            export_rank_df = df_rank.loc[:, column_order_rank].copy()
            export_rank_df = _format_summary_export_df(export_rank_df, country_code=country_code, kind="rank")
            buffer_rank = StringIO()
            export_rank_df.to_csv(buffer_rank, sep="\t", index=False, lineterminator="\n")
            rank_text = f"{rank_title}\n{buffer_rank.getvalue().rstrip()}"

        # 2. 보유 상세 데이터 (Holdings)
        df_hold = load_real_holdings_table(
            account_id,
            preloaded_exchange_rates=rates,
            preloaded_kor_realtime_snapshot=kor_snapshot,
        )
        master_data = load_portfolio_master(account_id)
        cash_val = master_data.get("cash_balance", 0.0) if master_data else 0.0
        hold_title = f"[{account_name}] 보유 상세"

        if df_hold is None or df_hold.empty:
            total_assets = cash_val
            hold_val = 0.0
            if total_assets > 0:
                df_cash = pd.DataFrame(
                    [
                        {
                            "버킷": "6. 현금",
                            "티커": "CASH",
                            "종목명": "현금",
                            "현재가": 1.0,
                            "수량": int(cash_val),
                            "평균 매입가": 1.0,
                            "매입금액(KRW)": int(cash_val),
                            "평가금액(KRW)": int(cash_val),
                            "평가손익(KRW)": 0,
                            "수익률(%)": 0.0,
                            "비중(%)": 100.0,
                            "보유일": "-",
                        }
                    ]
                )
                buffer_hold = StringIO()
                export_cash_df = _format_summary_export_df(df_cash, country_code=country_code, kind="holdings")
                export_cash_df.to_csv(buffer_hold, sep="\t", index=False, lineterminator="\n")
                hold_text = f"{hold_title}\n{buffer_hold.getvalue().rstrip()}"
            else:
                hold_text = f"{hold_title}\n(보유 자산 없음)"
        else:
            valuation_total = df_hold["평가금액(KRW)"].sum()
            total_assets = valuation_total + cash_val
            hold_val = valuation_total

            if total_assets > 0:
                df_hold["비중(%)"] = ((df_hold["평가금액(KRW)"] / total_assets) * 100.0).round(2)
                cash_row = {
                    "버킷": "6. 현금",
                    "티커": "CASH",
                    "종목명": "현금",
                    "현재가": 1.0,
                    "수량": int(cash_val),
                    "평균 매입가": 1.0,
                    "매입금액(KRW)": int(cash_val),
                    "평가금액(KRW)": int(cash_val),
                    "평가손익(KRW)": 0,
                    "수익률(%)": 0.0,
                    "비중(%)": round((cash_val / total_assets) * 100.0, 2),
                    "보유일": "-",
                }
                df_hold = pd.concat([df_hold, pd.DataFrame([cash_row])], ignore_index=True)
            else:
                df_hold["비중(%)"] = 0.0

            export_hold_df = df_hold.loc[:, column_order_holdings].copy()
            export_hold_df = _format_summary_export_df(export_hold_df, country_code=country_code, kind="holdings")
            buffer_hold = StringIO()
            export_hold_df.to_csv(buffer_hold, sep="\t", index=False, lineterminator="\n")
            hold_text = f"{hold_title}\n{buffer_hold.getvalue().rstrip()}"

        # 3. 요약 지표 (Summary Metrics)
        prev_snap = get_latest_daily_snapshot(account_id, before_today=True)
        prev_assets_krw = prev_snap.get("total_assets", 0.0) if prev_snap else 0.0
        daily_profit_krw = total_assets - prev_assets_krw if prev_assets_krw > 0 else 0.0
        daily_rt_pct = (daily_profit_krw / prev_assets_krw) * 100.0 if prev_assets_krw > 0 else None

        hold_pct = (hold_val / total_assets) * 100.0 if total_assets > 0 else 0.0
        cash_pct = (cash_val / total_assets) * 100.0 if total_assets > 0 else 0.0

        rt_str = f"{daily_rt_pct:+.2f}%" if daily_rt_pct is not None else "정보 없음"
        profit_str = f"{int(daily_profit_krw):+,}원" if daily_rt_pct is not None else "정보 없음"

        if country_code == "au":
            # 호주 계좌 특수 처리 (KRW/AUD 병기)
            aud_info = rates.get("AUD", {})
            aud_rate = float(aud_info.get("rate") or 1.0)
            aud_change = float(aud_info.get("change_pct") or 0.0)

            # AUD 기준 값 계산
            total_assets_aud = total_assets / aud_rate
            hold_val_aud = hold_val / aud_rate
            cash_val_aud = cash_val / aud_rate

            # 전일 AUD 기준 자산 추정 (간소화: 어제 환율 추정)
            prev_aud_rate = aud_rate / (1 + aud_change / 100.0)
            prev_assets_aud = prev_assets_krw / prev_aud_rate if prev_assets_krw > 0 else 0.0
            daily_profit_aud = total_assets_aud - prev_assets_aud if prev_assets_aud > 0 else 0.0

            aud_rt_str = f"{daily_rt_pct:+.2f}%" if daily_rt_pct is not None else "정보 없음"
            aud_profit_str = f"{daily_profit_aud:+,.2f} AUD" if daily_rt_pct is not None else "정보 없음"

            change_sign = "+" if aud_change > 0 else ""
            summary_header = (
                f"[{account_name}] 요약\n"
                f"- KRW 기준\n"
                f"  - 총 자산: {int(total_assets):,}원\n"
                f"  - 보유액: {int(hold_val):,}원 ({hold_pct:.1f}%)\n"
                f"  - 현금: {int(cash_val):,}원 ({cash_pct:.1f}%)\n"
                f"  - 전일 대비 수익: {profit_str} ({rt_str})\n"
                f"- AUD 기준\n"
                f"  - 총 자산: {total_assets_aud:,.2f} AUD\n"
                f"  - 보유액: {hold_val_aud:,.2f} AUD\n"
                f"  - 현금: {cash_val_aud:,.2f} AUD\n"
                f"  - 전일 대비 수익: {aud_profit_str} ({aud_rt_str})\n"
                f"- AUD/KRW: {aud_rate:,.2f}원({change_sign}{aud_change:.2f}%)\n"
            )
        else:
            # 일반(한국) 계좌
            summary_header = (
                f"[{account_name}] 요약\n"
                f"- 총 자산: {int(total_assets):,}원\n"
                f"- 보유액: {int(hold_val):,}원 ({hold_pct:.1f}%)\n"
                f"- 현금: {int(cash_val):,}원 ({cash_pct:.1f}%)\n"
                f"- 전일 대비 수익: {profit_str} ({rt_str})\n"
            )

        # 최종적으로 계좌 섹션 구성 (요약 -> 순위 -> 보유)
        sections.append(f"{summary_header}\n{rank_text}\n\n{hold_text}")
        progress_bar.progress(index / total_accounts if total_accounts else 1.0)

    separator = "\n\n" + "=" * 50 + "\n\n"
    final_text = separator.join(sections)
    if memo_content.strip():
        final_text = f"{memo_content.strip()}\n\n" + final_text

    return final_text, warnings_list


def render_system_page() -> None:
    st.subheader("🛠️ 시스템 정보")

    account_ids = [item["account_id"] for item in load_account_configs()]

    summary_rows = [
        {"구분": "계좌", "개수": len(account_ids), "대상": ", ".join(account_ids) if account_ids else "-"},
    ]
    st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

    schedule_rows = [
        {
            "작업": "종목 메타데이터 업데이트",
            "대상": "모든 계좌",
            "자동 주기": "매일 09:00 KST",
            "실행 명령": "python scripts/stock_meta_updater.py",
        },
        {
            "작업": "가격 캐시 업데이트",
            "대상": "모든 계좌",
            "자동 주기": "매시 정각 KST",
            "실행 명령": "python scripts/update_price_cache.py",
        },
        {
            "작업": "전체 자산 요약 알림",
            "대상": "전체 계좌 요약",
            "자동 주기": "매일 11:00, 18:00, 23:00, 06:00 KST",
            "실행 명령": "python scripts/slack_asset_summary.py",
        },
    ]
    st.dataframe(pd.DataFrame(schedule_rows), width="stretch", hide_index=True)
    st.caption("자동 주기는 현재 `.github/workflows` 기준입니다.")

    st.write("")
    st.subheader("🚀 전체 수동 실행")

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        if st.button("모든 메타데이터 업데이트", width="stretch", key="btn_system_meta_all"):
            _run_background(
                ["python", "scripts/stock_meta_updater.py"],
                "✅ 전체 메타데이터 업데이트를 시작했습니다.",
            )
    with row1_col2:
        if st.button("모든 가격 캐시 업데이트", width="stretch", key="btn_system_cache_all"):
            _run_background(
                ["python", "scripts/update_price_cache.py"],
                "✅ 전체 가격 캐시 업데이트를 시작했습니다.",
            )

    if st.button("전체 자산 요약 알림 전송", width="stretch", key="btn_system_asset_summary"):
        _run_background(
            ["python", "scripts/slack_asset_summary.py"],
            "✅ 전체 자산 요약 알림 전송을 시작했습니다.",
        )

    # (노트북LM 연동 주소 안내 제거됨 - 신규 전용 페이지로 대체 예정)
    st.caption("ℹ️ 위 주소는 데이터 확인용(브라우저)으로 편리하며, 향후 다양한 데이터 확장이 가능한 동적 경로입니다.")


def render_note_page() -> None:
    st.subheader("메모")

    _, account_ids, account_label_map = _load_account_options()
    blocked_warning_key = "note_account_change_blocked"
    reset_token_key = "note_account_selector_reset_token"
    current_id = _resolve_target_account_id(account_ids)
    selector_token = int(st.session_state.get(reset_token_key) or 0)
    selector_key = f"note_account_selector_{selector_token}"

    current_memo_key = f"note_memo_editor_{current_id}"
    current_saved_key = f"note_memo_saved_content_{current_id}"
    current_is_dirty = str(st.session_state.get(current_memo_key) or "") != str(
        st.session_state.get(current_saved_key) or ""
    )

    selected_id = st.selectbox(
        "계좌 선택",
        options=account_ids,
        index=account_ids.index(current_id),
        format_func=lambda acc_id: account_label_map.get(acc_id, acc_id),
        key=selector_key,
    )
    if selected_id != current_id:
        if current_is_dirty:
            st.session_state[blocked_warning_key] = True
            st.session_state[reset_token_key] = selector_token + 1
            st.rerun()
        st.session_state.pop(blocked_warning_key, None)
        st.session_state[CURRENT_ACCOUNT_STATE_KEY] = selected_id
        st.session_state[reset_token_key] = selector_token + 1
        st.rerun()

    target_account_id = selected_id
    memo_key = f"note_memo_editor_{target_account_id}"
    memo_loaded_key = f"note_memo_loaded_{target_account_id}"
    memo_saved_content_key = f"note_memo_saved_content_{target_account_id}"
    memo_updated_key = f"note_memo_updated_at_{target_account_id}"
    memo_error_key = f"note_memo_save_error_{target_account_id}"

    if (
        not st.session_state.get(memo_loaded_key)
        or memo_key not in st.session_state
        or memo_saved_content_key not in st.session_state
    ):
        note_doc = load_account_note(target_account_id)
        content = str((note_doc or {}).get("content") or "")
        st.session_state[memo_key] = content
        st.session_state[memo_saved_content_key] = content
        st.session_state[memo_updated_key] = (note_doc or {}).get("updated_at")
        st.session_state[memo_error_key] = ""
        st.session_state[memo_loaded_key] = True

    if st.session_state.pop(blocked_warning_key, False):
        st.warning("저장되지 않은 변경이 있어 계좌 이동이 취소되었습니다. 먼저 저장한 뒤 다시 이동하세요.")

    st.markdown("### 📝 계좌 메모 관리")
    _render_summary_note_styles()
    st.text_area(
        "계좌 메모 (AI 분석 시 상단에 포함됨)",
        height=250,
        placeholder="이 계좌에 대한 투자 전략이나 주의사항을 메모하세요. AI가 요약할 때 함께 참고합니다.",
        key=memo_key,
    )

    memo_editor_content = str(st.session_state.get(memo_key) or "")
    saved_content = str(st.session_state.get(memo_saved_content_key) or "")
    is_dirty = memo_editor_content != saved_content
    _render_summary_beforeunload_warning(enabled=is_dirty)

    save_col, _ = st.columns([1, 5])
    with save_col:
        if st.button("메모 저장", key=f"btn_note_save_{target_account_id}", type="primary", width="stretch"):
            try:
                updated_at = save_account_note(target_account_id, memo_editor_content)
            except Exception as exc:
                st.session_state[memo_error_key] = str(exc)
            else:
                st.session_state[memo_saved_content_key] = memo_editor_content
                st.session_state[memo_updated_key] = updated_at
                st.session_state[memo_error_key] = ""
                st.rerun()

    saved_at_text = _format_note_saved_at(st.session_state.get(memo_updated_key))
    save_error = str(st.session_state.get(memo_error_key) or "").strip()
    if save_error:
        st.caption(f"저장 오류: {save_error}")
    elif is_dirty:
        st.caption("저장되지 않은 변경이 있습니다.")
    elif saved_at_text:
        st.markdown(
            f"""
            <div style="color:#6b7280;font-size:0.95rem;">
                마지막 저장: {saved_at_text}
                <span style="display:inline-flex;align-items:center;gap:0.35rem;margin-left:0.35rem;">
                    <span style="width:0.9rem;height:0.9rem;border-radius:999px;background:#7fd36a;display:inline-block;border:1px solid #69bb55;"></span>
                    <span>저장됨</span>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.caption("아직 저장된 메모가 없습니다.")


def render_summary_for_ai_page() -> None:
    st.subheader("AI용 요약 (TSV)")
    st.info("내 투자에 관한 조언을 받기 위해 AI(제미나이 등)에게 전달할 텍스트 데이터를 생성합니다.")

    _, account_ids, account_label_map = _load_account_options()
    current_id = _resolve_target_account_id(account_ids)
    selector_key = "summary_account_selector"

    selected_id = st.selectbox(
        "계좌 선택",
        options=account_ids,
        index=account_ids.index(current_id),
        format_func=lambda acc_id: account_label_map.get(acc_id, acc_id),
        key=selector_key,
    )
    if selected_id != current_id:
        st.session_state[CURRENT_ACCOUNT_STATE_KEY] = selected_id
        st.rerun()

    target_account_id = selected_id
    note_doc = load_account_note(target_account_id)
    memo_content = str((note_doc or {}).get("content") or "")
    _render_summary_note_styles(readonly=True)
    st.text_area(
        "계좌 메모 (AI 분석 시 상단에 포함됨)",
        value=memo_content,
        height=_compute_readonly_note_height(memo_content),
        disabled=True,
        key=f"summary_note_readonly_{target_account_id}",
    )

    st.divider()

    if "system_manual_rank_extract_tsv" not in st.session_state:
        st.session_state["system_manual_rank_extract_tsv"] = ""
    if "system_manual_rank_extract_warnings" not in st.session_state:
        st.session_state["system_manual_rank_extract_warnings"] = []

    if st.button("AI용 요약 텍스트 생성", width="stretch", key="btn_system_manual_rank_extract"):
        try:
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            progress_bar = progress_placeholder.progress(0.0)
            extract_text, warnings_list = _build_manual_rank_extract_tsv(
                progress_bar=progress_bar,
                status_placeholder=status_placeholder,
                target_account_id=target_account_id,
                memo_content=memo_content,
            )
            st.session_state["system_manual_rank_extract_tsv"] = extract_text
            st.session_state["system_manual_rank_extract_warnings"] = warnings_list
            progress_bar.progress(1.0)
            status_placeholder.empty()
            st.success("✅ AI용 요약 텍스트(TSV) 생성을 완료했습니다.")
        except Exception as exc:
            st.error(f"⚠️ 추출 오류: {exc}")

    warnings_list_gemini = st.session_state.get("system_manual_rank_extract_warnings", [])
    if warnings_list_gemini:
        st.warning("\n".join(warnings_list_gemini))

    st.text_area(
        "AI용 요약 결과 (TSV)",
        height=250,  # 약 10줄
        key="system_manual_rank_extract_tsv",
    )
