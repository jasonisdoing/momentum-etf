from __future__ import annotations

import subprocess
from io import StringIO
from typing import Any

import pandas as pd
import streamlit as st

from services.price_service import get_exchange_rates, get_realtime_snapshot
from utils.account_registry import load_account_configs
from utils.db_manager import get_db_connection
from utils.notebook_exporter import update_notebook_rank_cache
from utils.portfolio_io import load_portfolio_master, load_real_holdings_table
from utils.rankings import build_account_rankings, get_account_rank_defaults
from utils.stock_list_io import get_etfs


def _run_background(command: list[str], success_message: str) -> None:
    try:
        subprocess.Popen(command)
        st.success(success_message)
    except Exception as exc:
        st.error(f"⚠️ 실행 시작 오류: {exc}")


def _build_empty_rank_header() -> str:
    return "보유\t버킷\t티커\t종목명\t현재가\t점수\t일간(%)\t1주(%)\t2주(%)\t1달(%)\t3달(%)\t6달(%)\t12달(%)\t고점대비\tRSI\t지속"


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
            export_rank_df = export_rank_df.fillna("")
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
        hold_text = ""
        if df_hold is None or df_hold.empty:
            total_assets = cash_val
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
                df_cash.to_csv(buffer_hold, sep="\t", index=False, lineterminator="\n")
                hold_text = f"{hold_title}\n{buffer_hold.getvalue().rstrip()}"
            else:
                hold_text = f"{hold_title}\n(보유 자산 없음)"
        else:
            valuation_total = df_hold["평가금액(KRW)"].sum()
            total_assets = valuation_total + cash_val

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
                    "비중(%)": round((cash_val / total_assets) * 100.0, 2) if total_assets > 0 else 0.0,
                    "보유일": "-",
                }
                df_hold = pd.concat([df_hold, pd.DataFrame([cash_row])], ignore_index=True)
            else:
                df_hold["비중(%)"] = 0.0

            export_hold_df = df_hold.loc[:, column_order_holdings].copy()
            export_hold_df = export_hold_df.fillna("")
            buffer_hold = StringIO()
            export_hold_df.to_csv(buffer_hold, sep="\t", index=False, lineterminator="\n")
            hold_text = f"{hold_title}\n{buffer_hold.getvalue().rstrip()}"

        sections.append(f"{rank_text}\n\n{hold_text}")
        progress_bar.progress(index / total_accounts if total_accounts else 1.0)

    separator = "\n\n" + "=" * 50 + "\n\n"
    return separator.join(sections), warnings_list


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

    st.write("")
    if st.button("노트북LM 캐시 강제 갱신", width="stretch", key="btn_system_notebook_refresh"):
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0.0)

        def _status_cb(msg):
            status_placeholder.info(msg)

        if update_notebook_rank_cache(force=True, progress_bar=progress_bar, status_callback=_status_cb):
            progress_bar.progress(1.0)
            status_placeholder.empty()
            st.success("✅ 노트북LM용 정적 마크다운 파일(static/notebook_rank.md)을 갱신했습니다.")
        else:
            status_placeholder.empty()
            st.warning("⚠️ 이미 다른 프로세스에서 갱신 중입니다.")

    # 현재 접속 환경에 따라 주소 동적으로 결정
    host = st.context.headers.get("host", "etf.dojason.com")
    is_local = "localhost" in host or "127.0.0.1" in host
    base_url = f"http://{host}" if is_local else "https://etf.dojason.com"

    st.info("**노트북LM 연동 주소 (클릭 시 복사 가능):**")
    st.code(f"{base_url}/static/notebook_rank.md", language="text")
    st.caption("⚠️ 위 주소가 비어 보일 경우, 아래 **비상용 주소**를 사용하세요:")
    st.code(f"{base_url}/?notebook=true", language="text")


def render_gemini_page() -> None:
    st.subheader("🤖 구글 제미나이용 텍스트 생성 (TSV)")
    st.info("내 투자에 관한 조언을 받기 위해 제미나이에게 전달할 텍스트 데이터를 생성합니다.")

    if "system_manual_rank_extract_tsv" not in st.session_state:
        st.session_state["system_manual_rank_extract_tsv"] = ""
    if "system_manual_rank_extract_warnings" not in st.session_state:
        st.session_state["system_manual_rank_extract_warnings"] = []

    if st.button("전체 구글 제미나이용 텍스트 생성", width="stretch", key="btn_system_manual_rank_extract"):
        try:
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            progress_bar = progress_placeholder.progress(0.0)
            extract_text, warnings_list = _build_manual_rank_extract_tsv(
                progress_bar=progress_bar,
                status_placeholder=status_placeholder,
            )
            st.session_state["system_manual_rank_extract_tsv"] = extract_text
            st.session_state["system_manual_rank_extract_warnings"] = warnings_list
            progress_bar.progress(1.0)
            status_placeholder.empty()
            st.success("✅ 구글 제미나이용 텍스트(TSV) 생성을 완료했습니다.")
        except Exception as exc:
            st.error(f"⚠️ 추출 오류: {exc}")

    warnings_list_gemini = st.session_state.get("system_manual_rank_extract_warnings", [])
    if warnings_list_gemini:
        st.warning("\n".join(warnings_list_gemini))

    st.text_area(
        "제미나이용 결과 (TSV)",
        height=500,
        key="system_manual_rank_extract_tsv",
    )
