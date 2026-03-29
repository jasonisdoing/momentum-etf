from __future__ import annotations

from io import StringIO
from typing import Any

import pandas as pd

from services.price_service import get_exchange_rates, get_realtime_snapshot
from utils.account_notes import load_account_note
from utils.account_registry import load_account_configs
from utils.db_manager import get_db_connection
from utils.portfolio_io import get_latest_daily_snapshot, load_portfolio_master, load_real_holdings_table
from utils.rankings import build_account_rankings, get_account_rank_defaults
from utils.stock_list_io import get_etfs


class _NoopProgressBar:
    def progress(self, _value: float) -> None:
        return None


class _NoopStatusPlaceholder:
    def info(self, _message: str) -> None:
        return None

    def empty(self) -> None:
        return None


def _build_empty_rank_header() -> str:
    return "보유\t버킷\t티커\t종목명\t현재가\t일간(%)\t추세\t고점\t1주(%)\t2주(%)\t1달(%)\t3달(%)\t6달(%)\t12달(%)\tRSI\t지속"


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
    status_placeholder: Any,
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


def build_manual_rank_extract_tsv(
    *,
    progress_bar: Any | None = None,
    status_placeholder: Any | None = None,
    target_account_id: str | None = None,
    memo_content: str = "",
) -> tuple[str, list[str]]:
    progress_bar = progress_bar or _NoopProgressBar()
    status_placeholder = status_placeholder or _NoopStatusPlaceholder()

    column_order_rank = [
        "보유",
        "버킷",
        "티커",
        "종목명",
        "현재가",
        "일간(%)",
        "추세",
        "고점",
        "1주(%)",
        "2주(%)",
        "1달(%)",
        "3달(%)",
        "6달(%)",
        "12달(%)",
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
        if df_rank.empty:
            rank_text = f"{rank_title}\n{_build_empty_rank_header()}"
        else:
            export_rank_df = df_rank.loc[:, column_order_rank].copy()
            export_rank_df = _format_summary_export_df(export_rank_df, country_code=country_code, kind="rank")
            buffer_rank = StringIO()
            export_rank_df.to_csv(buffer_rank, sep="\t", index=False, lineterminator="\n")
            rank_text = f"{rank_title}\n{buffer_rank.getvalue().rstrip()}"

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
                            "버킷": "5. 현금",
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
                    "버킷": "5. 현금",
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

        prev_snap = get_latest_daily_snapshot(account_id, before_today=True)
        prev_assets_krw = prev_snap.get("total_assets", 0.0) if prev_snap else 0.0
        daily_profit_krw = total_assets - prev_assets_krw if prev_assets_krw > 0 else 0.0
        daily_rt_pct = (daily_profit_krw / prev_assets_krw) * 100.0 if prev_assets_krw > 0 else None

        hold_pct = (hold_val / total_assets) * 100.0 if total_assets > 0 else 0.0
        cash_pct = (cash_val / total_assets) * 100.0 if total_assets > 0 else 0.0

        rt_str = f"{daily_rt_pct:+.2f}%" if daily_rt_pct is not None else "정보 없음"
        profit_str = f"{int(daily_profit_krw):+,}원" if daily_rt_pct is not None else "정보 없음"

        if country_code == "au":
            aud_info = rates.get("AUD", {})
            aud_rate = float(aud_info.get("rate") or 1.0)
            aud_change = float(aud_info.get("change_pct") or 0.0)

            total_assets_aud = total_assets / aud_rate
            hold_val_aud = hold_val / aud_rate
            cash_val_aud = cash_val / aud_rate

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
            summary_header = (
                f"[{account_name}] 요약\n"
                f"- 총 자산: {int(total_assets):,}원\n"
                f"- 보유액: {int(hold_val):,}원 ({hold_pct:.1f}%)\n"
                f"- 현금: {int(cash_val):,}원 ({cash_pct:.1f}%)\n"
                f"- 전일 대비 수익: {profit_str} ({rt_str})\n"
            )

        sections.append(f"{summary_header}\n{rank_text}\n\n{hold_text}")
        progress_bar.progress(index / total_accounts if total_accounts else 1.0)

    separator = "\n\n" + "=" * 50 + "\n\n"
    final_text = separator.join(sections)
    if memo_content.strip():
        final_text = f"{memo_content.strip()}\n\n" + final_text

    return final_text, warnings_list


def generate_ai_summary_payload(account_id: str) -> dict[str, Any]:
    account_id = str(account_id or "").strip().lower()
    if not account_id:
        raise ValueError("account_id가 필요합니다.")

    note_doc = load_account_note(account_id)
    memo_content = str((note_doc or {}).get("content") or "")
    text, warnings = build_manual_rank_extract_tsv(target_account_id=account_id, memo_content=memo_content)
    return {
        "account_id": account_id,
        "memo_content": memo_content,
        "text": text,
        "warnings": warnings,
    }
