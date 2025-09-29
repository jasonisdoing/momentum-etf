"""Pipeline logic extracted from root signals module.

Currently only main() is migrated. generate_signal_report() will be migrated next,
so we import it from the root signals for now to keep behavior identical.
"""
from __future__ import annotations

from typing import Optional, Dict, Any, List


from signals import (
    generate_signal_report as _root_generate_signal_report,  # temporary import until migration
    SignalExecutionResult,
)
from logic.signals.formatting import (
    _get_header_money_formatter,
    _load_display_precision,
    _load_precision_all,
)
from utils.account_registry import get_account_info
from utils.db_manager import save_signal_report_to_db
from utils.report import format_kr_money, render_table_eaw
from utils.stock_list_io import get_etfs
from logic.momentum import DECISION_CONFIG


def main(
    account: str,
    date_str: Optional[str] = None,
) -> Optional[SignalExecutionResult]:
    """Run signal generation and return a structured result for notifications/UI."""
    if not account:
        raise ValueError("account is required for signal generation")

    account_info = get_account_info(account)
    if not account_info:
        raise ValueError(f"등록되지 않은 계좌입니다: {account}")

    country = str(account_info.get("country") or "").strip()
    if not country:
        raise ValueError(f"'{account}' 계좌에 국가 정보가 없습니다.")

    result = generate_signal_report(account, date_str)

    if not result:
        return None

    (
        header_line,
        headers,
        rows_sorted,
        report_base_date,
        slack_message_lines,
        summary_data,
    ) = result

    # Save report to DB for later UI retrieval
    try:
        save_signal_report_to_db(
            country,
            account,
            report_base_date.to_pydatetime(),
            (header_line, headers, rows_sorted),
            summary_data,
        )
    except Exception:
        pass

    # Console formatting (kept for parity with previous behavior)
    col_indices: Dict[str, int] = {}
    try:
        score_header_candidates = ["점수", "모멘텀점수", "MA스코어"]
        for h in score_header_candidates:
            if h in headers:
                col_indices["score"] = headers.index(h)
                break
        col_indices["day_ret"] = headers.index("일간수익률")
        col_indices["cum_ret"] = headers.index("누적수익률")
        col_indices["weight"] = headers.index("비중")
    except (ValueError, KeyError):
        pass

    display_rows: List[List[Any]] = []
    prec = _load_display_precision()
    p_daily = max(0, int(prec.get("daily_return_pct", 2)))
    p_cum = max(0, int(prec.get("cum_return_pct", 2)))
    p_w = max(0, int(prec.get("weight_pct", 2)))

    all_prec = _load_precision_all()
    cprec = (all_prec.get("country") or {}).get(country, {}) if isinstance(all_prec, dict) else {}
    curmap = (all_prec.get("currency") or {}) if isinstance(all_prec, dict) else {}
    stock_ccy = str(cprec.get("stock_currency", "KRW")) if isinstance(cprec, dict) else "KRW"
    qty_p = int(cprec.get("stock_qty_precision", 0)) if isinstance(cprec, dict) else 0
    if isinstance(cprec, dict) and ("stock_amt_precision" in cprec):
        amt_p = int(cprec.get("stock_amt_precision", 0))
    else:
        amt_p = int(((curmap.get(stock_ccy) or {}).get("precision", 0)))

    try:
        col_price = headers.index("현재가")
    except ValueError:
        col_price = None
    try:
        col_amount = headers.index("금액")
    except ValueError:
        col_amount = None

    for row in rows_sorted:
        display_row = list(row)

        idx = col_indices.get("score")
        if idx is not None:
            val = display_row[idx]
            if isinstance(val, (int, float)):
                display_row[idx] = f"{val:.1f}"
            elif val is None or not str(val).strip():
                display_row[idx] = "-"

        idx = col_indices.get("day_ret")
        if idx is not None:
            val = display_row[idx]
            if isinstance(val, (int, float)):
                display_row[idx] = ("{:+." + str(p_daily) + "f}%").format(val)
            else:
                display_row[idx] = "-"

        idx = col_indices.get("cum_ret")
        if idx is not None:
            val = display_row[idx]
            if isinstance(val, (int, float)):
                display_row[idx] = ("{:+." + str(p_cum) + "f}%").format(val)
            else:
                display_row[idx] = "-"

        idx = col_indices.get("weight")
        if idx is not None:
            val = display_row[idx]
            if isinstance(val, (int, float)):
                display_row[idx] = ("{:." + str(p_w) + "f}%").format(val)
            else:
                display_row[idx] = "-"

        idx = col_indices.get("shares")
        if idx is not None:
            val = display_row[idx]
            if isinstance(val, (int, float)):
                if qty_p > 0:
                    s = f"{float(val):.{qty_p}f}".rstrip("0").rstrip(".")
                    display_row[idx] = s if s != "" else "0"
                else:
                    display_row[idx] = f"{int(round(val)):,d}"
            else:
                display_row[idx] = val

        if col_price is not None and isinstance(display_row[col_price], (int, float)):
            fmt = ("{:, ." + str(amt_p) + "f}") if amt_p > 0 else "{:, .0f}"
            fmt = fmt.replace(" ", "")
            display_row[col_price] = fmt.format(float(display_row[col_price]))
        if col_amount is not None and isinstance(display_row[col_amount], (int, float)):
            fmt = ("{:, ." + str(amt_p) + "f}") if amt_p > 0 else "{:, .0f}"
            fmt = fmt.replace(" ", "")
            display_row[col_amount] = fmt.format(float(display_row[col_amount]))

        display_rows.append(display_row)

    aligns = [
        "right",  # #
        "right",  # 티커
        "left",  # 종목명
        "left",  # 카테고리
        "center",  # 상태
        "left",  # 매수일자
        "right",  # 보유일
        "right",  # 현재가
        "right",  # 일간수익률
        "right",  # 보유수량
        "right",  # 금액
        "right",  # 누적수익률
        "right",  # 비중
        "right",  # 고점대비
        "right",  # 점수
        "center",  # 지속
        "left",  # 문구
    ]

    render_table_eaw(headers, display_rows, aligns=aligns)

    summary_line_plain = None
    try:
        from utils.notification import build_summary_line_from_summary_data

        summary_line_plain = build_summary_line_from_summary_data(
            summary_data, _get_header_money_formatter(country), use_html=False, prefix=None
        )
    except Exception:
        # Fallback: KRW formatter
        total_equity = float(summary_data.get("total_equity", 0.0) or 0.0)
        summary_line_plain = f"금액: {format_kr_money(total_equity)}"

    print("\n" + (summary_line_plain or ""))

    cash_amount = float(summary_data.get("total_cash", 0.0) or 0.0)
    total_equity = float(summary_data.get("total_equity", 0.0) or 0.0)

    try:
        idx_ticker = headers.index("티커")
        idx_amount = headers.index("금액")
    except ValueError:
        idx_ticker = idx_amount = None

    breakdown_items: List[tuple[float, str]] = []
    if idx_ticker is not None and idx_amount is not None:
        for row in rows_sorted:
            amount = row[idx_amount]
            try:
                value = float(amount)
            except (TypeError, ValueError):
                continue
            if value <= 0:
                continue
            ticker = row[idx_ticker]
            breakdown_items.append((value, ticker))

    breakdown_items.sort(key=lambda x: x[0], reverse=True)

    if breakdown_items or cash_amount:
        print("보유 자산 구성:")
        ticker_name_map: Dict[str, str] = {}
        try:
            for item in get_etfs(country) or []:
                code = item.get("ticker")
                if code:
                    ticker_name_map[str(code)] = item.get("name", "")
        except Exception:
            ticker_name_map = {}

        for value, ticker in breakdown_items:
            name_lookup = ticker_name_map.get(ticker) or ticker
            display_name = (
                f"{ticker}({name_lookup})" if name_lookup and name_lookup != ticker else ticker
            )
            print(f"  - {display_name}: {format_kr_money(value)}")
        print(f"  - 현금: {format_kr_money(cash_amount)}")
        print(f"  = 합계: {format_kr_money(total_equity)}")

    return SignalExecutionResult(
        report_date=report_base_date.to_pydatetime(),
        summary_data=summary_data,
        header_line=header_line,
        detail_headers=headers,
        detail_rows=rows_sorted,
        detail_extra_lines=slack_message_lines,
        decision_config=DECISION_CONFIG,
    )


# Temporary passthrough for generate_signal_report until it is migrated
def generate_signal_report(account: str, date_str: Optional[str] = None, prefetched_data=None):
    return _root_generate_signal_report(
        account=account, date_str=date_str, prefetched_data=prefetched_data
    )


__all__ = ["main", "generate_signal_report"]
