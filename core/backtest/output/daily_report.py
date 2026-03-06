from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from core.backtest.output.formatters import (
    _format_date_kor,
    _format_quantity,
    _is_finite_number,
    _resolve_formatters,
)
from core.backtest.output.snapshot_rows import advance_snapshot_state, build_snapshot_rows, create_snapshot_build_state
from utils.data_loader import get_exchange_rate_series
from utils.formatters import format_trading_days
from utils.notification import build_summary_line_from_summary_data
from utils.report import format_kr_money, render_table_eaw

if TYPE_CHECKING:
    from core.backtest.domain import AccountBacktestResult


def _build_daily_table_rows(
    *,
    result: AccountBacktestResult,
    target_date: pd.Timestamp,
    total_value: float,
    total_cash: float,
    price_formatter,
    money_formatter,
    qty_precision: int,
    buy_date_map: dict[str, pd.Timestamp | None],
    holding_days_map: dict[str, int],
    prev_rows_cache: dict[str, pd.Series | None],
    prev_decisions_map: dict[str, str],
    price_overrides: dict[str, float] | None = None,
) -> list[list[str]]:
    snapshot_state = create_snapshot_build_state()
    snapshot_state.buy_date_map = buy_date_map
    snapshot_state.holding_days_map = holding_days_map
    snapshot_state.prev_rows_cache = prev_rows_cache
    snapshot_state.prev_decisions_map = prev_decisions_map

    snapshot_rows = build_snapshot_rows(
        result=result,
        target_date=target_date,
        total_value=total_value,
        total_cash=total_cash,
        state=snapshot_state,
        price_overrides=price_overrides,
    )

    table_rows: list[list[str]] = []
    for snapshot_row in snapshot_rows:
        is_cash = bool(snapshot_row["is_cash"])
        price = float(snapshot_row["price"])
        shares = float(snapshot_row["shares"])
        pv = float(snapshot_row["pv"])
        avg_cost = snapshot_row["avg_cost"]
        evaluation_profit = snapshot_row["evaluation_profit"]
        score = snapshot_row["score"]
        row_data = [
            snapshot_row["row_index"],
            snapshot_row["bucket_display"],
            snapshot_row["ticker"],
            snapshot_row["name"],
            snapshot_row["display_decision"],
            format_trading_days(int(snapshot_row["holding_days"])),
            "1" if is_cash else (price_formatter(price) if _is_finite_number(price) else "-"),
            "-" if avg_cost is None else price_formatter(float(avg_cost)),
            f"{float(snapshot_row['daily_pct']):+.1f}%",
            "-" if snapshot_row["evaluation_pct"] is None else f"{float(snapshot_row['evaluation_pct']):+.1f}%",
            "1" if is_cash else _format_quantity(shares, qty_precision),
            money_formatter(0.0 if is_cash and abs(pv) < 0.01 else pv),
            "-" if evaluation_profit is None else money_formatter(float(evaluation_profit)),
            f"{float(snapshot_row['weight']):.1f}%",
            f"{float(score):.1f}" if _is_finite_number(score) else "-",
            str(snapshot_row["message"] or ""),
        ]
        table_rows.append(row_data)

    return table_rows


def _generate_daily_report_lines(result: AccountBacktestResult, account_settings: dict[str, Any]) -> list[str]:
    currency, money_formatter, price_formatter, qty_precision, _ = _resolve_formatters(
        account_settings, result.account_id
    )
    portfolio_df = result.portfolio_timeseries
    lines = []

    headers = [
        "#",
        "버킷",
        "티커",
        "종목명",
        "상태",
        "보유일",
        "현재가",
        "평균단가",
        "일간(%)",
        "평가(%)",
        "수량",
        "금액",
        "평가손익",
        "비중",
        "점수",
        "문구",
    ]
    aligns = [
        "right",  # #
        "left",  # 버킷
        "left",  # 티커
        "left",  # 종목명
        "center",  # 상태
        "right",  # 보유일
        "right",  # 현재가
        "right",  # 평균단가
        "right",  # 일간(%)
        "right",  # 평가(%)
        "right",  # 수량
        "right",  # 금액
        "right",  # 평가손익
        "right",  # 비중
        "right",  # 점수
        "left",  # 문구
    ]

    snapshot_state = create_snapshot_build_state()

    fx_series = None
    if currency != "KRW":
        try:
            fx_series = get_exchange_rate_series(portfolio_df.index.min(), portfolio_df.index.max())
        except Exception:
            pass

    for target_date in portfolio_df.index:
        row = portfolio_df.loc[target_date]
        total_value = float(row.get("total_value", 0))
        total_cash = float(row.get("total_cash", 0))
        total_holdings = float(row.get("total_holdings", 0))
        daily_pl = float(row.get("daily_profit_loss", 0))
        daily_ret = float(row.get("daily_return_pct", 0))
        cum_ret = float(row.get("cumulative_return_pct", 0))
        held_count = int(row.get("held_count", 0))

        header_values = {
            "principal": float(result.initial_capital),
            "total_eq": total_value,
            "total_h": total_holdings,
            "total_c": total_cash,
            "d_pl": daily_pl,
            "c_pl": total_value - float(result.initial_capital),
        }

        header_formatter = money_formatter

        if currency != "KRW":
            try:
                rate = (
                    float(fx_series.asof(target_date))
                    if fx_series is not None and pd.notna(fx_series.asof(target_date))
                    else 1.0
                )
                if rate != 1.0 and not pd.isna(rate):
                    header_formatter = format_kr_money
                    header_values["principal"] = float(result.initial_capital_krw)
                    header_values["total_eq"] *= rate
                    header_values["total_h"] *= rate
                    header_values["total_c"] *= rate
                    header_values["d_pl"] *= rate
                    header_values["c_pl"] = header_values["total_eq"] - header_values["principal"]
            except Exception:
                pass

        summary_data = {
            "principal": header_values["principal"],
            "total_equity": header_values["total_eq"],
            "total_holdings_value": header_values["total_h"],
            "total_cash": header_values["total_c"],
            "daily_profit_loss": header_values["d_pl"],
            "daily_return_pct": daily_ret,
            "eval_profit_loss": 0,
            "eval_return_pct": 0,
            "cum_profit_loss": header_values["c_pl"],
            "cum_return_pct": cum_ret,
            "held_count": held_count,
            "bucket_topn": int(result.holdings_limit),
        }

        prefix = f"{_format_date_kor(target_date)} |"
        line_str = build_summary_line_from_summary_data(summary_data, header_formatter, use_html=False, prefix=prefix)
        lines.append("")
        lines.append(line_str)

        table_rows = _build_daily_table_rows(
            result=result,
            target_date=target_date,
            total_value=total_value,
            total_cash=total_cash,
            price_formatter=price_formatter,
            money_formatter=money_formatter,
            qty_precision=qty_precision,
            buy_date_map=snapshot_state.buy_date_map,
            holding_days_map=snapshot_state.holding_days_map,
            prev_rows_cache=snapshot_state.prev_rows_cache,
            prev_decisions_map=snapshot_state.prev_decisions_map,
        )

        lines.extend(render_table_eaw(headers, table_rows, aligns))
        advance_snapshot_state(result=result, target_date=target_date, state=snapshot_state)

    return lines
