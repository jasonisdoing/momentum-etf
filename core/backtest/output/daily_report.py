from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from core.backtest.output.formatters import (
    BUCKET_NAMES,
    _format_date_kor,
    _format_quantity,
    _is_finite_number,
    _resolve_formatters,
)
from strategies.maps.constants import DECISION_MESSAGES
from utils.data_loader import get_exchange_rate_series
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
    price_overrides: dict[str, float] | None = None,
) -> list[list[str]]:
    entries = []
    tickers_order = []
    if "CASH" in result.ticker_timeseries:
        tickers_order.append("CASH")

    other_tickers = sorted(
        [str(t) for t in result.ticker_timeseries.keys() if str(t).upper() != "CASH"], key=lambda x: str(x).upper()
    )
    tickers_order.extend(other_tickers)

    for idx, ticker in enumerate(tickers_order, 1):
        ts = result.ticker_timeseries.get(ticker)
        if ts is None or not isinstance(ts, pd.DataFrame):
            continue
        if target_date not in ts.index:
            continue

        row = ts.loc[target_date]
        ticker_key = str(ticker).upper()
        meta = result.ticker_meta.get(ticker_key, {})

        price_val = row.get("price")
        shares_val = row.get("shares")
        avg_cost_val = row.get("avg_cost")

        price = float(price_val) if pd.notna(price_val) else 0.0
        if price_overrides and ticker_key in price_overrides:
            price = float(price_overrides[ticker_key])

        shares = float(shares_val) if pd.notna(shares_val) else 0.0
        pv = price * shares
        avg_cost = float(avg_cost_val) if pd.notna(avg_cost_val) else 0.0

        decision = str(row.get("decision", "")).upper()
        score = row.get("score")
        note = str(row.get("note", "") or "")

        is_cash = ticker_key == "CASH"
        if is_cash:
            price = 1.0
            shares = pv if pv else 1.0

        prev_row = prev_rows_cache.get(ticker_key)
        prev_price = (
            float(prev_row.get("price")) if (prev_row is not None and pd.notna(prev_row.get("price"))) else None
        )

        daily_ret = ((price / prev_price) - 1.0) * 100.0 if prev_price else 0.0

        pv_safe = pv if _is_finite_number(pv) else 0.0
        total_value_safe = total_value if _is_finite_number(total_value) and total_value > 0 else 0.0
        weight = (pv_safe / total_value_safe * 100.0) if total_value_safe > 0 else 0.0

        if ticker_key not in buy_date_map:
            buy_date_map[ticker_key] = None
            holding_days_map[ticker_key] = 0

        if not is_cash:
            if shares > 0:
                if buy_date_map[ticker_key] is None or decision.startswith("BUY"):
                    buy_date_map[ticker_key] = target_date
                    holding_days_map[ticker_key] = 1
                else:
                    holding_days_map[ticker_key] += 1
            else:
                buy_date_map[ticker_key] = None
                holding_days_map[ticker_key] = 0
        else:
            buy_date_map[ticker_key] = None
            holding_days_map[ticker_key] = 0

        holding_days_display = str(holding_days_map.get(ticker_key, 0))
        price_display = "1" if is_cash else (price_formatter(price) if _is_finite_number(price) else "-")
        shares_display = "1" if is_cash else _format_quantity(shares, qty_precision)

        if is_cash and abs(pv) < 0.01:
            pv = 0.0
        pv_display = money_formatter(pv)

        cost_basis = avg_cost * shares if _is_finite_number(avg_cost) and shares > 0 else 0.0
        eval_profit_value = 0.0 if is_cash else (pv - cost_basis)
        cumulative_profit_value = eval_profit_value

        evaluated_profit_display = money_formatter(eval_profit_value)
        evaluated_pct = (eval_profit_value / cost_basis * 100.0) if cost_basis > 0 else 0.0
        evaluated_pct_display = f"{evaluated_pct:+.1f}%" if cost_basis > 0 else "-"
        cumulative_pct_display = evaluated_pct_display

        score_display = f"{float(score):.1f}" if _is_finite_number(score) else "-"
        weight_display = f"{weight:.1f}%"
        if is_cash and total_value_safe > 0:
            cash_ratio = (total_cash / total_value_safe) if _is_finite_number(total_cash) else 0.0
            weight_display = f"{cash_ratio * 100.0:.1f}%"

        message = note
        if not message:
            message = DECISION_MESSAGES.get(decision, "")

        # decision_conf = DECISION_CONFIG.get(decision, {})
        # decision_order = decision_conf.get("order", 99)
        score_val = float(score) if _is_finite_number(score) else float("-inf")

        bucket_id = meta.get("bucket")
        bucket_display = "-"
        if bucket_id and bucket_id in BUCKET_NAMES:
            bucket_display = f"{bucket_id}. {BUCKET_NAMES[bucket_id]}"
        elif bucket_id:
            bucket_display = str(bucket_id)

        name_display = str(meta.get("name") or ticker_key)
        if is_cash:
            name_display = "현금"

        row_data = [
            "-",  # Index placeholder
            bucket_display,
            ticker_key,
            name_display,
            decision or "-",
            holding_days_display,
            price_display,
            f"{daily_ret:+.1f}%",
            shares_display,
            pv_display,
            evaluated_profit_display,
            evaluated_pct_display,
            money_formatter(cumulative_profit_value),
            cumulative_pct_display,
            weight_display,
            score_display,
            message,
        ]

        sort_group = 2  # Default: WAIT / others
        if is_cash:
            sort_group = 0
        elif decision in ("HOLD", "BUY", "BUY_REBALANCE"):
            sort_group = 1

        bucket_sort_val = int(bucket_id) if (bucket_id and str(bucket_id).isdigit()) else 99

        # Final Sort Key: Group -> Bucket (for Group 1) -> Score (desc) -> Ticker
        sort_key_tuple = (sort_group, bucket_sort_val if sort_group == 1 else 0, -score_val, ticker_key)
        entries.append((sort_key_tuple, row_data))

    entries.sort(key=lambda x: x[0])

    sorted_rows = []
    current_idx = 1
    for _, row_data in entries:
        if row_data[2] == "CASH":
            row_data[0] = "0"
            row_data[1] = "-"
        else:
            row_data[0] = str(current_idx)
            current_idx += 1
        sorted_rows.append(row_data)

    return sorted_rows


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
        "일간(%)",
        "수량",
        "금액",
        "평가손익",
        "평가(%)",
        "누적손익",
        "누적(%)",
        "비중",
        "점수",
        "문구",
    ]
    aligns = [
        "right",
        "left",
        "left",
        "left",
        "center",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "left",
    ]

    buy_date_map = {}
    holding_days_map = {}
    prev_rows_cache = {}

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
            buy_date_map=buy_date_map,
            holding_days_map=holding_days_map,
            prev_rows_cache=prev_rows_cache,
        )

        lines.extend(render_table_eaw(headers, table_rows, aligns))

        for ticker in result.ticker_timeseries:
            ts = result.ticker_timeseries[ticker]
            if isinstance(ts, pd.DataFrame) and target_date in ts.index:
                prev_rows_cache[str(ticker).upper()] = ts.loc[target_date].copy()

    return lines
