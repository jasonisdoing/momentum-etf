"""실제 거래 퍼포먼스 로그 생성을 위한 헬퍼 모듈."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from utils.logger import get_app_logger
from utils.report import format_aud_money, format_kr_money, format_usd_money
from utils.stock_list_io import get_etfs

logger = get_app_logger()

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "data" / "results"


def _resolve_money_formatter(currency: str):
    currency_upper = (currency or "").strip().upper()
    if currency_upper in {"AUD", "AUS"}:
        return format_aud_money, "A$"
    if currency_upper in {"USD", "US"}:
        return format_usd_money, "$"
    return format_kr_money, "₩"


def _format_day_of_week(date: pd.Timestamp) -> str:
    weekdays = ["월", "화", "수", "목", "금", "토", "일"]
    return weekdays[date.weekday()] if 0 <= date.weekday() < 7 else ""


def _format_trade(detail: Dict[str, Any], money_formatter, name_map: Dict[str, str]) -> str:
    ticker = str(detail.get("ticker", "-"))
    action = str(detail.get("action", "-")).upper()
    name = name_map.get(ticker.upper(), "")
    shares = detail.get("shares")
    price = detail.get("price")
    amount = detail.get("amount")
    memo = detail.get("memo")

    base = f"{action} {ticker}"
    if name:
        base += f" ({name})"
    parts = [base]
    if isinstance(shares, (int, float)):
        parts.append(f"{shares:,.4f}주")
    if isinstance(price, (int, float, float)):
        parts.append(f"@ {money_formatter(price)}")
    if isinstance(amount, (int, float)):
        parts.append(f"= {money_formatter(amount)}")
    if memo:
        parts.append(f"({memo})")
    return " ".join(parts)


def _build_holdings_table(
    holdings: Iterable[Dict[str, Any]],
    money_formatter,
    total_value: float,
    name_map: Dict[str, str],
) -> List[str]:
    lines: List[str] = []
    holdings_list = list(holdings)
    if not holdings_list:
        return lines

    header = "  +----+----------+--------------------------+--------------+--------------+--------------+--------------+--------+"
    lines.append("  [보유 내역]")
    lines.append(header)
    lines.append("  |  # | 티커     | 종목명                    |   보유수량    |    평균단가    |    평가금액    |    평가손익    | 비중   |")
    lines.append(header)
    for idx, item in enumerate(sorted(holdings_list, key=lambda x: -float(x.get("value", 0.0))), 1):
        shares = float(item.get("shares", 0.0))
        avg_cost = float(item.get("avg_cost", 0.0))
        value = float(item.get("value", 0.0))
        profit = float(item.get("profit", 0.0))
        weight = float(item.get("weight_pct", 0.0))
        ticker = str(item.get("ticker", "-"))
        lines.append(
            "  | {idx:>2} | {ticker:<8} | {name:<24} | {shares:>12,.4f} | {avg_cost:>12,.2f} | {value:>12,.2f} | {profit:>12,.2f} | {weight:>5.1f}% |".format(
                idx=idx,
                ticker=ticker[:8],
                name=name_map.get(ticker.upper(), "-")[:24],
                shares=shares,
                avg_cost=avg_cost,
                value=value,
                profit=profit,
                weight=weight,
            )
        )
    lines.append(header)
    return lines


def build_performance_log_lines(
    account_id: str,
    performance_result: Dict[str, Any],
    account_settings: Dict[str, Any],
) -> List[str]:
    """실제 거래 기반 퍼포먼스 결과를 텍스트 라인으로 변환합니다."""

    if not performance_result:
        raise ValueError("performance_result가 비어 있습니다.")

    currency_code = str(performance_result.get("currency") or account_settings.get("country_code") or "kr").upper()
    money_formatter, currency_symbol = _resolve_money_formatter(currency_code)

    start_date = pd.to_datetime(performance_result.get("start_date")).date()
    end_date = pd.to_datetime(performance_result.get("end_date")).date()
    initial_capital = float(performance_result.get("initial_capital", 0.0))
    final_value = float(performance_result.get("final_value", performance_result.get("current_value", 0.0)))
    cumulative_return = (final_value / initial_capital - 1) * 100 if initial_capital > 0 else 0.0

    lines: List[str] = []
    lines.append(f"퍼포먼스 로그 생성: {pd.Timestamp.now().isoformat(timespec='seconds')}")
    lines.append("1. ========= 기본정보 ==========")
    lines.append(f"계정: {account_id.upper()} ({currency_code}) | 기간: {start_date} ~ {end_date}")
    lines.append(
        f"초기 자본: {money_formatter(initial_capital)} | 최종 자산: {money_formatter(final_value)} | 누적 수익률: {cumulative_return:+.2f}%"
    )
    lines.append("")
    lines.append("2. ========= 일자별 상세 ==========")

    daily_records = performance_result.get("daily_records") or []

    country_code_lower = str(account_settings.get("country_code") or performance_result.get("currency") or "kr").lower()
    try:
        universe_meta = {
            str(stock.get("ticker", "")).upper(): str(stock.get("name") or "") for stock in get_etfs(country_code_lower) if stock.get("ticker")
        }
    except Exception:
        universe_meta = {}

    for record in daily_records:
        for trade in record.get("trades") or []:
            raw = trade.get("raw") or {}
            ticker_upper = str(raw.get("ticker") or trade.get("ticker") or "").upper()
            name_raw = raw.get("name") or trade.get("name")
            if ticker_upper and name_raw:
                universe_meta.setdefault(ticker_upper, str(name_raw))

    for record in daily_records:
        record_date = pd.to_datetime(record.get("date"))
        day_label = record_date.strftime("%Y-%m-%d")
        dow = _format_day_of_week(record_date)
        total_value = float(record.get("total_value", 0.0))
        cash = float(record.get("cash", 0.0))
        holdings_value = float(record.get("holdings_value", total_value - cash))
        daily_pct = float(record.get("daily_return_pct", 0.0))
        cumulative_pct = float(record.get("cumulative_return_pct", 0.0))
        trade_count = int(record.get("trade_count", 0))

        lines.append(
            f"{day_label}({dow}) | 총자산: {money_formatter(total_value)} | 현금: {money_formatter(cash)} | 보유: {money_formatter(holdings_value)} | 일간: {daily_pct:+.2f}% | 누적: {cumulative_pct:+.2f}% | 거래: {trade_count}건"
        )

        trades = record.get("trades") or []
        if trades:
            lines.append("  [거래]")
            for detail in trades:
                lines.append(f"    - {_format_trade(detail, money_formatter, universe_meta)}")

        holdings = record.get("holdings") or []
        holdings_lines = _build_holdings_table(holdings, money_formatter, total_value, universe_meta)
        if holdings_lines:
            lines.extend(holdings_lines)
        else:
            lines.append("  [보유 내역] 없음")

        lines.append("")

    return lines


def dump_performance_log(
    account_id: str,
    performance_result: Dict[str, Any],
    account_settings: Dict[str, Any],
    *,
    results_dir: Path | str | None = None,
) -> Path:
    """실제 거래 기반 퍼포먼스 결과를 텍스트 로그로 저장합니다."""

    lines = build_performance_log_lines(account_id, performance_result, account_settings)

    base_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    account_dir = base_dir / account_id.lower()
    account_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"performance_{pd.Timestamp.now().strftime('%Y-%m-%d')}.log"
    path = account_dir / file_name
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    logger.info("퍼포먼스 로그를 '%s'에 저장했습니다.", path)
    return path


__all__ = ["build_performance_log_lines", "dump_performance_log"]
