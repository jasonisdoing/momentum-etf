"""합성 오늘의 액션 조립 — 화면·슬랙 알람이 같은 결과를 쓰는 핵심 계산.

계좌·시세 조회는 하지 않는다 — 목표 보유 행과 슬리브별 이벤트를 인자로 받아
체결일 묶음을 만든다. 조립 규칙이 여기 한 곳에 있어야 화면과 알람이 갈리지 않는다.
"""

from __future__ import annotations

import math
from typing import Any

_WEEKDAYS_KO = ("월", "화", "수", "목", "금", "토", "일")


def _format_date_weekday(date: str) -> str:
    from datetime import date as date_cls

    try:
        parsed = date_cls.fromisoformat(date)
    except ValueError:
        return date
    return f"{date} ({_WEEKDAYS_KO[parsed.weekday()]})"


def _is_forecast(slot: dict[str, Any], event: dict[str, Any]) -> bool:
    """장중이라도 엔진이 체결일을 지정한 확정 주문은 예상이 아니다."""
    return bool(slot.get("live")) and not event.get("fill_date")


def _action_reasons(ticker: str, side: str, actions: dict[str, Any]) -> list[dict[str, str]]:
    """주문 원인을 추측하지 않고 엔진 이벤트와 배분 일정만 함께 표시한다."""
    reasons = []
    for slot in actions["slots"].values():
        event = "entries" if side == "buy" else "sells"
        events = [row for row in slot[event] if row["ticker"] == ticker]
        if events:
            signal = "진입" if side == "buy" else "청산"
            forecast = " 예상" if all(_is_forecast(slot, row) for row in events) else ""
            reasons.append({"code": "strategy_signal", "label": f"{slot['label']} {signal}{forecast}"})
        for trade in slot.get("engine_trades", []):
            if trade["ticker"] == ticker and trade["side"] == side:
                reasons.append(
                    {
                        "code": "engine_trade",
                        "label": f"{slot['label']} {trade['date']} 엔진 {trade['reason']} 반영",
                    }
                )
    if actions.get("sleeve_rebalance_today"):
        reasons.append({"code": "mix_rebalance", "label": "합성 월초 재배분 반영"})
    reasons.append({"code": "target_difference", "label": "목표 수량과 실제 보유 차이"})
    return reasons


def build_action_groups(
    holdings: list[dict[str, Any]],
    actions: dict[str, Any],
    next_trading_day: str | None,
    *,
    excess_holding_allowance: float,
    cash_balance: float,
    total_assets: float,
    fixed_asset_value: float,
    target_schedule: dict[str, dict[str, Any]],
    adjustment_day: str | None = None,
    adjustment_intraday: bool = False,
    currency: str = "KRW",
) -> list[dict[str, Any]]:
    """날짜별 엔진 목표 차이를 주문으로 만든다. 이후 주문은 앞선 목표 달성을 전제로 한다.

    ``adjustment_day`` 는 이벤트 없는 조정의 기준일(마감 전이면 오늘) — 엔진 예정 주문의
    체결일(다음 거래일 시가)과 다를 수 있다. ``adjustment_intraday`` 면 그 그룹 제목을
    '시가'가 아니라 '장중'으로 단다(오늘 시가는 이미 지났다)."""
    baseline_day = adjustment_day or next_trading_day
    # 다른 종목의 오늘 주문 때문에 미래 진입 종목을 오늘의 0주 목표로 먼저 청산하지 않는다.
    first_event: dict[str, str] = {}
    for slot in actions["slots"].values():
        for event in (*slot["entries"], *slot["sells"]):
            date = event.get("fill_date") or next_trading_day
            if date is not None:
                ticker = event["ticker"]
                first_event[ticker] = min(first_event.get(ticker, date), date)
    previous = {row["ticker"]: int(row.get("held_quantity") or 0) for row in holdings}
    projected_cash = cash_balance
    groups = []
    for index, (day, target) in enumerate(sorted(target_schedule.items())):
        stage_rows = []
        for source in holdings:
            if source.get("is_cash") or source.get("is_fixed_asset") or source.get("target_quantity") is None:
                continue
            ticker = source["ticker"]
            first_date = first_event.get(ticker, baseline_day)
            if first_date is not None and day < first_date:
                continue
            quantity = target["quantities"].get(ticker, 0)
            held = previous.get(ticker, 0)
            if baseline_day is not None and day < baseline_day and quantity != held:
                event_name = "entries" if quantity > held else "sells"
                # 오늘 매수 신호만 있는데 이미 더 보유했다고 조정 매도를 앞당기지 않는다.
                matching_signal = any(
                    event["ticker"] == ticker and (event.get("fill_date") or next_trading_day) == day
                    for slot in actions["slots"].values()
                    for event in slot[event_name]
                )
                if not matching_signal:
                    continue
            stage_rows.append(
                {
                    **source,
                    "held_quantity": held,
                    "target_quantity": quantity,
                    "trade_quantity": quantity - held,
                    "weight_pct": target["weights"].get(ticker, 0.0),
                    "is_sell_all": quantity == 0 and held > 0,
                }
            )
        stage_actions = {
            **actions,
            "sleeve_rebalance_today": bool(actions.get("sleeve_rebalance_today")) and index == 0,
            "slots": {
                key: {
                    **slot,
                    **{
                        event: [row for row in slot[event] if (row.get("fill_date") or next_trading_day) == day]
                        for event in ("entries", "sells")
                    },
                    "engine_trades": slot.get("engine_trades", []) if index == 0 else [],
                }
                for key, slot in actions["slots"].items()
            },
        }
        # 엔진이 남긴 현금만 보호한다. 내림으로 생긴 잔여 현금은 초과 보유에 사용할 수 있다.
        protected_cash = max(total_assets * (1 - sum(target["weights"].values()) / 100) - fixed_asset_value, 0)
        projected_cash = _apply_excess_allowance(
            stage_rows,
            stage_actions,
            allowance=excess_holding_allowance,
            cash_balance=projected_cash,
            protected_cash=protected_cash,
        )
        stage_groups = _build_action_group_stage(stage_rows, stage_actions, day, currency=currency)
        if projected_cash < -0.01 and stage_groups:
            # 체결가·수수료를 추정해 목표를 줄이지 않는다. 부족분은 주문과 함께 명시한다.
            stage_groups[0]["funding_warning"] = (
                f"표시된 매도 후에도 매수 자금 {abs(projected_cash):,.2f} {currency} 부족"
                " · 기준 가격 추정이며 실제 체결가·수수료는 별도입니다."
            )
        groups.extend(stage_groups)
        # 허용한 초과 보유는 팔린 것으로 가정하지 않고 다음 날짜에도 실제 예상 보유로 넘긴다.
        previous.update({row["ticker"]: row["held_quantity"] + row["trade_quantity"] for row in stage_rows})
    # 장중 조정 그룹 — 오늘 시가는 지났으니 '시가'가 아니라 '장중(지금 주문)'으로 단다.
    if adjustment_intraday and adjustment_day:
        for group in groups:
            if group["key"] == adjustment_day:
                group["title"] = f"{_format_date_weekday(adjustment_day)} 장중 — 지금 주문"
    # 날짜를 키에 포함하면 앞선 주문이 사라진 뒤에도 다음 주문의 키가 유지된다.
    for group in groups:
        for item in group["items"]:
            item["key"] = f"{item['key']}-{item['date']}"
    return groups


def _apply_excess_allowance(
    rows: list[dict[str, Any]],
    actions: dict[str, Any],
    *,
    allowance: float,
    cash_balance: float,
    protected_cash: float,
) -> float:
    """목표는 유지하고 계좌 전체 한도·매수 자금 안에서 조정 매도만 생략한다.

    티커 순으로 정수 초과분을 허용한다. 시세 순위로 허용 종목이 뒤집히지 않도록 한다.
    모든 부족분 매수와 필수 매도를 먼저 반영한 현금에서만 허용 예산을 꺼낸다.
    """
    prices = {}
    for row in rows:
        if not row["trade_quantity"]:
            continue
        price = row.get("price")
        if price is None or not math.isfinite(float(price)) or float(price) <= 0:
            raise ValueError(f"초과 보유·매수 자금 계산에 필요한 가격이 없습니다: {row['ticker']}")
        prices[row["ticker"]] = float(price)
    after_cash = cash_balance - math.fsum(row["trade_quantity"] * prices.get(row["ticker"], 0) for row in rows)
    remaining = min(allowance, max(after_cash - protected_cash, 0))
    for row in sorted(rows, key=lambda item: item["ticker"]):
        trade = row["trade_quantity"]
        if trade >= 0 or row["target_quantity"] <= 0:
            continue
        reasons = _action_reasons(row["ticker"], "sell", actions)
        if any(reason["code"] != "target_difference" for reason in reasons):
            continue
        price = prices[row["ticker"]]
        retained = min(-trade, math.floor(remaining / price))
        row["trade_quantity"] += retained
        row["retained_excess_quantity"] = retained
        remaining -= retained * price
        after_cash -= retained * price
    return after_cash


def _build_action_group_stage(
    holdings: list[dict[str, Any]],
    actions: dict[str, Any],
    next_trading_day: str | None,
    *,
    currency: str = "KRW",
) -> list[dict[str, Any]]:
    """오늘의 액션 — 체결일 묶음(매도 먼저, 같은 방향은 티커 순).

    화면과 슬랙 알람이 **이 결과를 그대로** 쓴다 — 조립을 한 곳에 두어 둘이 어긋나지
    않게 한다. 규칙:
      · 계좌 보유와 목표 주수의 차이를 **거르지 않고 전부** 낸다. 예전에는 「목표비중의
        10% 이상 차이만」 이라는 문턱(밴드)을 종목마다 따로 걸었는데, 백테스트에 없는
        규칙이라 화면과 백테스트가 갈라졌다. 게다가 목표 주수 배분은 12종목을 한 번에
        계산해 매수 합이 매도 합 + 현금을 넘지 않는데, 문턱이 큰 매수만 통과시키고
        작은 매도를 걸러 **살 돈이 없는 매수 지시**를 만들었다.
    슬리브가 어떤 전략인지는 보지 않는다 — 있는 액션만 읽는다.
    """
    slots: dict[str, dict[str, Any]] = actions["slots"]

    entry_tickers = {row["ticker"] for slot in slots.values() for row in slot["entries"]}
    live_entry_tickers = {
        row["ticker"] for slot in slots.values() for row in slot["entries"] if _is_forecast(slot, row)
    }
    sell_pending = [row["ticker"] for slot in slots.values() for row in slot["sells"]]

    # 장중 판정은 오늘 종가로 확정되기 전이라 **예상**이다 — 문구로 구분한다.
    # 장중을 쓰는지는 슬리브마다 다르므로 그 슬리브의 플래그를 본다.
    sell_reason: dict[str, str] = {}
    forecast_sell_tickers: set[str] = set()
    for slot in slots.values():
        for row in slot["sells"]:
            live_tag = " · 예상" if _is_forecast(slot, row) else ""
            # 수익률이 함께 오는 건 진입가를 아는 전략뿐이다(모멘텀 자격 상실은 사유만).
            suffix = f", {row['return_pct']:+.2f}%" if row.get("return_pct") is not None else ""
            sell_reason[row["ticker"]] = f"{row['reason']}{suffix}{live_tag}"
            if live_tag:
                forecast_sell_tickers.add(row["ticker"])

    def label(ticker: str, name: str, quantity: float | None) -> str:
        base = f"{name}({ticker})"
        return base if not quantity else f"{base} {abs(int(quantity)):,}주"

    row_by_ticker = {row["ticker"]: row for row in holdings}
    items: list[dict[str, Any]] = []
    for row in holdings:
        trade = row.get("trade_quantity")
        if not trade:
            continue
        ticker = row["ticker"]
        date = next_trading_day
        reason = sell_reason.get(ticker)
        weight = float(row.get("weight_pct") or 0)
        held = float(row.get("held_quantity") or 0) > 0
        sell_reason_applies = bool(reason) and trade < 0 and weight <= 0
        if row.get("is_sell_all"):
            title = "전량 매도"
        elif trade < 0:
            if sell_reason_applies:
                title = "매도 예정(예상)" if ticker in forecast_sell_tickers else "매도 예정"
            else:
                title = "목표 수량 조정 매도"
        elif held:
            title = "목표 수량 조정 매수"
        elif ticker in entry_tickers:
            title = "진입(예상)" if ticker in live_entry_tickers else "진입"
        else:
            title = "신규 매수"
        after = f" → 목표 {int(row['target_quantity']):,}주" if row.get("target_quantity") is not None else ""
        amount = _format_trade_amount(trade, row.get("price"), currency)
        amount_note = f" · {amount}" if amount else ""
        if row.get("retained_excess_quantity"):
            amount_note += f" · 목표 초과 {row['retained_excess_quantity']:,}주 허용"
        if sell_reason_applies:
            note = f"{after} ({reason}){amount_note}".strip()
        elif row.get("is_sell_all"):
            note = f"· 목표에 없는 보유 종목{amount_note}"
        else:
            note = f"{after} · {weight:.2f}%{amount_note}".strip()
        items.append(
            {
                "key": f"act-{ticker}",
                "ticker": ticker,
                "side": "buy" if trade > 0 else "sell",
                "title": title,
                "text": f"{label(ticker, row.get('name') or ticker, trade)} {note}".strip(),
                "date": date,
                # 알람 비교용 — 새 지시·수량 증가만 발송하고 감소(체결 반영)는 조용히 넘긴다.
                "quantity": abs(int(trade)),
            }
        )

    # 매도 예정인데 매매수량이 0인 경우(목표가 아직 그대로라 차이가 없음)도 알려야 한다.
    seen_keys = {item["key"] for item in items}
    for ticker in sell_pending:
        if f"act-{ticker}" in seen_keys:
            continue
        row = row_by_ticker.get(ticker)
        if not row or float(row.get("held_quantity") or 0) <= 0 or float(row.get("weight_pct") or 0) > 0:
            continue
        items.append(
            {
                "key": f"act-{ticker}",
                "ticker": ticker,
                "side": "sell",
                "title": "매도 예정",
                "text": (
                    f"{label(ticker, row.get('name') or ticker, row.get('held_quantity'))}"
                    f" ({sell_reason.get(ticker) or '이탈'})"
                    + (
                        f" · {amt}"
                        if (amt := _format_trade_amount(row.get("held_quantity"), row.get("price"), currency))
                        else ""
                    )
                ),
                "date": next_trading_day,
                "quantity": abs(int(float(row.get("held_quantity") or 0))),
            }
        )
        seen_keys.add(f"act-{ticker}")

    by_date: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        item["reasons"] = _action_reasons(item["ticker"], item["side"], actions)
        reason_text = " · ".join(reason["label"] for reason in item["reasons"])
        item["text"] += f" · 사유: {reason_text}"
        by_date.setdefault(item["date"] or "", []).append(item)
    groups = []
    for date in sorted(by_date):
        group_items = sorted(by_date[date], key=lambda x: (0 if x["side"] == "sell" else 1, x["ticker"]))
        title = f"{_format_date_weekday(date)} 시가" if date else "체결일 미정"
        groups.append({"key": date or "unscheduled", "title": title, "items": group_items})

    # 주중 이탈 **예상** 그룹 — 판정(오늘 종가) 확정 전의 미리보기. 같은 체결일(다음 거래일)
    # 시가지만 확정 지시와 섞이지 않게 별도 그룹으로 뒤에 둔다. 슬랙 알람은 이 그룹을 보내지
    # 않는다(장중 출렁일 때마다 알람이 나가면 노이즈 — notify 가 forecast 그룹을 거른다).
    # 이탈 예상 종목의 **매수** 지시는 유예한다 — "오늘 사서 내일 팔라"가 된다. 조용히 지우지
    # 않고 '(예상)' 그룹에 유예 사유를 남긴다(수량이 부족해 보여도 오늘은 사지 말라는 안내).
    forecast_rows = [(key, row) for key, slot in slots.items() for row in slot["exit_forecast"]]
    forecast_tickers = {row["ticker"] for _, row in forecast_rows}
    suspended_buys: list[dict[str, Any]] = []
    if forecast_tickers:
        for group in groups:
            kept = []
            for item in group["items"]:
                if item["side"] == "buy" and item["ticker"] in forecast_tickers:
                    suspended_buys.append(item)
                else:
                    kept.append(item)
            group["items"] = kept
        groups = [group for group in groups if group["items"]]

    confirmed_sells = {item["ticker"] for group in groups for item in group["items"] if item["side"] == "sell"}
    forecast_items = []
    forecast_slot_by_ticker: dict[str, str] = {}
    for key, row in forecast_rows:
        ticker = row["ticker"]
        held = row_by_ticker.get(ticker)
        if ticker in confirmed_sells or not held:
            continue
        # 파는 건 **그 슬리브 몫**뿐이다 — 예상 수량 = 보유 − 이탈 후 남을 목표(다른 슬리브 몫).
        # 계좌가 이미 그 몫을 팔아뒀으면 0 이 되어 자동으로 표시되지 않는다.
        held_qty = float(held.get("held_quantity") or 0)
        target_qty = float(held.get("target_quantity") or 0)
        weight_all = float(held.get("weight_pct") or 0)
        slot_weight = float(held.get(f"{key}_weight") or 0)
        remain_qty = round(target_qty * (weight_all - slot_weight) / weight_all) if weight_all > 0 else 0
        slot_qty = int(round(held_qty - remain_qty))
        if slot_qty <= 0:
            continue
        both = slot_weight > 0 and (weight_all - slot_weight) > 0
        reason = f"{row.get('reason')} · {slots[key]['label']} 몫" if both else str(row.get("reason"))
        forecast_slot_by_ticker[ticker] = key
        forecast_items.append(
            {
                "key": f"forecast-{ticker}",
                "ticker": ticker,
                "side": "sell",
                "title": "매도 예정(예상)",
                "text": (
                    f"{label(ticker, row.get('name') or ticker, slot_qty)} ({reason})"
                    + (f" · {amt}" if (amt := _format_trade_amount(slot_qty, held.get("price"), currency)) else "")
                ),
                "date": next_trading_day,
                # 알람 상태 비교용 — 예상도 '처음 등장할 때 1건' 발송되도록 실제 수량을 싣는다.
                "quantity": abs(slot_qty),
            }
        )
    # 매수 유예 안내는 매도 예상이 **없는** 종목에만 — 전량 매도 예상이 이미 '사지 말라'를
    # 내포하므로 같은 종목에 두 줄이 나오면 소음이다.
    forecast_sold = {item["ticker"] for item in forecast_items}
    forecast_slot_all = {row["ticker"]: key for key, row in forecast_rows}
    for item in suspended_buys:
        if item["ticker"] in forecast_sold:
            continue
        slot_label = slots[forecast_slot_all[item["ticker"]]]["label"]
        forecast_items.append(
            {
                **item,
                "key": f"suspend-{item['ticker']}",
                "title": "매수 유예(예상)",
                "text": f"{item['text']} — 내일 {slot_label} 몫 매도 예상이라 오늘은 사지 않음",
                "date": next_trading_day,
            }
        )
    if forecast_items and next_trading_day:
        groups.append(
            {
                "key": f"{next_trading_day}-forecast",
                "title": f"{_format_date_weekday(next_trading_day)} 시가 (예상 — 오늘 종가 확정 시)",
                "forecast": True,
                "items": sorted(forecast_items, key=lambda x: x["ticker"]),
            }
        )
    return groups


def _format_trade_amount(quantity: float | None, price: float | None, currency: str) -> str:
    """지시 금액 표기 — 원화는 'N억 1,234만원', 미국·호주는 현지 통화($ / A$)."""
    if not quantity or not price or float(price) <= 0:
        return ""
    amount = abs(float(quantity)) * float(price)
    code = str(currency or "KRW").strip().upper()
    if code == "KRW":
        if amount >= 1_0000_0000:
            uk = int(amount // 1_0000_0000)
            man = int(round((amount - uk * 1_0000_0000) / 1_0000))
            return f"{uk}억 {man:,}만원" if man else f"{uk}억원"
        if amount >= 1_0000:
            return f"{int(round(amount / 1_0000)):,}만원"
        return f"{amount:,.0f}원"
    symbol = {"USD": "$", "AUD": "A$"}.get(code, f"{code} ")
    return f"{symbol}{amount:,.0f}"
