"""합성전략 — SM(모멘텀 전략) + 신고가 돌파를 매월 50:50 리밸런싱으로 합친 백테스트.

`/strategy-mix` 열람 전용 화면의 백엔드. 설정은 이 화면이 갖지 않는다 —
**선택한 풀의 각 전략 화면 저장 설정**을 그대로 가져와 두 백테스트를 돌리고,
일별 곡선을 **매월 50:50 리밸런싱**으로 합성한 누적 시계열을 돌려준다.
화면은 신고가 화면과 같은 방식으로 이 일별 누적에서 연간·월간·일간 표를 만든다.

캐시는 두지 않는다 — 각 전략 화면의 백테스트와 같은 패턴(요청 시 계산)이다.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from utils.logger import get_app_logger

logger = get_app_logger()


def _sm_settings_map() -> dict[str, Any]:
    # SM 은 `{pool, settings_by_pool}` 래핑 구조라 풀 맵만 꺼낸다 (신고가는 평면 풀 맵).
    from utils.momentum_service import load_settings_map

    return dict(load_settings_map().get("settings_by_pool") or {})


def _nh_settings_map() -> dict[str, Any]:
    from utils.new_high_service import load_settings_map

    return load_settings_map()


def available_mix_pools() -> list[str]:
    """두 전략 모두 설정이 저장된 풀 — 기본 풀을 고를 때 쓴다."""
    return sorted(set(_sm_settings_map()) & set(_nh_settings_map()))


def _all_active_pools() -> list[str]:
    """활성 종목풀 전체 — 셀렉트에는 전부 보여주고, 설정 없는 풀은 선택 시 명시적 에러를 낸다."""
    from utils.settings_loader import list_available_ticker_types

    return list(list_available_ticker_types())


def _pool_options(pools: list[str]) -> list[dict[str, Any]]:
    """화면 셀렉트용 풀 정보 — 공용 formatPoolLabel 이 쓰는 필드(ticker_type·name·icon·order)."""
    from utils.settings_loader import get_ticker_type_settings

    options = []
    for pool in pools:
        try:
            settings = get_ticker_type_settings(pool) or {}
        except Exception:
            settings = {}
        options.append(
            {
                "ticker_type": pool,
                "name": str(settings.get("name") or pool),
                "icon": str(settings.get("icon") or ""),
                "order": settings.get("order"),
                # 적용 계좌 셀렉터가 같은 통화의 계좌만 보여주는 데 쓴다.
                "currency": str(settings.get("currency") or "").strip().upper(),
            }
        )
    options.sort(key=lambda o: (o["order"] is None, o["order"]))
    return options


# 백테스트 기간 선택지 — 신고가 화면과 동일한 목록 (상한 60개월 = 신고가 엔진의 최대).
MONTH_OPTIONS = (6, 12, 24, 36, 48, 60)

_CONFIG_COLLECTION = "system_config"
_SETTINGS_KEY = "strategy_mix_settings"
# 풀별로 보관하는 설정 — 다른 전략 화면과 같은 `{pool, settings_by_pool}` 구조다.
PER_POOL_SETTING_KEYS = ("account_id",)


def _db():
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")
    return db


def load_settings_map() -> dict[str, Any]:
    """풀별 저장 설정 — 화면이 풀 셀렉트를 바꿀 때 즉시 전환하는 데 쓴다."""
    doc = _db()[_CONFIG_COLLECTION].find_one({"_id": _SETTINGS_KEY}) or {}
    return dict(doc.get("settings_by_pool") or {})


def save_settings(pool: str, account_id: str | None) -> dict[str, Any]:
    """선택한 풀에 적용 계좌를 저장한다. 다른 풀의 저장분은 건드리지 않는다."""
    from utils.settings_loader import get_account_settings, list_available_accounts

    pool_norm, _, _ = _resolve_pool_and_settings(pool)
    account_norm = str(account_id or "").strip()
    if account_norm and account_norm not in list_available_accounts():
        raise ValueError(f"알 수 없는 계좌입니다: {account_id}")
    if account_norm:
        # 통화가 다르면 총자산을 그 풀의 종목 가격으로 나눌 수 없다 — 저장 자체를 막는다.
        from utils.settings_loader import get_ticker_type_settings

        pool_currency = str((get_ticker_type_settings(pool_norm) or {}).get("currency") or "").strip().upper()
        account_settings = get_account_settings(account_norm) or {}
        account_currency = str((account_settings.get("settings") or account_settings).get("currency") or "").strip().upper()
        if pool_currency and account_currency and pool_currency != account_currency:
            raise ValueError(f"통화가 다른 계좌입니다: 종목풀 {pool_currency} / 계좌 {account_currency}")
    per_pool = {"account_id": account_norm or None}
    _db()[_CONFIG_COLLECTION].update_one(
        {"_id": _SETTINGS_KEY},
        {"$set": {"pool": pool_norm, f"settings_by_pool.{pool_norm}": per_pool}},
        upsert=True,
    )
    return {"pool": pool_norm, **per_pool}


def mix_meta() -> dict[str, Any]:
    """화면 초기용 — 풀 셀렉트 목록과 기본 풀만 반환한다 (백테스트 계산 없음)."""
    all_pools = _all_active_pools()
    ready = available_mix_pools()
    from utils.momentum_service import load_settings_map as sm_map

    sm_current = str(sm_map().get("pool") or "").strip().lower()
    if sm_current in ready:
        default = sm_current
    elif ready:
        default = [o["ticker_type"] for o in _pool_options(ready)][0]
    else:
        default = all_pools[0] if all_pools else ""
    from utils.settings_loader import get_account_settings, list_available_accounts

    accounts = []
    for account_id in list_available_accounts():
        try:
            settings = get_account_settings(account_id) or {}
        except Exception:
            settings = {}
        inner = settings.get("settings") or settings
        accounts.append(
            {
                "account_id": account_id,
                "name": str(inner.get("name") or account_id),
                "icon": str(inner.get("icon") or ""),
                "order": inner.get("order"),
                # 통화가 다른 계좌는 목표 금액·주수를 낼 수 없다 — 화면이 풀별로 걸러 쓴다.
                "currency": str(inner.get("currency") or "").strip().upper(),
            }
        )
    accounts.sort(key=lambda item: (item["order"] is None, item["order"]))
    return {
        "pool": default,
        "pool_options": _pool_options(all_pools),
        "month_options": list(MONTH_OPTIONS),
        "accounts": accounts,
        # 풀별 저장 설정 — 화면이 풀을 바꿀 때 즉시 전환한다.
        "settings_by_pool": load_settings_map(),
    }


def _resolve_pool_and_settings(pool: str | None) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """풀을 확정하고 그 풀의 SM·신고가 저장 설정을 검증해 돌려준다 (백테스트·운영 공용)."""
    from utils.momentum_service import validate_settings as sm_validate
    from utils.new_high_service import validate_settings as nh_validate

    all_pools = _all_active_pools()
    ready = available_mix_pools()
    if not ready:
        raise RuntimeError("두 전략 모두 설정이 저장된 종목풀이 없습니다 — 각 전략 화면에서 먼저 저장하세요.")
    pool_norm = str(pool or "").strip().lower()
    if not pool_norm:
        # 기본 풀 — SM 화면이 현재 보고 있는 풀 (양쪽 설정이 없으면 번호가 가장 빠른 준비된 풀).
        from utils.momentum_service import load_settings_map as sm_map

        sm_current = str(sm_map().get("pool") or "").strip().lower()
        ordered_ready = [o["ticker_type"] for o in _pool_options(ready)]
        pool_norm = sm_current if sm_current in ready else ordered_ready[0]
    if pool_norm not in all_pools:
        raise ValueError(f"알 수 없는 종목풀입니다: {pool_norm}")

    sm_saved = _sm_settings_map().get(pool_norm)
    nh_saved = _nh_settings_map().get(pool_norm)
    missing = [name for name, saved in (("모멘텀 전략", sm_saved), ("신고가 돌파", nh_saved)) if not saved]
    if missing:
        raise RuntimeError(
            f"'{pool_norm}' 풀에 {' · '.join(missing)} 설정이 저장돼 있지 않습니다 — 해당 전략 화면에서 먼저 저장하세요."
        )
    return pool_norm, sm_validate({**sm_saved, "pool": pool_norm}), nh_validate({**nh_saved, "pool": pool_norm})


def _load_account_state(account_id: str) -> dict[str, Any]:
    """적용 계좌의 실제 보유 수량·평단·현금 — portfolio_master 가 단일 소스다."""
    from utils.portfolio_io import load_portfolio_master

    master = load_portfolio_master(account_id) or {}
    holdings = {}
    for row in master.get("holdings") or []:
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker or ticker in {"IS", "__CASH__"}:
            continue
        try:
            quantity = float(row.get("quantity") or 0)
        except (TypeError, ValueError):
            quantity = 0.0
        if quantity <= 0:
            continue
        holdings[ticker] = {
            "quantity": quantity,
            "name": str(row.get("name") or ticker),
            "average_buy_price": float(row.get("average_buy_price") or 0) or None,
        }
    try:
        cash = float(master.get("cash_balance") or 0)
    except (TypeError, ValueError):
        cash = 0.0
    return {"account_id": account_id, "holdings": holdings, "cash_balance": cash}


def mix_positions(pool: str | None = None, as_of: str | None = None) -> dict[str, Any]:
    """오늘 기준 합성 운영 상태 — 보유 목록(목표 비중)·현금 비중·오늘의 액션.

    두 전략 화면이 계산하는 것(SM compute_picks · 신고가 current_positions)을 합쳐서
    보여줄 뿐, 새 판정 로직은 없다. 비중은 슬리브 50% ÷ 슬롯 수(빈 슬롯 = 현금).
    겹치는 종목도 합치지 않고 슬리브별 행으로 각각 둔다 — 슬롯 하나가 행 하나다.
    ``as_of`` 를 주면 두 전략 모두 그 날짜의 상태를 재현한다 (실시간 없음).
    """
    import pandas as pd

    from utils.momentum_service import compute_picks
    from utils.new_high_backtest import current_positions

    pool_norm, sm_settings, nh_settings = _resolve_pool_and_settings(pool)

    sm = compute_picks(sm_settings, as_of=as_of)
    nh = current_positions(nh_settings, as_of=as_of)
    sm_top_n = int(sm_settings["top_n"])
    nh_top_n = int(nh_settings["top_n"])
    weight_sm = 50.0 / sm_top_n
    weight_nh = 50.0 / nh_top_n

    # SM 목표 포트폴리오 — 확정된 선정(다음 교체분)을 기준으로 보여준다. 이 화면은
    # '무엇을 보유해야 하는지'를 보는 곳이고, 신고가 슬리브도 진입 예정이 슬롯을
    # 채우는 방식이라 두 슬리브의 기준을 맞춘 것이다. 매도분은 오늘의 액션에만 남는다.
    sm_selected = [
        row
        for row in sm["rows"]
        if not row.get("is_reserve") and not row.get("is_expected_only") and (row.get("rank") or 999) <= sm_top_n
    ]
    sm_current = list(sm.get("holdings") or [])
    sm_current_tickers = {str(row["ticker"]).strip() for row in sm_current}
    sm_selected_tickers = {str(row["ticker"]).strip() for row in sm_selected}
    # 교체일에 새로 사는 종목 / 빠지는 종목 — 오늘의 액션이 쓴다.
    sm_rebalance_buys = [row for row in sm_selected if str(row["ticker"]).strip() not in sm_current_tickers]
    sm_rebalance_sells = [row for row in sm_current if str(row["ticker"]).strip() not in sm_selected_tickers]
    # 주중 매도 예정 — 자격 상실 판정(이미 보유 중인 종목만 해당).
    sm_sell_pending = [row for row in sm_selected if row.get("is_exit_pending")]
    sm_holdings = sm_selected
    nh_holdings = list(nh.get("holdings") or [])

    # 신고가 빈 슬롯을 채울 진입 예정 — 다음 시가에 사므로 목표 포트폴리오에 포함한다.
    nh_free = max(nh_top_n - len(nh_holdings), 0)
    nh_planned = list(nh.get("planned_entries") or [])[:nh_free]

    # 목표 포트폴리오는 **종목 단위**다. 두 슬리브가 같은 종목을 담으면 비중을 합쳐
    # 한 행으로 둔다 — 계좌에는 그 종목이 하나뿐이라 슬리브별로 나누면 보유 수량이
    # 두 번 세어지고 매매 지시가 반대로 나온다.
    holdings: list[dict[str, Any]] = []
    by_ticker: dict[str, dict[str, Any]] = {}

    def add_target(ticker: str, name: str, source: str, weight: float, price: Any, change_pct: Any, status: str) -> None:
        ticker = str(ticker).strip()
        row = by_ticker.get(ticker)
        if row is None:
            row = {
                "ticker": ticker,
                "name": name or ticker,
                "sources": [],
                "weight_pct": 0.0,
                "price": price,
                "change_pct": change_pct,
                "sm_status": None,
                "nh_status": None,
            }
            by_ticker[ticker] = row
            holdings.append(row)
        if source not in row["sources"]:
            row["sources"].append(source)
        row["weight_pct"] += weight
        if row.get("price") is None:
            row["price"] = price
        if row.get("change_pct") is None:
            row["change_pct"] = change_pct
        row[f"{source}_status"] = status

    for row in sm_selected:
        ticker = str(row["ticker"]).strip()
        if ticker in sm_current_tickers:
            streak = row.get("streak_weeks")
            status = f"유지 ({streak}주째)" if streak else "유지"
        else:
            status = f"매수 예정 ({sm.get('rebalance_date')} 시가)"
        if row.get("is_exit_pending"):
            status += " · 매도 예정(자격 상실)"
        add_target(ticker, row.get("name"), "sm", weight_sm, row.get("price"), row.get("daily_change_pct"), status)
    for row in nh_holdings:
        status = "오늘 진입" if row.get("is_new") else f"{row.get('days')}일째"
        if str(row.get("status")) == "sell":
            reason = str(row.get("exit_reason") or "이탈")
            status += f" · 매도 예정({reason})"
        add_target(row["ticker"], row.get("name"), "nh", weight_nh, row.get("price"), row.get("change_pct"), status)
    for row in nh_planned:
        add_target(
            row["ticker"], row.get("name"), "nh", weight_nh, row.get("price"), row.get("change_pct"), "진입 예정 (다음 시가 매수)"
        )

    stock_pct = sum(row["weight_pct"] for row in holdings)
    sm_cash = (sm_top_n - len(sm_holdings)) * weight_sm
    nh_cash = (nh_top_n - len(nh_holdings) - len(nh_planned)) * weight_nh

    # 매월 첫 거래일 = 슬리브 50:50 리밸런싱 날 (그 시장 달력 기준).
    from config import MARKET_SCHEDULES
    from utils.settings_loader import get_ticker_type_settings
    from utils.trading_calendar import get_trading_days

    country = str((get_ticker_type_settings(pool_norm) or {}).get("country_code") or "kor").strip().lower()
    tz_name = str((MARKET_SCHEDULES.get(country) or {}).get("timezone") or "Asia/Seoul")
    # 과거 날짜 조회면 그 날짜 기준으로 첫 거래일 여부를 판정한다.
    today_local = pd.Timestamp(as_of).date() if as_of else pd.Timestamp.now(tz=tz_name).date()
    month_start = today_local.replace(day=1)
    month_days = get_trading_days(month_start.strftime("%Y-%m-%d"), today_local.strftime("%Y-%m-%d"), country)
    sleeve_rebalance_today = bool(month_days) and month_days[0].date() == today_local

    # 다음 거래일 — 모든 체결은 시가라 액션 묶음의 실제 날짜가 된다. 연휴가 끼면
    # 이 날짜가 모멘텀 교체일과 같아질 수 있고, 그러면 화면이 한 묶음으로 합친다.
    ahead = get_trading_days(
        today_local.strftime("%Y-%m-%d"), (today_local + timedelta(days=21)).strftime("%Y-%m-%d"), country
    )
    next_trading_day = next((str(day.date()) for day in ahead if day.date() > today_local), None)

    # ── 적용 계좌 — 저장된 연결이 있으면 실제 보유·현금을 붙여 매매 지시를 만든다.
    account_id = str((load_settings_map().get(pool_norm) or {}).get("account_id") or "").strip()
    account = _load_account_state(account_id) if account_id else None
    if account is not None:
        # 보유 종목 가격 — 목표 목록에 있으면 그 값을, 없으면 가격 캐시에서 마지막 종가를 쓴다.
        price_by_ticker = {row["ticker"]: row["price"] for row in holdings if row.get("price")}
        missing = [ticker for ticker in account["holdings"] if ticker not in price_by_ticker]
        if missing:
            from utils.cache_utils import load_cached_frames_bulk_from_all_ticker_types

            for ticker, frame in load_cached_frames_bulk_from_all_ticker_types(missing).items():
                if frame is None or frame.empty or "Close" not in frame.columns:
                    continue
                close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
                if not close.empty:
                    price_by_ticker[ticker] = float(close.iloc[-1])
        stock_value = 0.0
        for ticker, item in account["holdings"].items():
            price = price_by_ticker.get(ticker)
            item["price"] = round(float(price), 4) if price else None
            item["value"] = round(item["quantity"] * float(price), 2) if price else None
            if price:
                stock_value += item["quantity"] * float(price)
        total_assets = stock_value + account["cash_balance"]
        account["stock_value"] = round(stock_value, 2)
        account["total_assets"] = round(total_assets, 2)
        # 종목별 목표 금액·주수 → 현재 보유와의 차이가 그대로 매매 지시가 된다.
        # 행이 종목 단위라 계좌 보유와 1:1 로 비교된다(겹치는 종목도 한 번만 센다).
        for row in holdings:
            held = account["holdings"].get(row["ticker"])
            row["held_quantity"] = held["quantity"] if held else 0.0
            row["held_value"] = (held or {}).get("value")
            row["current_weight_pct"] = (
                round(float(row["held_value"]) / total_assets * 100.0, 2)
                if row.get("held_value") and total_assets > 0
                else 0.0
            )
            target_amount = total_assets * row["weight_pct"] / 100.0
            row["target_amount"] = round(target_amount, 2)
            price = row.get("price")
            target_qty = int(target_amount // float(price)) if price else None
            row["target_quantity"] = target_qty
            row["trade_quantity"] = None if target_qty is None else target_qty - int(row["held_quantity"])
        # 목표 포트폴리오에 없는 보유 종목 = 전량 매도 대상. 목표 비중 0% 행으로 표에 함께 넣는다
        # — 팔아야 할 종목이 표 밖에 있으면 계좌를 표 하나로 대조할 수 없다.
        target_tickers = {row["ticker"] for row in holdings}
        account["sell_all"] = []
        for ticker, item in sorted(account["holdings"].items()):
            if ticker in target_tickers:
                continue
            value = item.get("value")
            account["sell_all"].append(
                {"ticker": ticker, "name": item["name"], "quantity": item["quantity"], "value": value}
            )
            holdings.append(
                {
                    "ticker": ticker,
                    "name": item["name"],
                    "sources": [],
                    "weight_pct": 0.0,
                    "price": item.get("price"),
                    "change_pct": None,
                    "sm_status": None,
                    "nh_status": None,
                    "is_sell_all": True,
                    "held_quantity": item["quantity"],
                    "held_value": value,
                    "current_weight_pct": round(float(value) / total_assets * 100.0, 2)
                    if value and total_assets > 0
                    else 0.0,
                    "target_amount": 0.0,
                    "target_quantity": 0,
                    "trade_quantity": -int(item["quantity"]),
                }
            )

    return {
        "computed_at": datetime.now().astimezone().isoformat(),
        "pool": pool_norm,
        "account": account,
        "as_of": nh.get("as_of"),
        "next_trading_day": next_trading_day,
        "live": bool(nh.get("live")),
        # 과거 날짜 셀렉트용 — 신고가 화면과 같은 날짜 목록을 그대로 쓴다.
        "available_dates": [str(d.get("date")) for d in (nh.get("available_dates") or [])],
        "summary": {
            "stock_pct": round(stock_pct, 2),
            "cash_pct": round(100 - stock_pct, 2),
            # slots_used = 목표가 찬 슬롯, held_count = 지금 실제로 들고 있는 종목 수.
            # 둘이 다르면 아직 체결 전이라는 뜻이라 화면이 구분해서 보여준다.
            "sm": {
                "slots_used": len(sm_holdings),
                "held_count": len(sm_current),
                "top_n": sm_top_n,
                "cash_pct": round(sm_cash, 2),
            },
            "nh": {
                "slots_used": len(nh_holdings) + len(nh_planned),
                "held_count": len(nh_holdings),
                "top_n": nh_top_n,
                "cash_pct": round(nh_cash, 2),
            },
        },
        "holdings": [{**row, "weight_pct": round(row["weight_pct"], 2)} for row in holdings],
        "actions": {
            # SM 주중 매도 예정 — 보유 자격(장단기 이평선) 상실, 다음 거래일 시가 매도.
            "sm_sells": [
                {
                    "ticker": str(row["ticker"]).strip(),
                    "name": row.get("name") or row["ticker"],
                    "reason": "자격 상실(이평선 하회)",
                }
                for row in sm_sell_pending
            ],
            "nh_entries": [
                {
                    "ticker": str(row["ticker"]).strip(),
                    "name": row.get("name") or row["ticker"],
                    "price": row.get("price"),
                    "change_pct": row.get("change_pct"),
                    "value_mult": row.get("value_mult"),
                }
                for row in nh_planned
            ],
            "nh_sells": [
                {
                    "ticker": row["ticker"],
                    "name": row["name"],
                    "return_pct": row.get("return_pct"),
                    "reason": row.get("exit_reason") or "이탈",
                }
                for row in nh_holdings
                if str(row.get("status")) == "sell"
            ],
            "sm_rebalance": {
                # 확정된 다음 교체 — 판정은 끝났고 체결만 남았다(이미 체결됐으면 is_filled=True).
                "is_filled": bool(sm.get("is_filled")),
                "fill_date": sm.get("rebalance_date"),
                "signal_date": sm.get("signal_date"),
                "portfolio_week": sm.get("portfolio_week"),
                "buys": [
                    {"ticker": str(r["ticker"]).strip(), "name": r.get("name") or r["ticker"], "price": r.get("price")}
                    for r in sm_rebalance_buys
                ],
                "sells": [
                    {"ticker": str(r["ticker"]).strip(), "name": r.get("name") or r["ticker"]}
                    for r in sm_rebalance_sells
                ],
            },
            "sleeve_rebalance_today": sleeve_rebalance_today,
        },
    }


def _merge_trades(sm: dict[str, Any], nh: dict[str, Any]) -> list[dict[str, Any]]:
    """두 전략의 체결을 한 목록으로 — 보유중 행이 먼저, 그 아래는 청산일 최신순."""
    merged: list[dict[str, Any]] = []
    for row in sm.get("trades") or []:
        merged.append({**row, "strategy": "모멘텀"})
    for row in nh.get("open_positions") or []:
        merged.append(
            {
                "strategy": "신고가",
                "ticker": row["ticker"],
                "name": row["name"],
                "entry_date": row["entry_date"],
                "entry_price": row["entry_price"],
                "exit_date": None,
                "exit_price": row.get("price"),
                "return_pct": row.get("return_pct"),
                "days": row.get("days"),
                "reason": "보유중",
            }
        )
    for row in nh.get("trades") or []:
        merged.append(
            {
                "strategy": "신고가",
                "ticker": row["ticker"],
                "name": row["name"],
                "entry_date": row["entry_date"],
                "entry_price": row["entry_price"],
                "exit_date": row["exit_date"],
                "exit_price": row["exit_price"],
                "return_pct": row["return_pct"],
                "days": row["days"],
                "reason": row["reason"],
            }
        )
    holding = sorted(
        (row for row in merged if row["exit_date"] is None), key=lambda row: row["entry_date"], reverse=True
    )
    closed = sorted((row for row in merged if row["exit_date"]), key=lambda row: row["exit_date"], reverse=True)
    return holding + closed


def run_mix_backtest(pool: str | None = None, months: int | None = None) -> dict[str, Any]:
    """선택한 풀의 저장 설정으로 SM·신고가 백테스트를 돌려 50:50 합성 결과를 만든다.

    ``months`` 를 주면 그 기간으로 두 백테스트를 돌린다(화면의 기간 셀렉트). 없으면
    두 저장 설정 중 짧은 쪽.
    """
    import pandas as pd

    from utils.momentum_backtest import run_backtest as sm_backtest
    from utils.new_high_backtest import run_backtest as nh_backtest

    pool_norm, sm_settings, nh_settings = _resolve_pool_and_settings(pool)
    if months is None:
        months = min(int(sm_settings["backtest_months"]), int(nh_settings["backtest_months"]))
    months = int(months)
    if months not in MONTH_OPTIONS:
        raise ValueError(f"'months' 는 {list(MONTH_OPTIONS)} 중 하나여야 합니다 (받은 값: {months})")

    logger.info("[STRATEGY-MIX] %s 합성 백테스트 시작 (%d개월)", pool_norm, months)
    sm = sm_backtest(months, sm_settings, include_daily=True)
    nh = nh_backtest(months, nh_settings)

    # 형식 주의 — SM 의 daily 는 **일간 변동률(%)**, 신고가의 daily 는 **누적(%)** 이다.
    # 둘 다 누적 곡선(시작=1.0)으로 만든 뒤 날짜를 맞춘다. 벤치마크는 SM 결과의 것(같은 풀).
    sm_curve: dict[str, float] = {}
    bench_curve: dict[str, float] = {}
    s_value = b_value = 1.0
    # SM 의 daily 는 화면 표시용으로 **최신 날짜가 앞** — 복리 전에 반드시 날짜순으로 정렬한다.
    for row in sorted(sm["daily"], key=lambda r: r["date"]):
        if row.get("strategy_pct") is not None:
            s_value *= 1 + row["strategy_pct"] / 100
        if row.get("benchmark_pct") is not None:
            b_value *= 1 + row["benchmark_pct"] / 100
        sm_curve[row["date"]] = s_value
        bench_curve[row["date"]] = b_value

    nh_curve = {row["date"]: 1 + row["strategy_pct"] / 100 for row in nh["daily"]}

    dates = sorted(set(sm_curve) & set(nh_curve))
    if len(dates) < 2:
        raise RuntimeError("두 전략의 공통 백테스트 구간이 부족합니다.")

    # 매월 50:50 리밸런싱 합성 — 월이 바뀌는 첫 거래일에 두 슬리브를 절반씩으로 되돌린다.
    daily_rows = []
    mix = 1.0
    base_sm = base_nh = base_mix = None
    prev_month = None
    first_bench = bench_curve[dates[0]]
    for date in dates:
        s, n = sm_curve[date], nh_curve[date]
        if date[:7] != prev_month:
            base_sm, base_nh, base_mix = s, n, mix
            prev_month = date[:7]
        mix = base_mix * (0.5 * (s / base_sm) + 0.5 * (n / base_nh))
        daily_rows.append(
            {
                "date": date,
                "strategy_pct": round((mix - 1) * 100, 2),
                "benchmark_pct": round((bench_curve[date] / first_bench - 1) * 100, 2),
            }
        )

    # 요약 지표 — 일별 누적 곡선 기준. 계산은 신고가 엔진과 같은 방식이다
    # (총수익·기간 CAGR·일별 곡선 MDD·일별 수익률 소르티노).
    def _summarize(curve: pd.Series) -> dict[str, Any]:
        total = float((curve.iloc[-1] - 1) * 100)
        returns = curve.pct_change().dropna()
        downside = returns[returns < 0]
        deviation = float((downside**2).mean() ** 0.5) if not downside.empty else 0.0
        sortino = (
            round(float(returns.mean()) / deviation * (252**0.5), 2) if deviation > 0 and len(returns) >= 2 else None
        )
        return {
            "total_pct": round(total, 2),
            "cagr_pct": round(((1 + total / 100) ** (12 / months) - 1) * 100, 2) if months > 0 else None,
            "mdd_pct": round(float(((curve / curve.cummax()) - 1).min() * 100), 2),
            "sortino": sortino,
        }

    strategy_curve = pd.Series([1 + row["strategy_pct"] / 100 for row in daily_rows])
    benchmark_curve = pd.Series([1 + row["benchmark_pct"] / 100 for row in daily_rows])
    strategy_stats, benchmark_stats = _summarize(strategy_curve), _summarize(benchmark_curve)

    return {
        "computed_at": datetime.now().astimezone().isoformat(),
        "pool": pool_norm,
        "months": months,
        "start_date": dates[0],
        "end_date": dates[-1],
        "benchmark_name": str(sm["benchmark_name"]),
        "benchmark_ticker": str(sm.get("benchmark_ticker") or ""),
        "strategy_total_pct": strategy_stats["total_pct"],
        "strategy_cagr_pct": strategy_stats["cagr_pct"],
        "strategy_mdd_pct": strategy_stats["mdd_pct"],
        "strategy_sortino": strategy_stats["sortino"],
        "benchmark_total_pct": benchmark_stats["total_pct"],
        "benchmark_cagr_pct": benchmark_stats["cagr_pct"],
        "benchmark_mdd_pct": benchmark_stats["mdd_pct"],
        "benchmark_sortino": benchmark_stats["sortino"],
        # 일별 누적(%) — 화면이 연간·월간·주간·일간 표를 이 시계열에서 만든다.
        "daily": daily_rows,
        # 체결 목록 — 두 전략을 합쳐 보여준다(보유중 행이 위, 그 아래 청산일 최신순).
        "trades": _merge_trades(sm, nh),
    }
