"""합성 전략 — SM(모멘텀 전략) + 신고가 돌파를 한 계좌에서 함께 굴린 백테스트.

`/strategy-mix` 열람 전용 화면의 백엔드. 설정은 이 화면이 갖지 않는다 —
**선택한 풀의 각 전략 화면 저장 설정**을 그대로 가져와 두 백테스트를 돌리고,
매월 첫 거래일에 계좌 설정의 배분(모멘텀·신고가·비워 두는 현금)으로 되돌리되,
**현금 우선**으로 이관한다 —
넘기는 슬리브의 현금부터 쓰고, 모자랄 때만 주식을 비례 매도한다(오르는 종목 유지).
화면은 신고가 화면과 같은 방식으로 이 일별 누적에서 연간·월간·일간 표를 만든다.

캐시는 두지 않는다 — 각 전략 화면의 백테스트와 같은 패턴(요청 시 계산)이다.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from utils.logger import get_app_logger
from utils.trade_stats import summarize_trades

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
def month_options() -> list[int]:
    """기간 선택지 — 종목풀 백테스트와 같은 목록(`get_month_options`)이 단일 소스.

    전략별로 목록을 따로 두면 화면마다 고를 수 있는 기간이 달라진다.
    """
    from utils.pool_signal_backtest_service import get_month_options

    return get_month_options()


# 배분 기본값(%) — 계좌 설정에 저장이 없을 때 쓰는 시스템 기본 배분이며, 지금까지의
# 동작(모멘텀 50 : 신고가 50, 비워 두는 현금 없음)과 같다. 화면이 이 값을 그대로 보여준다.
DEFAULT_MIX_WEIGHTS: dict[str, float] = {"sm_pct": 50.0, "nh_pct": 50.0, "cash_pct": 0.0}


def mix_weights(account_settings: dict[str, Any]) -> dict[str, float]:
    """계좌 설정의 합성 배분(%) — 저장이 없으면 `DEFAULT_MIX_WEIGHTS`.

    셋은 항상 함께 저장되므로(계좌 설정 검증), 하나라도 없으면 미저장으로 보고
    기본 배분을 쓴다 — 일부만 읽어 섞으면 합이 100 이 아닌 배분이 만들어진다.
    """
    keys = (("sm_pct", "mix_sm_pct"), ("nh_pct", "mix_nh_pct"), ("cash_pct", "mix_cash_pct"))
    if any(account_settings.get(stored) is None for _, stored in keys):
        return dict(DEFAULT_MIX_WEIGHTS)
    return {name: float(account_settings[stored]) for name, stored in keys}


def mix_accounts() -> list[dict[str, Any]]:
    """합성 전략을 운용하는 계좌 목록 — 계좌 설정의 `mix_pool` 이 지정된 계좌만.

    계좌 하나에 종목풀 하나다(계좌 설정 화면에서 지정). 지정이 없으면 이 화면에 오르지 않는다.
    """
    from utils.settings_loader import get_account_settings, list_available_accounts

    pool_names = {option["ticker_type"]: option for option in _pool_options(_all_active_pools())}
    accounts: list[dict[str, Any]] = []
    for account_id in list_available_accounts():
        try:
            settings = get_account_settings(account_id) or {}
        except Exception:
            continue
        inner = settings.get("settings") or settings
        pool = str(inner.get("mix_pool") or "").strip().lower()
        if not pool:
            continue
        accounts.append(
            {
                "account_id": account_id,
                "name": str(inner.get("name") or account_id),
                "icon": str(inner.get("icon") or ""),
                "order": inner.get("order"),
                "currency": str(inner.get("currency") or "").strip().upper(),
                "pool": pool,
                # 오늘의 액션 슬랙 알람 토글 상태 — 화면 헤더가 그대로 보여준다.
                "mix_slack_enabled": bool(inner.get("mix_slack_enabled")),
                # 합성 배분(%) — 저장이 없으면 기본 배분(50:50:0).
                **{f"mix_{name}": value for name, value in mix_weights(inner).items()},
                "pool_label": pool_names.get(pool),
            }
        )
    accounts.sort(key=lambda item: (item["order"] is None, item["order"]))
    return accounts


def mix_meta() -> dict[str, Any]:
    """화면 초기용 — 운용 계좌 목록과 기간 선택지 (백테스트 계산 없음)."""
    accounts = mix_accounts()
    return {
        "accounts": accounts,
        "month_options": month_options(),
        # 기본 선택 — 목록의 첫 계좌. 화면이 마지막 선택을 로컬스토리지에 기억한다.
        "account_id": accounts[0]["account_id"] if accounts else "",
    }


def mix_weights_for_pool(pool: str) -> dict[str, float]:
    """그 풀을 운용하는 계좌의 합성 배분(%). 계좌가 없으면 기본 배분.

    계좌 하나에 풀 하나이므로(계좌 설정의 `mix_pool`), 풀로 계좌를 되찾을 수 있다.
    백테스트·운영 화면·슬랙이 모두 같은 배분을 쓰도록 여기서만 읽는다.
    """
    target = str(pool or "").strip().lower()
    for account in mix_accounts():
        if account["pool"] == target:
            return {
                "sm_pct": float(account["mix_sm_pct"]),
                "nh_pct": float(account["mix_nh_pct"]),
                "cash_pct": float(account["mix_cash_pct"]),
            }
    return dict(DEFAULT_MIX_WEIGHTS)


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
        # 기본 풀 — 두 전략 설정이 모두 있는 풀 중 번호가 가장 빠른 것.
        pool_norm = [o["ticker_type"] for o in _pool_options(ready)][0]
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


def _sleeve_shares(
    pool: str, sm_settings: dict[str, Any], nh_settings: dict[str, Any], as_of: str | None
) -> tuple[float, float, float]:
    """(모멘텀 몫 %, 신고가 몫 %, 현금 몫 %) — 직전 월초 배분 이후 흘러간 비율.

    각 엔진의 일별 곡선에서 이번 달 첫 거래일 이후 성장률을 읽어, 월초 배분에 곱한 뒤
    셋을 100 으로 정규화한다. 비워 두는 현금은 자라지 않으므로 그 몫은 그대로 두고,
    두 슬리브가 오르면 상대적으로 현금 비중이 줄어든다.

    곡선을 못 구하면(데이터 부족 등) 저장된 월초 배분을 그대로 쓴다 — 임의 보정 대신
    '아직 안 흘러간 상태' 로 명시한다.
    """
    from utils.momentum_backtest import run_backtest as sm_backtest
    from utils.new_high_backtest import run_backtest as nh_backtest

    weights = mix_weights_for_pool(pool)
    base = (weights["sm_pct"], weights["nh_pct"], weights["cash_pct"])

    try:
        sm_daily = sm_backtest(2, sm_settings, include_daily=True)["daily"]
        nh_daily = nh_backtest(2, nh_settings)["daily"]
    except Exception:
        logger.warning("[STRATEGY-MIX] %s 슬리브 몫 계산 실패 — 월초 배분 그대로 둔다", pool, exc_info=True)
        return base

    cutoff = as_of or "9999-12-31"
    sm_rows = sorted((r for r in sm_daily if r["date"] <= cutoff), key=lambda r: r["date"])
    nh_rows = sorted((r for r in nh_daily if r["date"] <= cutoff), key=lambda r: r["date"])
    if not sm_rows or not nh_rows:
        return base
    month = sm_rows[-1]["date"][:7]

    # 모멘텀 일별은 전일 대비 변동률 — 월초 첫 거래일 **다음 날부터** 곱한다
    # (첫 거래일 종가가 기준점이라, 그날까지의 변동은 직전 달 몫이다).
    sm_month = [r for r in sm_rows if r["date"][:7] == month and r.get("strategy_pct") is not None]
    sm_growth = 1.0
    for row in sm_month[1:]:
        sm_growth *= 1.0 + row["strategy_pct"] / 100.0

    # 신고가 일별은 구간 시작 대비 누적 — 월초 첫 거래일 값을 기준으로 나눈다.
    nh_month = [r for r in nh_rows if r["date"][:7] == month and r.get("strategy_pct") is not None]
    nh_growth = 1.0
    if len(nh_month) >= 2:
        base = 1.0 + nh_month[0]["strategy_pct"] / 100.0
        if base > 0:
            nh_growth = (1.0 + nh_month[-1]["strategy_pct"] / 100.0) / base

    sm_value = weights["sm_pct"] * sm_growth
    nh_value = weights["nh_pct"] * nh_growth
    cash_value = weights["cash_pct"]  # 비워 둔 현금은 자라지 않는다.
    total = sm_value + nh_value + cash_value
    if total <= 0:
        return base
    return (
        round(sm_value / total * 100.0, 4),
        round(nh_value / total * 100.0, 4),
        round(cash_value / total * 100.0, 4),
    )


# 비중 조정 지시 밴드 — **슬롯 크기에 비례**한다(슬롯 목표비중 × 비율, 최소 0.5%p).
# 고정 0.5%p 로 두면 슬롯이 큰 종목은 하루 가격 변동(드리프트)만으로 지시가 떠서, 백테스트에
# 없는 주중 재조정 매매를 시키게 된다. 드리프트는 지시로 만들지 않는 게 전략 규칙이고,
# 입출금은 큰 단위로 들어오므로(총자산 15%+) 이 밴드를 넘어 지시가 생성된다.
_REBALANCE_BAND_RATIO = 0.15
_REBALANCE_BAND_MIN_PCT = 0.5

_WEEKDAYS_KO = ("월", "화", "수", "목", "금", "토", "일")


def _format_date_weekday(date: str) -> str:
    from datetime import date as date_cls

    try:
        parsed = date_cls.fromisoformat(date)
    except ValueError:
        return date
    return f"{date} ({_WEEKDAYS_KO[parsed.weekday()]})"


def _build_action_groups(
    holdings: list[dict[str, Any]],
    actions: dict[str, Any],
    next_trading_day: str | None,
) -> list[dict[str, Any]]:
    """오늘의 액션 — 체결일 묶음(매도 먼저, 같은 방향은 티커 순).

    화면과 슬랙 알람이 **이 결과를 그대로** 쓴다 — 조립을 한 곳에 두어 둘이 어긋나지
    않게 한다. 규칙은 화면에 있던 것 그대로:
      · 목표는 흘러간 비중을 따라가므로, 밴드(슬롯 비중의 15%, 최소 0.5%p) 이상 차이만 지시로 만든다
        — 가격 드리프트는 지시가 안 되고, 큰 단위 입출금·교체·진입·이탈만 지시가 된다.
      · 모멘텀 교체 확정분(미체결)은 교체일 시가 그룹, 나머지는 다음 거래일 그룹.
      · 전량 매도·손절·이탈은 금액과 무관하게 항상 남긴다.
    """
    rebalance = actions["sm_rebalance"]
    rebalance_buys = {row["ticker"] for row in rebalance["buys"]}
    rebalance_sells = {row["ticker"] for row in rebalance["sells"]}
    entry_tickers = {row["ticker"] for row in actions["nh_entries"]}
    nh_event_tickers = entry_tickers | {row["ticker"] for row in actions["nh_sells"]}
    sell_pending = [row["ticker"] for row in actions["sm_sells"]] + [row["ticker"] for row in actions["nh_sells"]]

    rebalance_date = rebalance["fill_date"] if (not rebalance["is_filled"] and rebalance["fill_date"]) else None

    sell_reason: dict[str, str] = {row["ticker"]: row["reason"] for row in actions["sm_sells"]}
    for row in actions["nh_sells"]:
        suffix = f", {row['return_pct']:+.2f}%" if row.get("return_pct") is not None else ""
        sell_reason[row["ticker"]] = f"{row['reason']}{suffix}"

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
        is_rebalance = ticker in rebalance_buys or ticker in rebalance_sells
        momentum_pending = (
            rebalance_date is not None
            and "sm" in (row.get("sources") or [])
            and ticker not in nh_event_tickers
            and not row.get("is_sell_all")
        )
        date = rebalance_date if (momentum_pending or is_rebalance) else next_trading_day
        reason = sell_reason.get(ticker)
        weight = float(row.get("weight_pct") or 0)
        if not row.get("is_sell_all") and not (reason and trade < 0 and weight <= 0):
            gap = abs(weight - float(row.get("current_weight_pct") or 0))
            band = max(_REBALANCE_BAND_MIN_PCT, weight * _REBALANCE_BAND_RATIO)
            if gap < band:
                continue
        held = float(row.get("held_quantity") or 0) > 0
        sell_reason_applies = bool(reason) and trade < 0 and weight <= 0
        if row.get("is_sell_all"):
            title = "교체 매도" if ticker in rebalance_sells else "전량 매도"
        elif trade < 0:
            title = "매도 예정" if sell_reason_applies else "비중 조정 매도"
        elif held:
            title = "비중 조정 매수"
        elif ticker in rebalance_buys:
            title = "교체 매수"
        elif ticker in entry_tickers:
            title = "신고가 진입"
        else:
            title = "신규 매수"
        after = f" → 목표 {int(row['target_quantity']):,}주" if row.get("target_quantity") is not None else ""
        if sell_reason_applies:
            note = f"{after} ({reason})".strip()
        elif row.get("is_sell_all"):
            value = row.get("held_value")
            note = (f"({value:,.0f}) " if value is not None else "") + "· 목표에 없는 보유 종목"
        else:
            note = f"{after} · {weight:.2f}%".strip()
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
                "text": f"{label(ticker, row.get('name') or ticker, row.get('held_quantity'))} ({sell_reason.get(ticker) or '이탈'})",
                "date": next_trading_day,
                "quantity": abs(int(float(row.get("held_quantity") or 0))),
            }
        )
        seen_keys.add(f"act-{ticker}")

    by_date: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_date.setdefault(item["date"] or "", []).append(item)
    groups = []
    for date in sorted(by_date):
        group_items = sorted(by_date[date], key=lambda x: (0 if x["side"] == "sell" else 1, x["ticker"]))
        title = (
            f"{_format_date_weekday(date)} 시가" + (" · 모멘텀 교체 포함" if date == rebalance_date else "")
            if date
            else "체결일 미정"
        )
        groups.append({"key": date or "unscheduled", "title": title, "items": group_items})

    # 주중 이탈 **예상** 그룹 — 판정(오늘 종가) 확정 전의 미리보기. 같은 체결일(다음 거래일)
    # 시가지만 확정 지시와 섞이지 않게 별도 그룹으로 뒤에 둔다. 슬랙 알람은 이 그룹을 보내지
    # 않는다(장중 출렁일 때마다 알람이 나가면 노이즈 — notify 가 forecast 그룹을 거른다).
    confirmed = {item["ticker"] for group in groups for item in group["items"]}
    forecast_items = []
    for row in actions.get("sm_exit_forecast") or []:
        ticker = row["ticker"]
        held = row_by_ticker.get(ticker)
        if ticker in confirmed or not held or float(held.get("held_quantity") or 0) <= 0:
            continue
        forecast_items.append(
            {
                "key": f"forecast-{ticker}",
                "ticker": ticker,
                "side": "sell",
                "title": "매도 예정(예상)",
                "text": f"{label(ticker, row.get('name') or ticker, held.get('held_quantity'))} ({row.get('reason')})",
                "date": next_trading_day,
                # 알람 상태 비교용 — 예상도 '처음 등장할 때 1건' 발송되도록 실제 수량을 싣는다.
                "quantity": abs(int(float(held.get("held_quantity") or 0))),
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


def _attach_disparity(holdings: list[dict[str, Any]], sm_settings: dict[str, Any]) -> None:
    """행마다 단기·장기 이격(%)을 붙인다 — 종목풀 설정의 이평선이 기준.

    순위·모멘텀 화면과 같은 계산(`momentum_metrics`)을 그대로 쓴다. 화면은 이 값으로
    종목명 옆 추세 이탈 배지(❗)를 붙이므로, 다른 기준으로 계산하면 화면마다 다른
    종목에 배지가 붙는다. 값을 못 구하면 None 으로 둔다(임의 값으로 채우지 않는다).
    """
    import pandas as pd

    from utils.cache_utils import load_cached_frames_bulk_from_all_ticker_types
    from utils.momentum_service import momentum_metrics

    tickers = [str(row["ticker"]).strip() for row in holdings if row.get("ticker")]
    if not tickers:
        return
    short_days = int(sm_settings["short_ma_days"])
    long_days = int(sm_settings["long_ma_days"])
    frames = load_cached_frames_bulk_from_all_ticker_types(tickers)
    for row in holdings:
        row["current_short_pct"] = None
        row["current_long_pct"] = None
        frame = frames.get(str(row.get("ticker") or "").strip())
        if frame is None or frame.empty or "Close" not in frame.columns:
            continue
        close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
        if close.empty:
            continue
        metrics = momentum_metrics(close, short_ma_days=short_days, long_ma_days=long_days, as_of=None)
        if not metrics:
            continue
        row["current_short_pct"] = round(metrics["short_disparity_pct"], 1)
        row["current_long_pct"] = round(metrics["disparity_pct"], 1)


def _krw_rate(currency: str) -> float:
    """종목 통화 → 원화 환율. 원화면 1.0.

    계좌 원장의 현금·평가액은 전부 원화 기준인데 종목 가격은 그 시장 통화다. 섞어서
    더하거나 나누면 미국·호주 풀에서 총자산과 목표 수량이 환율 배수만큼 어긋난다.
    환율을 못 받으면 0 을 돌려 호출부가 '계산 불가' 로 명시한다 — 1.0 으로 두면
    달러 가격을 원화로 착각한 값이 조용히 나간다.
    """
    code = str(currency or "KRW").strip().upper()
    if code == "KRW":
        return 1.0
    from services.price_service import get_exchange_rates

    try:
        return float(((get_exchange_rates() or {}).get(code) or {}).get("rate") or 0.0)
    except Exception:
        logger.warning("[STRATEGY-MIX] %s 환율 조회 실패 — 목표 수량을 계산하지 않는다", code, exc_info=True)
        return 0.0


def _attach_account_targets(
    holdings: list[dict[str, Any]], account: dict[str, Any], krw_rate: float = 1.0
) -> list[dict[str, Any]]:
    """계좌 보유와 목표를 대조해 수량 지시를 붙인다. 전량 매도 요약 목록을 돌려준다.

    종목별 목표 금액·주수 → 현재 보유와의 차이가 그대로 매매 지시가 된다.
    행이 종목 단위라 계좌 보유와 1:1 로 비교된다(겹치는 종목도 한 번만 센다).
    목표 포트폴리오에 없는 보유 종목은 전량 매도 대상 — 목표 비중 0% 행으로 표에
    함께 넣는다 (팔아야 할 종목이 표 밖에 있으면 계좌를 표 하나로 대조할 수 없다).
    """
    total_assets = float(account.get("total_assets") or 0)
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
        # 목표 금액은 원화, 가격은 그 시장 통화다 — 환율로 맞춘 뒤 나눠야 한다.
        price = row.get("price")
        price_krw = float(price) * krw_rate if price and krw_rate > 0 else None
        # 반올림 — 버림이면 1주 값이 목표 금액보다 조금만 커도 목표가 0 이 되어 슬롯이
        # 통째로 빈다(비중 0% vs 목표 5%). 반올림이 목표 비중에 더 가깝다.
        target_qty = round(target_amount / price_krw) if price_krw else None
        # 1주 값이 목표 금액보다 커서 0 이 된 경우에는 최소 1주를 목표로 둔다 — 비중을
        # 배정받은 슬롯이 0주로 비는 것보다 낫다. 비중이 0 인 행(전량 매도 등)은 그대로 0.
        if target_qty == 0 and row["weight_pct"] > 0:
            target_qty = 1
        row["target_quantity"] = target_qty
        row["trade_quantity"] = None if target_qty is None else target_qty - int(row["held_quantity"])
    target_tickers = {row["ticker"] for row in holdings}
    sell_all: list[dict[str, Any]] = []
    for ticker, item in sorted(account["holdings"].items()):
        if ticker in target_tickers:
            continue
        value = item.get("value")
        sell_all.append({"ticker": ticker, "name": item["name"], "quantity": item["quantity"], "value": value})
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
    return sell_all


def _build_next_week_preview(
    sm: dict[str, Any],
    holdings: list[dict[str, Any]],
    actions: dict[str, Any],
    weight_sm: float,
    account: dict[str, Any] | None,
    ahead: list,
    today_local,
) -> dict[str, Any] | None:
    """다음주 교체를 **지금 순위 그대로 확정된다고 가정**했을 때의 월요일 전체 액션.

    오늘의 액션과 **같은 조립기**(`_build_action_groups`)를 다음주 가정 목표에
    돌린다 — 전량 매도(목표에 없는 보유 종목)까지 포함한 완전한 그림이라, 종목
    교체가 없으면 오늘의 액션과 똑같이 나온다. 모멘텀 슬리브만 주간 교체가 있으므로
    다음주 예상(`next_week_expected` — 실시간 순위 기준)으로 모멘텀 목표만 바꾼다.

    이번 교체가 아직 체결 전이면(오늘의 액션에 교체일 그룹이 이미 있으면) 만들지
    않는다 — 같은 교체가 두 번 보이면 헷갈린다. 수량은 현재가·현재 총자산 기준
    추정치라 실제 체결 수량과 다를 수 있다.
    """
    if not sm.get("is_filled") or account is None:
        return None

    expected_rows = {
        str(row["ticker"]).strip(): row for row in (sm.get("rows") or []) if row.get("next_week_expected")
    }

    # ── 다음주 가정 목표 — 현재 목표에서 모멘텀 슬리브만 다음주 예상으로 바꾼다 ──
    hypo: list[dict[str, Any]] = []
    for row in holdings:
        if row.get("is_sell_all"):
            continue  # 전량 매도 행은 계좌 대조가 다시 만든다
        copy = {
            "ticker": row["ticker"],
            "name": row.get("name"),
            "sources": list(row.get("sources") or []),
            "weight_pct": float(row.get("weight_pct") or 0),
            "sm_weight": float(row.get("sm_weight") or 0),
            "nh_weight": float(row.get("nh_weight") or 0),
            "price": row.get("price"),
        }
        if "sm" in copy["sources"] and copy["ticker"] not in expected_rows:
            # 편출 예상 — 모멘텀 몫을 뺀다. 신고가 몫이 없으면 행 자체가 빠진다.
            copy["weight_pct"] -= copy["sm_weight"]
            copy["sm_weight"] = 0.0
            copy["sources"] = [s for s in copy["sources"] if s != "sm"]
            if not copy["sources"] and copy["weight_pct"] <= 0:
                continue
        hypo.append(copy)
    hypo_tickers = {row["ticker"] for row in hypo}
    preview_buys: list[dict[str, Any]] = []
    for ticker in sorted(expected_rows):
        if ticker in hypo_tickers and any(
            row["ticker"] == ticker and "sm" in row["sources"] for row in hypo
        ):
            continue  # 유지 — 이미 모멘텀 몫이 있다
        row = expected_rows[ticker]
        existing = next((r for r in hypo if r["ticker"] == ticker), None)
        if existing is None:
            existing = {
                "ticker": ticker,
                "name": row.get("name") or ticker,
                "sources": [],
                "weight_pct": 0.0,
                "sm_weight": 0.0,
                "nh_weight": 0.0,
                "price": row.get("price"),
            }
            hypo.append(existing)
        existing["sources"].append("sm")
        existing["weight_pct"] += weight_sm
        existing["sm_weight"] += weight_sm
        preview_buys.append({"ticker": ticker, "name": row.get("name") or ticker, "price": row.get("price")})

    _attach_account_targets(hypo, account)

    # 다음주 첫 거래일 = 다음 교체 체결일 (모멘텀 주간 리듬).
    this_week = today_local.isocalendar()[:2]
    fill_date = next(
        (str(day.date()) for day in ahead if day.date().isocalendar()[:2] > tuple(this_week)), None
    )

    # 오늘의 액션과 같은 조립 — 교체 예상분만 rebalance 로 넘겨 '교체 매수/매도' 라벨을 받는다.
    hypo_actions = {
        "sm_sells": actions["sm_sells"],
        "nh_entries": actions["nh_entries"],
        "nh_sells": actions["nh_sells"],
        "sm_rebalance": {
            "is_filled": True,
            "fill_date": None,
            "buys": preview_buys,
            "sells": [
                {"ticker": row["ticker"], "name": row.get("name") or row["ticker"]}
                for row in holdings
                if not row.get("is_sell_all")
                and "sm" in (row.get("sources") or [])
                and row["ticker"] not in expected_rows
            ],
        },
    }
    groups = _build_action_groups(hypo, hypo_actions, fill_date)
    # 주중 이탈 예상 종목은 뺀다 — 다음주가 아니라 내일 시가에 팔릴 예상이라 오늘의 액션의
    # '(예상)' 그룹이 담당한다. 여기 두면 '교체 매도 · 체결일 미정' 으로 잘못 안내된다.
    forecast_tickers = {row["ticker"] for row in actions.get("sm_exit_forecast") or []}
    if forecast_tickers:
        groups = [
            {**group, "items": [item for item in group["items"] if item["ticker"] not in forecast_tickers]}
            for group in groups
            if not group.get("forecast")
        ]
        groups = [group for group in groups if group["items"]]
    return {"fill_date": fill_date, "groups": groups}


def mix_positions(pool: str | None = None, as_of: str | None = None) -> dict[str, Any]:
    """오늘 기준 합성 운영 상태 — 보유 목록(목표 비중)·현금 비중·오늘의 액션.

    두 전략 화면이 계산하는 것(SM compute_picks · 신고가 current_positions)을 합쳐서
    보여줄 뿐, 새 판정 로직은 없다. 비중은 슬리브 몫 ÷ 슬롯 수(빈 슬롯 = 현금).
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

    # ── 슬리브 몫 — 월초 배분에서 두 슬리브가 각자 흘러간 비율을 역산한다 ──
    # 재조정은 매월 첫 거래일에만 하므로, 그 사이에는 잘 나간 슬리브의 몫이 커진 채로
    # 가는 것이 백테스트다. 항상 월초 배분으로 보면 승자 슬리브를 매주 깎는 지시가 나온다.
    base_weights = mix_weights_for_pool(pool_norm)
    sm_share, nh_share, reserved_cash_share = _sleeve_shares(pool_norm, sm_settings, nh_settings, as_of)

    # 새로 담는 슬롯의 몫 — 진입·교체 시점에는 그 슬리브 몫의 1/N 을 배정한다.
    weight_sm = sm_share / sm_top_n
    weight_nh = nh_share / nh_top_n

    # 이미 들고 있는 종목은 **흘러간 실제 비중**을 목표로 쓴다. 진입할 때 1/N 이었다가
    # 시세대로 벌어진 값이고, 백테스트도 그 상태를 그대로 들고 간다. 고정 1/N 을 목표로
    # 두면 목표와 보유가 매일 어긋나 실제로는 하지 않을 매매가 지시로 나온다.
    # 다만 모멘텀 교체가 확정됐는데 아직 체결 전이면(is_filled=False) 그 교체분은
    # 체결일에 1/N 으로 다시 맞춰지므로 균등 몫을 쓴다.
    sm_drift_weight: dict[str, float] = {}
    if sm.get("is_filled"):
        for row in sm.get("holdings") or []:
            weight = row.get("sleeve_weight_pct")
            if weight is not None:
                sm_drift_weight[str(row["ticker"]).strip()] = sm_share * float(weight) / 100.0
    nh_drift_weight: dict[str, float] = {}
    for row in nh.get("holdings") or []:
        weight = row.get("sleeve_weight_pct")
        if weight is not None:
            nh_drift_weight[str(row["ticker"]).strip()] = nh_share * float(weight) / 100.0

    # SM 목표 포트폴리오 — 확정된 선정(다음 교체분)을 기준으로 보여준다. 이 화면은
    # '무엇을 보유해야 하는지'를 보는 곳이고, 신고가 슬리브도 진입 예정이 슬롯을
    # 채우는 방식이라 두 슬리브의 기준을 맞춘 것이다. 매도분은 오늘의 액션에만 남는다.
    #
    # 주중에 **이미 팔린** 종목(`is_exited`)은 뺀다. 선정 목록에는 자리를 유지한 채로
    # 남아 있는데(모멘텀 화면이 '매도' 로 표시하는 그 행), 계좌 수량은 0 이라 그대로
    # 두면 방금 판 종목을 다시 사라는 지시가 나온다. 그 슬롯은 다음 교체까지 현금이다.
    sm_selected = [
        row
        for row in sm["rows"]
        if not row.get("is_reserve")
        and not row.get("is_expected_only")
        and not row.get("is_exited")
        and (row.get("rank") or 999) <= sm_top_n
    ]
    sm_current = list(sm.get("holdings") or [])
    sm_current_tickers = {str(row["ticker"]).strip() for row in sm_current}
    sm_selected_tickers = {str(row["ticker"]).strip() for row in sm_selected}
    # 교체일에 새로 사는 종목 / 빠지는 종목 — 오늘의 액션이 쓴다.
    sm_rebalance_buys = [row for row in sm_selected if str(row["ticker"]).strip() not in sm_current_tickers]
    sm_rebalance_sells = [row for row in sm_current if str(row["ticker"]).strip() not in sm_selected_tickers]
    # 주중 매도 예정 — 자격 상실 판정(이미 보유 중인 종목만 해당).
    sm_sell_pending = [row for row in sm_selected if row.get("is_exit_pending")]
    nh_holdings = list(nh.get("holdings") or [])

    # 신고가 빈 슬롯을 채울 진입 예정 — 다음 시가에 사므로 목표 포트폴리오에 포함한다.
    nh_free = max(nh_top_n - len(nh_holdings), 0)
    nh_planned = list(nh.get("planned_entries") or [])[:nh_free]

    # 목표 포트폴리오는 **종목 단위**다. 두 슬리브가 같은 종목을 담으면 비중을 합쳐
    # 한 행으로 둔다 — 계좌에는 그 종목이 하나뿐이라 슬리브별로 나누면 보유 수량이
    # 두 번 세어지고 매매 지시가 반대로 나온다.
    holdings: list[dict[str, Any]] = []
    by_ticker: dict[str, dict[str, Any]] = {}

    def add_target(
        ticker: str, name: str, source: str, weight: float, price: Any, change_pct: Any, status: str
    ) -> None:
        ticker = str(ticker).strip()
        row = by_ticker.get(ticker)
        if row is None:
            row = {
                "ticker": ticker,
                "name": name or ticker,
                "sources": [],
                "weight_pct": 0.0,
                # 슬리브별 몫 — 현금 비중과 화면 요약이 이 값을 쓴다.
                "sm_weight": 0.0,
                "nh_weight": 0.0,
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
        row[f"{source}_weight"] += weight
        if row.get("price") is None:
            row["price"] = price
        if row.get("change_pct") is None:
            row["change_pct"] = change_pct
        row[f"{source}_status"] = status

    # 매도 예정(자격 상실·이탈)은 목표 비중 0 이다 — 다음 시가에 전량 팔고 그 슬롯은
    # 다음 교체까지 현금이다. 비중을 남겨두면 팔아야 할 종목의 매매수량이 0 으로 보인다.
    for row in sm_selected:
        ticker = str(row["ticker"]).strip()
        if ticker in sm_current_tickers:
            streak = row.get("streak_weeks")
            status = f"유지 ({streak}주째)" if streak else "유지"
        else:
            status = f"매수 예정 ({sm.get('rebalance_date')} 시가)"
        exiting = bool(row.get("is_exit_pending"))
        if exiting:
            status += f" · 매도 예정({row.get('exit_reason') or '주중 이탈'})"
        weight = 0.0 if exiting else sm_drift_weight.get(ticker, weight_sm)
        add_target(ticker, row.get("name"), "sm", weight, row.get("price"), row.get("daily_change_pct"), status)
    for row in nh_holdings:
        status = "오늘 진입" if row.get("is_new") else f"{row.get('days')}일째"
        exiting = str(row.get("status")) == "sell"
        if exiting:
            reason = str(row.get("exit_reason") or "이탈")
            status += f" · 매도 예정({reason})"
        ticker_nh = str(row["ticker"]).strip()
        weight = 0.0 if exiting else nh_drift_weight.get(ticker_nh, weight_nh)
        add_target(ticker_nh, row.get("name"), "nh", weight, row.get("price"), row.get("change_pct"), status)
    for row in nh_planned:
        add_target(
            row["ticker"],
            row.get("name"),
            "nh",
            weight_nh,
            row.get("price"),
            row.get("change_pct"),
            "진입 예정 (다음 시가 매수)",
        )

    sm_active = [row for row in sm_selected if not row.get("is_exit_pending")]
    nh_active = [row for row in nh_holdings if str(row.get("status")) != "sell"]

    # 매월 첫 거래일 = 슬리브 배분 리밸런싱 날 (그 시장 달력 기준).
    from config import MARKET_SCHEDULES
    from utils.settings_loader import get_ticker_type_settings
    from utils.trading_calendar import get_trading_days

    pool_settings = get_ticker_type_settings(pool_norm) or {}
    country = str(pool_settings.get("country_code") or "kor").strip().lower()
    # 종목 가격의 통화 — 계좌 원장(원화)과 맞추려면 환율이 필요하다.
    pool_currency = str(pool_settings.get("currency") or "KRW").strip().upper()
    tz_name = str((MARKET_SCHEDULES.get(country) or {}).get("timezone") or "Asia/Seoul")
    # 과거 날짜 조회면 그 날짜 기준으로 첫 거래일 여부를 판정한다.
    today_local = pd.Timestamp(as_of).date() if as_of else pd.Timestamp.now(tz=tz_name).date()
    month_start = today_local.replace(day=1)
    month_days = get_trading_days(month_start.strftime("%Y-%m-%d"), today_local.strftime("%Y-%m-%d"), country)
    sleeve_rebalance_today = bool(month_days) and month_days[0].date() == today_local

    # ── 월초 배분 되돌리기는 **현금 우선**으로 이관한다 (백테스트와 같은 규칙) ──
    # 종목 비중은 흘러간 그대로 두고(오르는 종목 유지), 장부상 현금만 슬리브 사이에서
    # 옮긴다. 한 슬리브의 주식만으로 제 몫을 넘을 때만 — 현금으로 이관액을 다 못 채울
    # 때만 — 초과분을 비례 매도한다. 그래서 여기서는 주식 비중을 그 몫에 맞춰 깎고,
    # 슬리브 몫 표시는 월초 배분으로 되돌린다. 비워 두는 현금 몫도 여기서 함께 복구된다.
    if sleeve_rebalance_today:
        for prefix, target in (("sm", base_weights["sm_pct"]), ("nh", base_weights["nh_pct"])):
            sleeve_stock = sum(row[f"{prefix}_weight"] for row in holdings)
            if sleeve_stock > target:
                scale = (target / sleeve_stock) if sleeve_stock > 0 else 0.0
                for row in holdings:
                    trimmed = row[f"{prefix}_weight"] * scale
                    row["weight_pct"] += trimmed - row[f"{prefix}_weight"]
                    row[f"{prefix}_weight"] = trimmed
        sm_share = base_weights["sm_pct"]
        nh_share = base_weights["nh_pct"]
        reserved_cash_share = base_weights["cash_pct"]

    stock_pct = sum(row["weight_pct"] for row in holdings)
    # 슬리브 현금 = 그 슬리브 몫에서 담긴 종목 비중을 뺀 나머지. 빈 슬롯 수로 세면
    # 흘러간 비중과 맞지 않는다(종목이 오르면 남는 현금은 그만큼 줄어든다).
    sm_stock = sum(row["sm_weight"] for row in holdings)
    nh_stock = sum(row["nh_weight"] for row in holdings)
    sm_cash = max(sm_share - sm_stock, 0.0)
    nh_cash = max(nh_share - nh_stock, 0.0)

    # 다음 거래일 — 모든 체결은 시가라 액션 묶음의 실제 날짜가 된다. 연휴가 끼면
    # 이 날짜가 모멘텀 교체일과 같아질 수 있고, 그러면 화면이 한 묶음으로 합친다.
    ahead = get_trading_days(
        today_local.strftime("%Y-%m-%d"), (today_local + timedelta(days=21)).strftime("%Y-%m-%d"), country
    )
    next_trading_day = next((str(day.date()) for day in ahead if day.date() > today_local), None)

    # ── 적용 계좌 — 이 풀로 합성을 운용하는 계좌(계좌 설정의 mix_pool)를 찾는다.
    account_id = next((row["account_id"] for row in mix_accounts() if row["pool"] == pool_norm), "")
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
        # 계좌 원장의 현금은 원화다 — 종목 평가액도 원화로 맞춰야 총자산이 성립한다.
        # (환율을 못 받으면 0 이라 평가액이 비고, 목표 수량도 계산하지 않는다.)
        krw_rate = _krw_rate(pool_currency)
        stock_value = 0.0
        for ticker, item in account["holdings"].items():
            price = price_by_ticker.get(ticker)
            # 화면에 보이는 현재가는 그 시장 통화 그대로 둔다(달러 종목은 달러로 본다).
            item["price"] = round(float(price), 4) if price else None
            value_krw = item["quantity"] * float(price) * krw_rate if price and krw_rate > 0 else None
            item["value"] = round(value_krw, 2) if value_krw is not None else None
            if value_krw:
                stock_value += value_krw
        total_assets = stock_value + account["cash_balance"]
        account["stock_value"] = round(stock_value, 2)
        account["total_assets"] = round(total_assets, 2)
        account["sell_all"] = _attach_account_targets(holdings, account, krw_rate)

    # 종목명 옆 추세 이탈 배지(❗)용 — 종목풀 설정 이평선 기준 이격.
    _attach_disparity(holdings, sm_settings)

    payload = {
        "computed_at": datetime.now().astimezone().isoformat(),
        "pool": pool_norm,
        # 화면이 표시용 시세를 60초마다 갱신할 때 쓴다(시세 소스가 국가별로 다르다).
        "country": country,
        "account": account,
        "as_of": nh.get("as_of"),
        "next_trading_day": next_trading_day,
        "live": bool(nh.get("live")),
        # 과거 날짜 셀렉트용 — 신고가 화면과 같은 날짜 목록을 그대로 쓴다.
        "available_dates": [str(d.get("date")) for d in (nh.get("available_dates") or [])],
        "summary": {
            "stock_pct": round(stock_pct, 2),
            "cash_pct": round(100 - stock_pct, 2),
            # 총 현금 중 **두 전략에 아예 주지 않고 비워 둔 몫**. 나머지는 빈 슬롯에서 생긴다.
            "reserved_cash_pct": round(reserved_cash_share, 2),
            # 월초에 되돌릴 배분 — 화면이 "지금 몫"과 "목표 배분"을 함께 보여준다.
            "base_weights": {
                "sm_pct": round(base_weights["sm_pct"], 2),
                "nh_pct": round(base_weights["nh_pct"], 2),
                "cash_pct": round(base_weights["cash_pct"], 2),
            },
            # slots_used = 목표가 찬 슬롯, held_count = 지금 실제로 들고 있는 종목 수.
            # 둘이 다르면 아직 체결 전이라는 뜻이라 화면이 구분해서 보여준다.
            "sm": {
                "alloc_pct": round(sm_share, 2),
                "slots_used": len(sm_active),
                "held_count": len(sm_current),
                "top_n": sm_top_n,
                "cash_pct": round(sm_cash, 2),
            },
            "nh": {
                "alloc_pct": round(nh_share, 2),
                "slots_used": len(nh_active) + len(nh_planned),
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
                    # 발동 사유 — 손절선과 이평선 이탈을 구분한다(판정 함수가 정한 값).
                    "reason": {"주중 손절": "주중 손절선 하회", "주중 이탈": "자격 상실(이평선 하회)"}.get(
                        str(row.get("exit_reason") or ""), "자격 상실(이평선 하회)"
                    ),
                }
                for row in sm_sell_pending
            ],
            # SM 주중 이탈 **예상** — 현재(장중) 가격 기준으로 보유 자격을 잃은 종목.
            # 판정은 오늘 종가로 확정되므로 화면에만 '(예상)' 그룹으로 보여주고 알람은 안 보낸다.
            # 종가 확정·캐시 갱신 후에는 위 sm_sells(확정)로 올라온다.
            "sm_exit_forecast": [
                {
                    "ticker": str(row["ticker"]).strip(),
                    "name": row.get("name") or row["ticker"],
                    "reason": (
                        "주중 손절 예상"
                        if (
                            sm_settings.get("intraweek_stop_pct") is not None
                            and row.get("entry_return_pct") is not None
                            and float(row["entry_return_pct"]) <= float(sm_settings["intraweek_stop_pct"])
                        )
                        else "주중 이탈 예상(이평선 하회)"
                    ),
                }
                for row in sm_selected
                if bool(sm_settings.get("intraweek_exit"))
                and not row.get("is_exit_pending")
                and not row.get("is_exited")
                and row.get("current_long_pct") is not None
                and row.get("current_short_pct") is not None
                and (float(row["current_long_pct"]) <= 0 or float(row["current_short_pct"]) < 0)
            ]
            if not as_of
            else [],
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
    # 주중 이탈 예상 — 표의 매매수량·상태 칸에 예상을 겹쳐 보여주기 위한 행 플래그.
    # 목표수량·목표비중은 확정 기준 그대로 둔다(장중 값으로 표 전체를 뒤집지 않는다).
    forecast_exit_tickers = {row["ticker"] for row in payload["actions"].get("sm_exit_forecast") or []}
    for row in payload["holdings"]:
        if row["ticker"] in forecast_exit_tickers and float(row.get("held_quantity") or 0) > 0:
            row["is_exit_forecast"] = True
            row["forecast_trade_quantity"] = -int(float(row["held_quantity"]))

    # 오늘의 액션 — 화면·슬랙 알람이 같은 결과를 쓴다(조립 단일 소스).
    payload["actions"]["groups"] = _build_action_groups(payload["holdings"], payload["actions"], next_trading_day)
    # 다음주 교체 가정 미리보기 — 실시간 순위 기준 잠정치라 과거 재현(as_of)에는 없다.
    payload["actions"]["next_week_preview"] = (
        _build_next_week_preview(
            sm, payload["holdings"], payload["actions"], weight_sm, account, ahead, today_local
        )
        if not as_of
        else None
    )
    return payload


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


def _simulate_mix_daily(
    sm: dict[str, Any],
    nh_settings: dict[str, Any],
    nh_context: dict[str, Any],
    months: int,
) -> Any:  # 반환은 pandas.Series — 임포트는 함수 안에서 한다(모듈 로드를 가볍게 유지)
    """한 계좌(현금·주수)에서 두 슬리브를 함께 굴린 일별 자산 곡선(시작 1.0).

    - 모멘텀: **엔진 체결 내역으로 보유를 복원**한다(진입·청산일이 명시돼 있고, 선정은
      순위 기반이라 포지션 크기와 무관 — 후보 재계산 없이 정확히 재현된다).
      교체일마다 슬리브 자산/N 으로 동일가중, 주중 매도는 체결일에 판다.
    - 신고가: 신호(돌파·손절·이탈)에서 다시 판정한다 — 진입 가능 여부가 슬리브
      현금에 달려 있어, 이관으로 현금이 달라지면 엔진 체결 내역과 어긋날 수 있다.
    - 매월 첫 거래일 시가에 계좌 설정 배분으로 — **현금 우선** 이관. 넘기는 쪽 현금부터 쓰고,
      모자랄 때만 주식을 비례 매도한다. 받는 쪽은 현금으로만 받는다(기존 보유 불변).
    모든 체결은 시가, 편도 슬리피지를 물린다.
    """
    import pandas as pd

    from utils.momentum_backtest import _rebalance_dates
    from utils.new_high_backtest import _cap_by_industry, _meets_min_mult
    from utils.pool_settings_store import get_pool_slippage

    pool = nh_context["pool"]
    panel, signals = nh_context["panel"], nh_context["signals"]
    close_df, open_df = panel["close"], panel["open"]
    breakout, below_ma = signals["breakout"], signals["below_ma"]
    value_mult = signals["value_mult"]
    industry_by = nh_context["industry_by"]

    buy_slip, sell_slip = get_pool_slippage(pool)
    buy_slip, sell_slip = buy_slip / 100.0, sell_slip / 100.0
    nh_slots = int(nh_settings["top_n"])
    stop_pct = float(nh_settings["stop_loss_pct"])
    min_mult = nh_settings["min_value_mult"]

    # ── 모멘텀 이벤트 복원 — 체결 목록의 진입·청산일이 곧 매매 일정이다 ──
    sm_slots = 0
    sm_buys: dict[str, list[str]] = {}
    sm_sells: dict[str, list[str]] = {}
    for trade in sm["trades"]:
        sm_buys.setdefault(trade["entry_date"], []).append(trade["ticker"])
        if trade.get("exit_date"):
            sm_sells.setdefault(trade["exit_date"], []).append(trade["ticker"])
    from utils.momentum_service import load_benchmark_close

    benchmark_close = load_benchmark_close(pool)
    rebalance_days = {str(d.date()) for d in _rebalance_dates(benchmark_close, months)}
    from utils.momentum_service import load_settings as _sm_load

    sm_slots = int(_sm_load(pool)["top_n"])

    first = pd.Timestamp(min(sm_buys)) if sm_buys else close_df.index[0]
    span = [d for d in close_df.index if d >= first]

    # 시작 배분 — 계좌 설정의 합성 배분(합 100)을 1.0 기준으로 쪼갠다.
    weights = mix_weights_for_pool(pool)
    sm_w, nh_w, cash_w = weights["sm_pct"] / 100.0, weights["nh_pct"] / 100.0, weights["cash_pct"] / 100.0

    sm_shares: dict[str, float] = {}
    nh_shares: dict[str, float] = {}
    nh_entry: dict[str, float] = {}
    cash_sm, cash_nh = sm_w, nh_w
    # 두 전략에 주지 않고 비워 두는 몫 — 자라지 않고, 월초에만 다시 맞춘다.
    cash_reserved = cash_w
    prev_month: str | None = None
    curve: dict[str, float] = {}

    def px(df: pd.DataFrame, day: pd.Timestamp, ticker: str) -> float | None:
        if ticker not in df.columns:
            return None
        value = df.at[day, ticker]
        return float(value) if pd.notna(value) else None

    def sleeve_value(shares: dict[str, float], cash: float, day: pd.Timestamp, df: pd.DataFrame) -> float:
        total = cash
        for ticker, qty in shares.items():
            price = px(df, day, ticker) or px(close_df, day, ticker)
            if price:
                total += qty * price
        return total

    for i in range(1, len(span)):
        prev, day = span[i - 1], span[i]
        day_key = str(day.date())

        # 1) 신고가 청산 — prev 종가 판정 → 오늘 시가 체결
        for ticker in list(nh_shares):
            price = px(close_df, prev, ticker)
            if price is None:
                continue
            hit_stop = (price / nh_entry[ticker] - 1) * 100 <= stop_pct
            if not (hit_stop or bool(below_ma.at[prev, ticker])):
                continue
            fill = px(open_df, day, ticker) or price
            cash_nh += nh_shares.pop(ticker) * fill * (1 - sell_slip)
            nh_entry.pop(ticker, None)

        # 2) 모멘텀 매도 체결(교체 편출·주중 이탈 — 체결 내역의 청산일)
        for ticker in sm_sells.get(day_key, []):
            if ticker in sm_shares:
                fill = px(open_df, day, ticker) or px(close_df, prev, ticker)
                if fill:
                    cash_sm += sm_shares.pop(ticker) * fill * (1 - sell_slip)
                else:
                    sm_shares.pop(ticker)

        # 3) 월초 배분 되돌리기 — 현금 우선 이관(시가). 교체·진입보다 먼저.
        #    넘치는 슬리브에서 뽑아 한 곳에 모은 뒤 모자란 슬리브에 현금으로만 넘긴다.
        #    남는 것이 비워 두는 현금 몫이 된다(대수적으로 목표와 정확히 일치한다).
        if prev_month is not None and day_key[:7] != prev_month:
            sm_value = sleeve_value(sm_shares, cash_sm, day, open_df)
            nh_value = sleeve_value(nh_shares, cash_nh, day, open_df)
            total_value = sm_value + nh_value + cash_reserved
            target_sm, target_nh = total_value * sm_w, total_value * nh_w
            pool_cash = cash_reserved
            cash_reserved = 0.0

            for shares, value, cash_now, target in (
                (sm_shares, sm_value, cash_sm, target_sm),
                (nh_shares, nh_value, cash_nh, target_nh),
            ):
                excess = value - target
                if excess <= 1e-12:
                    continue
                from_cash = min(cash_now, excess)
                remain = excess - from_cash
                proceeds = 0.0
                stock_value = value - cash_now
                if remain > 1e-12 and stock_value > 0:
                    fraction = min(remain / stock_value, 1.0)
                    for ticker in list(shares):
                        fill = px(open_df, day, ticker) or px(close_df, prev, ticker)
                        if not fill:
                            continue
                        sell_qty = shares[ticker] * fraction
                        proceeds += sell_qty * fill * (1 - sell_slip)
                        shares[ticker] -= sell_qty
                pool_cash += from_cash + proceeds
                if shares is sm_shares:
                    cash_sm = cash_now - from_cash
                else:
                    cash_nh = cash_now - from_cash

            for name, value, target in (("sm", sm_value, target_sm), ("nh", nh_value, target_nh)):
                need = target - value
                if need <= 1e-12:
                    continue
                give = min(need, pool_cash)
                pool_cash -= give
                if name == "sm":
                    cash_sm += give
                else:
                    cash_nh += give

            cash_reserved = pool_cash
        prev_month = day_key[:7]

        # 4) 모멘텀 편입 + 교체일 동일가중(슬리브 자산/N, 시가)
        for ticker in sm_buys.get(day_key, []):
            sm_shares.setdefault(ticker, 0.0)
        if day_key in rebalance_days:
            sm_value = sleeve_value(sm_shares, cash_sm, day, open_df)
            unit = sm_value / sm_slots if sm_slots else 0.0
            for ticker in list(sm_shares):
                fill = px(open_df, day, ticker)
                if not fill:
                    continue
                delta = unit / fill - sm_shares[ticker]
                if delta > 0:
                    cost = delta * fill * (1 + buy_slip)
                    if cost > cash_sm:
                        delta = cash_sm / (fill * (1 + buy_slip))
                        cost = delta * fill * (1 + buy_slip)
                    cash_sm -= cost
                    sm_shares[ticker] += delta
                elif delta < 0:
                    cash_sm += -delta * fill * (1 - sell_slip)
                    sm_shares[ticker] += delta

        # 5) 신고가 진입 — prev 돌파 → 오늘 시가, 배정은 min(슬리브/N, 현금)
        free = nh_slots - len(nh_shares)
        if free > 0:
            row = breakout.loc[prev]
            picks = [
                t
                for t in row[row].index
                if t not in nh_shares
                and px(open_df, day, t) is not None
                and _meets_min_mult(value_mult.at[prev, t], min_mult)
            ]
            score_row = value_mult.loc[prev]

            def nh_priority(ticker: str) -> float:
                score = score_row.get(ticker)
                return float(score) if pd.notna(score) else 0.0

            picks.sort(key=nh_priority, reverse=True)
            picks, _ = _cap_by_industry(picks, list(nh_shares), industry_by, nh_settings["max_per_industry"], free)
            nh_open_value = sleeve_value(nh_shares, cash_nh, day, open_df)
            for ticker in picks:
                alloc = min(nh_open_value / nh_slots, cash_nh)
                if alloc <= 0:
                    break
                fill = px(open_df, day, ticker) * (1 + buy_slip)
                nh_shares[ticker] = alloc / fill
                nh_entry[ticker] = fill
                cash_nh -= alloc

        # 비워 둔 현금도 계좌 자산이다 — 곡선에서 빼면 배분을 늘릴수록 총자산이 줄어 보인다.
        curve[day_key] = (
            sleeve_value(sm_shares, cash_sm, day, close_df)
            + sleeve_value(nh_shares, cash_nh, day, close_df)
            + cash_reserved
        )

    return pd.Series(curve).sort_index()


def run_mix_backtest(pool: str | None = None, months: int | None = None) -> dict[str, Any]:
    """선택한 풀의 저장 설정으로 SM·신고가 백테스트를 돌려 합성 결과를 만든다.

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
    allowed = month_options()
    if months not in allowed:
        raise ValueError(f"'months' 는 {allowed} 중 하나여야 합니다 (받은 값: {months})")

    logger.info("[STRATEGY-MIX] %s 합성 백테스트 시작 (%d개월)", pool_norm, months)
    from utils.new_high_backtest import load_context as nh_load_context

    nh_context = nh_load_context(nh_settings)
    sm = sm_backtest(months, sm_settings, include_daily=True)
    nh = nh_backtest(months, nh_settings, nh_context)

    # 벤치마크 곡선은 SM 결과의 것(같은 풀). SM daily 는 일간 변동률(%)·최신 날짜가 앞.
    bench_curve: dict[str, float] = {}
    b_value = 1.0
    for row in sorted(sm["daily"], key=lambda r: r["date"]):
        if row.get("benchmark_pct") is not None:
            b_value *= 1 + row["benchmark_pct"] / 100
        bench_curve[row["date"]] = b_value

    # 합성 곡선 — 한 계좌 금액 기반 시뮬레이션(월초 배분 복구는 현금 우선 이관).
    mix_curve = _simulate_mix_daily(sm, nh_settings, nh_context, months)
    dates = [d for d in mix_curve.index if d in bench_curve]
    if len(dates) < 2:
        raise RuntimeError("두 전략의 공통 백테스트 구간이 부족합니다.")

    first_mix = float(mix_curve[dates[0]])
    first_bench = bench_curve[dates[0]]
    daily_rows = [
        {
            "date": date,
            "strategy_pct": round((float(mix_curve[date]) / first_mix - 1) * 100, 2),
            "benchmark_pct": round((bench_curve[date] / first_bench - 1) * 100, 2),
        }
        for date in dates
    ]

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
    merged_trades = _merge_trades(sm, nh)

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
        "trades": merged_trades,
        # 거래 수·승률·평균 손익 — 각 전략 화면과 같은 공용 계산.
        **summarize_trades(merged_trades),
    }
