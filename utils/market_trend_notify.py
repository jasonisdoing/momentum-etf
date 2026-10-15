"""시장 추세 알림 — 지수·환율의 현재 추세를 나열하고, 어제와 달라진 것이 있으면 @channel 로 알린다.

대상은 `config.MARKET_TREND_ALERT_TICKERS`(코스피~미국달러), 추세는 /market-trend 의 SuperTrend
방향(상승/하락)이다. 이전 값은 ``system_config.market_trend_alert.regimes`` 에 {티커: 방향} 으로 남기며,
**발송에 성공했을 때만** 갱신한다(실패하면 다음 실행이 같은 변화를 다시 알린다). 비교 기준이 아직
없는 첫 실행은 목록만 보내고 @channel 은 붙이지 않는다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from config import MARKET_TREND_ALERT_TICKERS

KST = ZoneInfo("Asia/Seoul")
_CONFIG_COLLECTION = "system_config"
_STATE_KEY = "market_trend_alert"
_REGIME_LABEL = {"accel_up": "상승", "accel_down": "하락"}
_REGIME_EMOJI = {"accel_up": ":small_red_triangle:", "accel_down": ":chart_with_downwards_trend:"}


def _load_previous() -> dict[str, str] | None:
    """직전에 알린 추세({티커: 방향}). 한 번도 안 보냈으면 None."""
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패해 이전 추세를 읽을 수 없습니다.")
    doc = db[_CONFIG_COLLECTION].find_one({"_id": _STATE_KEY}) or {}
    regimes = doc.get("regimes")
    return dict(regimes) if isinstance(regimes, dict) else None


def _save_current(regimes: dict[str, str]) -> None:
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패해 추세를 저장할 수 없습니다.")
    db[_CONFIG_COLLECTION].update_one(
        {"_id": _STATE_KEY},
        {"$set": {"regimes": regimes, "updated_at": datetime.now(KST).isoformat()}},
        upsert=True,
    )


def _current_items() -> list[dict[str, Any]]:
    """알림 대상의 현재 추세 — /market-trend 와 같은 계산. 하나라도 비면 에러(임의 값 없음)."""
    from utils.market_trend_service import compute_market_trend

    by_ticker = {item["ticker"]: item for item in compute_market_trend()["items"]}
    items: list[dict[str, Any]] = []
    for ticker in MARKET_TREND_ALERT_TICKERS:
        item = by_ticker.get(ticker)
        if item is None or item.get("current_regime") not in _REGIME_LABEL:
            raise RuntimeError(f"시장 추세를 계산하지 못했습니다: {ticker}")
        items.append(item)
    return items


def build_message(items: list[dict[str, Any]], previous: dict[str, str] | None) -> tuple[str, list[str]]:
    """슬랙 본문과 달라진 지수 이름 목록. 이전 값이 없으면 달라진 것은 없다(비교 기준 없음)."""
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
    # (이름, 이전 방향, 지금 방향) — 이전 값이 없으면 비교하지 않는다.
    flips = [
        (item["name"], previous[item["ticker"]], item["current_regime"])
        for item in items
        if previous is not None
        and previous.get(item["ticker"]) in _REGIME_LABEL
        and previous[item["ticker"]] != item["current_regime"]
    ]
    changed = [f"{name} {_REGIME_LABEL[before]}→{_REGIME_LABEL[after]}" for name, before, after in flips]
    changed_names = {name for name, _, _ in flips}

    lines = []
    if changed:
        lines.append(f"<!channel> 🔔 *추세 전환 {len(changed)}건* — {', '.join(changed)}")
    lines.append(f"*📊 시장 추세* ({now})" + ("" if previous is not None else " · 첫 실행이라 어제와 비교하지 않습니다"))
    for item in items:
        regime = item["current_regime"]
        days = item.get("current_regime_days")
        move = item.get("regime_change_pct")
        text = f"{_REGIME_EMOJI[regime]} *{item['name']}* {_REGIME_LABEL[regime]}"
        if days is not None:
            text += f" {days}일째"
        if move is not None:
            text += f" ({move:+.1f}%)"
        if item["name"] in changed_names:
            text += " 🔔 전환"
        lines.append(text)
    return "\n".join(lines), changed


def notify_market_trend(*, dry_run: bool = False) -> dict[str, Any]:
    """추세를 나열해 보내고, 어제와 달라진 것이 있으면 @channel 로 알린다. ``dry_run`` 은 보내지도 저장하지도 않는다."""
    from utils.notification import send_slack_message_v2

    items = _current_items()
    previous = _load_previous()
    message, changed = build_message(items, previous)
    if dry_run:
        return {"sent": False, "message": message, "changed": changed}
    if not send_slack_message_v2(message):
        raise RuntimeError("슬랙 전송에 실패했습니다.")
    _save_current({item["ticker"]: item["current_regime"] for item in items})
    return {"sent": True, "message": message, "changed": changed}
