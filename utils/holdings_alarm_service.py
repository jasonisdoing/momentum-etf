"""보유종목 알람 서비스 (화면·배치 공용).

여러 알람 종류를 한 번에 처리해 **한 건의 슬랙 메시지**로 보낸다.
- 이동선 이탈: 보유 종가 < 이동평균(계좌별 이평선 일수)
- 손절: 보유 수익률 <= 계좌별 손절 기준(예: -7%)

설정은 **계좌별**이다(각 계좌 문서):
    ma20_alarm_enabled / ma20_ma_days,  stoploss_alarm_enabled / stoploss_threshold_pct
계좌마다 On/Off 와 기준(이평선 일수·손절 %)을 다르게 둘 수 있다. 새 알람 종류는 계산 로직만
추가하면 함께 발송된다. 이격은 종목풀 순위와 동일하게 실시간 스냅샷을 종가에 반영해 계산한다.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from utils.account_settings_store import load_account_docs, save_account_settings
from utils.data_loader import fetch_ohlcv
from utils.holdings_detail_service import load_all_holdings_detail
from utils.logger import get_app_logger
from utils.moving_averages import calculate_moving_average, get_moving_average_type
from utils.notification import send_slack_message_v2
from utils.rankings import build_effective_close_series

logger = get_app_logger()

_WARMUP_EXTRA_BDAYS = 40
_DEFAULT_MA_DAYS = 20
_DEFAULT_STOPLOSS_PCT = -7.0
# 화면 셀렉트 선택지(백엔드는 값만 검증하고, 목록은 화면과 공유)
MA_DAYS_OPTIONS: tuple[int, ...] = (5, 10, 20, 40, 60, 120, 200)
STOPLOSS_PCT_OPTIONS: tuple[float, ...] = (-3.0, -5.0, -7.0, -10.0, -15.0, -20.0)


# 화면 배지용 기본 아이콘 — 계좌별로 저장해 덮어쓸 수 있다(빈값 저장 = 배지 미표시).
_DEFAULT_MA_ICON = "🚫"
_DEFAULT_STOPLOSS_ICON = "✂️"


def _account_ma_days(doc: dict[str, Any]) -> int:
    value = doc.get("ma20_ma_days")
    return int(value) if isinstance(value, (int, float)) and int(value) >= 2 else _DEFAULT_MA_DAYS


def _account_ma_icon(doc: dict[str, Any]) -> str:
    value = doc.get("ma20_alarm_icon")
    return str(value).strip() if isinstance(value, str) else _DEFAULT_MA_ICON


def _account_stoploss_icon(doc: dict[str, Any]) -> str:
    value = doc.get("stoploss_alarm_icon")
    return str(value).strip() if isinstance(value, str) else _DEFAULT_STOPLOSS_ICON


def _account_stoploss_pct(doc: dict[str, Any]) -> float:
    value = doc.get("stoploss_threshold_pct")
    return float(value) if isinstance(value, (int, float)) and float(value) < 0 else _DEFAULT_STOPLOSS_PCT


def _safe_realtime_snapshot(country: str, tickers: list[str]) -> dict[str, dict[str, float]]:
    """실시간 스냅샷(종목풀 순위와 동일 소스). 실패 시 빈 dict."""
    from services.price_service import get_realtime_snapshot

    if not tickers:
        return {}
    try:
        return get_realtime_snapshot(country, tickers)
    except Exception as exc:
        logger.warning("[HOLDINGS ALARM] 실시간 스냅샷 조회 실패 (%s): %s", country, exc)
        return {}


def _ma_status(
    fetch_ticker: str,
    ticker_type: str,
    country: str,
    ma_days: int,
    realtime_entry: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    """종가와 기준 이평선을 계산해 이탈 여부를 반환한다. 데이터 부족/실패 시 None.

    종목풀 순위 화면과 **동일하게** 실시간 현재가(스냅샷)를 종가 시리즈에 덮어씌워 계산한다
    (``build_effective_close_series``). 실시간이 없으면 캐시 일봉 종가만으로 계산한다.
    """
    start = (pd.Timestamp.today().normalize() - pd.offsets.BDay(ma_days + _WARMUP_EXTRA_BDAYS)).strftime("%Y-%m-%d")
    df = fetch_ohlcv(fetch_ticker, country, months_back=None, date_range=[start, None], ticker_type=ticker_type)
    if df is None or df.empty or "Close" not in df.columns:
        return None
    close = df["Close"].astype(float).dropna()
    close.index = pd.to_datetime(close.index)
    effective = build_effective_close_series(close, realtime_entry)
    if effective is not None and not effective.empty:
        close = effective
    if len(close) < ma_days:
        return None
    # 이동평균 종류는 config(SMA/EMA). 최신 봉의 이동평균 값으로 이탈을 판정한다.
    sma = float(calculate_moving_average(close, ma_days, min_periods=ma_days).iloc[-1])
    if sma == 0.0:
        return None
    last = float(close.iloc[-1])
    return {
        "last": last,
        "sma": round(sma, 2),
        "deviation_pct": round((last / sma - 1.0) * 100.0, 2),
        "below": last < sma,
    }


def compute_account_alerts(account_doc: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """한 계좌의 (켜진) 알람 종류별 트리거와 실패 사유를 계좌 자체 기준으로 계산한다."""
    account_id = account_doc["account_id"]
    detail = load_all_holdings_detail(account_id)
    rows = [r for r in detail.get("rows", []) if str(r.get("ticker") or "").strip() and str(r.get("ticker")).strip() != "IS"]

    ma_days = _account_ma_days(account_doc)
    threshold = _account_stoploss_pct(account_doc)
    ma20_on = bool(account_doc.get("ma20_alarm_enabled"))
    stoploss_on = bool(account_doc.get("stoploss_alarm_enabled"))

    ma20_hits: list[dict[str, Any]] = []
    stoploss_hits: list[dict[str, Any]] = []
    errors: list[str] = []

    # 손절: 보유 수익률(return_pct) 기준 — 시세 재조회 불필요.
    if stoploss_on:
        for row in rows:
            ret = row.get("return_pct")
            if isinstance(ret, (int, float)) and not isinstance(ret, bool) and ret <= threshold:
                stoploss_hits.append({
                    "ticker": str(row["ticker"]).strip(),
                    "name": str(row.get("name") or row["ticker"]),
                    "return_pct": round(float(ret), 2),
                })

    # 이동선 이탈: 종가 vs 이동평균. 종목풀 순위와 동일하게 실시간 스냅샷을 국가별로 미리 조회해 반영.
    if ma20_on:
        candidates: list[tuple[dict[str, Any], str, str, str]] = []  # (row, fetch_ticker, ticker_type, country)
        by_country: dict[str, list[str]] = {}
        for row in rows:
            ticker_type = str(row.get("ticker_type") or "").strip()
            country = str(row.get("country_code") or "").strip().lower()
            if not ticker_type or not country:
                errors.append(f"{str(row['ticker']).strip()}: ticker_type/country 정보 없음")
                continue
            fetch_ticker = str(row["ticker"]).strip().split(":")[-1]  # 'ASX:XXX' → 'XXX'
            candidates.append((row, fetch_ticker, ticker_type, country))
            by_country.setdefault(country, []).append(fetch_ticker)

        snapshots = {c: _safe_realtime_snapshot(c, tks) for c, tks in by_country.items()}
        for row, fetch_ticker, ticker_type, country in candidates:
            entry = snapshots.get(country, {}).get(fetch_ticker)
            try:
                status = _ma_status(fetch_ticker, ticker_type, country, ma_days, entry)
            except Exception as exc:  # 한 종목 실패가 전체를 막지 않게
                errors.append(f"{fetch_ticker}: {exc}")
                status = None
            if status is None:
                errors.append(f"{fetch_ticker}: 가격 데이터 부족")
            elif status["below"]:
                ma20_hits.append({"ticker": str(row["ticker"]).strip(), "name": str(row.get("name") or row["ticker"]), **status})

    return {"ma20": ma20_hits, "stoploss": stoploss_hits, "ma_days": ma_days, "threshold": threshold}, errors


# 배지 계산은 보유 종목별 가격 시계열 조회라 수 초 걸린다 — 계좌별 TTL 캐시로 재계산을 줄인다.
# (알람 설정 저장 시 무효화. 판정 기반인 종가·실시간 스냅샷은 이 TTL 내 변화가 미미하다.)
_BADGES_CACHE_TTL_SECONDS = 120.0
_badges_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def _invalidate_badges_cache() -> None:
    _badges_cache.clear()


def compute_account_alert_badges(account_id: str) -> dict[str, Any]:
    """자산 화면(종목명 배지)용: 계좌의 알람 트리거를 티커→아이콘 문자열 맵으로 반환한다.

    슬랙 알람(compute_account_alerts)과 **같은 설정·같은 판정**을 쓴다. 꺼진 알람 종류나
    아이콘이 빈값(미설정)이면 해당 배지는 붙지 않는다. 티커는 접두사 없는 형태로 정규화한다.
    """
    norm_id = str(account_id or "").strip().lower()
    import time as _time

    cached = _badges_cache.get(norm_id)
    if cached is not None and _time.monotonic() - cached[0] < _BADGES_CACHE_TTL_SECONDS:
        return dict(cached[1])
    account_doc = next((doc for doc in load_account_docs() if doc["account_id"] == norm_id), None)
    if account_doc is None:
        raise ValueError(f"알 수 없는 계좌입니다: {account_id}")

    ma_icon = _account_ma_icon(account_doc)
    stoploss_icon = _account_stoploss_icon(account_doc)
    alerts, _errors = compute_account_alerts(account_doc)

    badge_by_ticker: dict[str, str] = {}

    def _norm_ticker(value: str) -> str:
        return str(value or "").strip().upper().split(":")[-1]

    if ma_icon:
        for hit in alerts["ma20"]:
            key = _norm_ticker(hit["ticker"])
            if key:
                badge_by_ticker[key] = badge_by_ticker.get(key, "") + ma_icon
    if stoploss_icon:
        for hit in alerts["stoploss"]:
            key = _norm_ticker(hit["ticker"])
            if key:
                badge_by_ticker[key] = badge_by_ticker.get(key, "") + stoploss_icon

    result = {
        "account_id": norm_id,
        "badge_by_ticker": badge_by_ticker,
        "ma20_icon": ma_icon,
        "stoploss_icon": stoploss_icon,
        "ma_days": alerts["ma_days"],
        "threshold": alerts["threshold"],
    }
    _badges_cache[norm_id] = (_time.monotonic(), dict(result))
    return result


def get_alarm_view() -> dict[str, Any]:
    """알람 화면용: 계좌별 알람 On/Off + 기준(이평선 일수·손절 %) 목록(시세 계산 없음)."""
    return {
        "ma_days_options": list(MA_DAYS_OPTIONS),
        "stoploss_pct_options": list(STOPLOSS_PCT_OPTIONS),
        "ma_type": get_moving_average_type(),
        "accounts": [
            {
                "account_id": doc["account_id"],
                "name": str(doc.get("name") or doc["account_id"]),
                "icon": str(doc.get("icon") or ""),
                "order": int(doc.get("order") or 0),
                "ma20_enabled": bool(doc.get("ma20_alarm_enabled", False)),
                "ma20_ma_days": _account_ma_days(doc),
                "ma20_icon": _account_ma_icon(doc),
                "stoploss_enabled": bool(doc.get("stoploss_alarm_enabled", False)),
                "stoploss_threshold_pct": _account_stoploss_pct(doc),
                "stoploss_icon": _account_stoploss_icon(doc),
            }
            for doc in load_account_docs()
        ],
    }


def set_account_alarm(
    account_id: str, alarm_type: str, *, enabled: bool, value: float, icon: str | None = None
) -> dict[str, Any]:
    """계좌별 알람 On/Off + 기준값(+화면 배지 아이콘) 저장. alarm_type: 'ma20'|'stoploss'."""
    if alarm_type == "ma20":
        payload: dict[str, Any] = {"ma20_alarm_enabled": bool(enabled), "ma20_ma_days": int(value)}
        if icon is not None:
            payload["ma20_alarm_icon"] = str(icon).strip()
    elif alarm_type == "stoploss":
        payload = {"stoploss_alarm_enabled": bool(enabled), "stoploss_threshold_pct": float(value)}
        if icon is not None:
            payload["stoploss_alarm_icon"] = str(icon).strip()
    else:
        raise ValueError(f"알 수 없는 알람 종류입니다: {alarm_type}")
    save_account_settings(account_id, payload, save_method="알람 센터")
    _invalidate_badges_cache()  # 설정(기준·아이콘·On/Off) 변경 즉시 배지에 반영
    return get_alarm_view()


def _post_slack(sections: list[dict[str, Any]], *, manual: bool) -> bool:
    tag = "🖐 [수동]" if manual else "🔔"
    total = sum(len(s["ma20"]) + len(s["stoploss"]) for s in sections)
    header = f"{tag} 보유종목 알람"
    blocks: list[dict] = [{"type": "header", "text": {"type": "plain_text", "text": header, "emoji": True}}]
    for s in sections:
        parts: list[str] = [f"*{s['account']}*"]
        if s["ma20"]:
            parts.append(f"📉 *이동선 이탈* ({get_moving_average_type()} {s['ma_days']}일)")
            parts += [f"  • {b['name']}({b['ticker']}): 이격 {b['deviation_pct']:+.2f}%" for b in s["ma20"]]
        if s["stoploss"]:
            parts.append(f"🛑 *손절* ({s['threshold']:.1f}% 이하)")
            parts += [f"  • {b['name']}({b['ticker']}): 수익률 {b['return_pct']:+.2f}%" for b in s["stoploss"]]
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(parts)}})
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": f"총 {len(sections)}개 계좌 · {total}건"}]})

    ts = send_slack_message_v2(text=header, blocks=blocks)
    if ts:
        logger.info("[HOLDINGS ALARM] 발송 완료 (계좌 %d, 건수 %d)", len(sections), total)
        return True
    logger.warning("[HOLDINGS ALARM] 발송 실패")
    return False


def send_holdings_alarms(*, manual: bool = False) -> dict[str, Any]:
    """계좌별 켜진 알람 종류의 트리거를 계좌 자체 기준으로 모아 슬랙 1건으로 발송한다.

    트리거가 하나도 없으면 발송하지 않는다.
    """
    sections: list[dict[str, Any]] = []
    all_errors: list[str] = []
    for doc in load_account_docs():
        if not (doc.get("ma20_alarm_enabled") or doc.get("stoploss_alarm_enabled")):
            continue
        alerts, errors = compute_account_alerts(doc)
        all_errors.extend(f"{doc['account_id']}/{e}" for e in errors)
        if alerts["ma20"] or alerts["stoploss"]:
            sections.append({"account": str(doc.get("name") or doc["account_id"]), **alerts})

    if not sections:
        return {"sent": False, "reason": "알림 대상(이탈/손절 종목)이 없습니다.", "errors": all_errors}
    sent = _post_slack(sections, manual=manual)
    return {"sent": sent, "accounts": len(sections), "errors": all_errors}
