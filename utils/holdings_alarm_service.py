"""보유종목 알람 서비스 (화면·배치 공용).

여러 알람 종류를 한 번에 처리해 **한 건의 슬랙 메시지**로 보낸다.
- 이동선 이탈: 보유 종가가 **단기·장기 이평선 중 하나라도** 아래 (종목풀 순위 화면의
  회색 처리 기준과 같다 — 둘 중 하나만 꺾여도 보유 대상이 아니라고 본다)
- 손절: 보유 수익률 <= 계좌별 손절 기준(예: -7%)

**On/Off 는 계좌별**이다(각 계좌 문서의 ``ma_alarm_enabled`` / ``stoploss_alarm_enabled``).
기준은 알람 종류마다 소속이 다르다.
- 이평선 일수: **종목이 속한 종목풀**의 ``SHORT_MA_DAYS`` / ``LONG_MA_DAYS``.
  한 티커는 한 풀에만 들어가므로(``add_active_stock`` 이 중복을 막는다) 보유 종목의 소속 풀이
  유일하게 정해진다. 계좌가 여러 풀의 종목을 섞어 들고 있으면 종목마다 기준이 달라진다.
- 손절 기준(%): 계좌별(``stoploss_threshold_pct``).

기준을 정할 수 없는 종목은 조용히 넘기지 않고 **판정 불가**로 모아 슬랙에 함께 알린다
(소속 종목풀 없음 / 가격 데이터 부족). 이격은 종목풀 순위와 동일하게 실시간 스냅샷을 종가에
반영해 계산한다.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from config import CACHE_TTL_COMPUTE
from utils.account_settings_store import load_account_docs, save_account_settings
from utils.data_loader import fetch_ohlcv
from utils.holdings_detail_service import load_all_holdings_detail
from utils.logger import get_app_logger
from utils.moving_averages import calculate_moving_average, get_moving_average_type
from utils.notification import send_slack_message_v2
from utils.rankings import build_effective_close_series
from utils.stock_list_io import pools_by_ticker

logger = get_app_logger()

_WARMUP_EXTRA_BDAYS = 40
# 화면 셀렉트 선택지(백엔드는 값만 검증하고, 목록은 화면과 공유)
STOPLOSS_PCT_OPTIONS: tuple[float, ...] = (-7.0, -10.0)  # 전략 손절선 선택지와 통일

# 자산 화면 종목명 배지 아이콘. 계좌마다 다르게 둘 이유가 없어 코드에 고정한다.
_MA_ICON = "❗"  # 이동선 이탈(단기·장기 공용)
_STOPLOSS_ICON = "🚫"


def _pool_ma_days(pool: str) -> tuple[int, int]:
    """종목풀의 (단기, 장기) 이평선 일수. 설정이 없으면 에러 — 임의 기본값을 쓰지 않는다."""
    from utils.settings_loader import get_ticker_type_settings

    config = get_ticker_type_settings(pool) or {}
    short = config.get("SHORT_MA_DAYS")
    long = config.get("LONG_MA_DAYS")
    if not isinstance(short, (int, float)) or not isinstance(long, (int, float)):
        raise ValueError(f"종목풀 '{pool}' 에 이평선 설정이 없습니다.")
    return int(short), int(long)


def _pool_country(pool: str) -> str:
    """종목풀의 국가 코드. 종목이 실제로 거래되는 시장이라 시세 조회는 이 값을 쓴다."""
    from utils.settings_loader import get_ticker_type_settings

    country = str((get_ticker_type_settings(pool) or {}).get("country_code") or "").strip().lower()
    if not country:
        raise ValueError(f"종목풀 '{pool}' 에 국가 코드가 없습니다.")
    return country


def _account_stoploss_pct(doc: dict[str, Any]) -> float | None:
    """계좌의 손절 기준(%). 미설정/양수면 None — 판정 기준이 없다는 뜻."""
    value = doc.get("stoploss_threshold_pct")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) < 0:
        return float(value)
    return None


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
    ma_days: tuple[int, int],
    realtime_entry: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    """종가와 단기·장기 이평선을 계산해 이탈 여부를 반환한다. 데이터 부족/실패 시 None.

    종목풀 순위 화면과 **동일하게** 실시간 현재가(스냅샷)를 종가 시리즈에 덮어씌워 계산하고
    (``build_effective_close_series``), **하나라도 아래면 이탈**로 본다(순위 화면 회색 기준).
    실시간이 없으면 캐시 일봉 종가만으로 계산한다.

    둘 중 하나라도 이평선을 못 구하면(상장 직후 등) 판정 불가로 None — 절반만 보고
    이탈 여부를 정하면 장기가 꺾인 종목을 놓치기 때문이다.
    """
    short_days, long_days = ma_days
    warmup_days = max(short_days, long_days) + _WARMUP_EXTRA_BDAYS
    start = (pd.Timestamp.today().normalize() - pd.offsets.BDay(warmup_days)).strftime("%Y-%m-%d")
    df = fetch_ohlcv(fetch_ticker, country, months_back=None, date_range=[start, None], ticker_type=ticker_type)
    if df is None or df.empty or "Close" not in df.columns:
        return None
    close = df["Close"].astype(float).dropna()
    close.index = pd.to_datetime(close.index)
    effective = build_effective_close_series(close, realtime_entry)
    if effective is not None and not effective.empty:
        close = effective
    if len(close) < max(short_days, long_days):
        return None

    # 이동평균 종류는 config(SMA/EMA). 최신 봉의 이동평균 값으로 이탈을 판정한다.
    last = float(close.iloc[-1])
    result: dict[str, Any] = {"last": last}
    for label, days in (("short", short_days), ("long", long_days)):
        ma = float(calculate_moving_average(close, days, min_periods=days).iloc[-1])
        if ma == 0.0:
            return None
        result[f"{label}_days"] = days
        result[f"{label}_ma"] = round(ma, 2)
        result[f"{label}_deviation_pct"] = round((last / ma - 1.0) * 100.0, 2)
        result[f"{label}_below"] = last < ma

    result["below"] = bool(result["short_below"] or result["long_below"])
    return result


def compute_account_alerts(account_doc: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """한 계좌의 (켜진) 알람 종류별 트리거·판정 불가 종목과 실패 사유를 계산한다.

    이평선 일수는 **종목이 속한 종목풀** 설정에서 가져온다. 소속 풀을 찾지 못하거나 가격이
    모자라 판정할 수 없는 종목은 조용히 빠지지 않고 ``unknown`` 으로 모아 슬랙에 함께 알린다.
    """
    account_id = account_doc["account_id"]
    detail = load_all_holdings_detail(account_id)
    rows = [
        r for r in detail.get("rows", []) if str(r.get("ticker") or "").strip() and str(r.get("ticker")).strip() != "IS"
    ]

    threshold = _account_stoploss_pct(account_doc)
    ma_on = bool(account_doc.get("ma_alarm_enabled"))
    stoploss_on = bool(account_doc.get("stoploss_alarm_enabled"))

    ma_hits: list[dict[str, Any]] = []
    stoploss_hits: list[dict[str, Any]] = []
    unknown: list[dict[str, Any]] = []
    errors: list[str] = []

    def _label(row: dict[str, Any]) -> dict[str, str]:
        return {"ticker": str(row["ticker"]).strip(), "name": str(row.get("name") or row["ticker"])}

    def _unknown(row: dict[str, Any], reason: str) -> None:
        unknown.append({**_label(row), "reason": reason})
        errors.append(f"{str(row['ticker']).strip()}: {reason}")

    # 손절: 보유 수익률(return_pct) 기준 — 시세 재조회 불필요.
    if stoploss_on:
        if threshold is None:
            errors.append("손절 기준(%)이 설정되지 않아 손절 알림을 건너뜁니다.")
        else:
            for row in rows:
                ret = row.get("return_pct")
                if isinstance(ret, (int, float)) and not isinstance(ret, bool) and ret <= threshold:
                    stoploss_hits.append({**_label(row), "return_pct": round(float(ret), 2)})

    # 이동선 이탈: 종가 vs 이동평균. 종목풀 순위와 동일하게 실시간 스냅샷을 국가별로 미리 조회해 반영.
    if ma_on:
        pool_by_ticker = pools_by_ticker(str(r.get("ticker") or "") for r in rows)
        candidates: list[tuple[dict[str, Any], str, str, str, tuple[int, int]]] = []
        by_country: dict[str, list[str]] = {}
        for row in rows:
            ticker = str(row["ticker"]).strip().upper()
            pool = pool_by_ticker.get(ticker)
            if not pool:
                # 종목풀에서 빠졌는데 계좌가 아직 들고 있는 경우 — 기준으로 삼을 이평선이 없다.
                _unknown(row, "소속 종목풀 없음")
                continue
            try:
                ma_days = _pool_ma_days(pool)
                country = _pool_country(pool)
            except Exception as exc:
                _unknown(row, str(exc))
                continue
            fetch_ticker = ticker.split(":")[-1]  # 'ASX:XXX' → 'XXX'
            candidates.append((row, fetch_ticker, pool, country, ma_days))
            by_country.setdefault(country, []).append(fetch_ticker)

        snapshots = {c: _safe_realtime_snapshot(c, tks) for c, tks in by_country.items()}
        for row, fetch_ticker, pool, country, ma_days in candidates:
            entry = snapshots.get(country, {}).get(fetch_ticker)
            try:
                status = _ma_status(fetch_ticker, pool, country, ma_days, entry)
            except Exception as exc:  # 한 종목 실패가 전체를 막지 않게
                _unknown(row, f"판정 실패 ({exc})")
                continue
            if status is None:
                _unknown(row, "가격 데이터 부족")
            elif status["below"]:
                ma_hits.append({**_label(row), **status})

    return {"ma": ma_hits, "stoploss": stoploss_hits, "unknown": unknown, "threshold": threshold}, errors


# 배지 계산은 보유 종목별 가격 시계열 조회라 수 초 걸린다 — 계좌별 TTL 캐시로 재계산을 줄인다.
# (알람 설정 저장 시 무효화. 판정 기반인 종가·실시간 스냅샷은 이 TTL 내 변화가 미미하다.)
_BADGES_CACHE_TTL_SECONDS = CACHE_TTL_COMPUTE
_badges_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def _invalidate_badges_cache() -> None:
    _badges_cache.clear()


def compute_account_alert_badges(account_id: str) -> dict[str, Any]:
    """자산 화면(종목명 배지)용: 계좌의 알람 트리거를 티커→아이콘 문자열 맵으로 반환한다.

    슬랙 알람(compute_account_alerts)과 **같은 설정·같은 판정**을 쓴다. 꺼진 알람 종류의
    배지는 붙지 않는다. 티커는 접두사 없는 형태로 정규화한다.
    """
    norm_id = str(account_id or "").strip().lower()
    import time as _time

    cached = _badges_cache.get(norm_id)
    if cached is not None and _time.monotonic() - cached[0] < _BADGES_CACHE_TTL_SECONDS:
        return dict(cached[1])
    account_doc = next((doc for doc in load_account_docs() if doc["account_id"] == norm_id), None)
    if account_doc is None:
        raise ValueError(f"알 수 없는 계좌입니다: {account_id}")

    alerts, _errors = compute_account_alerts(account_doc)

    badge_by_ticker: dict[str, str] = {}

    def _norm_ticker(value: str) -> str:
        return str(value or "").strip().upper().split(":")[-1]

    # 이동선 이탈 종목 — 배지와 별개로 화면이 행 전체를 회색 처리하는 데 쓴다.
    ma_tickers: list[str] = []
    for hit in alerts["ma"]:
        key = _norm_ticker(hit["ticker"])
        if key:
            badge_by_ticker[key] = badge_by_ticker.get(key, "") + _MA_ICON
            ma_tickers.append(key)
    for hit in alerts["stoploss"]:
        key = _norm_ticker(hit["ticker"])
        if key:
            badge_by_ticker[key] = badge_by_ticker.get(key, "") + _STOPLOSS_ICON

    result = {
        "account_id": norm_id,
        "badge_by_ticker": badge_by_ticker,
        "ma_tickers": sorted(set(ma_tickers)),
    }
    _badges_cache[norm_id] = (_time.monotonic(), dict(result))
    return result


def get_alarm_view() -> dict[str, Any]:
    """알람 화면용: 계좌별 알람 On/Off + 손절 기준 목록(시세 계산 없음).

    이평선 일수는 계좌가 아니라 종목의 종목풀에서 오므로 여기 없다.
    """
    return {
        "stoploss_pct_options": list(STOPLOSS_PCT_OPTIONS),
        "accounts": [
            {
                "account_id": doc["account_id"],
                "name": str(doc.get("name") or doc["account_id"]),
                "icon": str(doc.get("icon") or ""),
                "order": int(doc.get("order") or 0),
                "country_code": str(doc.get("country_code") or "").strip().lower(),
                "ma_enabled": bool(doc.get("ma_alarm_enabled", False)),
                "stoploss_enabled": bool(doc.get("stoploss_alarm_enabled", False)),
                "stoploss_threshold_pct": _account_stoploss_pct(doc),
            }
            for doc in load_account_docs()
        ],
    }


def set_account_alarm(account_id: str, alarm_type: str, *, enabled: bool, values: dict[str, Any]) -> dict[str, Any]:
    """계좌별 알람 On/Off + 기준값 저장. alarm_type: 'ma'|'stoploss'.

    values 는 알람 종류가 요구하는 기준값 전부다 — 없는 키는 채우지 않고 에러를 낸다.
      ma       : {} (기준이 종목풀에 있어 계좌가 저장할 값이 없다)
      stoploss : {"threshold_pct": float}
    """
    if alarm_type == "ma":
        payload: dict[str, Any] = {"ma_alarm_enabled": bool(enabled)}
    elif alarm_type == "stoploss":
        if "threshold_pct" not in values:
            raise ValueError("'stoploss' 알람에는 'threshold_pct' 값이 필요합니다.")
        payload = {"stoploss_alarm_enabled": bool(enabled), "stoploss_threshold_pct": float(values["threshold_pct"])}
    else:
        raise ValueError(f"알 수 없는 알람 종류입니다: {alarm_type}")
    save_account_settings(account_id, payload, save_method="알람 센터")
    _invalidate_badges_cache()  # 설정(기준·On/Off) 변경 즉시 배지에 반영
    return get_alarm_view()


def _post_slack(sections: list[dict[str, Any]], *, manual: bool) -> bool:
    tag = "🖐 [수동]" if manual else "🔔"
    total = sum(len(s["ma"]) + len(s["stoploss"]) for s in sections)
    header = f"{tag} 보유종목 알람"
    blocks: list[dict] = [{"type": "header", "text": {"type": "plain_text", "text": header, "emoji": True}}]
    for s in sections:
        parts: list[str] = [f"*{s['account']}*"]
        if s["ma"]:
            # 기준(이평선 일수)이 종목의 종목풀마다 달라 계좌 헤더에 못 쓴다 — 종목 줄 끝에 붙인다.
            parts.append(f"📉 *이동선 이탈* ({get_moving_average_type()})")
            for b in s["ma"]:
                # 어느 쪽이 꺾였는지 보이게 이탈한 선만 표시한다.
                broken = [
                    f"{label} {b[f'{key}_deviation_pct']:+.2f}%"
                    for key, label in (("short", "단기"), ("long", "장기"))
                    if b[f"{key}_below"]
                ]
                parts.append(
                    f"  • {b['name']}({b['ticker']}): {' / '.join(broken)}  `[{b['short_days']}/{b['long_days']}]`"
                )
        if s["stoploss"]:
            parts.append(f"🛑 *손절* ({s['threshold']:.1f}% 이하)")
            parts += [f"  • {b['name']}({b['ticker']}): 수익률 {b['return_pct']:+.2f}%" for b in s["stoploss"]]
        if s["unknown"]:
            # 조용히 빠지면 알림이 안 온 건지 판정이 안 된 건지 구분되지 않는다 — 사유까지 알린다.
            parts.append("⚠️ *판정 불가*")
            parts += [f"  • {b['name']}({b['ticker']}): {b['reason']}" for b in s["unknown"]]
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(parts)}})
    blocks.append(
        {"type": "context", "elements": [{"type": "mrkdwn", "text": f"총 {len(sections)}개 계좌 · {total}건"}]}
    )

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
        if not (doc.get("ma_alarm_enabled") or doc.get("stoploss_alarm_enabled")):
            continue
        alerts, errors = compute_account_alerts(doc)
        all_errors.extend(f"{doc['account_id']}/{e}" for e in errors)
        # 판정 불가도 알려야 할 내용이라 발송 대상에 넣는다(정리가 필요하다는 신호다).
        if alerts["ma"] or alerts["stoploss"] or alerts["unknown"]:
            sections.append({"account": str(doc.get("name") or doc["account_id"]), **alerts})

    if not sections:
        return {"sent": False, "reason": "알림 대상(이탈/손절 종목)이 없습니다.", "errors": all_errors}
    sent = _post_slack(sections, manual=manual)
    return {"sent": sent, "accounts": len(sections), "errors": all_errors}
