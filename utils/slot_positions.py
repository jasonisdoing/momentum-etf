"""운용 현황 공용 — 신고가·모멘텀이 함께 쓰는 **오늘 상태** 조회 도구.

백테스트 엔진(`core.strategy.slot_backtest`)이 굴린 결과를 화면이 읽을 수 있게 만들 때 필요한
것들이다: 진행 중인 세션의 실시간 시세, 시장 현지 날짜, 다음 거래일, 시가총액·표시 시세.
두 전략이 같은 표를 그리므로 판정 내용만 각자 하고 이 부분은 함께 쓴다.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from utils.logger import get_app_logger

logger = get_app_logger()

# 장전에 화면을 주기적으로 다시 받기 시작할 시점 — 개장 몇 분 전부터인가.
# 실제로 예상체결가가 움직이는 구간은 동시호가(개장 30분 전~개장)라 한 시간이면 넉넉하다.
# 시세 제공처의 '장전' 플래그는 새벽부터 켜져 있을 수 있어 그것만 믿고 돌리지 않는다.
_PRE_MARKET_REFRESH_LEAD_MINUTES = 60


def adr_entry_gate(pool: str, adr_floor: Any) -> tuple[Any, Any]:
    """ADR 진입 게이트와 표시값 조회 함수를 한 쌍으로 만든다 — (entry_blocked, adr_at).

    게이트는 **신규 진입만** 막는다 — 보유 청산은 그대로 돈다. 하한이 없거나 레짐 시장이
    없는 풀이면 아무것도 막지 않고, 표시값도 None 이다. ADR 이력 이전 날짜는 미적용이다.
    DB 조회(ADR 이력)가 있어 엔진 밖에 둔다 — 엔진은 이 함수 쌍을 인자로 받는다.
    """
    from utils.momentum_service import adr_market_of_pool, load_adr_series

    market = adr_market_of_pool(pool)
    series = load_adr_series(market) if market else pd.Series(dtype=float)

    def adr_at(stamp: pd.Timestamp) -> float | None:
        if series.empty:
            return None
        value = series.asof(pd.Timestamp(stamp))
        return round(float(value), 1) if pd.notna(value) else None

    def entry_blocked(stamp: pd.Timestamp) -> bool:
        if adr_floor is None or series.empty:
            return False
        value = series.asof(pd.Timestamp(stamp))
        return bool(pd.notna(value) and float(value) < float(adr_floor))

    return entry_blocked, adr_at


def load_slot_market(pool: str, adr_floor: Any) -> dict[str, Any]:
    """슬롯 엔진 실행에 필요한 시장 파라미터 수집 — 엔진은 조회 없이 이 값들만 받는다.

    슬리피지·시작 자본은 운용 현황 **캐시 키에도 들어간다** — 키에 없으면 풀 설정을 바꿔도
    5분 동안 이전 값으로 계산된 결과가 보인다.
    """
    from utils.benchmark_curve import benchmark_growth
    from utils.new_high_service import benchmark_info
    from utils.pool_settings_store import get_pool_slippage
    from utils.share_allocation import backtest_initial_capital

    buy_slippage, sell_slippage = get_pool_slippage(pool)
    entry_blocked, adr_at = adr_entry_gate(pool, adr_floor)
    return {
        "buy_slippage": buy_slippage,
        "sell_slippage": sell_slippage,
        "initial_capital": backtest_initial_capital(pool),
        "entry_blocked": entry_blocked,
        "adr_at": adr_at,
        "benchmark_growth": lambda index: benchmark_growth(pool, index),
        "benchmark_name": benchmark_info(pool)["name"],
    }


def _market_caps(pool: str) -> dict[str, float]:
    """티커 → 시가총액. 배치 B 가 메타 캐시에 적어 둔 값을 읽기만 한다.

    한국 개별주는 예전에 여기서 네이버 시세표를 직접 순회했다(424종목에 4초). 그런데 그
    목록은 시총 **순위**를 매기려고 배치가 이미 받아 오는 값이라, 배치가 금액까지 적게
    하고(`utils/market_cap_rank`) 화면은 DB 만 읽는다. 국가별 분기도 함께 사라졌다.

    값이 없는 종목은 맵에서 빠진다 — 화면은 '-' 로 둔다(임의 보정 없음).
    현재 값만 있고 과거 이력이 없다. 그래서 백테스트 우선순위에는 쓰지 않는다.
    """
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        return {}
    caps: dict[str, float] = {}
    for doc in db["stock_cache_meta"].find({"ticker_type": pool}, {"ticker": 1, "meta_cache": 1}):
        value = (doc.get("meta_cache") or {}).get("total_net_assets")
        if value:
            caps[str(doc.get("ticker") or "").strip().upper()] = float(value)
    return caps


def _live_quotes(pool: str, tickers: list[str], cached_last: pd.Timestamp) -> dict[str, Any]:
    """진행 중인 세션의 실시간 시세. 캐시에 아직 안 들어온 날일 때만 의미가 있다.

    반환 ``{"live": bool, "pre_market": bool, "traded_at": str|None,
    "by_ticker": {티커: {price, high, change_pct}}}``.
    ``live`` 는 마지막 체결일이 가격 캐시의 마지막 거래일보다 **뒤**라는 뜻 —
    그날 종가가 아직 확정되지 않았으므로 화면은 '돌파중'처럼 잠정 상태로 표시한다.
    캐시와 같은 날이면 이미 확정된 세션이라 실시간을 쓰지 않는다.

    장전(동시호가) 구간은 ``live`` 로 보지 않는다. 그 시각 스냅샷의 고가·저가·시가는
    아직 **직전 세션의 값**이고 현재가만 오늘 예상체결가라, 둘을 섞으면 어제 확정된
    돌파가 오늘 예상가에 밀려 '터치 후 밀림'으로 뒤집힌다. 오늘 값이 다 갖춰지는
    정규장부터 쓴다.
    """
    from utils.settings_loader import get_ticker_type_settings

    try:
        country = str((get_ticker_type_settings(pool) or {}).get("country_code") or "").strip().lower()
    except Exception:
        # 설정이 없는 풀(테스트 등)은 실시간이 없다 — 시세 조회 실패와 같은 취급이다.
        country = ""
    if not country or not tickers:
        return {"live": False, "pre_market": False, "traded_at": None, "by_ticker": {}}

    from services.price_service import get_realtime_snapshot

    try:
        snapshot = get_realtime_snapshot(country, tickers)
    except Exception:
        logger.exception("[new_high] 실시간 시세 조회 실패 (%s)", pool)
        return {"live": False, "pre_market": False, "traded_at": None, "by_ticker": {}}

    by_ticker: dict[str, dict[str, float]] = {}
    traded_at: str | None = None
    pre_market = False
    for ticker, quote in snapshot.items():
        price = quote.get("nowVal")
        if price is None or float(price) <= 0:
            continue
        # 오늘 시가 — 어제 확정된 진입·청산이 체결된 가격이다. ETF 는 이 값이 안 와서
        # None 이 되고, 그런 종목은 체결로 처리하지 않는다(가격을 지어내지 않는다).
        open_val = quote.get("open")
        by_ticker[ticker] = {
            "price": float(price),
            "high": float(quote.get("high") or price),
            "open": float(open_val) if open_val is not None and float(open_val) > 0 else None,
            "change_pct": float(quote.get("changeRate")) if quote.get("changeRate") is not None else None,
        }
        if quote.get("is_pre_market"):
            pre_market = True
        stamp = str(quote.get("localTradedAt") or "")
        if stamp and (traded_at is None or stamp > traded_at):
            traded_at = stamp

    live = bool(traded_at) and not pre_market and str(traded_at)[:10] > str(cached_last.date())
    if not live and traded_at is None and by_ticker and not pre_market:
        # 체결 시각을 안 주는 종목(국내 ETF 등)은 시장 **세션 시계**로 장중을 판정한다 —
        # 정규장·애프터가 진행 중이면 스냅샷 현재가는 오늘 세션의 값이다. 체결 시각에만
        # 기대면 ETF 풀은 장중 판정(실시간 마지막 봉, AGENTS.md §10-6)이 영영 켜지지 않는다.
        from utils.market_session import AFTERMARKET, REGULAR, market_session

        try:
            session_now = market_session(country)["session"]
        except Exception:
            session_now = None
        today = _country_today(country)
        if session_now in (REGULAR, AFTERMARKET) and today and today > str(cached_last.date()):
            live = True
            traded_at = today
    return {
        "live": live,
        "pre_market": pre_market,
        "traded_at": traded_at,
        # 시세는 항상 담는다. 현재가·등락률은 어느 구간이든 오늘 값이라 표시에 쓰고,
        # 돌파 판정은 `live` 일 때만 한다 — ETF 처럼 체결 시각·고가를 안 주는 종목도
        # 일간(%) 은 정상으로 보여야 한다.
        "by_ticker": by_ticker,
    }


def _should_auto_refresh(pool: str, quotes: dict[str, Any]) -> bool:
    """화면이 주기 갱신을 걸어야 하는 시점인지.

    장중이면 늘 참이고, 장전이면 개장이 가까울 때만 참이다. 개장 시각은 시장마다 달라
    화면이 알 수 없으므로 여기서 판단해 내려준다.
    """
    if quotes["live"]:
        return True
    if not quotes["pre_market"]:
        return False

    from config import MARKET_SCHEDULES
    from utils.settings_loader import get_ticker_type_settings

    country = str((get_ticker_type_settings(pool) or {}).get("country_code") or "").strip().lower()
    schedule = (MARKET_SCHEDULES or {}).get(country)
    if not isinstance(schedule, dict):
        return False
    tz_name = str(schedule.get("timezone") or "").strip()
    open_time = schedule.get("open")
    if not tz_name or open_time is None:
        return False
    try:
        now_local = pd.Timestamp.now(tz=tz_name)
        opens_at = pd.Timestamp(f"{now_local.date()} {open_time.hour:02d}:{open_time.minute:02d}", tz=tz_name)
    except Exception:
        return False
    return opens_at - pd.Timedelta(minutes=_PRE_MARKET_REFRESH_LEAD_MINUTES) <= now_local <= opens_at


def _pool_country(pool: str) -> str:
    """종목풀의 국가 코드(kor·us·au). 시장별 규칙을 고르는 단일 소스."""
    from utils.settings_loader import get_ticker_type_settings

    return str((get_ticker_type_settings(pool) or {}).get("country_code") or "").strip().lower()


def _country_today(country: str) -> str | None:
    """그 시장의 **현지 오늘** 날짜(YYYY-MM-DD). 시간대를 모르면 None — 날짜를 지어내지 않는다."""
    from config import MARKET_SCHEDULES

    tz_name = str(((MARKET_SCHEDULES or {}).get(country) or {}).get("timezone") or "").strip()
    if not tz_name:
        return None
    try:
        return str(pd.Timestamp.now(tz=tz_name).date())
    except Exception:
        logger.exception("[slot] 시장 현지 날짜 계산 실패 (%s)", country)
        return None


def _market_today(pool: str) -> str | None:
    """그 종목풀 시장의 현지 오늘 날짜.

    미국 풀을 한국에서 보면 서버·브라우저의 날짜가 시장의 날짜와 하루 어긋난다. '그 세션이
    지났는지' 는 시장 현지 날짜로 따져야 한다.
    """
    return _country_today(_pool_country(pool))


def _next_session(pool: str, last: pd.Timestamp) -> str | None:
    """캐시 마지막 거래일 **다음**의 거래일 — 진입·청산이 체결되는 날.

    화면이 '오늘 매수'인지 '내일 매수'인지 가리는 데 쓴다. 장 시작 전에는 캐시의
    마지막 거래일이 아직 어제라, '다음 거래일' 이 곧 오늘이다.
    캘린더가 답할 수 없으면 None 을 돌려준다 — 날짜를 지어내지 않는다.
    """
    from utils.settings_loader import get_ticker_type_settings
    from utils.trading_calendar import get_trading_days

    country = str((get_ticker_type_settings(pool) or {}).get("country_code") or "").strip().lower()
    if not country:
        return None
    try:
        days = get_trading_days(
            str((last + pd.Timedelta(days=1)).date()),
            str((last + pd.Timedelta(days=14)).date()),
            country,
        )
    except Exception:
        logger.exception("[new_high] 다음 거래일 조회 실패 (%s)", pool)
        return None
    return str(days[0].date()) if days else None


def _apply_display_quotes(
    rows: list[dict[str, Any]],
    holdings: list[dict[str, Any]],
    by_ticker: dict[str, dict[str, Any]],
) -> None:
    """현재가·일간(%)·보유 수익률만 실시간으로 바꾼다. **판정에는 쓰지 않는다.**

    돌파 거리·터치·진입 예정은 확정 종가로 정해지고, 이 함수는 사람이 보는 숫자만 바꾼다.
    그래서 체결 시각이나 고가를 안 주는 종목(국내 ETF)도 일간(%) 은 정상으로 나온다.
    """
    for row in rows:
        quote = by_ticker.get(row["ticker"])
        if not quote:
            continue
        row["price"] = quote["price"]
        if quote["change_pct"] is not None:
            row["change_pct"] = round(quote["change_pct"], 2)
    for held in holdings:
        quote = by_ticker.get(held["ticker"])
        if not quote:
            continue
        held["price"] = quote["price"]
        held["return_pct"] = round((quote["price"] / held["entry_price"] - 1) * 100, 2)


def _cache_refreshed_at(pool: str) -> str | None:
    """이 종목풀 가격 캐시의 마지막 갱신 시각(ISO). 배치가 안 돌았으면 None."""
    from utils.cache_utils import get_cache_refresh_completed_at

    completed = get_cache_refresh_completed_at(pool)
    return completed.isoformat() if completed else None
