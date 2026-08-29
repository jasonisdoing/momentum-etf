"""신고가 돌파 전략 — 선정·설정 서비스 (UI/API 공용).

전략 규칙
--------
1. 유니버스: 설정에서 고른 종목풀 1개. 제외(exclude_from_ranking)는 제외.
2. 진입: 종가가 **직전 52주 최고가**를 넘어선 날(돌파). 다음 거래일 **시가** 체결.
   창은 거래일 수가 아니라 **달력 52주**다(`HIGH_WINDOW_WEEKS`). 화면 문구도 이
   값에서 만들어, 창을 바꾸면 문구가 따라온다.
3. 청산: 아래 둘 중 **먼저 걸리는 쪽**. 판정은 종가, 체결은 다음 거래일 시가.
     ① 진입가 대비 손절선(−7% 또는 −10%)
     ② 이탈 이동평균(기본 20일) 종가 하회
   목표가(익절)는 두지 않는다 — 오르는 종목은 계속 들고 간다.
4. 자리 배분: 동시 보유 상한 top_n, 균등 배분. 신호가 자리보다 많으면
   **거래대금 급증 배수**가 큰 순으로 담는다(돌파에 자금이 실린 쪽 우선).
   자리가 꽉 차 있으면 새 돌파가 와도 **교체하지 않는다** — 2026-08-14 kor(24개월)·
   us(60개월) 백테스트에서 최저수익/손실만/최장보유 교체 전부가 현행보다 나빴다
   (kor +1912% vs 교체 시 +142~779%, us +240% vs -76~+95%). 보유 중이라는 것 자체가
   손절·이탈에 안 걸린 살아있는 추세라는 뜻이라, 교체는 청산 규칙을 앞질러 이익
   종목을 자르고 슬리피지 왕복 비용만 쌓는다.

설정은 MongoDB `system_config.new_high_settings` 에 풀별로 저장한다(`settings_by_pool`).
선택지 밖 저장값 보정·튜닝은 모멘텀과 같은 공용 모듈(`strategy_settings`, `strategy_tuning`)을 쓴다.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from config import MIN_VALUE_MULT_OPTIONS, TOP_N_HOLD
from config import STOP_LOSS_PCT_OPTIONS as STOP_LOSS_OPTIONS
from utils.ma_options import SHORT_MA_OPTIONS
from utils.price_series import positive_prices as _positive
from utils.strategy_settings import coerce_to_options

# 신고가 판정 창 — 거래일 수가 아니라 달력 기간으로 자른다. 거래일로 고정하면
# 공휴일 수에 따라 실제 기간이 흔들려 이름과 어긋난다(12개월 × 20거래일 = 240거래일은
# 실제 52주보다 짧아서, 1년 전 고점이 창 밖으로 일찍 밀려난다).
HIGH_WINDOW_WEEKS = 52
HIGH_WINDOW = f"{HIGH_WINDOW_WEEKS * 7}D"
# 창 안에 이만큼 거래일이 있어야 판정한다. 364일 창의 거래일 수는 국내 238~249일,
# 미국 249~253일이라 정상 종목은 통과하고, 상장 1년 미만·장기 거래정지만 걸러진다.
HIGH_WINDOW_MIN_DAYS = 230

# 화면 셀렉트 선택지 — 백엔드가 단일 소스이고 화면은 응답으로 받는다.
EXIT_MA_OPTIONS = SHORT_MA_OPTIONS  # 이탈 이평선 = 시스템 공용 단기 이평 선택지

# 신호가 자리보다 많을 때는 20일 평균 대비 거래대금 배수가 큰 쪽(돌파에 자금이 실린 종목)부터 담는다.
# (시가총액 우선 옵션은 과거 시총 이력이 없어 검증이 불가능하고 결과 차이도 없어 제거했다.)

# 진입 자격 — 20일 평균 대비 거래대금 배수가 이 값 미만이면 돌파해도 사지 않는다.
# 거래대금이 실리지 않은 돌파는 실패 확률이 높다(오닐의 '돌파는 거래량 증가와 함께').
# None 은 '조건 없음'. 배수를 모르는 종목(상장 직후 등)도 자격 미달로 본다 — 추정하지 않는다.
# 백테스트 기간 기본값 — 화면에서 실행할 때 고르고, 저장하지 않는다.
DEFAULT_BACKTEST_MONTHS = 12

_CONFIG_COLLECTION = "system_config"
_SETTINGS_KEY = "new_high_settings"

# 풀을 바꾸면 그 풀의 값으로 전환되는 항목.
# 풀별로 따로 보관하는 설정. 여기 빠진 키는 저장을 눌러도 버려진다 — 설정을 추가하면
# 반드시 같이 넣어야 한다.
# 슬리피지는 종목풀 설정(BUY/SELL_SLIPPAGE_PCT)을 쓰고, 백테스트 기간은 실행할 때
# 화면에서 고른다 — 둘 다 여기 저장하지 않는다.
PER_POOL_SETTING_KEYS = (
    "stop_loss_pct",
    "exit_ma_days",
    "min_value_mult",
)

DEFAULT_SETTINGS: dict[str, Any] = {
    "stop_loss_pct": -7.0,
    "exit_ma_days": 20,
    # 기본은 조건 없음 — 풀마다 적정값이 달라 사용자가 시험해 보고 저장한다.
    "min_value_mult": None,
}


def _db():
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패했습니다.")
    return db


# ── 종목풀 ─────────────────────────────────────────────────────────────────
def available_pools() -> list[str]:
    from utils.settings_loader import list_available_ticker_types

    return list(list_available_ticker_types())


def pool_options() -> list[dict[str, Any]]:
    """풀 셀렉트 옵션 — 종목풀 설정(DB)의 이름·아이콘·순서를 단일 소스로 쓴다.

    화면은 이 목록을 공용 `formatPoolLabel`(다른 화면과 같은 표준 표기)에 그대로 넣는다.
    `order` 를 빼면 표기에서 번호가 조용히 사라지므로 반드시 함께 담는다.
    """
    from utils.settings_loader import get_ticker_type_settings

    options: list[dict[str, Any]] = []
    for pool in available_pools():
        try:
            settings = get_ticker_type_settings(pool) or {}
        except Exception:
            settings = {}
        options.append(
            {
                "ticker_type": pool,
                "name": str(settings.get("name") or "").strip() or pool,
                "icon": str(settings.get("icon") or "").strip(),
                "order": settings.get("order"),
                "country_code": str(settings.get("country_code") or "").strip().lower(),
                "currency": str(settings.get("currency") or "").strip().upper(),
                "pool_kind": str(settings.get("pool_kind") or "").strip(),
            }
        )
    return options


def load_universe(pool: str) -> list[dict[str, str]]:
    """선택한 종목풀의 종목 목록. 제외 종목은 투자 후보가 아니라 제외한다."""
    from utils.industry_map import industry_map
    from utils.stock_list_io import _load_ticker_type_stocks_raw

    # 업종은 공용 맵이 단일 소스 — 미국은 종목 문서에 없고 지수 구성종목(yfinance)에 있다.
    industry_by = industry_map(pool)

    universe: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in _load_ticker_type_stocks_raw(pool):
        ticker = str(item.get("ticker") or "").strip()
        if not ticker or ticker in seen or bool(item.get("exclude_from_ranking")):
            continue
        seen.add(ticker)
        universe.append(
            {
                "ticker": ticker,
                "name": str(item.get("name") or ticker),
                "pool": pool,
                "market": str(item.get("market") or "").strip(),
                "industry": industry_by.get(ticker, ""),
            }
        )
    return universe


def load_price_frames(universe: list[dict[str, str]]) -> dict[str, pd.DataFrame]:
    from utils.cache_utils import load_cached_frames_bulk_from_ticker_types

    frames: dict[str, pd.DataFrame] = {}
    for pool in sorted({row["pool"] for row in universe}):
        tickers = [row["ticker"] for row in universe if row["pool"] == pool]
        if tickers:
            frames.update(load_cached_frames_bulk_from_ticker_types([pool], tickers))
    return frames


def benchmark_info(pool: str) -> dict[str, str]:
    """벤치마크 {ticker, name} — 종목풀 설정(DB)이 단일 소스. 미설정이면 에러."""
    from utils.pool_settings_store import get_pool_benchmark_ticker
    from utils.settings_loader import get_ticker_type_settings

    settings = get_ticker_type_settings(pool) or {}
    ticker = get_pool_benchmark_ticker(settings)
    if not ticker:
        raise RuntimeError(f"종목풀({pool})에 벤치마크가 설정돼 있지 않습니다 — 종목풀 설정 화면에서 지정하세요.")
    benchmark = settings.get("BENCHMARK") or {}
    return {"ticker": ticker, "name": str(benchmark.get("name") or ticker)}


def load_benchmark_close(pool: str) -> pd.Series:
    from utils.cache_utils import load_cached_frames_bulk_from_all_ticker_types

    benchmark = benchmark_info(pool)
    frame = load_cached_frames_bulk_from_all_ticker_types([benchmark["ticker"]]).get(benchmark["ticker"])
    if frame is None or frame.empty:
        raise RuntimeError(f"벤치마크({benchmark['name']}) 가격 캐시를 불러올 수 없습니다.")
    close = _positive(frame["Close"]).dropna()
    if close.empty:
        raise RuntimeError(f"벤치마크({benchmark['name']}) 종가가 비어 있습니다.")
    return close


# ── 가격 ───────────────────────────────────────────────────────────────────
def build_price_panel(universe: list[dict[str, str]], frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """(날짜 × 티커) 종가·시가·고가·거래대금 표. 백테스트와 선정이 같은 값을 쓴다."""
    closes, opens, highs, values = {}, {}, {}, {}
    for row in universe:
        ticker = row["ticker"]
        frame = frames.get(ticker)
        if frame is None or frame.empty or "Close" not in frame or "Open" not in frame or "High" not in frame:
            continue
        close = _positive(frame["Close"])
        volume = pd.to_numeric(frame["Volume"], errors="coerce") if "Volume" in frame else None
        closes[ticker] = close
        opens[ticker] = _positive(frame["Open"])
        highs[ticker] = _positive(frame["High"])
        values[ticker] = close * volume if volume is not None else pd.Series(index=frame.index, dtype=float)

    if not closes:
        raise RuntimeError("가격 캐시를 불러오지 못해 신고가를 판정할 수 없습니다.")

    close_df = pd.DataFrame(closes).sort_index()
    return {
        "close": close_df,
        "open": pd.DataFrame(opens).reindex(close_df.index),
        "high": pd.DataFrame(highs).reindex(close_df.index),
        "value": pd.DataFrame(values).reindex(close_df.index),
    }


def compute_signals(panel: dict[str, pd.DataFrame], exit_ma_days: int) -> dict[str, pd.DataFrame]:
    """돌파·이탈 신호와 거래대금 급증 배수(진입 정렬 기준)를 한 번에 만든다."""
    close_df = panel["close"]
    # 진입 판정은 **직전 최고 종가** 기준이다. 종가끼리 비교하므로 종가가 오르는 동안
    # 신호가 끊기지 않는다. 장중 고가와 비교하면 전날 꼬리를 못 넘는 날 신호가 끊겨
    # 상승 중에도 진입 기회를 놓친다(한국 개별주 60개월 기준 그런 날이 34%).
    # 오늘은 창에서 뺀다 — 오늘 종가가 '직전' 최고를 넘었는지를 본다.
    prior_high = close_df.rolling(HIGH_WINDOW, min_periods=HIGH_WINDOW_MIN_DAYS).max().shift(1)
    # 관례상의 '52주 신고가'(장중 고가)는 화면 표시용으로만 쓴다 — 판정에는 쓰지 않는다.
    prior_high_intraday = panel["high"].rolling(HIGH_WINDOW, min_periods=HIGH_WINDOW_MIN_DAYS).max().shift(1)
    exit_ma = close_df.rolling(exit_ma_days, min_periods=exit_ma_days).mean()
    value_df = panel["value"]
    return {
        # 돌파 판정은 **종가** 기준이다 — 장중에 잠깐 찍고 밀린 것은 돌파로 보지 않는다.
        "breakout": close_df > prior_high,
        "below_ma": close_df < exit_ma,
        "prior_high": prior_high,
        "prior_high_intraday": prior_high_intraday,
        # 20일 평균 거래대금 대비 배수 — 돌파에 자금이 실렸는지 본다.
        # **당일을 분모에 포함**한다. 급증한 당일이 스스로 평균을 끌어올려 배수가 눌리고
        # (실제 20배가 10.3배로 표기) 표기값 상한이 20배가 되지만, 참고하는 외부 시스템이
        # 같은 정의를 쓴다(NHN 777억→5.3배, 855억→5.7배 — 분모가 당일 값을 따라 움직인다).
        # 하한 최적값(kor 5배 등)도 이 정의 위에서 찾은 것이라 바꾸면 다시 잡아야 한다.
        "value_mult": value_df / value_df.rolling(20, min_periods=20).mean(),
    }


# ── 설정 ───────────────────────────────────────────────────────────────────
def validate_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """화면·API 가 넘긴 설정을 검증한다. 선택지 밖의 값은 받지 않는다."""
    pool = str(settings.get("pool") or "").strip()
    pools = available_pools()
    if pool not in pools:
        raise ValueError(f"알 수 없는 종목풀입니다: {pool}")

    def pick(key: str, options: tuple, cast) -> Any:
        value = cast(settings.get(key, DEFAULT_SETTINGS[key]))
        if value not in options:
            raise ValueError(f"{key} 는 {list(options)} 중 하나여야 합니다 (받은 값: {value})")
        return value

    raw_min = settings.get("min_value_mult", DEFAULT_SETTINGS["min_value_mult"])
    min_value_mult = None if raw_min in (None, "", "none") else float(raw_min)
    if min_value_mult not in MIN_VALUE_MULT_OPTIONS:
        raise ValueError(f"min_value_mult 는 {list(MIN_VALUE_MULT_OPTIONS)} 중 하나여야 합니다 (받은 값: {raw_min})")

    return {
        "pool": pool,
        "min_value_mult": min_value_mult,
        # 종목 수(슬롯)는 풀별 설정이 아니라 시스템 공통(config.TOP_N_HOLD) — 업종 상한은
        # 개념째 폐기했다(집중 완화는 합성 배분 몫).
        "top_n": TOP_N_HOLD,
        "stop_loss_pct": pick("stop_loss_pct", STOP_LOSS_OPTIONS, float),
        "exit_ma_days": pick("exit_ma_days", EXIT_MA_OPTIONS, int),
    }


def _load_doc() -> dict[str, Any]:
    return _db()[_CONFIG_COLLECTION].find_one({"_id": _SETTINGS_KEY}) or {}


def load_settings_map() -> dict[str, Any]:
    """풀별 저장 설정 — 화면이 풀 셀렉트를 바꿀 때 즉시 전환하는 데 쓴다."""
    return dict(_load_doc().get("settings_by_pool") or {})


def default_pool() -> str:
    """설정이 저장된 풀 중 목록에서 가장 앞선 풀 — 화면이 기억한 값이 없을 때의 기준점.

    "마지막으로 고른 풀"은 브라우저 취향이라 DB 에 두지 않는다(화면이 로컬스토리지에 기억).
    """
    saved = set(_load_doc().get("settings_by_pool") or {})
    pools = available_pools()
    for pool in pools:
        if pool in saved:
            return pool
    return pools[0] if pools else ""


def load_settings(pool: str | None = None) -> dict[str, Any]:
    """그 풀의 설정을 반환한다. 풀을 주지 않으면 `default_pool()` 을 쓴다."""
    doc = _load_doc()
    pools = available_pools()
    selected = str(pool or default_pool()).strip()
    if selected not in pools:
        raise ValueError(f"지원하지 않는 종목풀입니다: {pool}")
    stored = dict((doc.get("settings_by_pool") or {}).get(selected) or {})
    return validate_settings({"pool": selected, **DEFAULT_SETTINGS, **stored})


# 화면 로드 때 선택지 밖 저장값을 보정할 항목 — (키, 라벨, 선택지)
_OPTION_FIELDS: tuple[tuple[str, str, tuple], ...] = (
    ("exit_ma_days", "이탈 이평선", EXIT_MA_OPTIONS),
    ("min_value_mult", "급증 하한", MIN_VALUE_MULT_OPTIONS),
    ("stop_loss_pct", "손절선", STOP_LOSS_OPTIONS),
)


def load_settings_for_view(pool: str | None = None) -> tuple[dict[str, Any], list[str]]:
    """화면용 로드 — 선택지 밖 저장값은 첫 선택지로 보정하고 내역을 함께 돌려준다
    (``utils.strategy_settings.coerce_to_options``). 배치·백테스트는 ``load_settings`` 를 쓴다."""
    doc = _load_doc()
    pools = available_pools()
    selected = str(pool or default_pool()).strip()
    if selected not in pools:
        raise ValueError(f"지원하지 않는 종목풀입니다: {pool}")
    merged = {"pool": selected, **DEFAULT_SETTINGS, **dict((doc.get("settings_by_pool") or {}).get(selected) or {})}
    return coerce_to_options(merged, _OPTION_FIELDS, validate_settings)


def save_settings(settings: dict[str, Any]) -> dict[str, Any]:
    normalized = validate_settings(settings)
    pool = normalized["pool"]
    per_pool = {key: normalized[key] for key in PER_POOL_SETTING_KEYS}
    _db()[_CONFIG_COLLECTION].update_one(
        {"_id": _SETTINGS_KEY},
        {"$set": {f"settings_by_pool.{pool}": per_pool}},
        upsert=True,
    )
    # 설정을 바꿨다 되돌리면 옛 키에 그대로 걸린다 — 그 사이 달라진 종목 목록이
    # 반영되지 않은 결과가 다시 나오므로, 저장할 때마다 비우고 새로 계산하게 한다.
    from utils.cache_invalidation import invalidate_strategy_caches

    invalidate_strategy_caches()
    return normalized


__all__ = [
    "DEFAULT_SETTINGS",
    "DEFAULT_BACKTEST_MONTHS",
    "MIN_VALUE_MULT_OPTIONS",
    "EXIT_MA_OPTIONS",
    "HIGH_WINDOW_WEEKS",
    "STOP_LOSS_OPTIONS",
    "benchmark_info",
    "build_price_panel",
    "compute_signals",
    "load_benchmark_close",
    "load_price_frames",
    "load_settings",
    "load_settings_map",
    "load_universe",
    "pool_options",
    "save_settings",
    "validate_settings",
]
