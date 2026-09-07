"""신고가 전략 — 선정·설정 서비스 (UI/API 공용).

전략 규칙
--------
1. 유니버스: 설정에서 고른 종목풀 1개. 제외(exclude_from_ranking)는 제외.
2. 진입: 종가가 **직전 52주 최고가**를 넘어선 날(돌파). 다음 거래일 **시가** 체결.
   창은 거래일 수가 아니라 **달력 52주**다(`HIGH_WINDOW_WEEKS`). 화면 문구도 이
   값에서 만들어, 창을 바꾸면 문구가 따라온다.
3. 청산: **이탈 이동평균(기본 20일) 종가 하회**. 판정은 종가, 체결은 다음 거래일 시가.
   목표가(익절)는 두지 않는다 — 오르는 종목은 계속 들고 간다.
4. 자리 배분: 동시 보유 상한 top_n, 균등 배분. 신호가 자리보다 많으면
   **거래대금 급증 배수**가 큰 순으로 담는다(돌파에 자금이 실린 쪽 우선).
   자리가 꽉 차 있으면 새 돌파가 와도 **교체하지 않는다** — 2026-08-14 kor(24개월)·
   us(60개월) 백테스트에서 최저수익/손실만/최장보유 교체 전부가 현행보다 나빴다
   (kor +1912% vs 교체 시 +142~779%, us +240% vs -76~+95%). 보유 중이라는 것 자체가
   이탈에 안 걸린 살아있는 추세라는 뜻이라, 교체는 청산 규칙을 앞질러 이익
   종목을 자르고 슬리피지 왕복 비용만 쌓는다.

설정은 MongoDB `system_config.new_high_settings` 에 풀별로 저장한다(`settings_by_pool`).
선택지 밖 저장값 보정·튜닝은 모멘텀과 같은 공용 모듈(`strategy_settings`, `strategy_tuning`)을 쓴다.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from config import ADR_FLOOR_OPTIONS, MIN_VALUE_MULT_OPTIONS_BY_COUNTRY
from utils.ma_options import SHORT_MA_OPTIONS
from utils.momentum_service import default_adr_floor
from utils.price_series import positive_prices as _positive
from utils.strategy_settings import coerce_to_options, require_start_date, validate_start_date

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
    "start_date",
    "exit_ma_days",
    "min_value_mult",
    "adr_floor",
)

DEFAULT_SETTINGS: dict[str, Any] = {
    "exit_ma_days": 20,
    # 기본은 조건 없음 — 풀마다 적정값이 달라 사용자가 시험해 보고 저장한다.
    "min_value_mult": None,
    # ADR 하한 — 전일 시장 ADR 이 미만이면 그날 **신규 진입만** 차단(보유는 이탈이 관리).
    # 기본 없음. 시장은 풀 설정의 시장 레짐 지수를 따른다(모멘텀과 같은 공용 판정).
    "adr_floor": default_adr_floor(),
}


def _db():
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패했습니다.")
    return db


# ── 종목풀 ─────────────────────────────────────────────────────────────────
def available_pools() -> list[str]:
    """이 전략을 쓰기로 켠 종목풀만(order 순) — 종목풀 설정의 「사용」 토글이 단일 소스다."""
    from utils.pool_strategy_use import pools_using

    return pools_using("new_high")


def pool_country(pool: str) -> str:
    """풀의 국가 코드(kor/us/aus…) — 이평 선택지·시간 비례 판정 여부가 이 값으로 갈린다."""
    from utils.settings_loader import get_ticker_type_settings

    return str((get_ticker_type_settings(pool) or {}).get("country_code") or "").strip().lower()


def min_value_mult_options(country_code: str | None) -> tuple[float | None, ...]:
    """그 국가의 거래대금 하한 선택지 — 목록은 `config.MIN_VALUE_MULT_OPTIONS_BY_COUNTRY`
    가 단일 소스다(이평 선택지의 `utils/ma_options` 와 같은 방식). 모르는 국가면 에러."""
    country = str(country_code or "").strip().lower()
    if country not in MIN_VALUE_MULT_OPTIONS_BY_COUNTRY:
        raise ValueError(f"거래대금 하한 선택지를 지원하지 않는 국가입니다: {country_code!r}")
    return MIN_VALUE_MULT_OPTIONS_BY_COUNTRY[country]


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
    """선택한 종목풀의 투자 후보 목록 — 정의는 `utils.stock_list_io` 한 곳이다."""
    from utils.stock_list_io import load_pool_universe

    return load_pool_universe(pool)


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

    # 거래대금 하한 — 선택지가 국가별이라 풀의 국가로 목록을 고른다.
    mult_options = min_value_mult_options(pool_country(pool))
    raw_min = settings.get("min_value_mult", DEFAULT_SETTINGS["min_value_mult"])
    min_value_mult = None if raw_min in (None, "", "none") else float(raw_min)
    if min_value_mult not in mult_options:
        raise ValueError(f"min_value_mult 는 {list(mult_options)} 중 하나여야 합니다 (받은 값: {raw_min})")

    raw_adr = settings.get("adr_floor", DEFAULT_SETTINGS["adr_floor"])
    adr_floor = None if raw_adr in (None, "", "none") else int(raw_adr)
    if adr_floor not in ADR_FLOOR_OPTIONS:
        allowed = ", ".join("없음" if v is None else str(v) for v in ADR_FLOOR_OPTIONS)
        raise ValueError(f"adr_floor 는 {allowed} 중 하나여야 합니다 (받은 값: {raw_adr})")
    if adr_floor is not None:
        from utils.momentum_service import adr_market_of_pool

        if adr_market_of_pool(pool) is None:
            raise ValueError("ADR 하한을 쓰려면 /pools-settings 에서 이 풀의 시장 레짐 지수를 먼저 설정하세요.")

    return {
        "pool": pool,
        "start_date": validate_start_date(settings.get("start_date")),
        "min_value_mult": min_value_mult,
        "adr_floor": adr_floor,
        # 종목 수(슬롯)는 순위·모멘텀·종목풀 백테스트와 같은 풀 설정을 쓴다.
        "top_n": _pool_top_n_hold(pool),
        "exit_ma_days": pick("exit_ma_days", EXIT_MA_OPTIONS, int),
    }


def _pool_top_n_hold(pool: str) -> int:
    from utils.pool_settings_store import get_pool_top_n_hold

    return get_pool_top_n_hold(pool)


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


# 화면 로드 때 선택지 밖 저장값을 보정할 항목 — (키, 라벨, 선택지).
# 거래대금 하한 선택지가 국가별이라 풀을 알아야 목록이 정해진다.
def _option_fields(pool: str) -> tuple[tuple[str, str, tuple], ...]:
    return (
        ("adr_floor", "ADR 하한", ADR_FLOOR_OPTIONS),
        ("exit_ma_days", "이탈 이평선", EXIT_MA_OPTIONS),
        ("min_value_mult", "거래대금 하한", min_value_mult_options(pool_country(pool))),
    )


def load_settings_for_view(pool: str | None = None) -> tuple[dict[str, Any], list[str]]:
    """화면용 로드 — 선택지 밖 저장값은 첫 선택지로 보정하고 내역을 함께 돌려준다
    (``utils.strategy_settings.coerce_to_options``). 배치·백테스트는 ``load_settings`` 를 쓴다."""
    doc = _load_doc()
    pools = available_pools()
    selected = str(pool or default_pool()).strip()
    if selected not in pools:
        # 그 화면이 기억한 풀이 이 전략에서 꺼졌을 수 있다 — 막지 말고 기본 풀로 연다.
        selected = default_pool()
        if not selected:
            raise ValueError("이 전략을 쓰는 종목풀이 없습니다 — 종목풀 설정에서 「사용」을 켜세요.")
    merged = {"pool": selected, **DEFAULT_SETTINGS, **dict((doc.get("settings_by_pool") or {}).get(selected) or {})}
    return coerce_to_options(merged, _option_fields(selected), validate_settings)


def save_settings(settings: dict[str, Any]) -> dict[str, Any]:
    normalized = validate_settings(settings)
    require_start_date(normalized)
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
    "min_value_mult_options",
    "EXIT_MA_OPTIONS",
    "benchmark_info",
    "load_benchmark_close",
    "load_price_frames",
    "load_settings",
    "load_settings_map",
    "load_universe",
    "pool_options",
    "save_settings",
    "validate_settings",
]


def delete_settings(pool: str) -> None:
    """그 풀의 신고가 설정을 지운다 — 종목풀 설정 화면에서 「사용」을 끄면 부른다."""
    _db()[_CONFIG_COLLECTION].update_one(
        {"_id": _SETTINGS_KEY},
        {"$unset": {f"settings_by_pool.{str(pool).strip()}": ""}},
    )
