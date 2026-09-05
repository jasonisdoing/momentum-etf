"""모멘텀 전략 — 시장 대비 '꾸준한 모멘텀' 선정 서비스 (UI/API/스크립트 공용).

전략 규칙
--------
1. 유니버스: 설정에서 고른 종목풀 1개(전체 활성 풀 중 선택 — 한국·미국·호주 등).
   제외 종목(exclude_from_ranking)은 제외. 코스피+코스닥 통합 같은 혼합
   유니버스는 종목풀 자체를 합쳐 만든다(풀 관리 화면).
2. 점수: **장기 이평선 이격(%)** — 계산식은 순위 화면과 같되, 이평선 일수는
   **전략 전용 값**(풀별 설정의 short/long_ma_days)을 쓴다.
   장기 이격 > 0 & 단기 이격 >= 0 만 후보. 점수 순 상위 top_n 동일가중.
   같은 규칙을 **주간 리듬으로 유지**하는 것이 이 화면의 차이다 —
   편입·편출을 주 단위로 본다.
3. 주간 리밸런싱: **주 마지막 거래일 종가로 판정하고 다음 주 첫 거래일 시가에 체결**한다.
   한 주 종가를 모두 보고 판정한 뒤 주말 동안 검토할 시간을 둔다. 종가 체결은 쓰지 않는다 —
   마감 동시호가에 포트폴리오 대부분을 동시에 팔고 사는 것이라 체결가를 모른 채 대량을
   던지게 되고 미체결을 정정할 시간도 없다.
4. 주중 매도: 보유 종목이 자격(장기 이격 > 0 & 단기 이격 >= 0)을 잃으면 다음 거래일
   시가에 판다. 판 슬롯은 다음 주 교체까지 현금이다(주중 재매수 없음).

벤치마크·국가·통화는 종목풀 설정(DB)을 단일 소스로 쓴다.
설정은 MongoDB `pool_settings` 의 **각 풀 문서**에 저장한다
(`{pool, settings_by_pool: {풀: {top_n, ...}}}`) — 풀을 바꾸면 그 풀의 설정으로 전환된다.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import pandas as pd

from config import ADR_FLOOR_OPTIONS
from core.strategy.scoring import (
    compute_ma_disparity,
    rank_score,
)
from utils.ma_options import LONG_MA_OPTIONS, SHORT_MA_OPTIONS
from utils.strategy_settings import coerce_to_options, require_start_date, validate_start_date

warnings.filterwarnings("ignore")

# ── 상수 ──────────────────────────────────────────────────────────────────
# 종목풀은 DB(pool_settings)의 활성 풀 전체 중 1개를 선택한다 — 목록·국가·통화의
# 단일 소스는 종목풀 설정이다 (하드코딩 목록을 두지 않는다).
# 한 업종에서 최대 몇 종목까지 담을지 — 화면 셀렉트와 검증이 같은 목록을 쓴다.
# 종목 수·업종 상한은 신고가와 **의미가 같아** 전략 공용 상수를 그대로 쓴다
# (`utils/strategy_options`). 여기서 다시 정의하면 한쪽만 고쳐져 두 화면이 갈린다.
# 월↔거래일 환산 — 공용 상수(config, =20)를 재사용한다 (자산헬퍼·시장추세와 동일 기준).

# ── 룩백 4 · 종목 수 6 · 업종 상한 2 를 쓰는 근거 ──────────────────────────
# ⚠ 아래 표는 **옛 점수 방식(연율화 상대기울기 × R²)·미국 풀** 기준이다. 점수를
#   t-통계량으로, 종목풀을 한국 개별주로 바꾼 뒤에는 최적 조합이 다를 수 있어
#   재검증이 필요하다.
# 2026-08-02 검증. 미국 풀 · 슬리피지 0.5% · **84개월** 단일 구간에서
# 룩백 3~6 × 종목수 5~10 × 업종상한 2~5 = 96조합을 전부 백테스트했다.
# (84개월인 이유: 룩백 6 의 상한이 84 라 그 이상은 룩백별로 구간이 달라져 비교가 안 된다)
#
# **소르티노 상위** — 선택 기준 (같은 기간 벤치마크 QQQ +275.6%, 소르티노 1.83)
#   순위  룩백  종목수  상한  소르티노    총수익    MDD   교체율
#     1    4     6     2    2.34   2845.9%  -28.7%  49.0%  ← 채택
#     2    4     5     2    2.24   3936.0%  -32.0%  49.4%
#     3    4     7     3    2.20   2275.4%  -31.3%  46.8%
#     4    4     5     3    2.18   4283.2%  -31.5%  48.7%
#     5    4     6     3    2.16   2943.6%  -28.6%  48.0%
#     6    4     6     4    2.16   3083.6%  -29.3%  47.4%
#
# 참고: 총수익 기준 1위는 4-5-3(4283.2%, 소르티노 2.18), 최하위는 3-10-2(903.8%)였다.
#
# 1. 종목 수가 총수익 순위를 지배한다. 총수익 1~12위가 전부 5~6종목,
#    하위 92~96위가 전부 9~10종목이다. 적을수록 수익이 크고 낙폭도 깊어진다.
# 2. 룩백 3 은 뚜렷하게 나쁘다. 교체율이 55~63% 로 다른 룩백(33~49%)보다 훨씬 높은데
#    그만큼의 대가를 얻지 못한다.
# 3. **소르티노는 룩백 4 가 지배한다.** 상위 12개 중 11개가 룩백 4 다. 룩백 6 은
#    회전이 느리고(33~37%) 낙폭이 얕지만(-26~-29%) 소르티노가 1.5~1.8 로 낮다.
# 4. 업종 상한은 **종목 수에 따라 효력이 갈린다.** 5종목에 상한 4·5 는 걸릴 수가 없어
#    값이 아예 같아진다(총수익 2·3위, 5·6위가 동일). 반면 6종목에 상한 2 는 최소 3개
#    업종으로 강제 분산되어 실제로 결과가 달라진다(4-6-2 2845.9% vs 4-6-4 3083.6%).
# => 4-6-2 를 쓴다. 소르티노 1위(2.34)이고, 상한 2 가 6종목을 최소 3업종으로 흩어
#    낙폭을 -28.7% 로 눌러 준다. 총수익은 4-5-3 보다 1437%p 낮지만 벤치마크의 10배다.
#    수익 극대화가 아니라 위험 대비 효율을 기준으로 고른 값이다.
#
# ⚠ 한계 — 이 값은 **표본 내 최적값**이다.
# 84개월을 앞뒤 42개월로 갈라 각각 순위를 매겨보니 **전·후반 순위상관이 0.056** 으로
# 사실상 무상관이었다. 전반 1위(룩백6·5종목·상한4)가 후반 11위, 전반 41위(룩백4·6종목·상한3)가
# 후반 6위로 뒤바뀐다. 즉 이 조합이 다음 구간에도 1위일 것이라는 근거는 없다.
# 여기에 강세장 편중과 생존 편향(유니버스가 현재 종목풀 기준)이 더해진다.
# 조합을 바꿀 때는 순위 몇 계단 차이가 아니라 성격 차이(회전 속도·낙폭)를 보고 정한다.
#
# 설정의 단일 소스는 DB(`pool_settings` 의 풀 문서)다. 코드에 기본값을
# 두지 않는다 — 값이 없거나 깨졌으면 임의 값으로 대체하지 않고 에러를 낸다.


# ── 풀 정보 (종목풀 설정 단일 소스) ─────────────────────────────────────────
def available_pools() -> list[str]:
    """DB(pool_settings)의 활성 종목풀 목록."""
    from utils.settings_loader import list_available_ticker_types

    return list_available_ticker_types()


def pool_info(pool: str) -> dict[str, str]:
    """풀의 국가·통화·이름 — 종목풀 설정(DB)이 단일 소스. 없으면 명시적 에러."""
    from utils.settings_loader import get_ticker_type_settings

    settings = get_ticker_type_settings(pool) or {}
    country = str(settings.get("country_code") or "").strip().lower()
    currency = str(settings.get("currency") or "").strip().upper()
    if not country or not currency:
        raise RuntimeError(f"종목풀({pool}) 설정에 country_code/currency 가 없습니다 — 종목풀 설정을 확인하세요.")
    return {"country": country, "currency": currency, "name": str(settings.get("name") or pool)}


# ── 설정 ──────────────────────────────────────────────────────────────────
# 풀별로 따로 저장되는 항목 — 풀을 바꾸면 이 값들이 그 풀의 저장분으로 전환된다.
# 슬리피지는 종목풀 설정(BUY/SELL_SLIPPAGE_PCT)을 쓰고, 백테스트 기간은 화면에서
# 실행할 때 고른다 — 둘 다 전략 설정으로 저장하지 않는다.
PER_POOL_SETTING_KEYS = (
    "start_date",
    "short_ma_days",
    "long_ma_days",
    "adr_floor",
)

# 교체 규칙·주중 매도는 **전략의 일부**라 설정이 아니다.
#   교체 규칙 = 자격 유지 : 후보 자격이 남아 있으면 순위와 무관하게 계속 들고 가고, 자격을
#               잃어 빈 자리만 그 시점 상위 후보로 채운다(신고가 전략의 편입 방식과 같은 결).
#               매주 상위 N 을 새로 뽑는 방식은 순위 몇 계단 차이로 왕복 비용만 쌓아 폐기했다.
#   주중 매도 = ADR 게이트만 : 시장 ADR 이 하한 미만으로 내려간 날 전량 매도한다.
#               개별 종목이 주중에 이평선을 이탈해도 ADR 이 하한 이상이면 주말 판정까지
#               보유한다 — 개별 이탈은 주 교체(자격 유지 재선정)가 처리한다. 주중 개별
#               매도는 거래만 늘리고 반등분을 놓쳐 폐기했다.

# 전략 전용 이평선 선택지 — 화면 셀렉트와 튜닝 축이 같은 값을 쓴다.
# 그리드 결과(kor_kr·kospi200·us)에서 장기 60~120 이 고원, 140 부터 열위라 140 까지만 둔다.
# 이평선 선택지는 시스템 공용(utils/ma_options) — 여기서 따로 정하지 않는다.


def validate_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """설정을 검증해 정규화된 dict 를 반환한다. 실패 시 ValueError."""
    if not isinstance(settings, dict):
        raise ValueError("설정 형식이 올바르지 않습니다.")

    def _num(key: str) -> float:
        value = settings.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"'{key}' 는 숫자여야 합니다.")
        return float(value)

    pool = str(settings.get("pool") or "").strip().lower()
    if pool not in available_pools():
        raise ValueError(f"지원하지 않는 종목풀입니다: {settings.get('pool')}")

    # 전략 전용 이평선 — 종목풀 설정과 별개다. 단기·장기 각자의 선택지만 허용한다.
    short_ma_days = int(_num("short_ma_days"))
    long_ma_days = int(_num("long_ma_days"))
    for key, value, options in (
        ("short_ma_days", short_ma_days, SHORT_MA_OPTIONS),
        ("long_ma_days", long_ma_days, LONG_MA_OPTIONS),
    ):
        if value not in options:
            allowed = ", ".join(str(v) for v in options)
            raise ValueError(f"'{key}' 는 {allowed} 중 하나여야 합니다.")
    if short_ma_days >= long_ma_days:
        raise ValueError("'short_ma_days' 는 'long_ma_days' 보다 작아야 합니다.")

    # ADR 하한 — 판정일의 시장 ADR 이 이 값 미만이면 그 주는 전량 현금. None = 게이트 없음(기본).
    # 시장은 풀 설정의 시장 레짐 지수(ADR 이 있는 4개 시장으로 제한됨)를 따른다.
    raw_adr = settings.get("adr_floor")
    adr_floor = None if raw_adr in (None, "", "none") else int(raw_adr)
    if adr_floor not in ADR_FLOOR_OPTIONS:
        allowed = ", ".join("없음" if v is None else str(v) for v in ADR_FLOOR_OPTIONS)
        raise ValueError(f"'adr_floor' 는 {allowed} 중 하나여야 합니다 (받은 값: {raw_adr}).")
    if adr_floor is not None and adr_market_of_pool(pool) is None:
        raise ValueError("ADR 하한을 쓰려면 /pools-settings 에서 이 풀의 시장 레짐 지수를 먼저 설정하세요.")

    return {
        "pool": pool,
        "start_date": validate_start_date(settings.get("start_date")),
        # 종목 수는 순위·신고가·종목풀 백테스트와 같은 풀 설정을 쓴다.
        "top_n": _pool_top_n_hold(pool),
        "short_ma_days": short_ma_days,
        "long_ma_days": long_ma_days,
        "adr_floor": adr_floor,
    }


def _pool_top_n_hold(pool: str) -> int:
    from utils.pool_settings_store import get_pool_top_n_hold

    return get_pool_top_n_hold(pool)


def pool_options() -> list[dict[str, Any]]:
    """풀 셀렉트 옵션 — 종목풀 설정(DB)의 이름·아이콘·순서를 단일 소스로 쓴다.

    화면은 이 목록을 공용 `formatPoolLabel`(pools-rank 와 같은 표준 표기)에 그대로
    넣는다. 반환 순서 = 종목풀 order 순.
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
                # 화면이 국가별 컬럼(마켓·시가총액)과 통화 표기를 결정하는 데 쓴다.
                "country_code": str(settings.get("country_code") or "").strip().lower(),
                "currency": str(settings.get("currency") or "").strip().upper(),
                # 풀 성격(stock/etf) — 섹터·업종 컬럼과 업종상한 노출 판단용. 미설정이면 빈 값.
                "pool_kind": str(settings.get("pool_kind") or "").strip().lower(),
            }
        )
    return sorted(options, key=lambda item: (item["order"] is None, item["order"]))


# 풀 설정 문서의 대문자 키 ↔ 전략 설정의 소문자 키. 저장 위치는 `pool_settings` 문서 하나다
# (예전에는 `system_config.momentum_settings` 에 따로 저장돼 같은 풀의 값이 갈렸다).
_POOL_KEY_BY_SETTING: dict[str, str] = {
    "start_date": "MOMENTUM_START_DATE",
    "short_ma_days": "SHORT_MA_DAYS",
    "long_ma_days": "LONG_MA_DAYS",
    "adr_floor": "ADR_FLOOR",
}


def _settings_from_pool_doc(config: dict[str, Any]) -> dict[str, Any] | None:
    """풀 설정 문서 → 전략 설정(평면). 필수 키가 하나라도 없으면 None(미설정)."""
    result: dict[str, Any] = {}
    for setting_key, pool_key in _POOL_KEY_BY_SETTING.items():
        if pool_key not in config:
            # None 을 값으로 갖는 항목(ADR 하한 등)은 키 자체는 있어야 한다.
            if setting_key in ("adr_floor", "start_date"):
                continue
            return None
        result[setting_key] = config[pool_key]
    # 없는 선택 항목은 '미설정' 기본값으로 채운다 — 임의 보정이 아니라 스키마 기본이다.
    result.setdefault("start_date", None)
    result.setdefault("adr_floor", min(ADR_FLOOR_OPTIONS))
    return result


def _load_settings_doc() -> dict[str, Any]:
    """모든 풀의 전략 설정 — `{settings_by_pool: {풀: {...}}}`.

    각 풀의 `pool_settings` 문서에서 읽는다. 설정이 없는 풀은 맵에서 빠진다.
    """
    from utils.settings_loader import get_ticker_type_settings

    by_pool: dict[str, Any] = {}
    for pool in available_pools():
        try:
            config = get_ticker_type_settings(pool) or {}
        except Exception:
            continue
        per_pool = _settings_from_pool_doc(config)
        if per_pool is not None:
            by_pool[pool] = per_pool
    if not by_pool:
        raise RuntimeError("모멘텀 전략 설정이 저장된 종목풀이 없습니다 — 종목풀 설정에서 먼저 저장하세요.")
    return {"settings_by_pool": by_pool}


def load_settings_map() -> dict[str, Any]:
    """`{settings_by_pool}` — 화면이 풀 전환 시 즉시 그 풀의 설정을 채우는 데 쓴다."""
    return _load_settings_doc()


def default_pool() -> str:
    """설정이 저장된 풀 중 번호가 가장 빠른 풀 — 화면이 기억한 값이 없을 때의 기준점.

    "마지막으로 고른 풀"은 브라우저 취향이라 DB 에 두지 않는다(화면이 로컬스토리지에 기억).
    """
    saved = set(_load_settings_doc()["settings_by_pool"])
    for option in pool_options():
        if option["ticker_type"] in saved:
            return str(option["ticker_type"])
    raise RuntimeError("모멘텀 전략 설정이 저장된 종목풀이 없습니다 — 종목풀 설정에서 먼저 저장하세요.")


def load_settings(pool: str | None = None) -> dict[str, Any]:
    """그 풀의 설정을 평면 dict 로 반환한다 (`{pool, top_n, ...}`).

    풀을 주지 않으면 `default_pool()` 을 쓴다. 저장분이 없거나 깨졌으면 대체값 없이
    에러 — 기본값으로 슬쩍 넘어가면 그대로 저장되는 순간 실제 설정이 덮어써진다.
    """
    stored = _load_settings_doc()
    pool = str(pool or default_pool()).strip().lower()
    per_pool = stored["settings_by_pool"].get(pool)
    if not isinstance(per_pool, dict):
        raise RuntimeError(f"종목풀({pool})의 모멘텀 전략 설정이 없습니다 — 화면에서 저장하세요.")
    try:
        return validate_settings({"pool": pool, **per_pool})
    except ValueError as error:
        raise ValueError(f"저장된 모멘텀 전략 설정이 올바르지 않습니다: {error}") from error


# 화면 로드 때 선택지 밖 저장값을 보정할 항목 — (키, 라벨, 선택지)
_OPTION_FIELDS: tuple[tuple[str, str, tuple], ...] = (
    ("short_ma_days", "단기 이평", SHORT_MA_OPTIONS),
    ("adr_floor", "ADR 하한", ADR_FLOOR_OPTIONS),
    ("long_ma_days", "장기 이평", LONG_MA_OPTIONS),
)


def load_settings_for_view(pool: str | None = None) -> tuple[dict[str, Any], list[str]]:
    """화면용 로드 — 선택지 밖 저장값은 첫 선택지로 보정하고 내역을 함께 돌려준다
    (``utils.strategy_settings.coerce_to_options``). 배치·백테스트는 ``load_settings`` 를 쓴다."""
    stored = _load_settings_doc()
    pool = str(pool or default_pool()).strip().lower()
    per_pool = stored["settings_by_pool"].get(pool)
    if not isinstance(per_pool, dict):
        raise RuntimeError(f"종목풀({pool})의 모멘텀 전략 설정이 없습니다 — 화면에서 저장하세요.")
    merged = {"pool": pool, **per_pool}
    try:
        return coerce_to_options(merged, _OPTION_FIELDS, validate_settings)
    except ValueError as error:
        raise ValueError(f"저장된 모멘텀 전략 설정이 올바르지 않습니다: {error}") from error


def save_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """검증 후 그 풀의 **종목풀 설정 문서**에 저장하고 정규화된 평면 설정을 반환한다.

    다른 풀의 저장분은 건드리지 않는다(풀별 독립). 저장이 풀 문서에 들어가므로 순위 화면·
    보유종목 알림·종목풀 백테스트가 곧바로 같은 값을 본다 — 튜닝 「적용」의 목적이다.
    """
    normalized = validate_settings(settings)
    require_start_date(normalized)
    from utils.pool_settings_store import save_pool_settings

    pool = normalized["pool"]
    values = {pool_key: normalized[setting_key] for setting_key, pool_key in _POOL_KEY_BY_SETTING.items()}
    save_pool_settings(pool, values, save_method="모멘텀 전략")
    # 설정을 바꿨다 되돌리면 옛 키에 그대로 걸린다 — 그 사이 달라진 종목 목록이
    # 반영되지 않은 결과가 다시 나오므로, 저장할 때마다 비우고 새로 계산하게 한다.
    from utils.cache_invalidation import invalidate_pool_caches

    invalidate_pool_caches(pool)
    return normalized


# ── 유니버스 · 가격 ────────────────────────────────────────────────────────


def load_universe(pool: str) -> list[dict[str, str]]:
    """선택한 종목풀의 투자 후보 목록 — 정의는 `utils.stock_list_io` 한 곳이다."""
    from utils.stock_list_io import load_pool_universe

    return load_pool_universe(pool)


def load_price_frames(universe: list[dict[str, str]]) -> dict[str, pd.DataFrame]:
    from utils.cache_utils import load_cached_frames_bulk_from_ticker_types

    frames: dict[str, pd.DataFrame] = {}
    pools = sorted({row["pool"] for row in universe})
    for pool in pools:
        tickers = [row["ticker"] for row in universe if row["pool"] == pool]
        if tickers:
            frames.update(load_cached_frames_bulk_from_ticker_types([pool], tickers))
    return frames


# ── 벤치마크 (종목풀 설정 단일 소스) ────────────────────────────────────────
def _pool_benchmark(pool: str) -> dict[str, str]:
    """풀 설정(DB)의 BENCHMARK — 종목풀 화면에서 관리하는 값을 단일 소스로 쓴다.

    미설정이면 명시적으로 에러를 낸다(임의 지수로 대체하지 않는다).
    """
    from utils.pool_settings_store import get_pool_benchmark_ticker
    from utils.settings_loader import get_ticker_type_settings

    settings = get_ticker_type_settings(pool) or {}
    ticker = get_pool_benchmark_ticker(settings)
    if not ticker:
        raise RuntimeError(f"종목풀({pool})에 벤치마크가 설정돼 있지 않습니다 — 종목풀 설정 화면에서 지정하세요.")
    benchmark = settings.get("BENCHMARK") or {}
    return {"ticker": ticker, "name": str(benchmark.get("name") or ticker)}


def benchmark_info(pool: str) -> dict[str, str]:
    """벤치마크 {ticker, name} — 화면 표기용."""
    return _pool_benchmark(pool)


def load_benchmark_close(pool: str) -> pd.Series:
    """풀 설정의 벤치마크 티커 종가 — 전체 종목풀 캐시에서 찾는다.

    (예: us 풀의 벤치마크 QQQ 는 us 캐시에 있다)
    """
    from utils.cache_utils import load_cached_frames_bulk_from_all_ticker_types

    benchmark = _pool_benchmark(pool)
    frames = load_cached_frames_bulk_from_all_ticker_types([benchmark["ticker"]])
    frame = frames.get(benchmark["ticker"])
    if frame is None or frame.empty:
        raise RuntimeError(f"벤치마크({benchmark['name']}) 가격 캐시를 불러올 수 없습니다.")
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if close.empty:
        raise RuntimeError(f"벤치마크({benchmark['name']}) 종가가 비어 있습니다.")
    return close


# ── 모멘텀 ─────────────────────────────────────────────────────────────────
def momentum_metrics(
    close: pd.Series,
    *,
    short_ma_days: int,
    long_ma_days: int,
    as_of: pd.Timestamp | None = None,
) -> dict[str, float] | None:
    """종목풀 설정 이평선 기준 이격 — 순위 화면과 같은 신호의 월간 버전.

    이격률 계산은 (종가 ÷ 이평 − 1) × 100 이고, 이평선 일수는 종목풀 설정
    (SHORT_MA_DAYS/LONG_MA_DAYS)을, 이평 종류(SMA/EMA)는 공통 설정을 그대로 쓴다 —
    순위/종목풀 백테스트와 신호가 같고 리듬(월간 유지)만 다르다.
    순위 점수(`momentum_score`)는 순위 화면과 같은 `rank_score` 가 정한다 — **장기 이격률**이다.
    단기 이격은 순위에 넣지 않고 후보 자격 판정(hold_eligible_mask)에만 쓴다.
    ``as_of`` 를 주면 그 날짜까지의 데이터만 사용한다(백테스트·판정일 재현).
    """
    series = pd.to_numeric(close, errors="coerce").dropna()
    if as_of is not None:
        series = series[series.index <= as_of]

    # 이평선이 자리 잡을 만큼 가격이 쌓이지 않았으면 점수를 내지 않는다 — 부분 계산으로
    # 낸 값은 그 종목만 유리·불리해진다. 이 문턱을 넘으면 마지막 값은 창이 꽉 찬 상태다.
    if len(series) < long_ma_days + 4:
        return None

    # 이격률 — 순위 화면·보유종목 알림과 **같은 함수**(core.strategy.scoring.compute_ma_disparity).
    # 예전에는 여기서 따로 계산해 경계 종목에서 순위 화면과 값이 갈렸다.
    disparity_pct = compute_ma_disparity(series, long_ma_days)
    short_disparity_pct = compute_ma_disparity(series, short_ma_days)
    if disparity_pct is None or short_disparity_pct is None:
        return None

    return {
        "disparity_pct": disparity_pct,
        "short_disparity_pct": short_disparity_pct,
        # 순위 점수 — 순위 화면과 **같은 함수**를 쓴다(core.strategy.scoring.rank_score).
        # 정의를 바꾸려면 그 함수 하나만 고친다.
        "momentum_score": rank_score(disparity_pct, short_disparity_pct),
    }


# ── ADR 게이트 ─────────────────────────────────────────────────────────────
# 시장 폭(ADR)이 무너진 주는 개별주 모멘텀의 승률이 급락한다는 검증 결과(메모리
# adr-gate-research)에 따른 주간 스위치. 판정은 주간 판정일의 ADR 로 하고, 여기(후보
# 선정)에서 빈 목록을 돌려주면 엔진이 자연히 전량 매도·현금 대기로 흘러간다 —
# 선정·백테스트·연속 추적·합성이 같은 경로라 한 곳으로 충분하다.

# 시리즈 캐시 — 백테스트가 판정일마다 부르므로 DB 를 반복해서 읽지 않는다.
# 튜닝 워커는 DB 를 건드리지 않는 규칙이라, 부모가 프리로드해 `seed_adr_series` 로 심는다.
_ADR_SERIES_CACHE: dict[str, pd.Series] = {}


def adr_market_of_pool(pool: str) -> str | None:
    """풀의 ADR 기준 → 시장 키. 미설정이면 None.

    지수 4개(KOSPI·KOSDAQ·SP500·NDX100) 또는 **그 풀 자신**(`pool:<풀id>`) 이다.
    어느 쪽이든 `market_breadth_daily` 의 같은 스키마라 읽기·판정은 구분하지 않는다.
    """
    from utils.market_breadth_service import (
        MARKET_BY_INDEX_TICKER,
        SELF_POOL_REGIME_TICKER,
        pool_market_key,
    )
    from utils.pool_settings_store import get_pool_market_regime_index
    from utils.settings_loader import get_ticker_type_settings

    try:
        pool_settings = get_ticker_type_settings(pool) or {}
    except Exception:
        return None
    index = get_pool_market_regime_index(pool_settings)
    if index is None:
        return None
    if index["ticker"] == SELF_POOL_REGIME_TICKER:
        return pool_market_key(pool)
    return MARKET_BY_INDEX_TICKER.get(index["ticker"])


def load_adr_series(market: str) -> pd.Series:
    """시장의 일별 ADR 시계열 (날짜 오름차순). 창이 차기 전 구간은 없다."""
    cached = _ADR_SERIES_CACHE.get(market)
    if cached is not None:
        return cached
    from utils.market_breadth_service import load_adr_series as load_points

    series = pd.Series(
        {pd.Timestamp(point["date"]): float(point["adr"]) for point in load_points(market) if point["adr"] is not None}
    ).sort_index()
    _ADR_SERIES_CACHE[market] = series
    return series


def seed_adr_series(market: str, series: pd.Series) -> None:
    """튜닝 워커용 — 부모가 읽은 시계열을 심어 워커의 DB 접근을 없앤다."""
    _ADR_SERIES_CACHE[market] = series


def available_backtest_months(benchmark_close: pd.Series, long_ma_days: int) -> int:
    """이 종목풀에서 실제로 돌릴 수 있는 최대 개월 수.

    이격 계산에 쓸 ``장기 이평선 일수 + 4`` 거래일이 쌓인 뒤부터가 유효 구간이다. 그 전은
    후보가 하나도 안 잡혀 성과가 통째로 비므로 백테스트 범위에서 아예 뺀다 — 그래야 전략과
    벤치마크가 **같은 달**을 비교한다. 가격 캐시 시작일만 보는 `get_max_backtest_months()`
    는 이 워밍업을 몰라 실제보다 크게 나온다.
    """
    index = benchmark_close.index
    required_bars = int(long_ma_days) + 4
    if len(index) <= required_bars:
        return 1
    usable_days = len(index) - required_bars
    return max(usable_days * 12 // 252, 1)


def adr_max_backtest_months(pool: str) -> int | None:
    """ADR 이력이 통째로 덮는 최대 백테스트 개월 수. 레짐 미설정이면 None."""
    market = adr_market_of_pool(pool)
    if market is None:
        return None
    series = load_adr_series(market)
    if series.empty:
        return 0
    span_days = (pd.Timestamp.now().normalize() - series.index[0]).days
    return max(0, span_days * 12 // 365)
