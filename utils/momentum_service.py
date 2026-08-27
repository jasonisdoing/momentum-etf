"""모멘텀 전략 — 시장 대비 '꾸준한 모멘텀' 선정 서비스 (UI/API/스크립트 공용).

전략 규칙
--------
1. 유니버스: 설정에서 고른 종목풀 1개(전체 활성 풀 중 선택 — 한국·미국·호주 등).
   고정 보유 종목(exclude_from_ranking)은 제외. 코스피+코스닥 통합 같은 혼합
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
설정은 MongoDB `system_config.momentum_settings` 에 **풀별로** 저장한다
(`{pool, settings_by_pool: {풀: {top_n, ...}}}`) — 풀을 바꾸면 그 풀의 설정으로 전환된다.
"""

from __future__ import annotations

import math
import warnings
from datetime import datetime
from typing import Any

import pandas as pd

from config import MAX_PER_INDUSTRY_OPTIONS, STOP_LOSS_PCT_OPTIONS, TOP_N_OPTIONS
from core.strategy.scoring import (
    cap_by_industry,
    compute_ma_disparity,
    drawdown_from_high_pct,
    rank_score,
)
from utils.ma_options import LONG_MA_OPTIONS, SHORT_MA_OPTIONS
from utils.price_series import positive_prices
from utils.strategy_settings import coerce_to_options

warnings.filterwarnings("ignore")

# ── 상수 ──────────────────────────────────────────────────────────────────
# 종목풀은 DB(pool_settings)의 활성 풀 전체 중 1개를 선택한다 — 목록·국가·통화의
# 단일 소스는 종목풀 설정이다 (하드코딩 목록을 두지 않는다).
# 한 업종에서 최대 몇 종목까지 담을지 — 화면 셀렉트와 검증이 같은 목록을 쓴다.
# 종목 수·업종 상한은 신고가와 **의미가 같아** 전략 공용 상수를 그대로 쓴다
# (`utils/strategy_options`). 여기서 다시 정의하면 한쪽만 고쳐져 두 화면이 갈린다.
# 차순위 후보를 종목 수의 몇 배까지 보여줄지 — 선정과 같은 수(합계 2배)만 보여
# 표를 짧게 유지한다. 표 밖 '다음 주 예상' 종목은 하단에 별도 행으로 붙는다.
RESERVE_MULTIPLIER = 1
# 월↔거래일 환산 — 공용 상수(config, =20)를 재사용한다 (자산헬퍼·시장추세와 동일 기준).
from config import CACHE_TTL_COMPUTE, TRADING_DAYS_PER_MONTH  # noqa: E402
from utils.ttl_cache import TtlCache  # noqa: E402

_CONFIG_COLLECTION = "system_config"
_SETTINGS_KEY = "momentum_settings"

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
# 설정의 단일 소스는 DB(`system_config.momentum_settings`)다. 코드에 기본값을
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
    "top_n",
    "max_per_industry",
    "short_ma_days",
    "long_ma_days",
    "intraweek_exit",
    "intraweek_stop_pct",
)

# 주중 손절선 선택지(%) — 교체일 시가 대비 낙폭. None 은 '손절 없음'.
# kor·kospi200 24개월 백테스트에서 -10 이 공통 피크였다(-5 는 휩쏘 손실, -15 는 효과 소멸).
# 주중 손절 — 공용 손절 목록에 '안 씀(None)' 을 더한 것이다. 퍼센트 자체는 한 곳에서만 정한다.
INTRAWEEK_STOP_OPTIONS: tuple[float | None, ...] = (None, *STOP_LOSS_PCT_OPTIONS)

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

    # 선택지 목록이 단일 소스다 — 여기에 범위를 따로 두면 목록을 늘려도 저장이 막힌다
    # (실제로 2 를 추가했을 때 하한 5 에 걸렸다). 아래 다른 값들과 같은 방식으로 검증한다.
    top_n = int(_num("top_n"))
    if top_n not in TOP_N_OPTIONS:
        allowed = ", ".join(str(v) for v in TOP_N_OPTIONS)
        raise ValueError(f"'top_n' 은 {allowed} 중 하나여야 합니다 (받은 값: {top_n}).")
    raw_cap = settings.get("max_per_industry")
    max_per_industry = None if raw_cap in (None, "", "none") else int(_num("max_per_industry"))
    if max_per_industry not in MAX_PER_INDUSTRY_OPTIONS:
        allowed = ", ".join("없음" if v is None else str(v) for v in MAX_PER_INDUSTRY_OPTIONS)
        raise ValueError(f"'max_per_industry' 는 {allowed} 중 하나여야 합니다 (받은 값: {raw_cap}).")
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

    # 주중 이탈 — 풀 성격에 따라 켜고 끈다. 개별주는 급락 방어가 필요하고,
    # ETF 는 20일선 부근을 오르내리며 하루짜리 이탈이 잦아 왕복 비용만 커진다.
    intraweek_exit = settings.get("intraweek_exit")
    if not isinstance(intraweek_exit, bool):
        raise ValueError("'intraweek_exit' 는 true/false 여야 합니다.")

    # 주중 손절선 — 주중 이탈의 추가 조건이라 이탈이 켜져 있을 때만 의미가 있다.
    # 미설정(None)은 '손절 없음'. 저장돼 있던 값이 선택지 밖이면 명시적으로 에러.
    stop_raw = settings.get("intraweek_stop_pct")
    if stop_raw in (None, ""):
        intraweek_stop_pct = None
    else:
        try:
            intraweek_stop_pct = float(stop_raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"'intraweek_stop_pct' 는 숫자여야 합니다: {stop_raw}") from error
        if intraweek_stop_pct not in INTRAWEEK_STOP_OPTIONS:
            allowed = ", ".join(str(v) for v in INTRAWEEK_STOP_OPTIONS if v is not None)
            raise ValueError(f"'intraweek_stop_pct' 는 {allowed} 중 하나여야 합니다.")

    return {
        "pool": pool,
        "top_n": top_n,
        "max_per_industry": max_per_industry,
        "short_ma_days": short_ma_days,
        "long_ma_days": long_ma_days,
        "intraweek_exit": intraweek_exit,
        "intraweek_stop_pct": intraweek_stop_pct,
    }


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


def _load_settings_doc() -> dict[str, Any]:
    """저장 문서를 새 스키마(`{pool, settings_by_pool}`)로 읽는다. 구 스키마는 1회 마이그레이션.

    마이그레이션 대상: ① 2풀 배열 스키마(`pools: [..]`) ② 단일 pool 평면 스키마.
    둘 다 선택 풀의 설정으로 승계하고, 원본은 백업 필드로 보존한다.
    """
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패해 모멘텀 전략 설정을 읽을 수 없습니다.")
    doc = db[_CONFIG_COLLECTION].find_one({"_id": _SETTINGS_KEY}) or {}
    stored = doc.get("settings")
    if not isinstance(stored, dict):
        raise RuntimeError(
            f"저장된 모멘텀 전략 설정이 없습니다 ({_CONFIG_COLLECTION}.{_SETTINGS_KEY} 문서를 먼저 저장하세요)."
        )

    # 풀별 맵이 있으면 정상 스키마다. 선택 풀(`pool`)은 화면이 로컬스토리지에 기억하므로
    # 문서에 없어도 된다(옛 문서에 남아 있으면 무시한다).
    if isinstance(stored.get("settings_by_pool"), dict):
        return stored

    # ── 일회성 마이그레이션 → 풀별 맵 스키마 ──
    if isinstance(stored.get("pools"), (list, tuple)) and stored["pools"]:
        pool = str(stored["pools"][0]).strip().lower()
    elif isinstance(stored.get("pool"), str):
        pool = str(stored["pool"]).strip().lower()
    else:
        raise RuntimeError("저장된 모멘텀 전략 설정에서 종목풀을 알 수 없습니다.")
    per_pool = {key: stored[key] for key in PER_POOL_SETTING_KEYS if key in stored}
    validate_settings({"pool": pool, **per_pool})  # 승계 값 검증 — 깨진 값은 여기서 드러난다
    upgraded = {"pool": pool, "settings_by_pool": {pool: per_pool}}
    update: dict[str, Any] = {"settings": upgraded, "updated_at": datetime.now().isoformat()}
    if "settings_before_per_pool_migration" not in doc:
        update["settings_before_per_pool_migration"] = stored
    db[_CONFIG_COLLECTION].update_one({"_id": _SETTINGS_KEY}, {"$set": update})
    return upgraded


def load_settings_map() -> dict[str, Any]:
    """`{pool, settings_by_pool}` — 화면이 풀 전환 시 즉시 그 풀의 설정을 채우는 데 쓴다."""
    return _load_settings_doc()


def default_pool() -> str:
    """설정이 저장된 풀 중 번호가 가장 빠른 풀 — 화면이 기억한 값이 없을 때의 기준점.

    "마지막으로 고른 풀"은 브라우저 취향이라 DB 에 두지 않는다(화면이 로컬스토리지에 기억).
    """
    saved = set(_load_settings_doc()["settings_by_pool"])
    for option in pool_options():
        if option["ticker_type"] in saved:
            return str(option["ticker_type"])
    raise RuntimeError("모멘텀 전략 설정이 저장된 종목풀이 없습니다 — 화면에서 먼저 저장하세요.")


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
    ("top_n", "종목 수", TOP_N_OPTIONS),
    ("max_per_industry", "업종 상한", MAX_PER_INDUSTRY_OPTIONS),
    ("short_ma_days", "단기 이평", SHORT_MA_OPTIONS),
    ("long_ma_days", "장기 이평", LONG_MA_OPTIONS),
    ("intraweek_stop_pct", "주중 손절선", INTRAWEEK_STOP_OPTIONS),
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
    """검증 후 선택 풀의 설정으로 저장하고 정규화된 평면 설정을 반환한다.

    다른 풀의 저장분은 건드리지 않는다(풀별 독립).
    """
    normalized = validate_settings(settings)
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패했습니다.")
    pool = normalized["pool"]
    per_pool = {key: normalized[key] for key in PER_POOL_SETTING_KEYS}
    db[_CONFIG_COLLECTION].update_one(
        {"_id": _SETTINGS_KEY},
        {
            "$set": {
                f"settings.settings_by_pool.{pool}": per_pool,
                "updated_at": datetime.now().isoformat(),
            }
        },
        upsert=True,
    )
    # 설정을 바꿨다 되돌리면 옛 키에 그대로 걸린다 — 그 사이 달라진 종목 목록이
    # 반영되지 않은 결과가 다시 나오므로, 저장할 때마다 비우고 새로 계산하게 한다.
    from utils.cache_invalidation import invalidate_strategy_caches

    invalidate_strategy_caches()
    return normalized


# ── 유니버스 · 가격 ────────────────────────────────────────────────────────


def load_universe(pool: str) -> list[dict[str, str]]:
    """선택한 종목풀 1개의 종목 목록. [{ticker, name, pool}]

    고정 보유 종목(exclude_from_ranking)은 투자 후보가 아니므로 제외한다
    (순위·종목풀 백테스트와 같은 규칙).
    """
    from utils.stock_list_io import _load_ticker_type_stocks_raw

    universe: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in _load_ticker_type_stocks_raw(pool):
        ticker = str(item.get("ticker") or "").strip()
        if not ticker or ticker in seen:
            continue
        if bool(item.get("exclude_from_ranking")):
            continue
        seen.add(ticker)
        universe.append(
            {
                "ticker": ticker,
                "name": str(item.get("name") or ticker),
                "pool": pool,
                # 한국 통합 풀(코스피+코스닥)에서 마켓 구분 표시용 — 없으면 빈 값.
                "market": str(item.get("market") or "").strip(),
            }
        )
    return universe


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
    순위 점수(`momentum_score`)는 순위 화면과 같은 `rank_score` 가 정한다.
    단기 이격은 후보 자격 판정(hold_eligible_mask)에도 쓴다.
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


def select_candidates(
    universe: list[dict[str, str]],
    frames: dict[str, pd.DataFrame],
    settings: dict[str, Any],
    *,
    as_of: pd.Timestamp | None = None,
) -> list[dict[str, Any]]:
    """이격 후보 목록 — 보유 가능 조건은 순위 화면과 같은 hold_eligible_mask 를 쓴다.

    이평선 일수는 **전략 전용 설정**(short/long_ma_days)이다 — 두 풀을 섞어도
    같은 이평선으로 점수를 매겨 공정하게 비교된다.
    """
    from utils.rankings import hold_eligible_mask

    short_ma_days = int(settings["short_ma_days"])
    long_ma_days = int(settings["long_ma_days"])

    candidates: list[dict[str, Any]] = []
    for row in universe:
        frame = frames.get(row["ticker"])
        if frame is None:
            continue
        metrics = momentum_metrics(
            frame["Close"],
            short_ma_days=short_ma_days,
            long_ma_days=long_ma_days,
            as_of=as_of,
        )
        if metrics is None:
            continue
        candidates.append({**row, **metrics})

    if not candidates:
        return []
    # 장기 이격 > 0 & 단기 이격 >= 0 — 순위/종목풀 백테스트와 같은 단일 규칙.
    eligible = hold_eligible_mask(
        pd.Series([c["disparity_pct"] for c in candidates]),
        pd.Series([c["short_disparity_pct"] for c in candidates]),
    )
    return [c for c, ok in zip(candidates, eligible, strict=True) if ok]


def rank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """꾸준한 모멘텀 점수 순 정렬 — 선정·백테스트·연속 추적이 같은 순위를 쓴다."""
    return sorted(candidates, key=lambda item: item["momentum_score"], reverse=True)


def select_top(
    scored: list[dict[str, Any]],
    top_n: int,
    max_per_industry: int | None,
    industry_by_ticker: dict[str, str],
) -> list[dict[str, Any]]:
    """점수 순서를 지키되 **한 업종이 상한을 넘지 않도록** 상위 top_n 을 고른다.

    상한 규칙 자체는 신고가 전략과 **같은 함수**(`core.strategy.scoring.cap_by_industry`)다 —
    같은 「업종 상한」 설정이 전략마다 다르게 동작하지 않도록 한 곳에서만 정의한다.
    여기서는 점수 순 dict 목록을 티커 순서로 바꿔 넘기고 결과를 다시 dict 로 되돌린다.
    선정·백테스트·연속 추적이 모두 이 함수를 써야 결과가 서로 어긋나지 않는다.
    """
    by_ticker = {item["ticker"]: item for item in scored}
    picked, _blocked = cap_by_industry(
        [item["ticker"] for item in scored],
        industry_by_ticker,
        max_per_industry,
        top_n,
    )
    return [by_ticker[ticker] for ticker in picked]


# ── 주간 리밸런싱 시점 ─────────────────────────────────────────────────────
def simulate_intraweek_exits(
    frames: dict[str, pd.DataFrame],
    settings: dict[str, Any],
    holdings: set[str],
    index: pd.DatetimeIndex,
    scan_start: pd.Timestamp,
    scan_end: pd.Timestamp,
) -> list[dict[str, Any]]:
    """주중 매도 — 보유 종목이 자격을 잃으면 다음 거래일 시가에 판다.

    [scan_start, scan_end] 의 각 거래일 종가로 보유 자격(hold_eligible_mask 와 같은
    장기 이격 > 0 그리고 단기 이격 >= 0)을 확인하고, 잃은 종목을 매도 목록에 넣는다.
    판 슬롯은 **다음 주 교체까지** 현금이다 — 주중 재매수는 하지 않는다(그 자리는
    다음 주 재선정이 채운다).

    반환 항목의 ``sell_date`` 가 None 이면 아직 체결 전(다음 거래일 시가 매도 예정)이다.
    선정 화면과 백테스트가 모두 이 함수를 써야 결과가 어긋나지 않는다.

    설정의 ``intraweek_exit`` 이 꺼져 있으면 주중에는 팔지 않는다(주 교체일에만 정리).
    ETF 풀처럼 20일선 부근을 오르내리는 종목이 많으면 하루짜리 이탈이 잦아 왕복
    비용만 커지기 때문이다 — 풀 성격에 맞춰 화면에서 켜고 끈다.
    """
    if not settings.get("intraweek_exit", True):
        return []

    from utils.moving_averages import calculate_moving_average

    short_ma_days = int(settings["short_ma_days"])
    long_ma_days = int(settings["long_ma_days"])
    stop_raw = settings.get("intraweek_stop_pct")
    stop_pct = float(stop_raw) if stop_raw is not None else None

    # 종목별 이격 시계열 — momentum_metrics 와 같은 이평선을 한 번만 계산해 재사용한다
    # (SMA/EMA 모두 날짜 d 의 값은 d 이후 데이터와 무관해 as_of 절단과 결과가 같다).
    disparity_cache: dict[str, tuple[pd.Series, pd.Series] | None] = {}

    def disparity(ticker: str) -> tuple[pd.Series, pd.Series] | None:
        if ticker not in disparity_cache:
            frame = frames.get(ticker)
            close = (
                pd.to_numeric(frame["Close"], errors="coerce").dropna()
                if frame is not None and not frame.empty and "Close" in frame.columns
                else None
            )
            if close is None or len(close) < long_ma_days + 4:
                disparity_cache[ticker] = None
            else:
                short_ma = calculate_moving_average(close, short_ma_days, min_periods=short_ma_days)
                long_ma = calculate_moving_average(close, long_ma_days, min_periods=long_ma_days)
                # 손절선 기준가 — 이번 구간(주) 시작 체결가(교체일 시가). 0 가격은 거래정지 칸.
                entry_price = None
                if stop_pct is not None and frame is not None and not frame.empty and "Open" in frame.columns:
                    opens = positive_prices(frame["Open"]).dropna()
                    if len(opens):
                        entry_raw = opens.asof(scan_start)
                        if pd.notna(entry_raw) and float(entry_raw) > 0:
                            entry_price = float(entry_raw)
                disparity_cache[ticker] = (
                    (close / short_ma - 1.0) * 100.0,
                    (close / long_ma - 1.0) * 100.0,
                    close,
                    entry_price,
                )
        return disparity_cache[ticker]

    remaining = set(holdings)
    exits: list[dict[str, Any]] = []
    for day in index[(index >= scan_start) & (index <= scan_end)]:
        later = index[index > day]
        sell_date = later[0] if len(later) > 0 else None
        for ticker in sorted(remaining):
            series = disparity(ticker)
            if series is None:
                continue  # 이격을 계산할 데이터가 없으면 신호 없음 — 유지
            try:
                short_value = series[0].asof(day)
                long_value = series[1].asof(day)
            except Exception:
                continue
            if pd.isna(short_value) or pd.isna(long_value):
                continue
            eligible = float(long_value) > 0 and float(short_value) >= 0
            # 주중 손절 — 교체일 시가 대비 종가 낙폭이 손절선 이하면 자격과 무관하게 판다.
            # kor·kospi200 백테스트에서 -10% 부근이 수익·MDD·소르티노를 함께 개선했다.
            hit_stop = False
            if stop_pct is not None and series[3] is not None:
                price = series[2].asof(day)
                if pd.notna(price):
                    hit_stop = (float(price) / series[3] - 1.0) * 100.0 <= stop_pct
            if eligible and not hit_stop:
                continue
            remaining.discard(ticker)
            exits.append(
                {
                    "ticker": ticker,
                    "signal_date": day,
                    "sell_date": sell_date,
                    # 발동 사유 — 화면·체결 목록이 그대로 쓴다. 둘 다 걸리면 손절을 앞세운다
                    # (손절이 더 급한 신호이고, 이평선 문구로 표시하면 원인을 오해한다).
                    "reason": "주중 손절" if hit_stop else "주중 이탈",
                }
            )
    return exits


def _signal_date_for(benchmark_close: pd.Series, rebalance_date: pd.Timestamp) -> pd.Timestamp:
    """교체일 직전의 캐시 거래일 — 과거 월 판정일 계산용."""
    prior = benchmark_close.index[benchmark_close.index < rebalance_date]
    if len(prior) == 0:
        raise RuntimeError("판정 기준일(교체일 직전 거래일)을 구할 수 없습니다.")
    return prior[-1]


# 업종 맵은 공용 모듈(utils/industry_map)이 단일 소스 — 순위·신고점 화면과 같은 값을 쓴다.
from utils.industry_map import industry_map  # noqa: E402


def available_backtest_months(benchmark_close: pd.Series, long_ma_days: int) -> int:
    """이 종목풀에서 실제로 돌릴 수 있는 최대 개월 수.

    두 가지 제약을 함께 본다.

    1. 각 교체일에는 그 **직전 거래일(판정일)** 이 있어야 한다.
    2. 그 판정일까지 이격 계산에 쓸 ``장기 이평선 일수 + 4`` 거래일이 쌓여 있어야
       한다. 이 조건 전 구간은 후보가 하나도 안 잡혀 성과가 통째로 비므로, 아예
       백테스트 범위에서 제외한다. 그래야 전략과 벤치마크가 **같은 달**을
       비교하게 된다.

    가격 캐시 시작일만 보는 `get_max_backtest_months()` 는 이 워밍업을 모르기
    때문에 실제보다 크게 나온다. ``long_ma_days`` 는 전략 전용 설정값이다.
    """
    index = benchmark_close.index
    month_ends = index.to_series().groupby(index.to_period("M")).max().tolist()
    required_bars = int(long_ma_days) + 4
    # 월말 직전 거래일까지 쌓인 봉 수가 required_bars 이상인 월말만 교체일이 될 수 있다.
    usable = sum(1 for month_end in month_ends if index.searchsorted(month_end, side="left") >= required_bars)
    # 개월 수 N 은 월말 N+1 개를 쓰므로, 쓸 수 있는 월말 개수보다 1 적다.
    return max(usable - 1, 1)


def completed_week_starts(benchmark_close: pd.Series) -> list[pd.Timestamp]:
    """캐시에 있는 주별 교체일(주 첫 거래일) 목록 (오름차순) — 연속 편입 추적 등 과거 이력용.

    어디까지가 '지난 주'인지는 호출부가 기준 교체일로 자른다 (오늘 기준으로 자르면
    아직 체결 전인 다음 교체를 보여줄 때 직전 주가 빠진다).
    """
    index = benchmark_close.index
    return list(index.to_series().groupby(index.to_period("W")).min())


def week_last_trading_day(country: str, day: pd.Timestamp) -> str:
    """그 날짜가 속한 주(월~일)의 마지막 거래일 — 주간 표의 기준일 라벨."""
    from utils.trading_calendar import get_trading_days

    monday = (pd.Timestamp(day) - pd.Timedelta(days=int(pd.Timestamp(day).weekday()))).normalize()
    try:
        days = get_trading_days(
            monday.strftime("%Y-%m-%d"), (monday + pd.Timedelta(days=6)).strftime("%Y-%m-%d"), country
        )
    except Exception:
        days = []
    return (pd.Timestamp(days[-1]) if days else pd.Timestamp(day)).strftime("%Y-%m-%d")


def week_rebalance_pair(country: str, week_monday: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """그 주의 (교체일 L = 첫 거래일, 판정일 L−1 = 직전 거래일) — 거래일 캘린더 기준.

    판정일은 보통 전주 마지막 거래일(금요일)이다 — 한 주의 종가를 모두 보고 판정한 뒤
    주말 동안 검토하고 다음 주 첫 거래일 시가에 체결한다.
    캘린더가 그 주를 커버하지 않으면 None.
    """
    from utils.trading_calendar import get_trading_days

    try:
        days = get_trading_days(
            week_monday.strftime("%Y-%m-%d"),
            (week_monday + pd.Timedelta(days=6)).strftime("%Y-%m-%d"),
            country,
        )
        if not days:
            return None
        first = pd.Timestamp(days[0]).normalize()
        prior = get_trading_days(
            (first - pd.Timedelta(days=14)).strftime("%Y-%m-%d"),
            (first - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            country,
        )
    except Exception:
        return None
    if not prior:
        return None
    return first, pd.Timestamp(prior[-1]).normalize()


def current_portfolio_dates(benchmark_close: pd.Series, country: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """현재 보여줄 포트폴리오의 (교체일 L, 판정일 L−1) — 주 단위, 거래일 캘린더 기준.

    교체일은 각 주의 **첫 거래일**, 판정일은 그 직전 거래일(= 전주 마지막 거래일)이다.
    판정일 종가가 확정된 다음날(달력)부터 그 교체분을 보여준다. 판정일이 고정이므로
    같은 구간 안에서는 몇 번을 실행해도 결과가 같다.
    """
    now = pd.Timestamp.now()
    today = now.normalize()
    index = benchmark_close.index
    cache_max = index[-1].normalize()
    this_monday = (today - pd.Timedelta(days=int(today.weekday()))).normalize()

    # 다음 주부터 거슬러 올라가며 '판정 종가가 확정된' 가장 최근 교체를 찾는다.
    # 금요일 종가로 판정이 끝나면(주말) 아직 체결 전이라도 그 교체분을 보여준다 —
    # 주말에 검토하라고 만든 리듬이라 확정된 다음 포트폴리오가 보여야 한다.
    for weeks_back in range(-1, 6):
        pair = week_rebalance_pair(country, this_monday - pd.Timedelta(weeks=weeks_back))
        if pair is None:
            continue
        rebalance_date, signal_calendar = pair
        if today <= signal_calendar or cache_max < signal_calendar:
            continue
        prior = index[index <= signal_calendar]
        if len(prior) == 0:
            continue
        return rebalance_date, prior[-1]
    raise RuntimeError("판정 가능한 주간 교체를 찾지 못했습니다 — 캘린더/캐시 데이터를 확인하세요.")


# ── 선정 (현재 포트폴리오) ─────────────────────────────────────────────────
# 유니버스 전체의 가격 프레임을 읽고 판정까지 하는 계산이라 수십 초 걸린다.
# 설정·기준일이 같으면 결과도 같으므로 짧게 재사용한다(설정을 바꾸면 키가 달라진다).
_PICKS_CACHE = TtlCache(CACHE_TTL_COMPUTE, name="momentum_picks")


def compute_picks(settings: dict[str, Any] | None = None, as_of: str | None = None) -> dict[str, Any]:
    """현재 적용 중인 월 확정 포트폴리오 — 화면 ③ 카드와 스크립트가 함께 쓴다.

    ``as_of`` 를 주면 그 날짜까지의 데이터만 사용해 그날의 상태를 재현한다
    (신고가 화면의 과거 날짜 조회와 같은 규칙 — 실시간은 섞지 않는다).
    """
    if settings is None:
        settings = load_settings()
    settings = validate_settings(settings)
    cache_key = _PICKS_CACHE.make_key(settings, as_of or "")
    return _PICKS_CACHE.get_or_compute(cache_key, lambda: _compute_picks(settings, as_of))


def _compute_picks(settings: dict[str, Any], as_of: str | None) -> dict[str, Any]:
    pool = str(settings["pool"])
    info = pool_info(pool)
    universe = load_universe(pool)
    frames = load_price_frames(universe)
    country = info["country"]
    currency = info["currency"]

    cutoff = pd.Timestamp(as_of) if as_of else None
    if cutoff is not None:
        frames = {
            ticker: (frame[frame.index <= cutoff] if frame is not None and not frame.empty else frame)
            for ticker, frame in frames.items()
        }

    # 실시간을 얹기 전 원본 — 체결가(시가)가 필요한 계산은 이 프레임을 쓴다.
    # 아래 실시간 반영본은 Close 만 남기므로 시가가 사라진다.
    cached_frames = frames

    # ── 실시간 반영 (종목풀 화면과 같은 규칙) ──────────────────────────────
    # 유니버스 전체의 종가 시리즈 끝에 실시간 현재가를 붙인다(공용
    # build_effective_close_series). 이후의 '현재' 기준 계산 — 다음 주 예상·예상
    # 순위·현재 이격·이번달 수익률·고점 — 이 전부 장중 가격을 본다.
    # 판정일 기준 계산은 as_of 필터가 오늘 행을 잘라내므로 영향이 없다.
    # 과거 날짜 재현 중에는 실시간을 섞지 않는다 — 그날 상태를 보는 화면이다.
    realtime: dict[str, dict[str, Any]] = {}
    if cutoff is None:
        try:
            from services.price_service import get_realtime_snapshot

            realtime = get_realtime_snapshot(country, [row["ticker"] for row in universe])
        except Exception:
            realtime = {}  # 실시간 실패는 캐시 폴백 — 화면이 값 없이 뜨는 것보단 어제 종가가 낫다.
    if realtime:
        from utils.rankings import build_effective_close_series

        effective_frames: dict[str, pd.DataFrame] = {}
        for frame_ticker, frame in frames.items():
            entry = realtime.get(frame_ticker)
            if not entry or frame is None or frame.empty:
                effective_frames[frame_ticker] = frame
                continue
            cached_close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
            eff_close = build_effective_close_series(cached_close, entry)
            if eff_close is None:
                effective_frames[frame_ticker] = frame
            else:
                # 시가는 원본을 보존한다 — 주중 손절선 기준가(교체일 시가)가 여기서 나온다.
                # 종가만 남기면 손절 판정이 조용히 무력화된다(오늘 행의 시가는 없음으로 둔다).
                columns = {"Close": eff_close}
                if "Open" in frame.columns:
                    columns["Open"] = pd.to_numeric(frame["Open"], errors="coerce")
                effective_frames[frame_ticker] = pd.DataFrame(columns)
        frames = effective_frames

    benchmark_close = load_benchmark_close(pool)
    if cutoff is not None:
        benchmark_close = benchmark_close[benchmark_close.index <= cutoff]
        if len(benchmark_close) == 0:
            raise RuntimeError(f"{as_of} 이전의 벤치마크 데이터가 없습니다.")
    rebalance_date, signal_date = current_portfolio_dates(benchmark_close, info["country"])
    candidates = select_candidates(universe, frames, settings, as_of=signal_date)
    scored = rank_candidates(candidates)

    top_n = int(settings["top_n"])
    max_per_industry = settings["max_per_industry"]  # None = 제한없음
    industry_map_by_ticker = industry_map(pool)
    industry_by_ticker = industry_map_by_ticker

    selected = select_top(scored, top_n, max_per_industry, industry_by_ticker)
    # 차순위 후보 — 선정에 못 든 종목 중 점수 상위 N개 (화면에서 흐리게 붙여 보여준다).
    # 선정에서 빠진 자리를 메울 후보라 업종 상한은 적용하지 않는다.
    selected_tickers = {item["ticker"] for item in selected}

    # 연속 편입 주 — 직전 최대 11주의 판정일마다 같은 규칙으로 선정을 재계산해,
    # 현재 종목이 연속으로 상위 N 에 들어 있던 횟수(+이번 주)를 센다. 끊기면 중단.
    streak_lookback = 11
    # 날짜가 아니라 '주'로 비교한다 — as_of 로 잘린 데이터에서는 교체일이 속한 주의
    # 마지막 캐시 거래일이 교체일보다 앞서 같은 주가 이중 계산된다.
    prior_rebalances = [
        stamp
        for stamp in completed_week_starts(benchmark_close)
        if stamp.to_period("W") < rebalance_date.to_period("W")
    ][-streak_lookback:]
    streaks = {item["ticker"]: 1 for item in selected}
    alive = set(streaks)
    # 연속 편입이 시작된 교체일 — '편입 후 수익률'(그날 시가 대비)의 기준점이다.
    streak_entry = {item["ticker"]: rebalance_date for item in selected}
    for prior_rebalance in reversed(prior_rebalances):
        if not alive:
            break
        prior_signal = _signal_date_for(benchmark_close, prior_rebalance)
        prior_candidates = select_candidates(universe, frames, settings, as_of=prior_signal)
        prior_top = {
            item["ticker"]
            for item in select_top(rank_candidates(prior_candidates), top_n, max_per_industry, industry_by_ticker)
        }
        for ticker in list(alive):
            if ticker in prior_top:
                streaks[ticker] += 1
                streak_entry[ticker] = prior_rebalance
            else:
                alive.discard(ticker)
    # 조회 창(11주)을 다 써도 살아 있으면 시작점을 모른다 — 수익률을 지어내지 않는다.
    streak_capped = alive if len(prior_rebalances) >= streak_lookback else set()

    # 이 포트폴리오를 들고 가는 주 — 체결일이 속한 주의 마지막 거래일로 부른다.
    portfolio_week = week_last_trading_day(info["country"], rebalance_date)

    # ── 주중 매도 (상시) — 보유 자격을 잃으면 다음 거래일 시가에 판다 (백테스트와 동일 함수).
    # 마지막 확정 종가에서 판정된 것은 아직 체결 전 → '매도 예정' 으로 표시한다.
    intraweek_exits = simulate_intraweek_exits(
        frames,
        settings,
        {item["ticker"] for item in selected},
        benchmark_close.index,
        rebalance_date,
        benchmark_close.index[-1],
    )
    exit_by_ticker = {exit_info["ticker"]: exit_info for exit_info in intraweek_exits}

    # ── 지금 실제로 들고 있는 종목 ──────────────────────────────────────────
    # 확정된 교체가 아직 체결 전(주말·판정 다음날)이면 **직전 교체분**이 현재 보유다.
    # 화면이 '보유'와 '체결 예정'을 섞지 않도록 여기서 갈라 준다.
    selected_tickers_now = [item["ticker"] for item in selected]
    is_filled = rebalance_date <= pd.Timestamp.now().normalize()
    # 현재 보유가 체결된 날과 그 구간에 주중 매도된 종목 — 슬리브 비중 계산이 쓴다.
    held_since: pd.Timestamp | None = None
    period_exits: dict[str, Any] = {}
    if is_filled:
        held_since = rebalance_date
        held_tickers = [t for t in selected_tickers_now if t not in exit_by_ticker]
        period_exits = {
            ticker: info.get("sell_date") for ticker, info in exit_by_ticker.items() if ticker in selected_tickers_now
        }
    else:
        prev_monday = rebalance_date - pd.Timedelta(days=int(rebalance_date.weekday())) - pd.Timedelta(weeks=1)
        prev_pair = week_rebalance_pair(country, prev_monday)
        held_tickers = []
        if prev_pair is not None:
            prev_rebalance, prev_signal_calendar = prev_pair
            prior_index = benchmark_close.index[benchmark_close.index <= prev_signal_calendar]
            if len(prior_index) > 0:
                prev_selected = {
                    item["ticker"]
                    for item in select_top(
                        rank_candidates(select_candidates(universe, frames, settings, as_of=prior_index[-1])),
                        top_n,
                        max_per_industry,
                        industry_by_ticker,
                    )
                }
                prev_exit_list = simulate_intraweek_exits(
                    frames,
                    settings,
                    prev_selected,
                    benchmark_close.index,
                    prev_rebalance,
                    benchmark_close.index[-1],
                )
                prev_exited = {info["ticker"] for info in prev_exit_list}
                held_tickers = sorted(prev_selected - prev_exited)
                held_since = prev_rebalance
                period_exits = {info["ticker"]: info["sell_date"] for info in prev_exit_list}

    def entry_return_pct(ticker: str) -> float | None:
        """편입 후 수익률(%) — 연속 편입 시작 교체일의 **시가** 대비 현재가.

        실제로 들고 있는 종목만 계산한다(신규 선정·미체결은 None). 조회 창(11주)을
        넘겨 시작점을 모르는 종목도 None — 값을 지어내지 않는다.
        """
        if ticker not in set(held_tickers) or ticker in streak_capped:
            return None
        entry_date = streak_entry.get(ticker)
        if entry_date is None:
            return None
        if entry_date > benchmark_close.index[-1]:
            # 체결 당일 — 오늘 일봉이 아직 가격 캐시에 없다. 실시간 스냅샷의 오늘 시가가
            # 곧 체결가이므로 그 대비로 계산한다(시가가 안 온 종목만 None).
            if entry_date != pd.Timestamp.now().normalize():
                return None
            snap = realtime.get(ticker) or {}
            today_open = snap.get("open")
            if not today_open or float(today_open) <= 0:
                return None
            # 현재가 — 스냅샷 값이 없으면(장 마감 직후 등) 화면 현재가와 같은 실시간 반영 종가.
            price_now = snap.get("price")
            if not price_now:
                current = frames.get(ticker)
                close = pd.to_numeric(current["Close"], errors="coerce").dropna() if current is not None else None
                if close is None or close.empty:
                    return None
                price_now = float(close.iloc[-1])
            return round((float(price_now) / float(today_open) - 1) * 100, 2)
        frame = cached_frames.get(ticker)
        if frame is None or frame.empty or "Open" not in frame.columns:
            return None
        opens = positive_prices(frame["Open"]).dropna()
        if opens.empty:
            return None
        entry_open = opens.asof(entry_date)
        if pd.isna(entry_open) or float(entry_open) <= 0:
            return None
        current = frames.get(ticker)
        close = pd.to_numeric(current["Close"], errors="coerce").dropna() if current is not None else None
        if close is None or close.empty:
            return None
        return round((float(close.iloc[-1]) / float(entry_open) - 1) * 100, 2)

    def _sleeve_weights() -> tuple[dict[str, float], float]:
        """슬리브 안에서의 현재 비중(%) — (종목별, 현금).

        체결일에 모든 슬롯이 1/top_n 이었고 그 뒤 시세대로 흘러간 값이다. 합성 화면이
        '지금 이 종목이 몇 % 여야 하는지' 를 이 값으로 잡는다 — 고정 1/N 을 쓰면 목표와
        보유가 매일 어긋나 실제로는 하지 않을 매매가 계속 지시로 나온다.
        주중 매도된 슬롯은 매도 시가에서 멈춘 현금으로 남는다(다음 교체까지 재매수 없음).
        """
        if held_since is None:
            return {}, 100.0

        def open_at(ticker: str, day: Any) -> float | None:
            frame = cached_frames.get(ticker)
            if day is None or frame is None or frame.empty or "Open" not in frame.columns:
                return None
            series = positive_prices(frame["Open"]).dropna()
            if series.empty:
                return None
            value = series.asof(day)
            return float(value) if pd.notna(value) else None

        def price_now(ticker: str) -> float | None:
            frame = frames.get(ticker)
            if frame is None or frame.empty or "Close" not in frame.columns:
                return None
            close = positive_prices(frame["Close"]).dropna()
            return float(close.iloc[-1]) if not close.empty else None

        # 슬롯 단위로 센다 — 시작은 전부 현금(top_n 슬롯), 채운 슬롯만 종목으로 옮긴다.
        units: dict[str, float] = {}
        cash_units = float(top_n)
        for ticker in held_tickers:
            base, price = open_at(ticker, held_since), price_now(ticker)
            if not base or not price:
                continue
            units[ticker] = price / base
            cash_units -= 1.0
        for ticker, sell_date in period_exits.items():
            base, sold = open_at(ticker, held_since), open_at(ticker, sell_date)
            if not base or not sold:
                continue
            cash_units += sold / base - 1.0  # 그 슬롯은 매도 시가에서 멈춘 현금이다
        total = sum(units.values()) + cash_units
        if total <= 0:
            return {}, 100.0
        return (
            {ticker: round(value / total * 100, 4) for ticker, value in units.items()},
            round(cash_units / total * 100, 4),
        )

    sleeve_weight_by_ticker, sleeve_cash_weight_pct = _sleeve_weights()

    def exit_flags(ticker: str) -> dict[str, Any]:
        """행에 얹는 주중 매도 상태 — 매도됨(체결 완료) / 매도 예정(다음 시가)."""
        exit_info = exit_by_ticker.get(ticker)
        if not exit_info:
            return {"is_exited": False, "is_exit_pending": False, "exit_date": None, "exit_reason": None}
        sold = exit_info["sell_date"] is not None
        return {
            "is_exited": sold,
            "is_exit_pending": not sold,
            "exit_date": exit_info["sell_date"].strftime("%Y-%m-%d") if sold else None,
            "exit_reason": exit_info.get("reason"),
        }

    def exit_forecast(ticker: str, current: dict[str, Any]) -> dict[str, Any]:
        """주중 이탈·손절 **예상** — 현재(장중) 가격으로 확정 판정(simulate_intraweek_exits)과
        같은 두 조건을 미리 본다: ① 자격 상실(장기 ≤ 0 또는 단기 < 0) ② 교체일 시가 대비
        낙폭이 손절선 이하. 오늘 종가로 확정되기 전의 예보라 화면 표시 전용이다.
        """
        none = {"is_exit_forecast": False, "exit_forecast_reason": None}
        if not settings.get("intraweek_exit", True) or cutoff is not None:
            return none
        if ticker not in set(held_tickers) or ticker in exit_by_ticker:
            return none
        # ② 손절 — 기준가는 이번 구간 시작 체결가(교체일 시가). 확정 판정과 같은 소스(캐시 시가)를
        #    쓰되, 체결 당일이라 캐시에 없으면 실시간 스냅샷의 오늘 시가로 대신한다.
        stop_raw = settings.get("intraweek_stop_pct")
        hit_stop = False
        if stop_raw is not None and held_since is not None:
            entry_price = None
            frame = cached_frames.get(ticker)
            if frame is not None and not frame.empty and "Open" in frame.columns:
                opens = positive_prices(frame["Open"]).dropna()
                if len(opens):
                    entry_raw = opens.asof(held_since)
                    if pd.notna(entry_raw) and float(entry_raw) > 0 and opens.index.asof(held_since) == held_since:
                        entry_price = float(entry_raw)
            if entry_price is None and held_since == pd.Timestamp.now().normalize():
                snap_open = (realtime.get(ticker) or {}).get("open")
                if snap_open and float(snap_open) > 0:
                    entry_price = float(snap_open)
            frame_now = frames.get(ticker)
            close_now = pd.to_numeric(frame_now["Close"], errors="coerce").dropna() if frame_now is not None else None
            if entry_price and close_now is not None and not close_now.empty:
                hit_stop = (float(close_now.iloc[-1]) / entry_price - 1.0) * 100.0 <= float(stop_raw)
        # ① 자격 상실 — 현재 이격(전략 이평선 기준).
        short_now, long_now = current.get("current_short_pct"), current.get("current_long_pct")
        lost = short_now is not None and long_now is not None and (float(long_now) <= 0 or float(short_now) < 0)
        if hit_stop:
            return {"is_exit_forecast": True, "exit_forecast_reason": "주중 손절 예상"}
        if lost:
            return {"is_exit_forecast": True, "exit_forecast_reason": "주중 이탈 예상(이평선 하회)"}
        return none

    # 다음 주 예상 — 오늘까지의 가격(실시간 반영 종가)으로 같은 규칙을 한 번 더 돌려,
    # 지금 교체한다면 뽑힐 종목을 표시한다. 실제 확정은 다음 판정일(주 마지막 거래일) 종가다.
    # 현재 기준 단기/장기 이격 — 표의 '현재-단기/장기' 컬럼용. 자격 필터와 무관하게
    # 행에 있는 종목은 전부 계산한다(단기 음수로 후보 탈락한 종목도 값은 보여준다).
    def current_disparity(ticker: str) -> dict[str, float | None]:
        frame = frames.get(ticker)
        metrics = (
            momentum_metrics(
                frame["Close"],
                short_ma_days=int(settings["short_ma_days"]),
                long_ma_days=int(settings["long_ma_days"]),
                as_of=None,
            )
            if frame is not None and not frame.empty
            else None
        )
        if metrics is None:
            return {"current_short_pct": None, "current_long_pct": None}
        return {
            "current_short_pct": round(metrics["short_disparity_pct"], 1),
            "current_long_pct": round(metrics["disparity_pct"], 1),
        }

    def signal_disparity(ticker: str) -> dict[str, float | None]:
        """판정일 기준 단기/장기 이격 — 후보 밖 종목(하단 예상 행)용 재계산."""
        frame = frames.get(ticker)
        metrics = (
            momentum_metrics(
                frame["Close"],
                short_ma_days=int(settings["short_ma_days"]),
                long_ma_days=int(settings["long_ma_days"]),
                as_of=signal_date,
            )
            if frame is not None and not frame.empty
            else None
        )
        if metrics is None:
            return {"signal_short_pct": None, "signal_long_pct": None}
        return {
            "signal_short_pct": round(metrics["short_disparity_pct"], 1),
            "signal_long_pct": round(metrics["disparity_pct"], 1),
        }

    current_scored = rank_candidates(select_candidates(universe, frames, settings, as_of=None))
    current_top = select_top(current_scored, top_n, max_per_industry, industry_by_ticker)
    next_expected: set[str] = {item["ticker"] for item in current_top}
    # 예상 순위 — 판정일 순위와 같은 규칙의 '현재 기준' 버전: 선정분 1~top_n,
    # 그 아래는 점수순으로 이어 붙인다(상한 미적용). 자격 미달(후보 밖)은 없음(None).
    expected_rank_by_ticker: dict[str, int] = {}
    for expected_rank, item in enumerate(
        [*current_top, *[c for c in current_scored if c["ticker"] not in next_expected]], start=1
    ):
        expected_rank_by_ticker[item["ticker"]] = expected_rank
    # 차순위 후보 — **현재 이격 기준** 선정 제외 상위 N. 주중에 새로 올라온 종목이
    # 보이도록 판정일이 아니라 지금 순위로 뽑는다(판정일 값은 컬럼에 재계산으로 표시).
    reserve = [item for item in current_scored if item["ticker"] not in selected_tickers][: top_n * RESERVE_MULTIPLIER]
    # 현재 표(선정+차순위)에 없는 예상 종목 — 하단에 별도 행으로 붙인다.
    table_tickers = {item["ticker"] for item in selected + reserve}
    extra_expected = [item for item in current_top if item["ticker"] not in table_tickers]

    # 참고용 가격 정보 — 현재가·일간(%)은 위에서 받은 실시간 스냅샷을 쓰고, 실패 시
    # 캐시 종가로 폴백. 월별 수익률은 pools-rank 와 같은 공용 계산이며, frames 에
    # 실시간이 반영돼 있어 이번 달 값은 장중 가격 기준이다.
    # 개월 수 = 이번 달 + 장기 이평선이 커버하는 과거 개월(장기 일수 ÷ 월 거래일)
    # — 점수가 보는 기간만큼의 월별 흐름을 함께 보여준다 (장기 120일·월 20일 → 1+6).
    from utils.rankings import build_recent_monthly_return_metrics, get_recent_monthly_return_labels

    month_count = 1 + max(1, int(settings["long_ma_days"]) // TRADING_DAYS_PER_MONTH)
    month_labels = get_recent_monthly_return_labels(month_count)

    row_tickers = [item["ticker"] for item in selected + reserve] + [item["ticker"] for item in extra_expected]

    # 시가총액(억) — /kor-market-stock 과 같은 네이버 marketValue 소스(10분 캐시).
    market_caps: dict[str, int] = {}
    if country == "kor":
        try:
            from utils.kor_stock_market_service import load_kor_market_caps

            market_caps = load_kor_market_caps(row_tickers)
        except Exception:
            market_caps = {}  # 보조 정보 — 실패해도 선정 표는 그대로 뜬다.
    # 시총 순위 — 배치 B 가 메타 캐시에 적어 둔 시장 전체 순위(개별주 풀만 값 있음). 화면은 읽기만 한다.
    from utils.market_cap_rank import market_cap_rank_of
    from utils.stock_cache_meta_io import get_stock_cache_meta_docs

    try:
        meta_docs = get_stock_cache_meta_docs(pool, row_tickers)
    except Exception:
        meta_docs = {}
    rank_by_ticker = {t: market_cap_rank_of((doc or {}).get("meta_cache")) for t, doc in meta_docs.items()}

    def price_info(ticker: str) -> dict[str, Any]:
        frame = frames.get(ticker)
        if frame is None or frame.empty:
            return {
                "price": None,
                "daily_change_pct": None,
                "high_drawdown_pct": None,
                "market_cap_eok": None,
                "market_cap_rank": rank_by_ticker.get(ticker),
                "monthly_returns": {label: None for label in month_labels},
            }
        close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
        # 고점 대비(%) — 순위 화면과 **같은 함수**(core.strategy.scoring.drawdown_from_high_pct).
        # frames 에 실시간 현재가가 반영돼 있어 pools-rank 와 값이 일치한다.
        raw_drawdown = drawdown_from_high_pct(close)
        high_drawdown_pct = None if raw_drawdown is None else round(raw_drawdown, 2)
        snap = realtime.get(ticker) or {}
        now_val = snap.get("nowVal")
        change_rate = snap.get("changeRate")
        price = (
            float(now_val)
            if isinstance(now_val, (int, float)) and float(now_val) > 0
            else (float(close.iloc[-1]) if not close.empty else None)
        )
        if isinstance(change_rate, (int, float)):
            daily_change_pct = round(float(change_rate), 2)
        elif len(close) >= 2:
            daily_change_pct = round((float(close.iloc[-1]) / float(close.iloc[-2]) - 1.0) * 100.0, 2)
        else:
            daily_change_pct = None
        return {
            "price": round(price, 4) if price is not None else None,
            "daily_change_pct": daily_change_pct,
            "high_drawdown_pct": high_drawdown_pct,
            "market_cap_eok": market_caps.get(ticker),
            "market_cap_rank": rank_by_ticker.get(ticker),
            "monthly_returns": build_recent_monthly_return_metrics(close, labels=month_labels),
        }

    name_by_ticker = {row["ticker"]: row.get("name") or row["ticker"] for row in universe}

    # 종목 메모 — 계좌가 아니라 **종목**에 붙는다(utils/stock_memo_store). 순위·자산 관리
    # 화면과 같은 값이고, 유니버스 전체를 한 번에 읽는다.
    from utils.stock_memo_store import get_stock_memos

    memo_by_ticker = get_stock_memos([row["ticker"] for row in universe])

    # 선정분(1~N) — 판정일 종가로 확정된 순서 그대로. 이 주에 실제로 들고 가는 목록이라
    # 지금 값으로 다시 세우면 확정된 편입 순서가 흔들린다.
    selected_rows = [
        {
            "rank": rank,
            "expected_rank": expected_rank_by_ticker.get(item["ticker"]),
            "is_reserve": False,
            "is_expected_only": False,
            "streak_weeks": streaks.get(item["ticker"], 1),
            # 편입 후 수익률 — 보유 중인 종목만(연속 편입 시작 교체일 시가 대비).
            "entry_return_pct": entry_return_pct(item["ticker"]),
            "next_week_expected": item["ticker"] in next_expected,
            "ticker": item["ticker"],
            "name": item["name"],
            "market": item.get("market", ""),
            "memo": memo_by_ticker.get(item["ticker"], ""),
            "industry": industry_map_by_ticker.get(item["ticker"], ""),
            "currency": currency,
            **price_info(item["ticker"]),
            "signal_short_pct": round(item["short_disparity_pct"], 1),
            "signal_long_pct": round(item["disparity_pct"], 1),
            # 순위 점수(장기·단기 평균) — 줄 세우기 기준. 이격률과 다른 값이라 따로 싣는다.
            "score": round(item["momentum_score"], 1),
            **(current := current_disparity(item["ticker"])),
            **exit_flags(item["ticker"]),
            **exit_forecast(item["ticker"], current),
        }
        for rank, item in enumerate(selected, start=1)
    ]

    # 선정 밖(차순위 + 예상 전용) — 구성원도 정렬도 **지금 이격** 기준이다. 이 구간의
    # 관심사는 '다음 교체에 무엇이 올라오는가' 라서, 판정일 기준으로 뽑으면 주중에 새로
    # 치고 올라온 종목(판정일엔 순위권 밖)이 정작 필요할 때 안 보인다. 판정일 이격은
    # 컬럼(판정일-단기/장기)에 재계산으로 채운다.
    other_rows = [
        {
            "rank": None,
            "expected_rank": expected_rank_by_ticker.get(item["ticker"]),
            "is_reserve": True,
            "is_expected_only": False,
            # 연속 편입은 선정분에만 의미가 있다 — 차순위는 표시하지 않는다(None → '-')
            "streak_weeks": None,
            "entry_return_pct": entry_return_pct(item["ticker"]),
            "next_week_expected": item["ticker"] in next_expected,
            "ticker": item["ticker"],
            "name": item["name"],
            "market": item.get("market", ""),
            "memo": memo_by_ticker.get(item["ticker"], ""),
            "industry": industry_map_by_ticker.get(item["ticker"], ""),
            "currency": currency,
            **price_info(item["ticker"]),
            # 판정일 기준 이격 — 현재 기준 후보라 판정일 값은 같은 이평선으로 재계산한다.
            **signal_disparity(item["ticker"]),
            "current_short_pct": round(item["short_disparity_pct"], 1),
            "current_long_pct": round(item["disparity_pct"], 1),
            "score": round(item["momentum_score"], 1),
            **exit_flags(item["ticker"]),
        }
        for item in reserve
    ] + [
        # 표 밖인데 다음 주 편입이 예상되는 종목 — 순위·현재 이격은 '지금' 기준이고,
        # 판정일-단기/장기는 같은 이평선으로 판정일 시점을 재계산해 채운다
        # (후보 필터 밖이었을 뿐 값 자체는 계산된다 — 단기 음수 등 탈락 사유가 그대로 보인다).
        {
            "rank": None,
            "expected_rank": expected_rank_by_ticker.get(item["ticker"]),
            "is_reserve": True,
            "is_expected_only": True,
            "streak_weeks": None,
            "next_week_expected": True,
            "ticker": item["ticker"],
            "name": item["name"],
            "market": item.get("market", ""),
            "memo": memo_by_ticker.get(item["ticker"], ""),
            "industry": industry_map_by_ticker.get(item["ticker"], ""),
            "currency": currency,
            **price_info(item["ticker"]),
            # 판정일 기준 이격 — 후보 밖이었어도 같은 이평선으로 재계산해 보여준다.
            **signal_disparity(item["ticker"]),
            "current_short_pct": round(item["short_disparity_pct"], 1),
            "current_long_pct": round(item["disparity_pct"], 1),
            "score": round(item["momentum_score"], 1),
        }
        for item in extra_expected
    ]
    other_rows.sort(key=lambda row: (row["score"] is None, -(row["score"] or 0.0)))
    for offset, row in enumerate(other_rows, start=len(selected_rows) + 1):
        row["rank"] = offset

    return {
        "as_of": signal_date.strftime("%Y-%m-%d"),
        "portfolio_week": portfolio_week,
        # 확정 교체가 이미 체결됐는지 — 거짓이면 rows 는 '다음 교체 예정' 목록이다.
        "is_filled": is_filled,
        # 지금 실제로 들고 있는 종목 — 화면의 '현재 보유' 표가 이걸 쓴다.
        "holdings": [
            {
                "ticker": ticker,
                "name": name_by_ticker.get(ticker, ticker),
                "industry": industry_map_by_ticker.get(ticker, ""),
                "currency": currency,
                # 다음 교체에서도 남는지 — 거짓이면 교체일에 매도된다.
                "keeps_next": ticker in set(selected_tickers_now),
                "entry_return_pct": entry_return_pct(ticker),
                # 이 슬리브 안에서의 비중(%) — 슬리브 전체를 100 으로 본다.
                "sleeve_weight_pct": sleeve_weight_by_ticker.get(ticker, 0.0),
                **price_info(ticker),
                **exit_flags(ticker),
            }
            for ticker in held_tickers
        ],
        # 빈 슬롯·주중 매도분 현금 비중 — 종목 비중과 합쳐 100 이 된다.
        "sleeve_cash_weight_pct": sleeve_cash_weight_pct,
        "rebalance_date": rebalance_date.strftime("%Y-%m-%d"),
        "signal_date": signal_date.strftime("%Y-%m-%d"),
        "universe_count": len(universe),
        "candidate_count": len(candidates),
        # 풀의 국가·통화 — 화면이 마켓·시가총액 컬럼 표시와 티커 표기(ASX: 등)를 정한다.
        "country": country,
        "currency": currency,
        # 월별 컬럼 라벨(최근 6개월, pools-rank 와 같은 형식) — 화면이 이 순서로 컬럼을 만든다.
        "monthly_return_labels": month_labels,
        "rows": selected_rows + other_rows,
    }
