"""Steady Momentum — 시장 대비 '꾸준한 모멘텀' 선정 서비스 (UI/API/스크립트 공용).

전략 규칙
--------
1. 유니버스: 설정에서 고른 한국 개별주 종목풀 1개. 고정 보유 종목(exclude_from_ranking)은 제외.
2. 점수: **장기 이평선 이격(%)** — 순위 화면·종목풀 백테스트와 같은 신호
   (이평선 일수는 종목풀 설정, 장기 이격 > 0 & 단기 이격 >= 0 만 후보).
   점수 순 상위 top_n 동일가중. 같은 규칙을 **월간 리듬으로 유지**하는 것이
   이 화면의 차이다 — 편입·편출을 월 단위로 보고, 한 달 동안 손절 없이 든다.
3. 월간 리밸런싱: 판정은 월말 직전 거래일(L−1) 종가, 체결은 월말(L) 종가.
   L−1 종가가 확정된 다음날부터 다음 달 포트폴리오를 보여준다(거래일 캘린더 기준).

벤치마크는 종목풀 설정(DB)의 BENCHMARK 를 단일 소스로 쓴다.
설정은 MongoDB `system_config.steady_momentum_settings` 에 저장한다.
"""

from __future__ import annotations

import math
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 상수 ──────────────────────────────────────────────────────────────────
# 선택 가능한 종목풀 — 한 번에 1개만 선택한다(통화·벤치마크·달력 혼합 방지).
# 한국 개별주 종목풀만 지원한다.
POOL_CONFIGS: dict[str, dict[str, Any]] = {
    "kor": {"label": "코스피 개별주", "country": "kor", "currency": "KRW"},
    "kor_kosdaq": {"label": "코스닥 개별주", "country": "kor", "currency": "KRW"},
}
AVAILABLE_POOLS = tuple(POOL_CONFIGS)
# 한 업종에서 최대 몇 종목까지 담을지 — 화면 셀렉트와 검증이 같은 목록을 쓴다.
MAX_PER_INDUSTRY_OPTIONS = (1, 2, 3, 4, 5, 10)
TRADING_DAYS_PER_MONTH = 21

_CONFIG_COLLECTION = "system_config"
_SETTINGS_KEY = "steady_momentum_settings"

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
# 설정의 단일 소스는 DB(`system_config.steady_momentum_settings`)다. 코드에 기본값을
# 두지 않는다 — 값이 없거나 깨졌으면 임의 값으로 대체하지 않고 에러를 낸다.


# ── 설정 ──────────────────────────────────────────────────────────────────
def validate_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """설정을 검증해 정규화된 dict 를 반환한다. 실패 시 ValueError."""
    if not isinstance(settings, dict):
        raise ValueError("설정 형식이 올바르지 않습니다.")

    def _num(key: str) -> float:
        value = settings.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"'{key}' 는 숫자여야 합니다.")
        return float(value)

    lookback_months = int(_num("lookback_months"))
    if not 1 <= lookback_months <= 24:
        raise ValueError("'lookback_months' 는 1~24 사이여야 합니다.")
    top_n = int(_num("top_n"))
    if not 5 <= top_n <= 100:
        raise ValueError("'top_n' 은 5~100 사이여야 합니다.")
    slippage_pct = _num("slippage_pct")
    if not 0.0 <= slippage_pct <= 1.0:
        raise ValueError("'slippage_pct' 는 0~1(%) 사이여야 합니다.")
    # 기간 상한은 종목풀 백테스트와 같은 계산(가격 캐시 시작일 기준)을 재사용한다.
    from utils.pool_signal_backtest_service import get_max_backtest_months

    max_months = get_max_backtest_months()
    max_per_industry = int(_num("max_per_industry"))
    if max_per_industry not in MAX_PER_INDUSTRY_OPTIONS:
        allowed = ", ".join(str(v) for v in MAX_PER_INDUSTRY_OPTIONS)
        raise ValueError(f"'max_per_industry' 는 {allowed} 중 하나여야 합니다.")
    backtest_months = int(_num("backtest_months"))
    if not 1 <= backtest_months <= max_months:
        raise ValueError(f"'backtest_months' 는 1~{max_months} 사이여야 합니다.")
    pool = str(settings.get("pool") or "").strip().lower()
    if pool not in AVAILABLE_POOLS:
        raise ValueError(f"지원하지 않는 종목풀입니다: {settings.get('pool')}")

    return {
        "pool": pool,
        "lookback_months": lookback_months,
        "top_n": top_n,
        "slippage_pct": slippage_pct,
        "max_per_industry": max_per_industry,
        "backtest_months": backtest_months,
    }


def pool_labels() -> dict[str, str]:
    """풀 표시 이름 — 종목풀 설정(DB)의 공식 이름을 단일 소스로 쓴다."""
    from utils.settings_loader import get_ticker_type_settings

    labels: dict[str, str] = {}
    for pool, config in POOL_CONFIGS.items():
        try:
            name = str((get_ticker_type_settings(pool) or {}).get("name") or "").strip()
        except Exception:
            name = ""
        labels[pool] = name or str(config["label"])
    return labels


def load_settings() -> dict[str, Any]:
    """저장된 설정을 반환한다. 없거나 읽을 수 없으면 대체값 없이 에러를 낸다.

    기본값으로 슬쩍 넘어가면 화면에는 그럴듯한 값이 뜨고, 그대로 저장하는 순간
    실제 저장돼 있던 설정이 덮어써진다. 그래서 실패는 실패로 드러낸다.
    """
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패해 Steady Momentum 설정을 읽을 수 없습니다.")
    doc = db[_CONFIG_COLLECTION].find_one({"_id": _SETTINGS_KEY}) or {}
    stored = doc.get("settings")
    if not isinstance(stored, dict):
        raise RuntimeError(
            f"저장된 Steady Momentum 설정이 없습니다 "
            f"({_CONFIG_COLLECTION}.{_SETTINGS_KEY} 문서를 먼저 저장하세요)."
        )
    try:
        return validate_settings(stored)
    except ValueError as error:
        # 스키마에 항목이 늘어난 경우 등 — 어느 값이 문제인지 그대로 드러낸다.
        raise ValueError(f"저장된 Steady Momentum 설정이 올바르지 않습니다: {error}") from error


def save_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """검증 후 저장하고 정규화된 설정을 반환한다.

    기간 상한은 룩백에 따라 달라지므로 여기서 실제 데이터로 한 번 더 막는다.
    (읽기 경로인 `load_settings` 에는 넣지 않는다 — 설정 조회가 가격 캐시에
    묶이면 캐시 문제로 설정 화면까지 못 여는 상황이 된다.)
    """
    normalized = validate_settings(settings)
    limit = available_backtest_months(
        load_benchmark_close(normalized["pool"]), normalized["lookback_months"]
    )
    if normalized["backtest_months"] > limit:
        raise ValueError(
            f"룩백 {normalized['lookback_months']}개월 기준으로 이 종목풀은 백테스트 "
            f"최대 {limit}개월입니다 (요청 {normalized['backtest_months']}개월). "
            f"기간을 줄이거나 룩백을 짧게 하세요."
        )
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패했습니다.")
    db[_CONFIG_COLLECTION].update_one(
        {"_id": _SETTINGS_KEY},
        {"$set": {"settings": normalized, "updated_at": datetime.now().isoformat()}},
        upsert=True,
    )
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
        universe.append({"ticker": ticker, "name": str(item.get("name") or ticker), "pool": pool})
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


def benchmark_name(pool: str) -> str:
    return _pool_benchmark(pool)["name"]


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
    benchmark_close: pd.Series,
    *,
    lookback_days: int,
    short_ma_days: int,
    long_ma_days: int,
    as_of: pd.Timestamp | None = None,
) -> dict[str, float] | None:
    """종목풀 설정 이평선 기준 이격 — 순위 화면과 같은 신호의 월간 버전.

    점수 = **장기 이평선 이격(%)** = (종가 ÷ 장기 이평 − 1) × 100. 이평선 일수는
    종목풀 설정(SHORT_MA_DAYS/LONG_MA_DAYS)을, 이평 종류(SMA/EMA)는 공통 설정을
    그대로 쓴다 — 순위/종목풀 백테스트와 신호가 같고 리듬(월간 유지)만 다르다.
    단기 이격은 후보 자격 판정(hold_eligible_mask)에 쓰며, 상대기울기·R²·월승률은
    참고 지표로 함께 계산한다.
    ``as_of`` 를 주면 그 날짜까지의 데이터만 사용한다(백테스트·판정일 재현).
    """
    series = pd.to_numeric(close, errors="coerce").dropna()
    bench = pd.to_numeric(benchmark_close, errors="coerce").dropna()
    if as_of is not None:
        series = series[series.index <= as_of]
        bench = bench[bench.index <= as_of]

    aligned = pd.concat([series.rename("stock"), bench.rename("bench")], axis=1, join="inner").dropna()
    min_rows = max(lookback_days, long_ma_days) + 4
    if len(aligned) < min_rows:
        return None

    # 이격 — 순위 화면과 같은 계산(공통 이동평균 헬퍼, min_periods=일수).
    from utils.moving_averages import calculate_moving_average

    stock_close = aligned["stock"]
    long_ma = float(calculate_moving_average(stock_close, long_ma_days, min_periods=long_ma_days).iloc[-1])
    short_ma = float(calculate_moving_average(stock_close, short_ma_days, min_periods=short_ma_days).iloc[-1])
    if long_ma <= 0 or short_ma <= 0:
        return None
    last_price = float(stock_close.iloc[-1])
    disparity_pct = (last_price / long_ma - 1.0) * 100.0
    short_disparity_pct = (last_price / short_ma - 1.0) * 100.0
    window = aligned.iloc[-lookback_days:]
    if (window["stock"] <= 0).any() or (window["bench"] <= 0).any():
        return None

    relative = window["stock"] / window["bench"]
    log_rel = np.log(relative.to_numpy())
    x = np.arange(len(log_rel), dtype=float)
    slope, intercept = np.polyfit(x, log_rel, 1)
    fitted = slope * x + intercept
    ss_res = float(np.sum((log_rel - fitted) ** 2))
    ss_tot = float(np.sum((log_rel - log_rel.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # t-통계량 — 기울기를 기울기 표준오차로 나눈다. 잔차가 0 에 가까운 완벽한
    # 직선은 표준오차가 0 이라 나눗셈이 불가한데, 그건 사실상 이상 데이터라 0 점 처리.
    n_points = len(log_rel)
    x_var = float(np.sum((x - x.mean()) ** 2))
    if n_points > 2 and x_var > 0 and ss_res > 0:
        slope_stderr = math.sqrt(ss_res / (n_points - 2) / x_var)
        t_stat = slope / slope_stderr if slope_stderr > 0 else 0.0
    else:
        t_stat = 0.0

    rel_slope_annual_pct = (math.exp(slope * 252) - 1.0) * 100.0
    rel_return_pct = (float(relative.iloc[-1]) / float(relative.iloc[0]) - 1.0) * 100.0
    absolute_return_pct = (float(window["stock"].iloc[-1]) / float(window["stock"].iloc[0]) - 1.0) * 100.0

    # 월승률 — 룩백 창의 월별 상대수익 중 플러스 달의 비율 (참고, 최대 6개월)
    monthly_last = relative.groupby(relative.index.to_period("M")).last()
    monthly_rel_returns = monthly_last.pct_change().dropna().tail(6)
    win_months = int((monthly_rel_returns > 0).sum())
    months_count = int(len(monthly_rel_returns))

    return {
        "slope_annual_pct": rel_slope_annual_pct,
        "r_squared": r_squared,
        "t_stat": t_stat,
        "disparity_pct": disparity_pct,
        "short_disparity_pct": short_disparity_pct,
        "momentum_score": disparity_pct,
        "return_lookback_pct": absolute_return_pct,
        "rel_return_pct": rel_return_pct,
        "win_months": win_months,
        "months_count": months_count,
    }


def select_candidates(
    universe: list[dict[str, str]],
    frames: dict[str, pd.DataFrame],
    settings: dict[str, Any],
    benchmark_close: pd.Series,
    *,
    as_of: pd.Timestamp | None = None,
) -> list[dict[str, Any]]:
    """이격 후보 목록 — 보유 가능 조건은 순위 화면과 같은 hold_eligible_mask 를 쓴다."""
    from utils.rankings import get_ticker_type_ma_rules, hold_eligible_mask

    lookback_days = int(settings["lookback_months"]) * TRADING_DAYS_PER_MONTH
    ma_rule = get_ticker_type_ma_rules(str(settings["pool"]))[0]
    short_ma_days = int(ma_rule["short_ma_days"])
    long_ma_days = int(ma_rule["long_ma_days"])

    candidates: list[dict[str, Any]] = []
    for row in universe:
        frame = frames.get(row["ticker"])
        if frame is None:
            continue
        metrics = momentum_metrics(
            frame["Close"],
            benchmark_close,
            lookback_days=lookback_days,
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
    max_per_industry: int,
    industry_by_ticker: dict[str, str],
) -> list[dict[str, Any]]:
    """점수 순서를 지키되 **한 업종이 상한을 넘지 않도록** 상위 top_n 을 고른다.

    상한에 걸린 종목은 건너뛰고 다음 순위가 그 자리를 채운다. 업종을 모르는
    종목(구성종목 파일에 없는 경우)은 묶을 근거가 없으므로 상한을 적용하지 않는다.
    선정·백테스트·연속 추적이 모두 이 함수를 써야 결과가 서로 어긋나지 않는다.
    """
    counts: dict[str, int] = {}
    picked: list[dict[str, Any]] = []
    for item in scored:
        if len(picked) >= top_n:
            break
        industry = industry_by_ticker.get(item["ticker"], "")
        if industry:
            if counts.get(industry, 0) >= max_per_industry:
                continue
            counts[industry] = counts.get(industry, 0) + 1
        picked.append(item)
    return picked


# ── 월간 리밸런싱 시점 ─────────────────────────────────────────────────────
def _signal_date_for(benchmark_close: pd.Series, rebalance_date: pd.Timestamp) -> pd.Timestamp:
    """교체일 직전의 캐시 거래일 — 과거 월 판정일 계산용."""
    prior = benchmark_close.index[benchmark_close.index < rebalance_date]
    if len(prior) == 0:
        raise RuntimeError("판정 기준일(교체일 직전 거래일)을 구할 수 없습니다.")
    return prior[-1]


def sector_industry_map(pool: str) -> dict[str, dict[str, str]]:
    """티커 → {sector, industry}. `/us-market-stock` 과 같은 지수 구성종목 파일을 쓴다.

    한국 개별주 풀은 stock_meta 의 sector/industry 를 쓴다 — 배치 B(식별·상세)가
    yfinance 로 수집한 값이라 미국 풀과 같은 분류 체계다. 아직 수집 전이거나
    yfinance 에 분류가 없는 종목(코스닥 소형주 등)은 업종 상한이 적용되지 않는다.

    분류 값은 yfinance 에서 받아 파일에 저장된 것이다(`scripts/update_us_market_stocks.py`).
    구성종목 목록만 위키피디아에서 오고, 섹터·업종은 두 지수가 한 체계를 쓰도록
    yfinance 로 통일했다.

    구성종목 파일이 아직 없으면 표시용 정보 하나 때문에 선정 전체가 막히지 않도록
    빈 맵으로 두되, 파일이 없다는 사실 자체는 로그로 남긴다.
    """
    from utils.index_constituents_loader import load_index_constituents

    if str(POOL_CONFIGS.get(pool, {}).get("country")) != "us":
        from utils.stock_list_io import _load_ticker_type_stocks_raw

        result: dict[str, dict[str, str]] = {}
        for item in _load_ticker_type_stocks_raw(pool):
            ticker = str(item.get("ticker") or "").strip()
            sector = str(item.get("sector") or "").strip()
            industry = str(item.get("industry") or "").strip()
            if ticker and (sector or industry):
                result[ticker] = {"sector": sector, "industry": industry or sector}
        return result

    result: dict[str, dict[str, str]] = {}
    for index_name in ("SP500", "NDX100"):
        try:
            constituents = load_index_constituents(index_name)
        except FileNotFoundError as error:
            warnings.warn(f"{index_name} 구성종목 파일이 없어 섹터·업종을 채우지 못했습니다: {error}", stacklevel=2)
            continue
        for item in constituents:
            ticker = str(item.get("ticker") or "").strip().upper()
            if not ticker or ticker in result:
                continue
            sector = str(item.get("sector") or "")
            result[ticker] = {"sector": sector, "industry": str(item.get("industry") or sector)}
    return result


def available_backtest_months(benchmark_close: pd.Series, lookback_months: int) -> int:
    """이 종목풀·룩백에서 실제로 돌릴 수 있는 최대 개월 수.

    두 가지 제약을 함께 본다.

    1. 각 교체일에는 그 **직전 거래일(판정일)** 이 있어야 한다.
    2. 그 판정일까지 회귀에 쓸 ``룩백 × 21 + 4`` 거래일이 쌓여 있어야 한다.
       이 조건 전 구간은 후보가 하나도 안 잡혀 성과가 통째로 비므로, 아예
       백테스트 범위에서 제외한다. 그래야 전략과 벤치마크가 **같은 달**을
       비교하게 된다.

    가격 캐시 시작일만 보는 `get_max_backtest_months()` 는 룩백을 모르기 때문에
    실제보다 크게 나온다. 룩백이 길수록 이 값은 줄어든다.
    """
    index = benchmark_close.index
    month_ends = index.to_series().groupby(index.to_period("M")).max().tolist()
    required_bars = int(lookback_months) * TRADING_DAYS_PER_MONTH + 4
    # 월말 직전 거래일까지 쌓인 봉 수가 required_bars 이상인 월말만 교체일이 될 수 있다.
    usable = sum(1 for month_end in month_ends if index.searchsorted(month_end, side="left") >= required_bars)
    # 개월 수 N 은 월말 N+1 개를 쓰므로, 쓸 수 있는 월말 개수보다 1 적다.
    return max(usable - 1, 1)


def completed_month_ends(benchmark_close: pd.Series) -> list[pd.Timestamp]:
    """지난 달들의 월말 거래일 목록 (오름차순) — 연속 편입 추적 등 과거 이력용."""
    index = benchmark_close.index
    today_period = pd.Timestamp.now().to_period("M")
    month_ends = index.to_series().groupby(index.to_period("M")).max()
    return [stamp for period, stamp in month_ends.items() if period < today_period]


def month_last_two_trading_days(
    country: str, period: pd.Period
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """해당 월의 (마지막 거래일 L, 그 직전 거래일 L−1) — 거래일 캘린더 기준.

    캘린더가 그 달을 커버하지 않거나 거래일이 2일 미만이면 None.
    """
    from utils.data_loader import get_trading_days

    try:
        days = get_trading_days(
            period.start_time.strftime("%Y-%m-%d"),
            period.end_time.strftime("%Y-%m-%d"),
            country,
        )
    except Exception:
        return None
    if len(days) < 2:
        return None
    return days[-1].normalize(), days[-2].normalize()


def current_portfolio_dates(
    benchmark_close: pd.Series, country: str
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """현재 보여줄 포트폴리오의 (교체일 L, 판정일 L−1) — 거래일 캘린더 기준.

    판정일(L−1) 종가가 확정된 다음날(달력)부터 다음 달 포트폴리오를 보여준다:
    오늘 > L−1 이고 캐시에 L−1 종가가 있으면 그 달의 교체(L)를 채택한다.
    예: 8월(L=8/31, L−1=8/28) → 8/29 00시부터 9월 포트폴리오. KRX 12월은
    캘린더가 연말 휴장을 알아 L=12/30, 12/30 00시부터 1월 포트폴리오.
    판정일이 고정이므로 같은 기간 안에서는 몇 번을 실행해도 결과가 같다.
    """
    now = pd.Timestamp.now()
    today = now.normalize()
    index = benchmark_close.index
    cache_max = index[-1].normalize()

    # 이번 달부터 거슬러 올라가며 '판정 종가가 확정된' 가장 최근 월의 교체를 찾는다.
    for months_back in range(0, 4):
        period = now.to_period("M") - months_back
        pair = month_last_two_trading_days(country, period)
        if pair is None:
            continue
        rebalance_date, signal_calendar = pair
        if today <= signal_calendar or cache_max < signal_calendar:
            continue
        prior = index[index <= signal_calendar]
        if len(prior) == 0:
            continue
        return rebalance_date, prior[-1]
    raise RuntimeError("판정 가능한 월말 교체를 찾지 못했습니다 — 캘린더/캐시 데이터를 확인하세요.")


# ── 선정 (현재 포트폴리오) ─────────────────────────────────────────────────
def compute_picks(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """현재 적용 중인 월 확정 포트폴리오 — 화면 ③ 카드와 스크립트가 함께 쓴다."""
    if settings is None:
        settings = load_settings()
    settings = validate_settings(settings)

    universe = load_universe(settings["pool"])
    frames = load_price_frames(universe)
    benchmark_close = load_benchmark_close(settings["pool"])
    rebalance_date, signal_date = current_portfolio_dates(
        benchmark_close, POOL_CONFIGS[settings["pool"]]["country"]
    )
    candidates = select_candidates(universe, frames, settings, benchmark_close, as_of=signal_date)
    scored = rank_candidates(candidates)

    top_n = int(settings["top_n"])
    max_per_industry = int(settings["max_per_industry"])
    sector_map = sector_industry_map(settings["pool"])
    industry_by_ticker = {ticker: meta["industry"] for ticker, meta in sector_map.items()}

    selected = select_top(scored, top_n, max_per_industry, industry_by_ticker)
    # 차순위 후보 — 선정에 못 든 종목 중 점수 상위 N개 (화면에서 흐리게 붙여 보여준다).
    # 선정에서 빠진 자리를 메울 후보라 업종 상한은 적용하지 않는다.
    selected_tickers = {item["ticker"] for item in selected}
    reserve = [item for item in scored if item["ticker"] not in selected_tickers][:top_n]

    # 연속 편입 개월 — 직전 최대 11개월의 판정일마다 같은 규칙으로 선정을 재계산해,
    # 현재 종목이 연속으로 상위 N 에 들어 있던 횟수(+이번 달)를 센다. 끊기면 중단.
    streak_lookback = 11
    prior_rebalances = [
        stamp for stamp in completed_month_ends(benchmark_close) if stamp < rebalance_date
    ][-streak_lookback:]
    streaks = {item["ticker"]: 1 for item in selected}
    alive = set(streaks)
    for prior_rebalance in reversed(prior_rebalances):
        if not alive:
            break
        prior_signal = _signal_date_for(benchmark_close, prior_rebalance)
        prior_candidates = select_candidates(
            universe, frames, settings, benchmark_close, as_of=prior_signal
        )
        prior_top = {
            item["ticker"]
            for item in select_top(
                rank_candidates(prior_candidates), top_n, max_per_industry, industry_by_ticker
            )
        }
        for ticker in list(alive):
            if ticker in prior_top:
                streaks[ticker] += 1
            else:
                alive.discard(ticker)

    portfolio_month = (rebalance_date.to_period("M") + 1).strftime("%Y-%m")

    # 참고용 가격 정보 — 현재가는 캐시의 최신 종가, 기간수익률은 나머지 컬럼과 같은
    # 판정일 기준이라 다음 교체 전까지 값이 바뀌지 않는다.
    from core.strategy.metrics import period_return_pct

    currency = str(POOL_CONFIGS[settings["pool"]]["currency"])

    def price_info(ticker: str) -> dict[str, Any]:
        frame = frames.get(ticker)
        if frame is None or frame.empty:
            return {"price": None, "return_1m_pct": None, "return_3m_pct": None, "return_12m_pct": None}
        close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
        return {
            "price": round(float(close.iloc[-1]), 4) if not close.empty else None,
            "return_1m_pct": period_return_pct(close, 1, signal_date),
            "return_3m_pct": period_return_pct(close, 3, signal_date),
            "return_12m_pct": period_return_pct(close, 12, signal_date),
        }

    return {
        "as_of": signal_date.strftime("%Y-%m-%d"),
        "portfolio_month": portfolio_month,
        "rebalance_date": rebalance_date.strftime("%Y-%m-%d"),
        "signal_date": signal_date.strftime("%Y-%m-%d"),
        "universe_count": len(universe),
        "candidate_count": len(candidates),
        "rows": [
            {
                "rank": rank,
                "is_reserve": rank > top_n,
                # 연속 편입은 선정분에만 의미가 있다 — 차순위는 표시하지 않는다(None → '-')
                "streak_months": streaks.get(item["ticker"], 1) if rank <= top_n else None,
                "ticker": item["ticker"],
                "name": item["name"],
                "pool": item["pool"],
                "sector": sector_map.get(item["ticker"], {}).get("sector", ""),
                "industry": sector_map.get(item["ticker"], {}).get("industry", ""),
                "currency": currency,
                **price_info(item["ticker"]),
                "return_lookback_pct": round(item["return_lookback_pct"], 1),
                "rel_return_pct": round(item["rel_return_pct"], 1),
                "win_label": f"{item['win_months']}/{item['months_count']}",
                "slope_annual_pct": round(item["slope_annual_pct"], 1),
                "r_squared": round(item["r_squared"], 3),
                "momentum_score": round(item["momentum_score"], 1),
            }
            for rank, item in enumerate([*selected, *reserve], start=1)
        ],
    }
