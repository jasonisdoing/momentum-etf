"""Steady Momentum — 시장 대비 '꾸준한 모멘텀' 선정 서비스 (UI/API/스크립트 공용).

전략 규칙
--------
1. 유니버스: 설정에서 고른 종목풀 1개. 한국 풀은 우선주 제외(티커 끝자리가 '0'
   이 아니면 우선주), 고정 보유 종목(exclude_from_ranking)은 전 풀에서 제외.
2. 점수: **상대 가격선(종가 ÷ 벤치마크)** 에 룩백 구간 로그 회귀를 돌려
   ``연율화 상대기울기 × R²``. 시장을 얼마나 빠르고(기울기) 꾸준하게(R²)
   이겨왔는지를 하나의 점수로 본다. 점수 순 상위 top_n 동일가중.
3. 시장 상대기울기 필터(체크박스): 켜면 상대기울기 음수(시장에 지는 추세)를
   후보에서 제외한다.
4. 월간 리밸런싱: 판정은 월말 직전 거래일(L−1) 종가, 체결은 월말(L) 종가.
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
POOL_CONFIGS: dict[str, dict[str, Any]] = {
    "kor": {"label": "코스피 개별주", "country": "kor", "currency": "KRW"},
    "kor_kosdaq": {"label": "코스닥 개별주", "country": "kor", "currency": "KRW"},
    "us": {"label": "나스닥 100 + S&P 100", "country": "us", "currency": "USD"},
    "us_nasdaq": {"label": "나스닥 100", "country": "us", "currency": "USD"},
    "us_snp": {"label": "S&P100", "country": "us", "currency": "USD"},
}
AVAILABLE_POOLS = tuple(POOL_CONFIGS)
TRADING_DAYS_PER_MONTH = 21

_CONFIG_COLLECTION = "system_config"
_SETTINGS_KEY = "steady_momentum_settings"

DEFAULT_SETTINGS: dict[str, Any] = {
    "pool": "kor",
    "lookback_months": 6,
    "top_n": 40,
    "slippage_pct": 0.1,
    "slope_filter": True,
}


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
    slope_filter = settings.get("slope_filter")
    if not isinstance(slope_filter, bool):
        raise ValueError("'slope_filter' 는 참/거짓이어야 합니다.")
    pool = str(settings.get("pool") or "").strip().lower()
    if pool not in AVAILABLE_POOLS:
        raise ValueError(f"지원하지 않는 종목풀입니다: {settings.get('pool')}")

    return {
        "pool": pool,
        "lookback_months": lookback_months,
        "top_n": top_n,
        "slippage_pct": slippage_pct,
        "slope_filter": slope_filter,
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


def load_settings() -> tuple[dict[str, Any], bool]:
    """(설정, 저장 여부). 저장 전이면 기본값과 False 를 반환한다."""
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        return dict(DEFAULT_SETTINGS), False
    doc = db[_CONFIG_COLLECTION].find_one({"_id": _SETTINGS_KEY}) or {}
    stored = doc.get("settings")
    if not isinstance(stored, dict):
        return dict(DEFAULT_SETTINGS), False
    try:
        return validate_settings(stored), True
    except ValueError:
        # 저장값이 깨졌으면(스키마 변경 등) 기본값으로 표시하되 미저장으로 알린다.
        return dict(DEFAULT_SETTINGS), False


def save_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """검증 후 저장하고 정규화된 설정을 반환한다."""
    normalized = validate_settings(settings)
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
def is_preferred_share(ticker: str) -> bool:
    """한국 우선주 판별 — 티커 끝자리가 '0' 이 아니면 우선주 (예: 005935, 00680K)."""
    return not str(ticker).endswith("0")


def load_universe(pool: str) -> list[dict[str, str]]:
    """선택한 종목풀 1개의 종목 목록. [{ticker, name, pool}]

    우선주 제외(끝자리 규칙)는 한국 풀에만 적용한다 — 미국 티커(AAPL 등)는
    알파벳이라 이 규칙을 적용하면 전부 걸러진다.
    고정 보유 종목(exclude_from_ranking)은 투자 후보가 아니므로 제외한다
    (순위·종목풀 백테스트와 같은 규칙).
    """
    from utils.stock_list_io import _load_ticker_type_stocks_raw

    is_korean = POOL_CONFIGS[pool]["country"] == "kor"
    universe: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in _load_ticker_type_stocks_raw(pool):
        ticker = str(item.get("ticker") or "").strip()
        if not ticker or ticker in seen or (is_korean and is_preferred_share(ticker)):
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

    (예: kor_kosdaq 의 벤치마크 069500 은 kor 캐시에, us 의 QQQ 는 us 캐시에 있다)
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
    as_of: pd.Timestamp | None = None,
) -> dict[str, float] | None:
    """상대 가격선(종가÷벤치마크) 로그 회귀 — 꾸준한 모멘텀 점수.

    점수 = 연율화 상대기울기 × R². 기울기는 시장을 이기는 속도, R² 는 그
    아웃퍼폼이 얼마나 꾸준했는지다. 월승률(최근 6개 월별 상대수익 중 플러스
    비율)은 참고 지표로 함께 계산한다.
    ``as_of`` 를 주면 그 날짜까지의 데이터만 사용한다(백테스트·판정일 재현).
    """
    series = pd.to_numeric(close, errors="coerce").dropna()
    bench = pd.to_numeric(benchmark_close, errors="coerce").dropna()
    if as_of is not None:
        series = series[series.index <= as_of]
        bench = bench[bench.index <= as_of]

    aligned = pd.concat([series.rename("stock"), bench.rename("bench")], axis=1, join="inner").dropna()
    min_rows = lookback_days + 4
    if len(aligned) < min_rows:
        return None
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
        "momentum_score": rel_slope_annual_pct * r_squared,
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
    """상대 모멘텀 후보 목록 (유동성 필터 없음 — 지수 기반 풀)."""
    lookback_days = int(settings["lookback_months"]) * TRADING_DAYS_PER_MONTH
    candidates: list[dict[str, Any]] = []
    for row in universe:
        frame = frames.get(row["ticker"])
        if frame is None:
            continue
        metrics = momentum_metrics(
            frame["Close"], benchmark_close, lookback_days=lookback_days, as_of=as_of
        )
        if metrics is None:
            continue
        # 시장 상대기울기 필터(설정) — 켜면 시장에 지는 추세를 후보에서 제외.
        if settings["slope_filter"] and metrics["slope_annual_pct"] <= 0:
            continue
        candidates.append({**row, **metrics})
    return candidates


def rank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """꾸준한 모멘텀 점수 순 정렬 — 선정·백테스트·연속 추적이 같은 순위를 쓴다."""
    return sorted(candidates, key=lambda item: item["momentum_score"], reverse=True)


# ── 월간 리밸런싱 시점 ─────────────────────────────────────────────────────
def _signal_date_for(benchmark_close: pd.Series, rebalance_date: pd.Timestamp) -> pd.Timestamp:
    """교체일 직전의 캐시 거래일 — 과거 월 판정일 계산용."""
    prior = benchmark_close.index[benchmark_close.index < rebalance_date]
    if len(prior) == 0:
        raise RuntimeError("판정 기준일(교체일 직전 거래일)을 구할 수 없습니다.")
    return prior[-1]


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
        settings, _ = load_settings()
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
    selected = scored[:top_n]
    # 차순위 후보 — 선정에 못 든 다음 N개 (화면에서 흐리게 붙여 보여준다)
    reserve = scored[top_n : top_n * 2]

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
        prior_top = {item["ticker"] for item in rank_candidates(prior_candidates)[:top_n]}
        for ticker in list(alive):
            if ticker in prior_top:
                streaks[ticker] += 1
            else:
                alive.discard(ticker)

    portfolio_month = (rebalance_date.to_period("M") + 1).strftime("%Y-%m")

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
                "return_6m_pct": round(item["return_lookback_pct"], 1),
                "rel_return_pct": round(item["rel_return_pct"], 1),
                "win_label": f"{item['win_months']}/{item['months_count']}",
                "slope_annual_pct": round(item["slope_annual_pct"], 1),
                "r_squared": round(item["r_squared"], 3),
                "momentum_score": round(item["momentum_score"], 1),
            }
            for rank, item in enumerate([*selected, *reserve], start=1)
        ],
    }
