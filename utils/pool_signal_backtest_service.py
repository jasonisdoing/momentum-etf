"""종목풀 신호(이격/기울기) 실증 백테스트 — 읽기 전용 분석.

향후 N거래일 수익이 이격·기울기와 실제로 관계있는지 분위별로 집계한다.

통계 주의(화면에도 함께 노출):
    - 전망 N일 수익률은 매일 겹치므로 행 수가 곧 표본 수가 아니다.
      유효 독립구간 ≈ 거래일수 / N.
    - 강세장에서는 기저율(아무 종목이나 N일 보유 시 상승확률)이 이미 높다.
      따라서 '기저율 대비' 차이만 신호로 볼 수 있다.
    - 종목 간 동조(같은 시장) 때문에 실제 유효표본은 위 값보다도 작다 → 보수적으로 해석해야 한다.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from config import CACHE_START_DATE
from utils.cache_utils import load_cached_close_series_bulk
from utils.logger import get_app_logger
from utils.pool_settings_store import MA_DAY_OPTIONS, SLOPE_DAY_OPTIONS, get_pool_benchmark_ticker
from utils.rankings import get_ticker_type_ma_rules, hold_eligible_mask
from utils.settings_loader import get_ticker_type_settings
from utils.stock_list_io import get_etfs

logger = get_app_logger()

FORWARD_DAY_OPTIONS: tuple[int, ...] = (5, 10, 20, 40, 60)
QUANTILE_COUNT = 10
_TRADING_DAYS_PER_MONTH = 21
# t 검정을 낼 수 있는 최소 독립구간 수. 이 미만이면 t 가 우연히 커져 '유의'로 오도할 수 있어
# 판정 자체를 내지 않는다(기간을 늘리거나 전망일수를 줄이면 확보된다).
MIN_INDEPENDENT_SAMPLES = 10


def get_max_backtest_months(today: date | None = None) -> int:
    """가격 캐시 시작일 기준으로 선택 가능한 최대 개월 수를 계산한다."""
    start = datetime.strptime(CACHE_START_DATE, "%Y-%m-%d").date()
    end = today or date.today()
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day < start.day:
        months -= 1
    return max(months, 1)


def get_month_options() -> list[int]:
    """기간 셀렉트 옵션. 60개월보다 긴 구간은 현재 캐시 기준 최대값만 노출한다."""
    max_months = get_max_backtest_months()
    options = [1, 2, 3, 4, 5, 6, 12, 24, 36, 48, 60]
    if max_months > 60:
        options.append(max_months)
    return [month for month in options if month <= max_months]


def _quantile_label(order: int) -> str:
    """1등급 = 신호 상위, 마지막 등급 = 신호 하위 (순위 화면과 같은 '1등이 최고' 관례)."""
    share = round(100 / QUANTILE_COUNT)
    if order == 1:
        return f"1등급 (상위 {share}%)"
    if order == QUANTILE_COUNT:
        return f"{QUANTILE_COUNT}등급 (하위 {share}%)"
    return f"{order}등급"


# 등급 경계 (0 ~ 1 을 QUANTILE_COUNT 등분)
_QUANTILE_EDGES = [i / QUANTILE_COUNT for i in range(QUANTILE_COUNT + 1)]


def _grade_buckets(df: pd.DataFrame, column: str) -> pd.Series:
    """날짜별로 신호가 **높을수록 1등급**이 되도록 등급을 매긴다(내림차순 순위).

    같은 날짜 안에서만 비교하므로 시장 전체의 등락(타이밍 효과)이 상쇄된다.
    """
    ranked = df.groupby("date")[column].rank(pct=True, ascending=False)
    return pd.cut(ranked, _QUANTILE_EDGES, labels=list(range(1, QUANTILE_COUNT + 1)))


def _summarize_group(
    group: pd.DataFrame,
    label: str,
    base_return: float,
    base_rate: float,
    effective_samples: int,
    rate_error: float,
) -> dict[str, Any]:
    """한 구간의 평균수익(주 지표)·상승확률(보조) + 기저 대비 및 유의 여부."""
    avg_return = float(group["fwd"].mean())
    return_diff = avg_return - base_return
    # 평균수익의 95% 오차: 표본은 행 수가 아니라 '독립 구간 수'로 본다(겹침·동조 보정).
    std = float(group["fwd"].std(ddof=1)) if len(group) > 1 else 0.0
    return_error = 1.96 * std / np.sqrt(max(effective_samples, 1))
    up_rate = float((group["fwd"] > 0).mean() * 100)
    rate_diff = up_rate - base_rate
    return {
        "label": label,
        "avg_return": round(avg_return, 2),
        "avg_return_diff": round(return_diff, 2),
        "avg_return_error": round(return_error, 2),
        "avg_return_significant": bool(abs(return_diff) > return_error),
        "up_rate": round(up_rate, 2),
        "up_rate_diff": round(rate_diff, 2),
        "up_rate_significant": bool(abs(rate_diff) > rate_error),
        "samples": int(len(group)),
    }


def _spread_stats(spread: pd.Series, forward_days: int, digits: int = 2) -> dict[str, Any] | None:
    """날짜별 스프레드 시계열의 통계.

    점추정(평균)은 전체 날짜로 구한다. t 값은 겹침 때문에 시작 오프셋마다 달라지므로
    0..forward_days-1 오프셋 전부로 계산해 **중앙값**을 쓴다(특정 표본에 운좋게 걸리는 것 방지).

    독립구간이 MIN_INDEPENDENT_SAMPLES 미만이면 t 검정이 무의미하므로 t/유의 판정을 내지 않고
    ``insufficient`` 로 표시한다(표본 3개짜리 t가 우연히 2를 넘어 '유의' 초록불이 켜지는 것 방지).
    """
    spread = spread.dropna()
    # 승패는 **겹치지 않는 독립 구간**으로만 센다. 전체 날짜로 세면 같은 사건을
    # forward_days 번씩 중복 계수해 승률이 부풀려진다(예: 5승 8패인데 '이긴 날 58%').
    independent_series = spread.iloc[::forward_days]
    independent = len(independent_series)
    wins = int((independent_series > 0).sum())
    losses = int(independent - wins)
    if independent < MIN_INDEPENDENT_SAMPLES:
        if spread.empty:
            return None
        return {
            "insufficient": True,
            "mean": round(float(spread.mean()), digits),
            "wins": wins,
            "losses": losses,
            "independent_samples": independent,
            "required_samples": MIN_INDEPENDENT_SAMPLES,
            "t_value": None,
            "significant": False,
        }
    t_values: list[float] = []
    for offset in range(forward_days):
        sample = spread.iloc[offset::forward_days]
        if len(sample) > 2:
            std = float(sample.std(ddof=1))
            if std > 0:
                t_values.append(float(sample.mean() / (std / np.sqrt(len(sample)))))
    if not t_values:
        return None
    t_median = float(np.median(t_values))
    return {
        "insufficient": False,
        "mean": round(float(spread.mean()), digits),
        "wins": wins,
        "losses": losses,
        "t_value": round(t_median, 2),
        "independent_samples": independent,
        "required_samples": MIN_INDEPENDENT_SAMPLES,
        # t>2 ≈ 우연일 확률 5% 미만
        "significant": bool(abs(t_median) > 2.0),
    }


def _quantile_long_short(df: pd.DataFrame, column: str, forward_days: int) -> dict[str, Any] | None:
    """날짜별 (1등급 − 최하위 등급) 스프레드 통계.

    같은 날 안에서 빼므로 시장 요인이 상쇄되고 신호의 순수 변별력만 남는다.
    """
    frame = df.assign(_bucket=_grade_buckets(df, column)).dropna(subset=["_bucket"])
    pivot = frame.groupby(["date", "_bucket"], observed=True)["fwd"].mean().unstack()
    if QUANTILE_COUNT not in pivot.columns or 1 not in pivot.columns:
        return None
    return _spread_stats(pivot[1] - pivot[QUANTILE_COUNT], forward_days)


def _information_coefficient(df: pd.DataFrame, column: str, forward_days: int) -> dict[str, Any] | None:
    """날짜별 상관(IC): 그날 신호 순위와 향후 수익 순위의 스피어만 상관.

    등급 평균의 단조성은 '10개의 증거'가 아니다(같은 날 같은 종목을 10칸으로 나눈 것이라
    서로 독립이 아님). 미약한 경향도 평균을 내면 매끄럽게 정렬돼 단조로 보인다.
    IC 는 **날짜마다 관계가 실제로 성립했는지**를 재므로 그 착시를 걸러낸다.
    """

    def _ic(group: pd.DataFrame) -> float:
        if len(group) < 6:
            return float("nan")
        # scipy 없이 스피어만 = 순위에 대한 피어슨
        return float(group[column].rank().corr(group["fwd"].rank()))

    per_date = df.groupby("date")[[column, "fwd"]].apply(_ic).dropna()
    return _spread_stats(per_date, forward_days, digits=3)


def _cross_sectional_table(
    df: pd.DataFrame,
    column: str,
    base_return: float,
    base_rate: float,
    effective_samples: int,
    rate_error: float,
) -> list[dict[str, Any]]:
    """날짜별 상대등급 표(1등급 = 신호 상위).

    같은 날짜 안에서 종목끼리 순위를 매겨 등급을 나눈다. 전체 기간을 섞어 자르면
    '이격 최하위 = 시장이 빠진 날'이 되어 종목 선택력이 아니라 시장 타이밍 효과가 섞인다.
    """
    frame = df.assign(_bucket=_grade_buckets(df, column)).dropna(subset=["_bucket"])
    rows: list[dict[str, Any]] = []
    for order, group in frame.groupby("_bucket", observed=True):
        rows.append(
            _summarize_group(
                group, _quantile_label(int(order)), base_return, base_rate, effective_samples, rate_error
            )
        )
        rows[-1]["order"] = int(order)
    return rows


def _max_drawdown_pct(segment_returns: pd.Series) -> float | None:
    """회차별 수익률(%) 시리즈의 최대낙폭(MDD, 음수 %). 자산 곡선의 고점 대비 최대 하락."""
    values = segment_returns.dropna()
    if len(values) < 2:
        return None
    curve = (1.0 + values / 100.0).cumprod()
    drawdown = curve / curve.cummax() - 1.0
    return round(float(drawdown.min()) * 100.0, 1)


def _sortino(segment_returns: pd.Series, forward_days: int) -> float | None:
    """회차별 수익률(%)의 소르티노 지수(연율화). 하방편차(0% 미만 수익만) 기준.

    표본이 적거나 하락 회차가 없으면 값을 내지 않는다(None) — 억지로 채우지 않는다.
    """
    values = segment_returns.dropna()
    if len(values) < 3:
        return None
    downside = values[values < 0.0]
    if downside.empty:
        return None
    downside_dev = float(np.sqrt((downside**2).mean()))
    if downside_dev == 0.0:
        return None
    periods_per_year = 252.0 / forward_days
    return round(float(values.mean() / downside_dev) * float(np.sqrt(periods_per_year)), 2)


def _rule_performance(
    df: pd.DataFrame,
    pool_id: str,
    top_n: int,
    forward_days: int,
    benchmark: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """현재 설정 그대로 규칙을 돌렸을 때의 기간 실적.

    순위 화면의 추천(✅)과 동일한 규칙으로 ``forward_days`` 마다 리밸런싱한다.
    조건을 만족하는 종목이 없는 회차는 현금 보유(0%)로 본다.

    **기대수익이 아니라 지나간 기간의 실적**이다. 표본은 기간/forward_days 회차뿐이라
    강세장 한 구간이 통째로 들어오면 숫자가 커진다. 그래서 기저(아무 종목이나 보유)와
    벤치마크(설정된 기준 종목)를 함께 반환해 '규칙이 기여한 몫'을 구분할 수 있게 한다.

    ``benchmark`` 는 ``{"ticker", "name", "returns"}`` 이며, 미설정이거나 데이터가 없으면
    None 이다. 조용히 기저로 대체하지 않고 그 사실을 그대로 노출한다.
    """
    # df 는 시작일이 고정된 구간 전체(fwd 가 NaN 인 최근 구간 포함). 리밸런싱 시작점을
    # 전망일수와 무관하게 고정하기 위해 여기서 잘라 쓴다. 실제 집계는 fwd 가 있는 행만.
    scored = df.dropna(subset=["fwd"])
    valid_dates = set(scored["date"])
    calendar = sorted(df["date"].unique())
    # 고정 시작점에서 forward_days 간격. 단 미래수익(fwd)이 있는 날짜까지만 매매한다.
    rebalance_dates = [d for d in calendar[::forward_days] if d in valid_dates]
    if len(rebalance_dates) < 2:
        return None

    eligible = scored[hold_eligible_mask(scored["이격"], scored["단기이격"])]
    segment_returns: list[float] = []
    baskets: list[set[str]] = []
    cash_rounds = 0  # 조건 종목이 0개라 통째로 현금인 회차
    partial_rounds = 0  # 0 < 종목 < top_n 이라 일부만 현금인 회차
    for as_of in rebalance_dates:
        picked = eligible[eligible["date"] == as_of].nlargest(top_n, "이격")
        if picked.empty:
            segment_returns.append(0.0)
            baskets.append(set())
            cash_rounds += 1
            continue
        # 조건 충족 종목이 top_n 보다 적으면 있는 만큼만 1/top_n 씩 투자하고 부족분은 현금(0%).
        # mean() 이면 3개를 100% 투자한 셈이 되어 실제(3/N만 투자)보다 부풀려진다 → sum()/top_n.
        if len(picked) < top_n:
            partial_rounds += 1
        segment_returns.append(float(picked["fwd"].sum()) / top_n)
        baskets.append(set(picked["ticker"]))

    segments = pd.Series(segment_returns, dtype="float64")

    # 회전율: 직전 회차 대비 바스켓이 바뀐 비율. 슬리피지는 여기에 비례한다.
    turnovers = [
        (1.0 - len(prev & cur) / len(cur)) if cur else (1.0 if prev else 0.0)
        for prev, cur in zip(baskets, baskets[1:])
    ]
    turnover = float(np.mean(turnovers)) if turnovers else 0.0
    pool_settings = get_ticker_type_settings(pool_id)
    missing_slippage = [
        key
        for key in ("BUY_SLIPPAGE_PCT", "SELL_SLIPPAGE_PCT")
        if pool_settings.get(key) in (None, "")
    ]
    if missing_slippage:
        raise ValueError(
            f"종목풀 '{pool_id}' 의 슬리피지 설정이 없습니다: {', '.join(missing_slippage)}. "
            "/pools-settings 에서 매수/매도 슬리피지를 저장하세요."
        )
    round_trip_pct = float(pool_settings["BUY_SLIPPAGE_PCT"]) + float(pool_settings["SELL_SLIPPAGE_PCT"])
    cost_per_round = turnover * round_trip_pct

    baseline = scored[scored["date"].isin(rebalance_dates)].groupby("date")["fwd"].mean().reindex(rebalance_dates)

    def _compound(values: pd.Series) -> float:
        return float((1.0 + values / 100.0).prod() - 1.0) * 100.0

    # ① 종목풀 규칙(슬리피지 차감) ② 종목풀 보유(전체 동일가중=기저) — 둘 다 회차별 수익 시리즈.
    rule_net = segments - cost_per_round
    rule_stats = {
        "cumulative_pct": round(_compound(rule_net), 1),
        "mdd_pct": _max_drawdown_pct(rule_net),
        "sortino": _sortino(rule_net, forward_days),
    }
    pool_hold_stats = {
        "cumulative_pct": round(_compound(baseline), 1),
        "mdd_pct": _max_drawdown_pct(baseline),
        "sortino": _sortino(baseline, forward_days),
    }

    # ③ 벤치마크: 운용 기간(첫 리밸런싱 ~ 마지막 청산일) 동안 그냥 계속 보유.
    # 누적은 시작·끝 종가비로만 계산해 텔레스코핑 오차를 없앤다. MDD·소르티노는 규칙·보유와
    # 같은 잣대로 회차별 수익 시리즈에서 구한다. 미설정/데이터 없음이면 None(기저로 대체 금지).
    benchmark_payload: dict[str, Any] | None = None
    if benchmark is not None:
        bclose = benchmark["close"]
        start_d = pd.Timestamp(rebalance_dates[0]).normalize()
        last_d = pd.Timestamp(rebalance_dates[-1]).normalize()
        if start_d in bclose.index and last_d in bclose.index:
            i0 = bclose.index.get_loc(start_d)
            # 마지막 리밸런싱에서 forward_days 뒤 = 규칙이 마지막에 청산한 날(범위 밖이면 최신 종가).
            i_end = min(bclose.index.get_loc(last_d) + forward_days, len(bclose) - 1)
            cumulative = (float(bclose.iloc[i_end]) / float(bclose.iloc[i0]) - 1.0) * 100.0
            # 회차별 벤치 수익(MDD·소르티노용): 각 리밸런싱 시점의 forward_days 수익.
            bench_seg = []
            for as_of in rebalance_dates:
                d = pd.Timestamp(as_of).normalize()
                if d not in bclose.index:
                    continue
                pos = bclose.index.get_loc(d)
                if pos + forward_days < len(bclose):
                    bench_seg.append((float(bclose.iloc[pos + forward_days]) / float(bclose.iloc[pos]) - 1.0) * 100.0)
            bench_series = pd.Series(bench_seg, dtype="float64")
            benchmark_payload = {
                "ticker": benchmark["ticker"],
                "name": benchmark["name"],
                "cumulative_pct": round(cumulative, 1),
                "mdd_pct": _max_drawdown_pct(bench_series),
                "sortino": _sortino(bench_series, forward_days),
            }

    return {
        "top_n_hold": int(top_n),
        "rounds": len(rebalance_dates),
        "cash_rounds": cash_rounds,
        "partial_rounds": partial_rounds,
        "mean_return": round(float(segments.mean()), 2),
        "wins": int((segments > 0).sum()),
        "losses": int((segments < 0).sum()),
        "turnover_pct": round(turnover * 100.0, 1),
        "round_trip_pct": round(round_trip_pct, 2),
        "cost_per_round_pct": round(cost_per_round, 2),
        "rule": rule_stats,
        "pool_hold": pool_hold_stats,
        "benchmark": benchmark_payload,
    }


def _load_benchmark_close(pool_id: str, pool_settings: dict[str, Any]) -> dict[str, Any] | None:
    """벤치마크 종목의 종가 시리즈. 미설정/데이터 없음이면 None.

    벤치마크는 매수 후보에서 빠지므로 종가를 여기서 따로 불러온다. 벤치마크 누적은
    '규칙 운용 기간 동안 이 종목을 그냥 계속 보유'로, 시작·끝 종가비로만 계산한다
    (리밸런싱마다 끊어 곱하면 거래일 경계에서 텔레스코핑이 깨져 부정확해진다).
    반환: ``{"ticker", "name", "close"}`` (close 는 normalize 된 날짜 인덱스).
    """
    benchmark = pool_settings.get("BENCHMARK")
    if not isinstance(benchmark, dict):
        return None
    ticker = str(benchmark.get("ticker") or "").strip().upper()
    name = str(benchmark.get("name") or "").strip()
    if not ticker:
        return None

    series_map = load_cached_close_series_bulk(pool_id, [ticker])
    close = pd.to_numeric(series_map.get(ticker, pd.Series(dtype="float64")), errors="coerce").dropna()
    if len(close) < 2:
        return None
    close.index = pd.to_datetime(close.index).normalize()
    return {"ticker": ticker, "name": name or ticker, "close": close}


def _resolve_int_override(value: int | None, fallback: int, allowed: tuple[int, ...], label: str) -> int:
    """오버라이드 값이 있으면 허용값인지 검증해 쓰고, 없으면 종목풀 설정값을 쓴다."""
    if value is None:
        return int(fallback)
    if int(value) not in allowed:
        options = ", ".join(str(day) for day in allowed)
        raise ValueError(f"{label}은(는) 다음 값 중 하나여야 합니다: {options}. 입력값: {value}")
    return int(value)


def compute_pool_signal_backtest(
    pool_id: str,
    forward_days: int = 20,
    months: int = 12,
    *,
    top_n: int | None = None,
    short_ma_days: int | None = None,
    long_ma_days: int | None = None,
    slope_days: int | None = None,
) -> dict[str, Any]:
    """종목풀의 이격/단기이격/기울기 → 향후 N일 수익 실증 결과를 반환한다.

    신호 정의는 순위 화면(`utils.rankings`)과 같다. 이격은 장기 이평선, 단기이격은
    단기 이평선 기준이며, 두 이평선의 역할(선택/손절)이 실제로 성립하는지 확인한다.

    MA 파라미터(단기/장기/기울기 일수)는 해당 종목풀 설정을 그대로 쓴다.
    고정 보유 종목(exclude_from_ranking)은 투자 후보가 아니므로 제외한다.
    """
    if forward_days not in FORWARD_DAY_OPTIONS:
        options = ", ".join(str(day) for day in FORWARD_DAY_OPTIONS)
        raise ValueError(f"전망일수는 {options} 중 하나여야 합니다: {forward_days}")
    max_months = get_max_backtest_months()
    if not (1 <= int(months) <= max_months):
        raise ValueError(f"기간은 1~{max_months}개월이어야 합니다: {months}")

    # MA/보유수 파라미터는 종목풀 설정이 기본. 화면에서 넘긴 오버라이드가 있으면 그 값으로
    # 실험한다(저장은 하지 않음). 오버라이드도 허용값인지 반드시 검증한다.
    rule = get_ticker_type_ma_rules(pool_id)[0]
    short_days = _resolve_int_override(short_ma_days, rule["short_ma_days"], MA_DAY_OPTIONS, "단기 이평선")
    long_days = _resolve_int_override(long_ma_days, rule["long_ma_days"], MA_DAY_OPTIONS, "장기 이평선")
    slope_days = _resolve_int_override(slope_days, rule["slope_days"], SLOPE_DAY_OPTIONS, "기울기 일수")
    window = int(months) * _TRADING_DAYS_PER_MONTH

    pool_settings = get_ticker_type_settings(pool_id)
    top_n_hold = int(pool_settings["TOP_N_HOLD"]) if top_n is None else int(top_n)
    if not (1 <= top_n_hold <= 100):
        raise ValueError(f"보유 종목수는 1~100 범위여야 합니다: {top_n_hold}")
    # 벤치마크는 비교 기준일 뿐 매수 대상이 아니다 — 순위 화면의 추천 규칙과 동일하게 뺀다.
    benchmark_ticker = get_pool_benchmark_ticker(pool_settings)

    all_etfs = get_etfs(pool_id)
    etfs = [
        item
        for item in all_etfs
        if not bool(item.get("exclude_from_ranking"))
        and str(item.get("ticker") or "").strip().upper() != benchmark_ticker
    ]
    excluded_count = len(all_etfs) - len(etfs)
    if not etfs:
        raise ValueError(f"'{pool_id}' 종목풀에 분석 가능한 종목이 없습니다(고정 보유·벤치마크 제외 후 0개).")

    series_map = load_cached_close_series_bulk(pool_id, [item["ticker"] for item in etfs])
    frames: list[pd.DataFrame] = []
    min_length = long_days + slope_days + 20
    for ticker, series in series_map.items():
        close = pd.to_numeric(series, errors="coerce").dropna()
        if len(close) < min_length:
            continue
        short_ma = close.rolling(short_days).mean()
        long_ma = close.rolling(long_days).mean()
        frame = pd.DataFrame(
            {
                "close": close,
                "이격": (close / long_ma - 1.0) * 100.0,
                # 단기이격: 순위 화면과 동일하게 이격과 같은 식에 단기 이평선을 넣은 값.
                "단기이격": (close / short_ma - 1.0) * 100.0,
                "기울기": (short_ma / short_ma.shift(slope_days) - 1.0) * 100.0,
                # 향후 N거래일 수익률(라벨). 마지막 N거래일은 미래가 없어 NaN.
                "fwd": (close.shift(-forward_days) / close - 1.0) * 100.0,
            }
        )
        # 신호가 유효한 구간만 남긴다. fwd 는 최근 구간에서 NaN 이어도 남겨서,
        # 분석 시작일을 전망일수와 무관하게 고정한다(전망일수를 바꿔도 기저·벤치가 안 흔들리게).
        frame = frame.dropna(subset=["이격", "단기이격", "기울기"])
        if frame.empty:
            continue
        frame["ticker"] = ticker
        frame["date"] = frame.index
        frames.append(frame)

    if not frames:
        raise ValueError(f"'{pool_id}' 종목풀에 분석에 충분한 가격 데이터가 없습니다.")

    df_all = pd.concat(frames, ignore_index=True)
    # 분석 구간을 전망일수와 무관하게 고정: 신호가 있는 거래일 기준 최근 window 개로 자른다.
    unique_dates = sorted(df_all["date"].unique())
    if len(unique_dates) > window:
        df_all = df_all[df_all["date"] >= unique_dates[-window]].reset_index(drop=True)
    # 분위/롱숏/IC/기저는 미래수익(fwd)이 있는 행만 쓴다. df_all(시작 고정)은 performance 리밸런싱용.
    df = df_all.dropna(subset=["fwd"]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"'{pool_id}' 종목풀에 분석에 충분한 가격 데이터가 없습니다.")
    base_return = float(df["fwd"].mean())
    base_rate = float((df["fwd"] > 0).mean() * 100)
    trading_days = int(df["date"].nunique())
    # 겹침 보정: N일 수익률은 N일마다 하나꼴로만 독립이다.
    effective_samples = max(trading_days // forward_days, 1)
    # 비율의 95% 오차 근사(p=0.5 최대분산 가정) — 이보다 작은 차이는 노이즈로 본다.
    rate_error = float(np.sqrt(0.25 / effective_samples) * 100 * 1.96)

    return {
        "pool_id": pool_id,
        "forward_days": forward_days,
        "months": int(months),
        "ma_rule": {"short_ma_days": short_days, "long_ma_days": long_days, "slope_days": slope_days},
        "ticker_count": int(df["ticker"].nunique()),
        "excluded_fixed_count": excluded_count,
        "row_count": int(len(df)),
        "trading_days": trading_days,
        # 시작일(date_from)은 전망일수와 무관하게 고정(df_all 기준). 끝일(date_to)은 미래 종가가
        # 필요하므로 최근 N거래일이 빠져 전망일수만큼 당겨진다(fwd 있는 마지막 = df 기준).
        "date_from": pd.Timestamp(df_all["date"].min()).strftime("%Y-%m-%d"),
        "date_to": pd.Timestamp(df["date"].max()).strftime("%Y-%m-%d"),
        "base_return": round(base_return, 2),
        "base_rate": round(base_rate, 2),
        "performance": _rule_performance(
            df_all,
            pool_id,
            top_n_hold,
            forward_days,
            _load_benchmark_close(pool_id, pool_settings),
        ),
        "effective_samples": effective_samples,
        "rate_error": round(rate_error, 2),
        "disparity": _cross_sectional_table(df, "이격", base_return, base_rate, effective_samples, rate_error),
        "short_disparity": _cross_sectional_table(
            df, "단기이격", base_return, base_rate, effective_samples, rate_error
        ),
        "slope": _cross_sectional_table(df, "기울기", base_return, base_rate, effective_samples, rate_error),
        "quantile_count": QUANTILE_COUNT,
        "disparity_long_short": _quantile_long_short(df, "이격", forward_days),
        "short_disparity_long_short": _quantile_long_short(df, "단기이격", forward_days),
        "slope_long_short": _quantile_long_short(df, "기울기", forward_days),
        "disparity_ic": _information_coefficient(df, "이격", forward_days),
        "short_disparity_ic": _information_coefficient(df, "단기이격", forward_days),
        "slope_ic": _information_coefficient(df, "기울기", forward_days),
    }
