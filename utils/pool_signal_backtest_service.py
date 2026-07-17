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

from typing import Any

import numpy as np
import pandas as pd

from utils.cache_utils import load_cached_close_series_bulk
from utils.logger import get_app_logger
from utils.rankings import get_ticker_type_ma_rules
from utils.stock_list_io import get_etfs

logger = get_app_logger()

FORWARD_DAY_OPTIONS: tuple[int, ...] = (5, 10, 20, 40, 60)
QUANTILE_COUNT = 10
_TRADING_DAYS_PER_MONTH = 21
# t 검정을 낼 수 있는 최소 독립구간 수. 이 미만이면 t 가 우연히 커져 '유의'로 오도할 수 있어
# 판정 자체를 내지 않는다(기간을 늘리거나 전망일수를 줄이면 확보된다).
MIN_INDEPENDENT_SAMPLES = 10


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
    independent = len(spread.iloc[::forward_days])
    if independent < MIN_INDEPENDENT_SAMPLES:
        if spread.empty:
            return None
        return {
            "insufficient": True,
            "mean": round(float(spread.mean()), digits),
            "win_rate": round(float((spread > 0).mean() * 100), 1),
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
        "win_rate": round(float((spread > 0).mean() * 100), 1),
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


def compute_pool_signal_backtest(pool_id: str, forward_days: int = 20, months: int = 36) -> dict[str, Any]:
    """종목풀의 이격/기울기 → 향후 N일 수익 실증 결과를 반환한다.

    MA 파라미터(단기/메인/기울기 일수)는 해당 종목풀 설정을 그대로 쓴다.
    고정 보유 종목(exclude_from_ranking)은 투자 후보가 아니므로 제외한다.
    """
    if forward_days not in FORWARD_DAY_OPTIONS:
        options = ", ".join(str(day) for day in FORWARD_DAY_OPTIONS)
        raise ValueError(f"전망일수는 {options} 중 하나여야 합니다: {forward_days}")
    if not (1 <= int(months) <= 120):
        raise ValueError(f"기간은 1~120개월이어야 합니다: {months}")

    rule = get_ticker_type_ma_rules(pool_id)[0]
    short_days = int(rule["short_ma_days"])
    main_days = int(rule["main_ma_days"])
    slope_days = int(rule["slope_days"])
    window = int(months) * _TRADING_DAYS_PER_MONTH

    all_etfs = get_etfs(pool_id)
    etfs = [item for item in all_etfs if not bool(item.get("exclude_from_ranking"))]
    excluded_count = len(all_etfs) - len(etfs)
    if not etfs:
        raise ValueError(f"'{pool_id}' 종목풀에 분석 가능한 종목이 없습니다(고정 보유 제외 후 0개).")

    series_map = load_cached_close_series_bulk(pool_id, [item["ticker"] for item in etfs])
    frames: list[pd.DataFrame] = []
    min_length = main_days + slope_days + forward_days + 20
    for ticker, series in series_map.items():
        close = pd.to_numeric(series, errors="coerce").dropna()
        if len(close) < min_length:
            continue
        short_ma = close.rolling(short_days).mean()
        main_ma = close.rolling(main_days).mean()
        frame = pd.DataFrame(
            {
                "이격": (close / main_ma - 1.0) * 100.0,
                "기울기": (short_ma / short_ma.shift(slope_days) - 1.0) * 100.0,
                # 향후 N거래일 수익률(라벨). 마지막 N일은 미래가 없어 자동 제외된다.
                "fwd": (close.shift(-forward_days) / close - 1.0) * 100.0,
            }
        ).dropna()
        if frame.empty:
            continue
        frame = frame.tail(window)
        frame["ticker"] = ticker
        frame["date"] = frame.index
        frames.append(frame)

    if not frames:
        raise ValueError(f"'{pool_id}' 종목풀에 분석에 충분한 가격 데이터가 없습니다.")

    df = pd.concat(frames, ignore_index=True)
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
        "ma_rule": {"short_ma_days": short_days, "main_ma_days": main_days, "slope_days": slope_days},
        "ticker_count": int(df["ticker"].nunique()),
        "excluded_fixed_count": excluded_count,
        "row_count": int(len(df)),
        "trading_days": trading_days,
        # 실제 분석에 쓰인 날짜 범위. 전망 N일은 미래 종가가 필요하므로 최근 N거래일은
        # 아직 결과가 없어 빠진다 → '최근 1개월'을 골라도 끝일자가 오늘이 아닐 수 있다.
        "date_from": pd.Timestamp(df["date"].min()).strftime("%Y-%m-%d"),
        "date_to": pd.Timestamp(df["date"].max()).strftime("%Y-%m-%d"),
        "base_return": round(base_return, 2),
        "base_rate": round(base_rate, 2),
        "effective_samples": effective_samples,
        "rate_error": round(rate_error, 2),
        "disparity": _cross_sectional_table(df, "이격", base_return, base_rate, effective_samples, rate_error),
        "slope": _cross_sectional_table(df, "기울기", base_return, base_rate, effective_samples, rate_error),
        "quantile_count": QUANTILE_COUNT,
        "disparity_long_short": _quantile_long_short(df, "이격", forward_days),
        "slope_long_short": _quantile_long_short(df, "기울기", forward_days),
        "disparity_ic": _information_coefficient(df, "이격", forward_days),
        "slope_ic": _information_coefficient(df, "기울기", forward_days),
    }
