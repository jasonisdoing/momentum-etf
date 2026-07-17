"""종목풀 신호(이격/기울기/배열) 실증 백테스트 — 읽기 전용 분석.

향후 N거래일 상승확률이 이격·기울기·배열과 실제로 관계있는지 분위별로 집계한다.

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
QUANTILE_COUNT = 5
_TRADING_DAYS_PER_MONTH = 21


_QUANTILE_LABELS = ("1분위 (하위 20%)", "2분위", "3분위", "4분위", "5분위 (상위 20%)")


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


def _cross_sectional_table(
    df: pd.DataFrame,
    column: str,
    base_return: float,
    base_rate: float,
    effective_samples: int,
    rate_error: float,
) -> list[dict[str, Any]]:
    """날짜별 상대분위 표.

    같은 날짜 안에서 종목끼리 순위를 매겨 분위를 나눈다. 전체 기간을 섞어 자르면
    '이격 최하위 = 시장이 빠진 날'이 되어 종목 선택력이 아니라 시장 타이밍 효과가 섞인다.
    """
    ranked = df.groupby("date")[column].rank(pct=True)
    bins = pd.cut(ranked, [0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=list(range(1, QUANTILE_COUNT + 1)))
    frame = df.assign(_bucket=bins).dropna(subset=["_bucket"])
    rows: list[dict[str, Any]] = []
    for order, group in frame.groupby("_bucket", observed=True):
        rows.append(
            _summarize_group(
                group, _QUANTILE_LABELS[int(order) - 1], base_return, base_rate, effective_samples, rate_error
            )
        )
        rows[-1]["order"] = int(order)
    return rows


def compute_pool_signal_backtest(pool_id: str, forward_days: int = 20, months: int = 36) -> dict[str, Any]:
    """종목풀의 이격/기울기/배열 → 향후 N일 상승확률 실증 결과를 반환한다.

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
                "배열": short_ma >= main_ma,
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

    arrangement: list[dict[str, Any]] = []
    for is_aligned, group in df.groupby("배열", observed=True):
        arrangement.append(
            _summarize_group(
                group,
                "정배열" if bool(is_aligned) else "역배열",
                base_return,
                base_rate,
                effective_samples,
                rate_error,
            )
        )
    arrangement.sort(key=lambda item: item["label"], reverse=True)

    return {
        "pool_id": pool_id,
        "forward_days": forward_days,
        "months": int(months),
        "ma_rule": {"short_ma_days": short_days, "main_ma_days": main_days, "slope_days": slope_days},
        "ticker_count": int(df["ticker"].nunique()),
        "excluded_fixed_count": excluded_count,
        "row_count": int(len(df)),
        "trading_days": trading_days,
        "base_return": round(base_return, 2),
        "base_rate": round(base_rate, 2),
        "effective_samples": effective_samples,
        "rate_error": round(rate_error, 2),
        "disparity": _cross_sectional_table(df, "이격", base_return, base_rate, effective_samples, rate_error),
        "slope": _cross_sectional_table(df, "기울기", base_return, base_rate, effective_samples, rate_error),
        "arrangement": arrangement,
    }
