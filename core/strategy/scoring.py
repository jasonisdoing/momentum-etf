"""RANK 전략 점수 계산 및 정규화 함수.

랭킹(utils/rankings.py)과 백테스트(backtest/engine.py)는 **반드시** 이 모듈의
공통 엔진 함수를 통해서만 점수를 계산해야 한다. 점수식이 양쪽으로 갈라지면 백테스트 결과는
의미가 없어진다.

설정 상수(MIN_TRADING_DAYS)도 이 모듈에서 단일 진입점으로 참조한다.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from config import METRIC_WINDOW_MONTHS, MIN_TRADING_DAYS
from utils.moving_averages import calculate_moving_average


def calculate_maps_score(
    close_prices: pd.Series,
    moving_average: pd.Series,
) -> pd.Series:
    """
    RANK(Moving Average Position Score) 점수를 계산합니다.

    Args:
        close_prices: 종가 시리즈
        moving_average: 이동평균 시리즈

    Returns:
        pd.Series: 이동평균 대비 수익률 (%)

    Examples:
        >>> close = pd.Series([110, 115, 120])
        >>> ma = pd.Series([100, 100, 100])
        >>> calculate_maps_score(close, ma)
        0    10.0
        1    15.0
        2    20.0
        dtype: float64
    """
    # 0으로 나누기 방지
    safe_moving_average = moving_average.replace(0, np.nan)
    ma_score = ((close_prices / safe_moving_average) - 1.0) * 100
    # 무한대 값 처리
    ma_score = ma_score.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return ma_score


def calculate_signed_percentile_score(data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Signed percentile 점수를 계산합니다.

    - 양수 값들은 양수 그룹 내에서 백분위 랭킹 → ``[0, 100]``
    - 음수 값들은 절댓값 기준 음수 그룹 내 백분위 랭킹 → ``[-100, 0]``
    - 0 은 그대로 ``0.0``, ``NaN`` 은 유지한다.

    Series 가 입력되면 전체 값들을 한 세트로 보고 랭킹한다.
    DataFrame 이 입력되면 **각 행(일자)** 내부에서 열(티커)들끼리 랭킹한다
    (``axis=1``). 즉 cross-section 점수 계산에 바로 사용할 수 있다.

    랭킹/백테스트 점수 식이 갈라지지 않도록 **반드시 이 함수 하나만** 사용한다.

    Examples:
        >>> s = pd.Series([3, -1, 2, 0])
        >>> calculate_signed_percentile_score(s).tolist()
        [100.0, -100.0, 50.0, 0.0]
    """
    if isinstance(data, pd.Series):
        numeric = pd.to_numeric(data, errors="coerce")
        pos_rank = numeric.where(numeric > 0).rank(method="average", pct=True) * 100.0
        neg_rank = numeric.where(numeric < 0).abs().rank(method="average", pct=True) * -100.0
        result = pos_rank.combine_first(neg_rank)
        result = result.mask(numeric == 0, 0.0)
        return result

    if isinstance(data, pd.DataFrame):
        numeric = data.apply(pd.to_numeric, errors="coerce")
        pos_rank = numeric.where(numeric > 0).rank(axis=1, method="average", pct=True) * 100.0
        neg_rank = numeric.where(numeric < 0).abs().rank(axis=1, method="average", pct=True) * -100.0
        result = pos_rank.combine_first(neg_rank)
        result = result.mask(numeric == 0, 0.0)
        return result

    raise TypeError(f"지원하지 않는 타입입니다: {type(data)!r}. Series 또는 DataFrame 을 전달하세요.")


# --------------------------- 공통 랭킹 엔진 --------------------------- #


def compute_trend_frame(
    close_frame: pd.DataFrame,
    ma_days: int,
) -> pd.DataFrame:
    """[일자 × 티커] 구조의 종가 프레임으로부터 MA 대비 트렌드(%) 프레임을 계산한다.

    각 MA 함수가 ``min_periods=1`` 로 부분 계산을 지원하므로, MA 성숙 기간
    을 만족하지 못하더라도 가용한 종가로 MA 를 계산해 트렌드 값을 반환한다.
    랭킹 포함 여부는 ``compute_eligibility_mask`` 가 ``MIN_TRADING_DAYS`` 기준으로 따로 판정한다.
    """
    days = int(ma_days)
    ma_cols: dict[str, pd.Series] = {}
    for ticker in close_frame.columns:
        series = close_frame[ticker].dropna()
        if series.empty:
            ma_cols[ticker] = pd.Series(np.nan, index=close_frame.index, dtype=float)
            continue
        ma_series = calculate_moving_average(series, days)
        ma_cols[ticker] = ma_series.reindex(close_frame.index)
    ma_frame = pd.DataFrame(ma_cols, index=close_frame.index)
    trend = pd.DataFrame(
        {ticker: calculate_maps_score(close_frame[ticker], ma_frame[ticker]) for ticker in close_frame.columns},
        index=close_frame.index,
    )
    return trend


def compute_ma_disparity(close_series: pd.Series, ma_days: int) -> float | None:
    """종목 하나의 **최신 이격률(%)** — 종가가 이동평균보다 몇 % 위/아래인가.

    `compute_trend_frame` 의 1종목 버전이다. 순위 화면(프레임 단위)과 보유종목 알림(종목
    단위)이 같은 정의를 쓰도록 계산을 여기 한 곳에 둔다 — 예전에는 알림이 따로 계산하면서
    가격 구간을 잘라(EMA 워밍업 부족) 순위 화면과 다른 값을 냈다.

    ``min_periods`` 를 두지 않는 것도 프레임 버전과 같다 — 상장 직후라도 있는 종가로
    부분 계산한다. 랭킹 포함 여부는 ``compute_eligibility_mask`` 가 따로 판정한다.
    """
    series = close_series.dropna()
    if series.empty:
        return None
    ma_series = calculate_moving_average(series, int(ma_days))
    score = calculate_maps_score(series, ma_series)
    if score.empty or pd.isna(score.iloc[-1]):
        return None
    return float(score.iloc[-1])


def rank_score(long_disparity_pct: Any, short_disparity_pct: Any) -> Any:
    """**순위 점수** — 종목을 줄 세우는 단일 기준.

    순위 화면(`/pools-rank`)의 정렬·추천(✅)과 모멘텀 전략(`/strategy-momentum`)의 선정이
    이 함수 하나만 쓴다. 예전에는 두 곳이 각자 "장기 이격률" 을 점수로 삼아, 정의를 바꾸려면
    양쪽을 따로 고쳐야 했고 한쪽만 고치면 두 화면의 순서가 조용히 갈렸다.

    현재 정의: **장기·단기 이격률의 평균**. 장기만 보면 장기 추세는 좋지만 지금 밀리는
    종목이 위에 남고, 단기만 보면 반등 한 번에 순위가 뒤집힌다. 둘을 같은 무게로 본다.

    스칼라(종목 하나)와 Series(프레임 전체) 둘 다 받는다 — 호출부가 형태를 맞출 필요가 없다.
    값이 없으면 None/NaN 을 그대로 돌려준다(임의 값으로 채우지 않는다).
    """
    if isinstance(long_disparity_pct, pd.Series) or isinstance(short_disparity_pct, pd.Series):
        long_series = pd.to_numeric(long_disparity_pct, errors="coerce")
        short_series = pd.to_numeric(short_disparity_pct, errors="coerce")
        return (long_series + short_series) / 2.0
    if long_disparity_pct is None or short_disparity_pct is None:
        return None
    if pd.isna(long_disparity_pct) or pd.isna(short_disparity_pct):
        return None
    return (float(long_disparity_pct) + float(short_disparity_pct)) / 2.0


def drawdown_from_high_pct(
    close_series: pd.Series,
    current_price: float | None = None,
    *,
    window_months: int = METRIC_WINDOW_MONTHS,
) -> float | None:
    """고점 대비(%) — 최근 ``window_months`` 최고가 대비 현재가. 0 이면 신고점.

    순위 화면(`/pools-rank`)과 모멘텀 전략이 같은 값을 쓰도록 여기서만 정의한다.
    창을 캐시 전 기간이 아니라 12개월로 자르는 것이 규칙의 핵심이라, 각자 계산하면
    창 길이가 갈려 같은 종목에 화면마다 다른 값이 붙는다.

    ``current_price`` 를 주면 그 값으로 비교한다(실시간 반영가). 없으면 마지막 종가.
    """
    series = pd.to_numeric(close_series, errors="coerce").dropna()
    if series.empty:
        return None
    price = float(series.iloc[-1]) if current_price is None else float(current_price)
    window = series.loc[series.index[-1] - pd.DateOffset(months=int(window_months)) :]
    if window.empty:
        return None
    max_price = float(window.max())
    if max_price <= 0:
        return None
    return (price / max_price - 1.0) * 100.0


def hold_eligible(long_disparity_pct: Any, short_disparity_pct: Any) -> Any:
    """보유 가능 조건 — **장기 이격 > 0 이고 단기 이격 >= 0**.

    장기 이평선은 종목 선택, 단기 이평선은 손절/익절을 담당한다. 장기 추세가 죽었거나
    단기 추세가 꺾이면 보유하지 않는다. 순위 화면의 추천(✅)·모멘텀 선정·백테스트가
    같은 규칙을 쓰도록 여기서만 정의한다.

    스칼라와 Series 둘 다 받는다 — `rank_score` 와 같다. 종목풀 조건(고정 보유)은
    호출부에서 따로 건다.
    """
    if isinstance(long_disparity_pct, pd.Series) or isinstance(short_disparity_pct, pd.Series):
        long_series = pd.to_numeric(long_disparity_pct, errors="coerce")
        short_series = pd.to_numeric(short_disparity_pct, errors="coerce")
        return long_series.notna() & (long_series > 0) & short_series.notna() & (short_series >= 0)
    if long_disparity_pct is None or short_disparity_pct is None:
        return False
    if pd.isna(long_disparity_pct) or pd.isna(short_disparity_pct):
        return False
    return float(long_disparity_pct) > 0 and float(short_disparity_pct) >= 0


def cap_by_industry(
    candidates: list[str],
    industry_by: Mapping[str, str],
    cap: int | None,
    limit: int,
    *,
    held: Sequence[str] = (),
) -> tuple[list[str], list[str]]:
    """우선순위를 지키되 **한 업종이 상한을 넘지 않도록** ``limit`` 개를 고른다.

    모멘텀·신고가 두 전략이 이 함수 하나만 쓴다. 예전에는 각자 구현을 들고 있어서
    같은 「업종 상한」 설정이 전략마다 다르게 동작했다(보유분을 세느냐가 갈렸다).

    `(고른 것, 상한에 걸려 밀린 것)` 을 돌려준다. 밀린 목록은 화면이 '조건은 맞지만
    업종 상한 때문에 못 산다' 를 표시하는 데 쓴다 — 그냥 후보로 두면 살 것처럼 보인다.

    ``held`` 를 주면 **이미 보유 중인 종목이 상한을 차지한다** — 오늘 새로 담는 것만
    세면 계좌 전체로는 한 업종에 몰릴 수 있다. 상한에 걸린 종목은 건너뛰고 다음 순위가
    그 자리를 채운다. 업종을 모르는 종목(ETF 풀 등)은 묶을 근거가 없어 상한을 적용하지 않는다.
    ``cap`` 이 None/0 이면 제한 없음이다.
    """
    if limit <= 0:
        return [], []
    if not cap:
        return candidates[:limit], []
    counts: dict[str, int] = {}
    for ticker in held:
        key = industry_by.get(ticker, "")
        if key:
            counts[key] = counts.get(key, 0) + 1
    picked: list[str] = []
    blocked: list[str] = []
    for ticker in candidates:
        if len(picked) >= limit:
            break
        key = industry_by.get(ticker, "")
        if key:
            if counts.get(key, 0) >= cap:
                blocked.append(ticker)
                continue
            counts[key] = counts.get(key, 0) + 1
        picked.append(ticker)
    return picked, blocked


def select_holdings(
    candidates: Sequence[Mapping[str, Any]],
    *,
    top_n: int,
    max_per_industry: int | None = None,
    industry_by: Mapping[str, str] | None = None,
) -> list[str]:
    """**보유 대상 선정 — 순위 화면과 모멘텀 전략이 함께 쓰는 단 하나의 규칙.**

    ① 자격(`hold_eligible`) → ② 순위 점수(`rank_score`) 내림차순 → ③ 업종 상한
    (`cap_by_industry`) 을 적용해 최대 `top_n` 개의 티커를 돌려준다.

    예전에는 순위 화면이 ①②만 하고 업종 상한을 빼먹어서, 상한이 걸린 풀에서 표의 ✅ 와
    실제 전략 선정이 어긋났다. 셋을 한 함수로 묶어 그런 누락이 다시 생기지 않게 한다.

    `candidates` 의 각 항목에 필요한 키:
        ticker · long_disparity_pct · short_disparity_pct
    이미 자격을 거른 목록을 넘겨도 된다(조건을 다시 통과할 뿐이다).
    """
    scored: list[tuple[float, str]] = []
    for row in candidates:
        ticker = str(row.get("ticker") or "").strip()
        if not ticker:
            continue
        long_pct, short_pct = row.get("long_disparity_pct"), row.get("short_disparity_pct")
        if not hold_eligible(long_pct, short_pct):
            continue
        score = rank_score(long_pct, short_pct)
        if score is None:
            continue
        scored.append((float(score), ticker))
    # 점수 내림차순. 동점은 티커 순으로 못 박아 실행할 때마다 순서가 흔들리지 않게 한다.
    scored.sort(key=lambda item: (-item[0], item[1]))
    picked, _blocked = cap_by_industry(
        [ticker for _, ticker in scored],
        industry_by or {},
        max_per_industry,
        top_n,
    )
    return picked


def compute_rule_percentile_frame(
    close_frame: pd.DataFrame,
    ma_days: int,
) -> pd.DataFrame:
    """단일 MA 규칙에 대한 signed-percentile 점수 프레임을 계산한다."""
    trend = compute_trend_frame(close_frame, ma_days)
    return calculate_signed_percentile_score(trend)


def compute_eligibility_mask(close_frame: pd.DataFrame) -> pd.DataFrame:
    """각 일자·티커가 ``MIN_TRADING_DAYS`` 이상 종가 데이터를 누적했는지 여부."""
    return close_frame.notna().cumsum() >= int(MIN_TRADING_DAYS)


def _resolve_rule_ma_days(rule: dict[str, Any]) -> int:
    """공통 엔진 MA 규칙에서 사용할 일수를 명시적으로 결정한다."""
    if "long_ma_days" in rule:
        return int(rule["long_ma_days"])
    raise KeyError("MA 규칙에는 long_ma_days 가 필요합니다.")


def combine_rule_percentiles(
    per_rule_frames: Iterable[pd.DataFrame],
    eligibility_mask: pd.DataFrame,
) -> pd.DataFrame:
    """여러 규칙의 percentile 프레임을 합산하고 자격 마스크를 적용한다.

    - 한 규칙이라도 NaN 이면 합산 결과도 NaN (pandas 기본 동작).
    - ``eligibility_mask`` 가 False 인 셀은 NaN 으로 만든다 (랭킹 제외).
    """
    frames = list(per_rule_frames)
    if not frames:
        return pd.DataFrame(
            index=eligibility_mask.index,
            columns=eligibility_mask.columns,
            dtype=float,
        )
    composite = frames[0].copy()
    for pf in frames[1:]:
        composite = composite + pf
    return composite.where(eligibility_mask)


def build_composite_rank_scores(
    close_frame: pd.DataFrame,
    ma_rules: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    """랭킹/백테스트 공통: MA 규칙들을 받아 ``(composite, trend_by_order, percentile_by_order)`` 반환.

    - ``composite`` : [일자 × 티커] 최종 점수 (자격 마스크 적용).
    - ``trend_by_order`` : ``{order: trend_frame}`` — 원천 추세(%) (화면 표시/보관용).
    - ``percentile_by_order`` : ``{order: percentile_frame}`` — 규칙별 signed-percentile.
    """
    trend_by_order: dict[int, pd.DataFrame] = {}
    percentile_by_order: dict[int, pd.DataFrame] = {}
    for rule in ma_rules:
        order = int(rule["order"])
        trend = compute_trend_frame(close_frame, _resolve_rule_ma_days(rule))
        trend_by_order[order] = trend
        percentile_by_order[order] = calculate_signed_percentile_score(trend)

    eligibility = compute_eligibility_mask(close_frame)
    composite = combine_rule_percentiles(
        [percentile_by_order[int(r["order"])] for r in ma_rules],
        eligibility,
    )
    return composite, trend_by_order, percentile_by_order


__all__ = [
    "calculate_maps_score",
    "calculate_signed_percentile_score",
    "compute_trend_frame",
    "rank_score",
    "hold_eligible",
    "cap_by_industry",
    "select_holdings",
    "drawdown_from_high_pct",
    "compute_rule_percentile_frame",
    "compute_eligibility_mask",
    "combine_rule_percentiles",
    "build_composite_rank_scores",
]
