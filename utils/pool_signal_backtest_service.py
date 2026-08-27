"""종목풀 신호(이격/단기이격) 규칙 백테스트 — 읽기 전용 분석.

이격 상위 N종목 보유 + 단기이격 손절 규칙을 포트폴리오로 시뮬레이션해
규칙과 벤치마크의 누적수익·MDD·소르티노를 비교한다.

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

from config import BACKTEST_MONTH_OPTIONS as MONTH_OPTIONS
from config import CACHE_START_DATE, FORWARD_DAY_OPTIONS
from utils.cache_utils import load_cached_close_series_bulk
from utils.logger import get_app_logger
from utils.ma_options import LONG_MA_OPTIONS, SHORT_MA_OPTIONS
from utils.moving_averages import calculate_moving_average
from utils.pool_settings_store import (
    get_pool_benchmark_ticker,
    get_pool_market_regime_index,
)
from utils.rankings import get_ticker_type_ma_rules, hold_eligible_mask
from utils.settings_loader import get_ticker_type_settings
from utils.stock_list_io import get_etfs

logger = get_app_logger()

_TRADING_DAYS_PER_MONTH = 21


def get_max_backtest_months(today: date | None = None) -> int:
    """가격 캐시 시작일 기준으로 선택 가능한 최대 개월 수를 계산한다."""
    start = datetime.strptime(CACHE_START_DATE, "%Y-%m-%d").date()
    end = today or date.today()
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day < start.day:
        months -= 1
    return max(months, 1)


def get_month_options() -> list[int]:
    """기간 셀렉트 옵션. 가격 캐시가 못 채우는 구간은 뺀다.

    예전에는 캐시가 60개월보다 길면 그 최대값(예 91개월)을 끝에 덧붙였는데, 캐시가
    자랄수록 값이 매달 달라져 화면마다 선택지가 어긋났다. 고정 목록만 노출한다.
    """
    max_months = get_max_backtest_months()
    return [month for month in MONTH_OPTIONS if month <= max_months]


def _format_date(value: Any) -> str:
    """응답에 넣을 날짜를 YYYY-MM-DD 문자열로 맞춘다."""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _curve_mdd_info(curve: pd.Series) -> dict[str, Any]:
    """자산 곡선(일별 등)에서 최대낙폭(MDD)과 고점~저점 구간을 계산한다.

    MDD 는 반드시 **일별** 곡선으로 계산한다. 리밸런싱 간격(예 20일)으로만 찍은 점으로
    구하면 그 사이의 급락을 건너뛰어 실제보다 훨씬 작게 나온다(예: 실제 −20% → −6%).
    """
    curve = curve.dropna()
    if len(curve) < 2:
        return {"mdd_pct": None, "mdd_start_date": None, "mdd_end_date": None}
    drawdown = curve / curve.cummax() - 1.0
    end_date = drawdown.idxmin()
    start_date = curve.loc[:end_date].idxmax()
    return {
        "mdd_pct": round(float(drawdown.loc[end_date]) * 100.0, 1),
        "mdd_start_date": _format_date(start_date),
        "mdd_end_date": _format_date(end_date),
    }


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


def _monthly_returns(curve: pd.Series) -> dict[str, float]:
    """일별 자산(또는 종가) 곡선을 월별 수익률(%)로 자른다. {"YYYY-MM": pct}.

    첫 달의 기준은 곡선의 **시작값**이다 — 그 달 도중에 운용이 시작됐다면 그 부분 기간의
    수익률이 된다(직전 달 종가가 아니라). 마지막 달도 마찬가지로 부분 기간일 수 있다.
    """
    series = pd.to_numeric(curve, errors="coerce").dropna()
    if series.empty:
        return {}
    series.index = pd.to_datetime(series.index)
    series = series.sort_index()

    monthly_last = series.groupby(series.index.to_period("M")).last()
    if monthly_last.empty:
        return {}
    # 첫 달을 계산하려면 그 앞에 기준점 하나가 필요하다 — 곡선의 시작값을 직전 달 자리에 둔다.
    seed = pd.Series([float(series.iloc[0])], index=[monthly_last.index[0] - 1])
    changes = pd.concat([seed, monthly_last]).pct_change().dropna() * 100.0
    return {str(period): round(float(value), 2) for period, value in changes.items()}


def _rule_performance(
    df: pd.DataFrame,
    pool_id: str,
    top_n: int,
    forward_days: int,
    benchmark: dict[str, Any],
    hold_threshold_k: float | None = None,
    down_market_invest_pct: float = 100.0,
) -> dict[str, Any] | None:
    """현재 설정 그대로 규칙을 돌렸을 때의 기간 실적.

    순위 화면의 추천(✅)과 동일한 규칙으로 ``forward_days`` 마다 리밸런싱한다.
    조건을 만족하는 종목이 없는 회차는 현금 보유(0%)로 본다.

    ``hold_threshold_k`` (상대 임계, 값 기반 히스테리시스): None 이면 매 회차 이격 상위 N 을
    새로 뽑는다(기본, 순위 화면과 동일). 값이 주어지면 **이미 보유한 종목은 이격이
    (그날 N등 이격 × k) 이상이면 유지**하고, 빈 자리만 상위 후보로 채운다. k<1 일수록
    기존 보유를 오래 들고 가 회전율이 준다. 순위가 아니라 이격 값으로 판단해 신호 크기를
    반영한다(1~2등 차이와 5~6등 차이를 다르게 본다).

    **기대수익이 아니라 지나간 기간의 실적**이다. 표본은 기간/forward_days 회차뿐이라
    강세장 한 구간이 통째로 들어오면 숫자가 커진다. 그래서 기저(아무 종목이나 보유)와
    벤치마크(설정된 기준 종목)를 함께 반환해 '규칙이 기여한 몫'을 구분할 수 있게 한다.
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

    pool_settings = get_ticker_type_settings(pool_id)
    missing_slippage = [
        key for key in ("BUY_SLIPPAGE_PCT", "SELL_SLIPPAGE_PCT") if pool_settings.get(key) in (None, "")
    ]
    if missing_slippage:
        raise ValueError(
            f"종목풀 '{pool_id}' 의 슬리피지 설정이 없습니다: {', '.join(missing_slippage)}. "
            "/pools-settings 에서 매수/매도 슬리피지를 저장하세요."
        )
    buy_pct = float(pool_settings["BUY_SLIPPAGE_PCT"]) / 100.0
    sell_pct = float(pool_settings["SELL_SLIPPAGE_PCT"]) / 100.0
    round_trip_pct = float(pool_settings["BUY_SLIPPAGE_PCT"]) + float(pool_settings["SELL_SLIPPAGE_PCT"])
    market_regime = (
        _load_market_regime_map(pool_settings, require=True) if float(down_market_invest_pct) < 100.0 else None
    )

    # 종목마다 상장일·거래정지가 달라 pivot 에 구멍이 생긴다. 포트폴리오는 금액을 이어 추적하므로
    # NaN 이 전파되면 곡선이 망가진다 → ffill(거래정지는 직전 종가 유지 = 그날 수익 0)로 메운다.
    # 상장 전(앞쪽) 결측은 그 종목이 아직 신호도 없어 선택되지 않으므로 그대로 둔다.
    close_wide = df.pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()
    short_disparity_wide = df.pivot_table(index="date", columns="ticker", values="단기이격").sort_index().ffill()

    # 포트폴리오 시뮬레이션(비중 방식):
    #  - 목표 비중 1/N. **유지 종목은 팔지 않고 비중을 그대로 둔다**(승자 비중이 커진 채 유지).
    #  - 이탈 종목(선택에서 빠짐)은 전량 매도.
    #  - 신규는 목표 1/N 까지만 매수. 현금이 부족하면 남은 현금을 신규끼리 균등 분배(1/N 미달),
    #    남으면 다음 회차로 이월한다. 슬리피지는 실제 거래금액에만 물린다(유지 종목은 0).
    target_w = 1.0 / top_n
    holdings: dict[str, float] = {}  # 종목 → 평가금액(초기자본 1.0 기준)
    cash = 1.0
    daily_curve_parts: list[pd.Series] = []
    round_returns: list[float] = []
    round_costs_pct: list[float] = []
    round_turnovers: list[float] = []
    baskets: list[set[str]] = []
    cash_rounds = 0
    partial_rounds = 0
    down_market_rounds = 0

    for i, as_of in enumerate(rebalance_dates):
        total_before = cash + sum(holdings.values())
        invest_ratio = 1.0
        if market_regime is not None:
            regime = _resolve_market_regime_for_date(market_regime["series"], as_of)
            if regime == "down":
                invest_ratio = float(down_market_invest_pct) / 100.0
                down_market_rounds += 1
        day = eligible[eligible["date"] == as_of]
        if day.empty:
            selected: list[str] = []
        else:
            ranked = day.sort_values("이격", ascending=False)
            ranked_tickers = list(ranked["ticker"])
            if hold_threshold_k is None:
                selected = ranked_tickers[:top_n]
            else:
                disparity_by_ticker = dict(zip(ranked["ticker"], ranked["이격"], strict=False))
                nth_disparity = float(ranked["이격"].iloc[min(top_n, len(ranked)) - 1])
                threshold = nth_disparity * hold_threshold_k
                keep = [t for t in ranked_tickers if t in holdings and disparity_by_ticker[t] >= threshold][:top_n]
                new_slots = top_n - len(keep)
                newcomers_sel = [t for t in ranked_tickers if t not in keep][:new_slots]
                selected = keep + newcomers_sel

        # 이탈 종목 전량 매도 → 현금
        sell_amount = 0.0
        for ticker in list(holdings):
            if ticker not in selected:
                sell_amount += holdings[ticker]
                cash += holdings.pop(ticker)

        # 매수는 레짐 켬/끔과 무관하게 항상 '승자 키우기': 신규만 목표 1/N(100% 기준)까지 사고,
        # 유지 종목은 비중 드리프트를 둔다(팔지 않음). 종목 간 상대 비중(승자 구조)은 여기서 결정된다.
        buy_amount = 0.0
        target_amount = target_w * total_before
        newcomers = [ticker for ticker in selected if ticker not in holdings]
        if newcomers:
            each = target_amount if cash >= target_amount * len(newcomers) else cash / len(newcomers)
            for ticker in newcomers:
                holdings[ticker] = each
                cash -= each
                buy_amount += each

        # 레짐 볼륨 다이얼: 총 투자금액만 목표 투자비중으로 맞춘다. 상대 비중은 안 건드린다(리셋 없음).
        #  - 하락(현재>목표): 모든 종목을 같은 배율로 축소, 차액은 현금.
        #  - 상승 복귀(현재<목표, 현금 있음): 남은 현금을 현재 비중에 비례해 재투입(같은 배율로 증가).
        if market_regime is not None:
            target_invested = total_before * invest_ratio
            current_invested = sum(holdings.values())
            if current_invested > target_invested and current_invested > 0:
                scale = target_invested / current_invested
                for ticker in list(holdings):
                    reduced = holdings[ticker] * (1.0 - scale)
                    holdings[ticker] *= scale
                    sell_amount += reduced
                    cash += reduced
                    if holdings[ticker] <= 1e-12:
                        holdings.pop(ticker)
            elif current_invested < target_invested and current_invested > 0 and cash > 0:
                add_total = min(target_invested - current_invested, cash)
                for ticker in list(holdings):
                    add = add_total * (holdings[ticker] / current_invested)
                    holdings[ticker] += add
                    buy_amount += add
                cash -= add_total

        # 정기 리밸런싱 비용. 긴급 매도 비용은 보유 기간 일별 루프에서 추가한다.
        cost = sell_amount * sell_pct + buy_amount * buy_pct
        cash -= cost
        round_sell_amount = sell_amount
        round_buy_amount = buy_amount
        round_cost = cost

        baskets.append(set(holdings))
        if not holdings:
            cash_rounds += 1
        elif len(holdings) < top_n:
            partial_rounds += 1

        # N일 홀드: 일별 총자산 곡선 + 회차 끝 종목별 평가금액 갱신(드리프트 유지)
        start_pos = close_wide.index.get_loc(as_of)
        end_pos = (
            close_wide.index.get_loc(rebalance_dates[i + 1]) if i + 1 < len(rebalance_dates) else len(close_wide) - 1
        )
        window = close_wide.iloc[start_pos : end_pos + 1]
        asset_values: list[float] = []
        for pos, current_date in enumerate(window.index):
            if pos > 0 and holdings:
                previous_date = window.index[pos - 1]
                for ticker in list(holdings):
                    prev_price = close_wide.at[previous_date, ticker] if ticker in close_wide.columns else np.nan
                    current_price = close_wide.at[current_date, ticker] if ticker in close_wide.columns else np.nan
                    if pd.notna(prev_price) and pd.notna(current_price) and float(prev_price) > 0:
                        holdings[ticker] *= float(current_price) / float(prev_price)

                is_next_rebalance_boundary = i + 1 < len(rebalance_dates) and current_date == rebalance_dates[i + 1]
                if not is_next_rebalance_boundary:
                    # 긴급 매도: 정기 리밸런싱일이 아니어도 단기이격이 음수로 내려가면 현금화한다.
                    # 매도 후 생긴 현금은 다음 리밸런싱일까지 대기한다.
                    for ticker in list(holdings):
                        short_disparity = (
                            short_disparity_wide.at[current_date, ticker]
                            if ticker in short_disparity_wide.columns
                            else np.nan
                        )
                        if pd.notna(short_disparity) and float(short_disparity) < 0:
                            emergency_sell_amount = holdings.pop(ticker)
                            emergency_cost = emergency_sell_amount * sell_pct
                            cash += emergency_sell_amount - emergency_cost
                            round_sell_amount += emergency_sell_amount
                            round_cost += emergency_cost

            asset_values.append(cash + sum(holdings.values()))

        asset_series = pd.Series(asset_values, index=window.index, dtype="float64")
        round_costs_pct.append(round_cost / total_before * 100.0 if total_before > 0 else 0.0)
        round_turnovers.append((round_sell_amount + round_buy_amount) / total_before if total_before > 0 else 0.0)
        # 첫 회차는 시작점 포함, 이후는 경계(직전 회차 끝과 중복) 제외.
        daily_curve_parts.append(asset_series if i == 0 else asset_series.iloc[1:])

        total_after = cash + sum(holdings.values())
        round_returns.append((total_after / total_before - 1.0) * 100.0 if total_before > 0 else 0.0)

    rule_curve = pd.concat(daily_curve_parts) if daily_curve_parts else pd.Series(dtype="float64")
    round_returns_s = pd.Series(round_returns, dtype="float64")
    turnover = float(np.mean(round_turnovers)) if round_turnovers else 0.0
    cost_per_round = float(np.mean(round_costs_pct)) if round_costs_pct else 0.0

    def _compound(values: pd.Series) -> float:
        return float((1.0 + values / 100.0).prod() - 1.0) * 100.0

    # ① 종목풀 규칙: 누적·소르티노는 개별 fwd 기반 회차수익(정확), MDD 는 일별 곡선(근사).
    rule_stats = {
        "cumulative_pct": round(_compound(round_returns_s), 1),
        **_curve_mdd_info(rule_curve),
        "sortino": _sortino(round_returns_s, forward_days),
    }

    # ② 벤치마크: 운용 기간(첫 리밸런싱 ~ 마지막 청산일) 동안 그냥 계속 보유.
    # 누적은 시작·끝 종가비로(텔레스코핑 오차 없음). MDD 는 그 구간 일별 종가곡선으로,
    # 소르티노는 회차별 수익으로. 못 쓰면 기저로 대체하지 않고 사유를 그대로 내려보낸다.
    benchmark_payload: dict[str, Any] | None = None
    benchmark_window: pd.Series | None = None
    # 설정·캐시는 있는데 운용 구간과 날짜가 안 맞는 경우도 구분해 둔다.
    benchmark_status = str(benchmark.get("status") or "unset")
    if benchmark["close"] is not None:
        benchmark_status = "no_overlap"
        bclose = benchmark["close"]
        start_d = pd.Timestamp(rebalance_dates[0]).normalize()
        last_d = pd.Timestamp(rebalance_dates[-1]).normalize()
        if start_d in bclose.index and last_d in bclose.index:
            i0 = bclose.index.get_loc(start_d)
            # 마지막 리밸런싱에서 forward_days 뒤 = 규칙이 마지막에 청산한 날(범위 밖이면 최신 종가).
            i_end = min(bclose.index.get_loc(last_d) + forward_days, len(bclose) - 1)
            cumulative = (float(bclose.iloc[i_end]) / float(bclose.iloc[i0]) - 1.0) * 100.0
            # 회차별 벤치 수익(소르티노용): 각 리밸런싱 시점의 forward_days 수익.
            bench_seg = []
            for as_of in rebalance_dates:
                d = pd.Timestamp(as_of).normalize()
                if d not in bclose.index:
                    continue
                pos = bclose.index.get_loc(d)
                if pos + forward_days < len(bclose):
                    bench_seg.append((float(bclose.iloc[pos + forward_days]) / float(bclose.iloc[pos]) - 1.0) * 100.0)
            bench_series = pd.Series(bench_seg, dtype="float64")
            benchmark_window = bclose.iloc[i0 : i_end + 1]
            benchmark_mdd = _curve_mdd_info(benchmark_window)
            benchmark_status = "ok"
            benchmark_payload = {
                "ticker": benchmark["ticker"],
                "name": benchmark["name"],
                "cumulative_pct": round(cumulative, 1),
                **benchmark_mdd,
                "sortino": _sortino(bench_series, forward_days),
            }

    # 월별 상세 — 전략은 일별 자산곡선, 벤치마크는 같은 운용구간의 종가곡선 기준.
    # 캐시 마지막 날짜가 종목마다 달라 두 곡선의 끝이 어긋난다(전략 쪽은 ffill 로 평평한
    # 꼬리가 붙기도 한다). 그대로 두면 한쪽만 있는 달이 생겨 비교가 안 되므로 **겹치는
    # 구간으로 잘라** 같은 기간끼리 비교한다.
    strategy_window = rule_curve
    if benchmark_window is not None and not benchmark_window.empty and not rule_curve.empty:
        strategy_index = pd.to_datetime(rule_curve.index)
        bench_index = pd.to_datetime(benchmark_window.index)
        overlap_start = max(strategy_index.min(), bench_index.min())
        overlap_end = min(strategy_index.max(), bench_index.max())
        strategy_window = rule_curve.loc[overlap_start:overlap_end]
        benchmark_window = benchmark_window.loc[overlap_start:overlap_end]

    strategy_monthly = _monthly_returns(strategy_window)
    benchmark_monthly = _monthly_returns(benchmark_window) if benchmark_window is not None else {}
    monthly_rows: list[dict[str, Any]] = []
    for month in sorted(set(strategy_monthly) | set(benchmark_monthly)):
        monthly_rows.append(
            {
                "month": month,
                "strategy_pct": strategy_monthly.get(month),
                "benchmark_pct": benchmark_monthly.get(month),
            }
        )

    return {
        "top_n_hold": int(top_n),
        "hold_threshold_k": hold_threshold_k,
        "rounds": len(rebalance_dates),
        "cash_rounds": cash_rounds,
        "partial_rounds": partial_rounds,
        "mean_return": round(float(round_returns_s.mean()), 2) if not round_returns_s.empty else 0.0,
        "wins": int((round_returns_s > 0).sum()),
        "losses": int((round_returns_s < 0).sum()),
        "turnover_pct": round(turnover * 100.0, 1),
        "round_trip_pct": round(round_trip_pct, 2),
        "cost_per_round_pct": round(cost_per_round, 2),
        "down_market_invest_pct": round(float(down_market_invest_pct), 1),
        "market_regime_index": market_regime["index"] if market_regime is not None else None,
        "down_market_rounds": down_market_rounds,
        "rule": rule_stats,
        "benchmark": benchmark_payload,
        # 벤치마크를 못 쓸 때 화면이 원인을 구분해 보여주기 위한 값.
        "benchmark_status": benchmark_status,
        "benchmark_ticker": benchmark["ticker"] or None,
        "benchmark_name": benchmark["name"] or None,
        "monthly": monthly_rows,
    }


def _load_market_regime_map(pool_settings: dict[str, Any], *, require: bool) -> dict[str, Any] | None:
    """종목풀의 시장 레짐 지수로 일별 상승/하락 맵을 만든다."""
    index = get_pool_market_regime_index(pool_settings)
    if index is None:
        if require:
            raise ValueError("하락시 투자비중을 100% 미만으로 쓰려면 /pools-settings 에서 시장 레짐 지수를 설정하세요.")
        return None

    from utils.market_trend_service import _calculate_supertrend, _resolve_supertrend_params, load_index_ohlc

    ticker = index["ticker"]
    ohlc = load_index_ohlc(ticker)
    if ohlc is None or ohlc.empty:
        raise ValueError(f"시장 레짐 지수 '{index['name']}'({ticker}) 의 가격 데이터를 불러오지 못했습니다.")

    period, multiplier = _resolve_supertrend_params(ticker)
    st = _calculate_supertrend(ohlc, period=period, multiplier=multiplier)
    if st.empty or "direction" not in st:
        raise ValueError(f"시장 레짐 지수 '{index['name']}'({ticker}) 의 레짐 계산 결과가 없습니다.")

    series = pd.Series(
        np.where(st["direction"] == 1, "up", "down"), index=pd.to_datetime(st.index).normalize()
    ).sort_index()
    return {"index": index, "series": series}


def _resolve_market_regime_for_date(regime: pd.Series, as_of: Any) -> str:
    """리밸런싱일 기준 가장 최근 시장 레짐을 반환한다. 이전 레짐이 없으면 실패한다."""
    target = pd.Timestamp(as_of).normalize()
    pos = regime.index.searchsorted(target, side="right") - 1
    if pos < 0:
        raise ValueError(f"{target.strftime('%Y-%m-%d')} 이전의 시장 레짐 데이터가 없습니다.")
    value = str(regime.iloc[pos])
    if value not in {"up", "down"}:
        raise ValueError(f"{target.strftime('%Y-%m-%d')} 시장 레짐 값이 올바르지 않습니다: {value}")
    return value


def _load_benchmark_close(pool_id: str, pool_settings: dict[str, Any]) -> dict[str, Any]:
    """벤치마크 종목의 종가 시리즈와 **왜 못 쓰는지**를 함께 반환한다.

    벤치마크는 매수 후보에서 빠지므로 종가를 여기서 따로 불러온다. 벤치마크 누적은
    '규칙 운용 기간 동안 이 종목을 그냥 계속 보유'로, 시작·끝 종가비로만 계산한다
    (리밸런싱마다 끊어 곱하면 거래일 경계에서 텔레스코핑이 깨져 부정확해진다).

    ``status`` 를 나눠 두는 이유: 예전에는 미설정과 가격 캐시 없음을 똑같이 None 으로
    돌려줘 화면이 둘 다 '미설정'으로 표시했다. 설정은 돼 있는데 캐시가 없는 경우
    (벤치마크가 종목풀 구성 종목이 아닐 때) 원인을 엉뚱한 곳에서 찾게 된다.

    반환: ``{"status", "ticker", "name", "close"}``
      - ``ok``       : close 사용 가능
      - ``unset``    : 종목풀에 벤치마크 미설정
      - ``no_cache`` : 설정은 있으나 해당 종목풀 가격 캐시에 종가가 없음
    """
    ticker = get_pool_benchmark_ticker(pool_settings)
    if not ticker:
        return {"status": "unset", "ticker": "", "name": "", "close": None}

    benchmark = pool_settings.get("BENCHMARK") or {}
    name = str(benchmark.get("name") or "").strip() or ticker

    series_map = load_cached_close_series_bulk(pool_id, [ticker])
    close = pd.to_numeric(series_map.get(ticker, pd.Series(dtype="float64")), errors="coerce").dropna()
    if len(close) < 2:
        return {"status": "no_cache", "ticker": ticker, "name": name, "close": None}

    close.index = pd.to_datetime(close.index).normalize()
    return {"status": "ok", "ticker": ticker, "name": name, "close": close}


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
    hold_threshold_k: float | None = None,
    down_market_invest_pct: float,
) -> dict[str, Any]:
    """종목풀의 이격/단기이격 규칙 → 최근 기간 실적(규칙/벤치마크)을 반환한다.

    신호 정의는 순위 화면(`utils.rankings`)과 같다. 이격은 장기 이평선, 단기이격은
    단기 이평선 기준이며, 두 이평선의 역할(선택/손절)로 포트폴리오를 시뮬레이션한다.

    MA 파라미터(단기/장기 이평선)는 해당 종목풀 설정을 기본으로 쓰되 화면 오버라이드가 있으면
    그 값으로 실험한다. 제외 종목(exclude_from_ranking)은 투자 후보가 아니므로 제외한다.
    """
    if forward_days not in FORWARD_DAY_OPTIONS:
        options = ", ".join(str(day) for day in FORWARD_DAY_OPTIONS)
        raise ValueError(f"전망일수는 {options} 중 하나여야 합니다: {forward_days}")
    max_months = get_max_backtest_months()
    if not (1 <= int(months) <= max_months):
        raise ValueError(f"기간은 1~{max_months}개월이어야 합니다: {months}")
    if hold_threshold_k is not None and not (0.0 < float(hold_threshold_k) <= 1.0):
        raise ValueError(f"보유 유지 기준(k)은 0 초과 1 이하여야 합니다: {hold_threshold_k}")
    if not (0.0 <= float(down_market_invest_pct) <= 100.0):
        raise ValueError(f"하락시 투자비중은 0~100 범위여야 합니다: {down_market_invest_pct}")

    # MA/보유수 파라미터는 종목풀 설정이 기본. 화면에서 넘긴 오버라이드가 있으면 그 값으로
    # 실험한다(저장은 하지 않음). 오버라이드도 허용값인지 반드시 검증한다.
    rule = get_ticker_type_ma_rules(pool_id)[0]
    short_days = _resolve_int_override(short_ma_days, rule["short_ma_days"], SHORT_MA_OPTIONS, "단기 이평선")
    long_days = _resolve_int_override(long_ma_days, rule["long_ma_days"], LONG_MA_OPTIONS, "장기 이평선")
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
        raise ValueError(f"'{pool_id}' 종목풀에 분석 가능한 종목이 없습니다(제외 종목·벤치마크를 뺀 후 0개).")

    series_map = load_cached_close_series_bulk(pool_id, [item["ticker"] for item in etfs])
    frames: list[pd.DataFrame] = []
    min_length = long_days + 20
    for ticker, series in series_map.items():
        close = pd.to_numeric(series, errors="coerce").dropna()
        if len(close) < min_length:
            continue
        short_ma = calculate_moving_average(close, short_days, min_periods=short_days)
        long_ma = calculate_moving_average(close, long_days, min_periods=long_days)
        frame = pd.DataFrame(
            {
                "close": close,
                "이격": (close / long_ma - 1.0) * 100.0,
                # 단기이격: 순위 화면과 동일하게 이격과 같은 식에 단기 이평선을 넣은 값.
                "단기이격": (close / short_ma - 1.0) * 100.0,
                # 향후 N거래일 수익률(라벨). 마지막 N거래일은 미래가 없어 NaN.
                "fwd": (close.shift(-forward_days) / close - 1.0) * 100.0,
            }
        )
        # 신호가 유효한 구간만 남긴다. fwd 는 최근 구간에서 NaN 이어도 남겨서,
        # 분석 시작일을 전망일수와 무관하게 고정한다(전망일수를 바꿔도 벤치가 안 흔들리게).
        frame = frame.dropna(subset=["이격", "단기이격"])
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
    # 끝일 표시는 미래수익(fwd)이 있는 마지막 날짜. df_all(시작 고정)은 performance 리밸런싱용.
    scored_dates = df_all.dropna(subset=["fwd"])["date"]
    if scored_dates.empty:
        raise ValueError(f"'{pool_id}' 종목풀에 분석에 충분한 가격 데이터가 없습니다.")

    return {
        "pool_id": pool_id,
        "forward_days": forward_days,
        "months": int(months),
        "excluded_fixed_count": excluded_count,
        # 시작일(date_from)은 전망일수와 무관하게 고정(df_all 기준). 끝일(date_to)은 미래 종가가
        # 필요하므로 최근 N거래일이 빠져 전망일수만큼 당겨진다(fwd 있는 마지막).
        "date_from": pd.Timestamp(df_all["date"].min()).strftime("%Y-%m-%d"),
        "date_to": pd.Timestamp(scored_dates.max()).strftime("%Y-%m-%d"),
        "performance": _rule_performance(
            df_all,
            pool_id,
            top_n_hold,
            forward_days,
            _load_benchmark_close(pool_id, pool_settings),
            hold_threshold_k=hold_threshold_k,
            down_market_invest_pct=float(down_market_invest_pct),
        ),
    }
