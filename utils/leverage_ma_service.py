"""이동평균선 크로스 레버리지 전략의 튜닝 뷰·추천 판정 서비스 (화면·배치 공용).

설정·상태의 단일 소스는 MongoDB(`leverage/config_store.py`). 시장(한국/미국)은
프로필 ``ma_cross_kor`` / ``ma_cross_us`` 로 분리되어 각자 설정·직전 추천 상태를 갖는다.

화면은 종목풀 백테스트처럼 **즉시(라이브)** 튜닝 sweep 결과를 표로 보여주고, 배치는
장 마감 후 그날 판정을 상태로 저장한다. 두 경로가 같은 계산을 쓰도록 공통화한다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from leverage.config_store import load_config, load_leverage_state, save_leverage_state
from leverage.engine.backtest.ma_cross import current_index_judgment, run_buy_hold, tune_ma_cross
from leverage.holding import count_trading_days_market, resolve_holding_start_date
from utils.data_loader import fetch_ohlcv
from utils.logger import get_app_logger
from utils.moving_averages import get_moving_average_type

# 이동평균 최장 후보(240)를 창 시작부터 계산하려면 그만큼의 사전 데이터가 필요하다.
_WARMUP_EXTRA_BDAYS = 30
_TICKER_TYPE = "etf"

logger = get_app_logger()


def _fetch_close(ticker: str, country: str, start_str: str) -> pd.Series:
    """단일 티커 종가 시계열. 실패 시 임의 보정 없이 에러."""
    df = fetch_ohlcv(ticker, country, months_back=None, date_range=[start_str, None], ticker_type=_TICKER_TYPE)
    if df is None or df.empty or "Close" not in df.columns:
        raise ValueError(f"가격 데이터를 받아오지 못했습니다: {ticker} ({country})")
    series = df["Close"].astype(float)
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def _load_series(
    config: dict[str, Any],
    months: int,
    max_ma: int,
) -> tuple[pd.Series, pd.Series, pd.Series | None, pd.Timestamp]:
    """지수/레버리지/방어 종가 + 평가 시작일. 방어가 현금이면 None(수익 0).

    평가 창은 최근 ``months`` 개월. 이동평균 최장값(``max_ma``)을 창 시작부터 계산할 수 있게
    그만큼 앞의 워밍업 데이터까지 받아온다.
    """
    country = config["market"]
    eval_start = pd.Timestamp.today().normalize() - pd.DateOffset(months=months)
    fetch_start = eval_start - pd.offsets.BDay(max_ma + _WARMUP_EXTRA_BDAYS)
    start_str = fetch_start.strftime("%Y-%m-%d")

    index = _fetch_close(config["index_ticker"], country, start_str)
    leverage = _fetch_close(config["leverage_ticker"], country, start_str)
    defense = None if config["defense_ticker"] == "CASH" else _fetch_close(config["defense_ticker"], country, start_str)
    return index, leverage, defense, eval_start


def _candidate_range(ma_min: int, ma_max: int, ma_step: int) -> list[int]:
    """이동선 범위를 min~max(끝값 포함) step 간격으로 나열한다. 임의 보정 없이 검증만."""
    if not all(isinstance(v, int) for v in (ma_min, ma_max, ma_step)):
        raise ValueError("이동선 범위(min/max/step)는 정수여야 합니다.")
    if ma_min < 2:
        raise ValueError(f"이동선 min 은 2 이상이어야 합니다: {ma_min}")
    if ma_step < 1:
        raise ValueError(f"이동선 step 은 1 이상이어야 합니다: {ma_step}")
    if ma_max < ma_min:
        raise ValueError(f"이동선 max({ma_max})는 min({ma_min}) 이상이어야 합니다.")
    return list(range(ma_min, ma_max + 1, ma_step))


def _pct_candidate_range(min_pct: float, max_pct: float, step_pct: float) -> list[float]:
    """고점대비 범위를 min~max(끝값 포함) step 간격으로 나열한다."""
    if not all(isinstance(v, (int, float)) for v in (min_pct, max_pct, step_pct)):
        raise ValueError("고점대비 범위(min/max/step)는 숫자여야 합니다.")
    if min_pct < 0:
        raise ValueError(f"고점대비 min 은 0 이상이어야 합니다: {min_pct}")
    if step_pct <= 0:
        raise ValueError(f"고점대비 step 은 0보다 커야 합니다: {step_pct}")
    if max_pct < min_pct:
        raise ValueError(f"고점대비 max({max_pct})는 min({min_pct}) 이상이어야 합니다.")

    values: list[float] = []
    current = float(min_pct)
    limit = float(max_pct)
    step = float(step_pct)
    while current <= limit + 1e-9:
        values.append(round(current, 4))
        current += step
    return values


def _asset_meta(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {
        "index": {"ticker": config["index_ticker"], "name": config["index_name"]},
        "leverage": {"ticker": config["leverage_ticker"], "name": config["leverage_name"]},
        "defense": {"ticker": config["defense_ticker"], "name": config["defense_name"]},
    }


def _holding_return(state: dict[str, Any], country: str) -> dict[str, Any]:
    """보유 중인 자산의 보유 시작일 대비 수익률.

    현금(CASH)을 들고 있으면 수익률이라는 개념이 없어 값을 채우지 않는다.
    시작일 종가가 없으면(그날 거래정지 등) 그 이후 첫 거래일 종가를 진입가로 본다.
    조회 실패는 화면 전체를 막을 이유가 아니라 값 없음으로 둔다.
    """
    ticker = str(state.get("target") or "").strip()
    start = str(state.get("holding_start_date") or "").strip()
    if not ticker or ticker.upper() == "CASH" or not start:
        return {}

    try:
        series = _fetch_close(ticker, country, start).dropna()
    except Exception as exc:
        logger.warning("[LEVERAGE] 보유 수익률 계산용 시세 조회 실패 (%s): %s", ticker, exc)
        return {}

    series = series[series > 0]
    if len(series) < 1:
        return {}
    entry_price = float(series.iloc[0])
    current_price = float(series.iloc[-1])
    if entry_price <= 0:
        return {}
    return {
        "holding_entry_price": round(entry_price, 2),
        "holding_current_price": round(current_price, 2),
        "holding_return_pct": round((current_price / entry_price - 1.0) * 100.0, 2),
    }


def compute_ma_cross_view(profile: str) -> dict[str, Any]:
    """현재 판정 + 추천 + 직전 상태를 반환한다(읽기 전용, 저장 안 함).

    설정된 ma_days 와 peak_drawdown_pct 기준으로 레버리지/방어를 판정한다. 튜닝 sweep 은
    별도(``compute_ma_cross_tune``)로 분리해 사용자가 범위·기간을 지정해 실행한다.
    """
    config = load_config(profile)
    ma_days = int(config["ma_days"])
    peak_drawdown_pct = float(config["peak_drawdown_pct"])
    country = config["market"]

    # 판정에는 지수 종가만 필요하다(레버리지/방어 자산 이름은 설정에서 온다).
    fetch_start = pd.Timestamp.today().normalize() - pd.offsets.BDay(ma_days + _WARMUP_EXTRA_BDAYS + 60)
    index = _fetch_close(config["index_ticker"], country, fetch_start.strftime("%Y-%m-%d"))
    judgment = current_index_judgment(index, ma_days, peak_drawdown_pct=peak_drawdown_pct)

    # 현재(실제) 보유 상태: 배치가 저장한 상태를 단일 소스로 쓰되, 보유일은 보유 시작일에서
    # 시장 거래일 달력으로 매번 계산한다(지수 티커를 달력 기준으로 사용).
    state = load_leverage_state(profile)
    if state.get("holding_start_date"):
        state = {
            **state,
            "holding_days": count_trading_days_market(country, config["index_ticker"], state["holding_start_date"]),
        }
        state = {**state, **_holding_return(state, country)}

    meta = _asset_meta(config)
    if judgment is None:
        recommendation: dict[str, Any] | None = None
    else:
        side = "leverage" if judgment["want_leverage"] else "defense"
        target = meta[side]
        prev_target = state.get("target")
        recommendation = {
            "side": side,
            "target_ticker": target["ticker"],
            "target_name": target["name"],
            "as_of": judgment["as_of"].date().isoformat(),
            "prev_target": prev_target,
            "is_changed": prev_target is not None and prev_target != target["ticker"],
        }

    return {
        "profile": profile,
        "market": config["market"],
        "ma_days": ma_days,
        "ma_type": get_moving_average_type(),
        "peak_drawdown_pct": peak_drawdown_pct,
        "slippage": float(config["slippage"]),
        "assets": meta,
        "judgment": None
        if judgment is None
        else {
            "as_of": judgment["as_of"].date().isoformat(),
            "index_close": judgment["index_close"],
            "ma": judgment["ma"],
            "gap_pct": judgment["gap_pct"],
            "peak_drawdown_pct": judgment["peak_drawdown_pct"],
            "peak_drawdown_limit_pct": judgment["peak_drawdown_limit_pct"],
            "ma_threshold_close": judgment["ma_threshold_close"],
            "peak_threshold_close": judgment["peak_threshold_close"],
            "required_index_close": judgment["required_index_close"],
            "want_leverage": judgment["want_leverage"],
        },
        "recommendation": recommendation,
        "state": state,
    }


def compute_ma_cross_tune(
    profile: str,
    *,
    months: int,
    ma_min: int,
    ma_max: int,
    ma_step: int,
    peak_min: float,
    peak_max: float,
    peak_step: float,
) -> dict[str, Any]:
    """사용자가 지정한 기간·이동선 범위로 튜닝 sweep 을 즉시 계산해 반환한다.

    - ``months``: 최근 N 개월(평가 창)
    - ``ma_min``/``ma_max``/``ma_step``: 이동선 후보 범위(끝값 포함)
    - ``peak_min``/``peak_max``/``peak_step``: 지수 고점대비 허용 하락폭 후보 범위(%)
    - ``rows``: 후보별 수익/MDD/소르티노(소르티노 내림차순) — 특정 값만 튀는지(과적합) 판단용
    """
    if not isinstance(months, int) or months < 1:
        raise ValueError(f"기간(개월)은 1 이상 정수여야 합니다: {months}")
    candidates = _candidate_range(ma_min, ma_max, ma_step)
    peak_candidates = _pct_candidate_range(peak_min, peak_max, peak_step)

    config = load_config(profile)
    index, leverage, defense, eval_start = _load_series(config, months, max(candidates))
    slip = float(config["slippage"]) / 100.0
    index_hold = run_buy_hold(index, eval_start=eval_start)
    leverage_hold = run_buy_hold(leverage, eval_start=eval_start)
    if index_hold is None:
        raise ValueError(f"지수 단순 보유 성과 계산 데이터가 부족합니다: {config['index_ticker']}")
    if leverage_hold is None:
        raise ValueError(f"레버리지 종목 단순 보유 성과 계산 데이터가 부족합니다: {config['leverage_ticker']}")

    rows = tune_ma_cross(
        index,
        leverage,
        defense,
        buy_pct=slip,
        sell_pct=slip,
        candidates=candidates,
        peak_drawdown_candidates=peak_candidates,
        eval_start=eval_start,
    )
    return {
        "profile": profile,
        "market": config["market"],
        "months": months,
        "ma_range": {"min": ma_min, "max": ma_max, "step": ma_step},
        "peak_drawdown_range": {"min": peak_min, "max": peak_max, "step": peak_step},
        "candidates": candidates,
        "peak_drawdown_candidates": peak_candidates,
        "benchmarks": [
            {
                "label": "지수 보유",
                "ticker": config["index_ticker"],
                "name": config["index_name"],
                **index_hold,
            },
            {
                "label": "레버리지 보유",
                "ticker": config["leverage_ticker"],
                "name": config["leverage_name"],
                **leverage_hold,
            },
        ],
        "rows": rows,
    }


def persist_ma_cross_state(profile: str) -> dict[str, Any]:
    """현재 판정을 계산하고, 장 마감(확정) 이후일 때만 직전 추천 상태로 저장한다.

    장중·장전은 종가 미확정이라 저장하지 않는다(직전 상태 유지). 배치 추천이 호출한다.
    """
    from leverage.notify import send_slack_ma_cross
    from leverage.recommend import MARKET_PHASE_LABEL, get_market_status

    view = compute_ma_cross_view(profile)
    recommendation = view.get("recommendation")
    status = get_market_status(view["market"])
    is_confirmed = status in ("CLOSED_JUST_NOW", "CLOSED")

    saved = False
    slack_sent = False
    if is_confirmed and recommendation is not None:
        prev = load_leverage_state(profile)
        as_of = recommendation["as_of"]
        # 포지션이 실제로 바뀐 날에만 보유 시작일을 새로 기록, 유지되면 기존값 보존(리셋 없음).
        holding_start_date = resolve_holding_start_date(
            prev.get("target"),
            prev.get("holding_start_date"),
            recommendation["target_ticker"],
            as_of,
        )

        # 슬랙: 설정에서 켜져 있고, '장 마감 직후'이며, 오늘 아직 안 보냈을 때만 1회 발송.
        config = load_config(profile)
        already_sent_today = prev.get("slack_sent_date") == as_of
        if config.get("slack_enabled") and status == "CLOSED_JUST_NOW" and not already_sent_today:
            slack_sent = send_slack_ma_cross(view, market_phase=MARKET_PHASE_LABEL.get(status, "장 마감 후"))
        slack_sent_date = as_of if slack_sent else prev.get("slack_sent_date")

        new_state = {
            "date": as_of,
            "target": recommendation["target_ticker"],
            "target_name": recommendation["target_name"],
            "side": recommendation["side"],
            "holding_start_date": holding_start_date,
            "ma_days": view["ma_days"],
            "peak_drawdown_pct": view["peak_drawdown_pct"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if slack_sent_date:
            new_state["slack_sent_date"] = slack_sent_date
        save_leverage_state(profile, new_state)
        saved = True

    return {**view, "market_status": status, "state_saved": saved, "slack_sent": slack_sent}
