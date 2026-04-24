"""모멘텀 ETF 파라미터 스윕 백테스트 엔진.

``backtest/run.py`` 에서 호출되며, 종목풀별 ``BACKTEST_CONFIG`` 와
전역 공통값 ``BACKTEST_START_DATE``, ``BACKTEST_INITIAL_KRW_AMOUNT`` 를 사용한다.
멀티프로세스 병렬 실행을 지원하며, 실행 중에도 중간 결과를 파일에 주기적으로 기록한다.
"""

from __future__ import annotations

import itertools
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    BACKTEST_INITIAL_KRW_AMOUNT,
    BACKTEST_START_DATE,
    MARKET_SCHEDULES,
    SLIPPAGE_CONFIG,
    TRADING_DAYS_PER_MONTH,
)
from core.strategy.scoring import (
    compute_eligibility_mask,
    compute_rule_percentile_frame,
)
from utils.cache_utils import load_cached_frames_bulk_with_fallback
from utils.data_loader import get_exchange_rate_series, get_trading_days
from utils.formatters import format_pct_change, format_price, format_trading_days
from utils.report import render_table_eaw
from utils.settings_loader import get_ticker_type_settings
from utils.stock_list_io import get_etfs
from services.price_service import get_realtime_snapshot

logger = logging.getLogger(__name__)
RSI_PERIOD = 14

# ----------------------------- 헬퍼 ----------------------------- #

def _select_close_column(columns: list[str]) -> str:
    for candidate in ("unadjusted_close", "Close", "close"):
        if candidate in columns:
            return candidate
    raise ValueError(f"OHLCV 프레임에서 종가 컬럼을 찾을 수 없습니다: {columns}")


def _select_open_column(columns: list[str]) -> str:
    for candidate in ("Open", "open"):
        if candidate in columns:
            return candidate
    raise ValueError(f"OHLCV 프레임에서 시가 컬럼을 찾을 수 없습니다: {columns}")


def _resolve_fx_symbol(country_code: str) -> str | None:
    """country_code → Yahoo Finance FX 심볼. ``kor`` 은 FX 불필요."""
    code = str(country_code or "").strip().lower()
    if code in ("kor", "kr"):
        return None
    if code == "us":
        return "KRW=X"
    if code in ("au", "aus"):
        return "AUDKRW=X"
    raise ValueError(f"알 수 없는 country_code 입니다: {country_code!r}")


def _load_fx_series(
    country_code: str,
    calendar_days: list[pd.Timestamp],
) -> pd.Series:
    """백테스트 캘린더에 맞춰 현지화/KRW 환율 시리즈를 생성한다.

    - 국내 풀(kor) 은 전부 1.0 (환전 없음).
    - 해외 풀은 yfinance 환율을 ffill + bfill 로 거래일에 정렬.
    """
    index = pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in calendar_days])
    symbol = _resolve_fx_symbol(country_code)
    if symbol is None:
        return pd.Series(1.0, index=index, dtype=float)

    start = index.min().strftime("%Y-%m-%d")
    end = index.max().strftime("%Y-%m-%d")
    raw = get_exchange_rate_series(start, end, symbol=symbol, allow_partial=True)
    if raw is None or raw.empty:
        raise RuntimeError(f"환율 데이터를 가져오지 못했습니다: symbol={symbol} {start}~{end}")

    rates = raw.copy()
    rates.index = pd.to_datetime(rates.index).normalize()
    rates = rates[~rates.index.duplicated(keep="last")].sort_index()
    rates = rates.reindex(index).ffill().bfill()
    if rates.isna().any():
        raise RuntimeError(f"환율 정렬 후 NaN 이 남았습니다: symbol={symbol}")
    return rates.astype(float)


def _pct_to_ratio(pct: float) -> float:
    """% 단위를 비율로 변환한다. 예: 0.5 → 0.005."""
    return float(pct) / 100.0


def _resolve_slippage(pool_id: str) -> tuple[float, float]:
    """종목풀별 매수/매도 슬리피지 비율을 반환한다."""
    if pool_id not in SLIPPAGE_CONFIG:
        raise ValueError(f"SLIPPAGE_CONFIG 에 '{pool_id}' 설정이 없습니다.")
    config = SLIPPAGE_CONFIG[pool_id]
    if "BUY_PCT" not in config or "SELL_PCT" not in config:
        raise ValueError(f"SLIPPAGE_CONFIG['{pool_id}']에 BUY_PCT/SELL_PCT가 모두 필요합니다.")
    buy_ratio = _pct_to_ratio(float(config["BUY_PCT"]))
    sell_ratio = _pct_to_ratio(float(config["SELL_PCT"]))
    if buy_ratio < 0 or sell_ratio < 0:
        raise ValueError(f"SLIPPAGE_CONFIG['{pool_id}']는 음수일 수 없습니다.")
    return buy_ratio, sell_ratio


def _should_include_latest_day(open_frame: pd.DataFrame, latest_day: pd.Timestamp, today: pd.Timestamp) -> bool:
    """오늘 거래일은 Open 데이터가 실제로 있는 경우에만 백테스트 말일에 포함한다."""
    normalized_latest = pd.Timestamp(latest_day).normalize()
    normalized_today = pd.Timestamp(today).normalize()
    if normalized_latest != normalized_today:
        return True
    if normalized_latest not in open_frame.index:
        return False
    latest_open_row = open_frame.loc[normalized_latest]
    if isinstance(latest_open_row, pd.DataFrame):
        latest_open_row = latest_open_row.iloc[0]
    valid_open = pd.to_numeric(latest_open_row, errors="coerce")
    return bool(((~valid_open.isna()) & (valid_open > 0)).any())


def _is_market_opened(country_code: str, today: pd.Timestamp) -> bool:
    schedule = MARKET_SCHEDULES.get(str(country_code or "").strip().lower())
    if not schedule:
        raise ValueError(f"MARKET_SCHEDULES 에 '{country_code}' 설정이 없습니다.")

    now_local = pd.Timestamp.now(tz=schedule["timezone"])
    today_local = pd.Timestamp(today).tz_localize(schedule["timezone"])
    return bool(
        (now_local.date() > today_local.date())
        or (now_local.date() == today_local.date() and now_local.time() >= schedule["open"])
    )


def _augment_frames_with_intraday_open(
    frames: dict[str, pd.DataFrame],
    tickers: list[str],
    country_code: str,
    latest_day: pd.Timestamp,
    today: pd.Timestamp,
) -> None:
    normalized_latest = pd.Timestamp(latest_day).normalize()
    normalized_today = pd.Timestamp(today).normalize()
    if normalized_latest != normalized_today:
        return

    today_str = normalized_today.strftime("%Y-%m-%d")
    trading_days = get_trading_days(today_str, today_str, country_code)
    if not trading_days:
        return
    if not _is_market_opened(country_code, normalized_today):
        return

    snapshots = get_realtime_snapshot(country_code, tickers)
    if not snapshots:
        return

    for ticker in tickers:
        frame = frames.get(ticker)
        if frame is None or frame.empty:
            continue

        snapshot = snapshots.get(ticker)
        if not snapshot:
            continue

        open_price = float(snapshot.get("nowVal") or 0.0)
        if open_price <= 0:
            continue

        row_payload = {column: np.nan for column in frame.columns}
        for candidate in ("Open", "open"):
            if candidate in row_payload:
                row_payload[candidate] = open_price
        for candidate in ("High", "high"):
            if candidate in row_payload:
                row_payload[candidate] = open_price
        for candidate in ("Low", "low"):
            if candidate in row_payload:
                row_payload[candidate] = open_price
        for candidate in ("Close", "close", "unadjusted_close"):
            if candidate in row_payload:
                row_payload[candidate] = open_price
        for candidate in ("Volume", "volume"):
            if candidate in row_payload:
                row_payload[candidate] = 0

        today_row = pd.DataFrame([row_payload], index=[normalized_today])
        merged = pd.concat([frame, today_row])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        frames[ticker] = merged


def _combine_rule_percentiles_array(
    per_rule_arrays: list[np.ndarray],
    eligibility_mask: np.ndarray,
) -> np.ndarray:
    """규칙별 percentile 배열을 합산하고 자격 마스크를 적용한다.

    공통 엔진의 ``combine_rule_percentiles()`` 와 같은 의미를 유지하되,
    백테스트 워커 내부에서는 DataFrame 대신 ndarray 로 계산한다.
    """
    if not per_rule_arrays:
        return np.full(eligibility_mask.shape, np.nan, dtype=np.float64)
    composite = per_rule_arrays[0].copy()
    for rule_values in per_rule_arrays[1:]:
        composite += rule_values
    composite[~eligibility_mask] = np.nan
    return composite


def _compute_rsi_frame(close_frame: pd.DataFrame, period: int) -> pd.DataFrame:
    """종가 프레임으로 RSI 프레임을 계산한다."""
    delta = close_frame.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rsi = pd.DataFrame(np.nan, index=close_frame.index, columns=close_frame.columns, dtype=float)
    zero_loss_mask = avg_loss == 0
    rsi[zero_loss_mask & ~avg_gain.isna()] = 100.0

    valid_mask = (~avg_gain.isna()) & (~avg_loss.isna()) & (~zero_loss_mask)
    rs = avg_gain[valid_mask] / avg_loss[valid_mask]
    rsi[valid_mask] = 100.0 - (100.0 / (1.0 + rs))
    return rsi


# -------------------- 워커 프로세스 (병렬 실행) -------------------- #


# 워커 프로세스에서 공유하는 전역 변수.
# _init_worker() 로 한 번만 초기화되며, 각 워커 프로세스 내에서만 유효하다.
_W_PCT_VALUES: dict[tuple[str, int], np.ndarray] = {}
_W_ELIG_VALUES: np.ndarray = np.empty((0, 0), dtype=bool)
_W_DAYS: list[pd.Timestamp] = []
_W_CASH_LOCAL: float = 0.0
_W_BUY_SLIPPAGE: float = 0.0
_W_SELL_SLIPPAGE: float = 0.0
_W_OPEN_VALUES: np.ndarray = np.empty((0, 0), dtype=np.float64)
_W_CLOSE_VALUES: np.ndarray = np.empty((0, 0), dtype=np.float64)
_W_FX_VALUES: np.ndarray = np.empty(0, dtype=np.float64)
_W_RSI_VALUES: np.ndarray = np.empty((0, 0), dtype=np.float64)


def _init_worker(
    pct_specs_values: dict[tuple[str, int], np.ndarray],
    eligibility_values: np.ndarray,
    open_values: np.ndarray,
    close_values: np.ndarray,
    bt_days: list[pd.Timestamp],
    init_cash_local: float,
    fx_values: np.ndarray,
    rsi_values: np.ndarray,
    buy_slippage: float,
    sell_slippage: float,
) -> None:
    """워커 프로세스 초기화: 공유 데이터를 전역 변수에 설정."""
    global _W_PCT_VALUES, _W_ELIG_VALUES, _W_DAYS, _W_CASH_LOCAL, _W_BUY_SLIPPAGE, _W_SELL_SLIPPAGE  # noqa: PLW0603
    global _W_OPEN_VALUES, _W_CLOSE_VALUES, _W_FX_VALUES, _W_RSI_VALUES  # noqa: PLW0603
    _W_PCT_VALUES = pct_specs_values
    _W_ELIG_VALUES = eligibility_values
    _W_DAYS = bt_days
    _W_CASH_LOCAL = init_cash_local
    _W_BUY_SLIPPAGE = buy_slippage
    _W_SELL_SLIPPAGE = sell_slippage
    _W_OPEN_VALUES = open_values
    _W_CLOSE_VALUES = close_values
    _W_FX_VALUES = fx_values
    _W_RSI_VALUES = rsi_values


def _run_single_combo(args: tuple[int, float, str, int, float | None]) -> dict[str, Any]:
    """워커에서 단일 파라미터 조합을 실행하고 결과 딕셔너리를 반환한다."""
    top_n, bonus, ma_t, ma_m, rsi_limit = args
    composite_values = _combine_rule_percentiles_array([_W_PCT_VALUES[(ma_t, ma_m)]], _W_ELIG_VALUES)
    total_ret, cagr, mdd, trades = _simulate_one_combo(
        initial_cash_local=_W_CASH_LOCAL,
        top_n=top_n,
        bonus=bonus,
        composite_values=composite_values,
        open_values=_W_OPEN_VALUES,
        close_values=_W_CLOSE_VALUES,
        backtest_days=_W_DAYS,
        fx_values=_W_FX_VALUES,
        rsi_values=_W_RSI_VALUES,
        rsi_limit=rsi_limit,
        buy_slippage=_W_BUY_SLIPPAGE,
        sell_slippage=_W_SELL_SLIPPAGE,
    )
    return {
        "TOP_N_HOLD": top_n,
        "HOLDING_BONUS_SCORE": bonus,
        "MA_TYPE": ma_t,
        "MA_MONTHS": ma_m,
        "RSI_LIMIT": rsi_limit,
        "TOTAL_RETURN_PCT": total_ret,
        "CAGR_PCT": cagr,
        "MDD_PCT": mdd,
        "TRADES": trades,
    }


# --------------------------- 시뮬레이션 --------------------------- #


def _simulate_one_combo(
    *,
    initial_cash_local: float,
    top_n: int,
    bonus: float,
    composite_values: np.ndarray,
    open_values: np.ndarray,
    close_values: np.ndarray,
    backtest_days: list[pd.Timestamp],
    fx_values: np.ndarray,
    rsi_values: np.ndarray,
    rsi_limit: float | None,
    buy_slippage: float,
    sell_slippage: float,
) -> tuple[float, float, float, int]:
    """단일 파라미터 조합에 대해 1회 백테스트.

    모든 체결/보유/현금은 현지 통화로 관리한다. 평가 일자마다 당일 환율을 곱해
    KRW 기준 value_curve 를 만들고 총수익률/CAGR/MDD 를 계산한다.

    Returns:
        (total_return_pct, cagr_pct, mdd_pct) — 모두 KRW 기준.
    """
    shares = np.zeros(composite_values.shape[1], dtype=np.int64)
    cash = float(initial_cash_local)

    value_curve: list[float] = []
    trade_count = 0
    if len(backtest_days) < 2:
        return 0.0, 0.0, 0.0, 0

    # 첫 거래일은 전일 종가 신호를 사용해 당일 시초가에 첫 진입한다.
    for exec_idx in range(1, len(backtest_days)):
        signal_idx = exec_idx - 1
        close_today = close_values[signal_idx]
        priced_mask = ~np.isnan(close_today)
        portfolio_value_local = cash + float(np.dot(shares[priced_mask], close_today[priced_mask]))

        composite_today = composite_values[signal_idx].copy()
        if bonus:
            held_bonus_mask = (shares > 0) & ~np.isnan(composite_today)
            composite_today[held_bonus_mask] += bonus

        rsi_sell_mask = np.zeros_like(shares, dtype=bool)
        if rsi_limit is not None:
            rsi_today = rsi_values[signal_idx]
            rsi_sell_mask = (~np.isnan(rsi_today)) & (rsi_today > rsi_limit)
            composite_today[rsi_sell_mask] = np.nan

        open_exec = open_values[exec_idx]
        valid_mask = ~np.isnan(composite_today) & ~np.isnan(open_exec) & (open_exec > 0)
        valid_idx = np.flatnonzero(valid_mask)
        if valid_idx.size == 0:
            close_exec = close_values[exec_idx]
            priced_exec_mask = ~np.isnan(close_exec)
            portfolio_value_local = cash + float(np.dot(shares[priced_exec_mask], close_exec[priced_exec_mask]))
            fx_today = float(fx_values[exec_idx])
            value_curve.append(portfolio_value_local * fx_today)
            continue

        # 동점 처리: 점수 desc → ticker asc(컬럼 순서가 이미 ticker asc).
        # 전체 정렬 대신 상위 top_n 후보만 partial sort로 추린 뒤 최종 정렬한다.
        valid_scores = composite_today[valid_idx]
        target_count = min(top_n, valid_idx.size)
        if target_count <= 0:
            close_exec = close_values[exec_idx]
            priced_exec_mask = ~np.isnan(close_exec)
            portfolio_value_local = cash + float(np.dot(shares[priced_exec_mask], close_exec[priced_exec_mask]))
            fx_today = float(fx_values[exec_idx])
            value_curve.append(portfolio_value_local * fx_today)
            continue
        if target_count < valid_idx.size:
            top_slice = np.argpartition(-valid_scores, target_count - 1)[:target_count]
            candidate_idx = valid_idx[top_slice]
            candidate_scores = valid_scores[top_slice]
        else:
            candidate_idx = valid_idx
            candidate_scores = valid_scores
        candidate_order = np.lexsort((candidate_idx, -candidate_scores))
        target_idx = candidate_idx[candidate_order]
        target_mask = np.zeros_like(valid_mask, dtype=bool)
        target_mask[target_idx] = True
        held_mask = shares > 0
        to_sell_idx = np.flatnonzero(held_mask & (~target_mask | rsi_sell_mask))
        to_buy_idx = target_idx[shares[target_idx] == 0]

        # 신호 시점 기준 총 자산 (= 목표 비중 계산용). 매수/매도는 현지 통화로 수행.
        total_equity_signal = portfolio_value_local

        # 1) 매도 먼저
        for ticker_idx in to_sell_idx:
            raw_open_price = float(open_exec[ticker_idx])
            if np.isnan(raw_open_price) or raw_open_price <= 0:
                # 매도 불가 → 보유 유지
                continue
            price = raw_open_price * (1.0 - sell_slippage)
            n = int(shares[ticker_idx])
            shares[ticker_idx] = 0
            cash += n * price
            trade_count += 1

        # 2) 매수 (방식 S3): 신규 K개에 ``min(현금/K, 총자산/N)`` 씩 균등 단주 매수.
        #    - 기존 보유 종목 절대 트리밍 없음 (상승 추세 보존).
        #    - 1/N 슬롯 상한으로 한 종목 몰빵 차단.
        #    - 잔액은 그대로 현금 보유 (모두 하락 시 자동 현금화).
        if to_buy_idx.size == 0 or cash <= 0:
            close_exec = close_values[exec_idx]
            priced_exec_mask = ~np.isnan(close_exec)
            portfolio_value_local = cash + float(np.dot(shares[priced_exec_mask], close_exec[priced_exec_mask]))
            fx_today = float(fx_values[exec_idx])
            value_curve.append(portfolio_value_local * fx_today)
            continue

        slot_target = total_equity_signal / float(top_n)
        per_new_budget = min(slot_target, cash / float(to_buy_idx.size))

        for ticker_idx in to_buy_idx:
            raw_open_price = float(open_exec[ticker_idx])
            if np.isnan(raw_open_price) or raw_open_price <= 0:
                continue
            price = raw_open_price * (1.0 + buy_slippage)
            buy_budget = min(cash, per_new_budget)
            if buy_budget <= 0:
                continue
            n_shares = int(buy_budget // price)
            if n_shares <= 0:
                continue
            cost = n_shares * price
            cash -= cost
            shares[ticker_idx] += n_shares
            trade_count += 1

        close_exec = close_values[exec_idx]
        priced_exec_mask = ~np.isnan(close_exec)
        portfolio_value_local = cash + float(np.dot(shares[priced_exec_mask], close_exec[priced_exec_mask]))
        fx_today = float(fx_values[exec_idx])
        value_curve.append(portfolio_value_local * fx_today)

    if not value_curve:
        return 0.0, 0.0, 0.0, 0

    values = np.asarray(value_curve, dtype=np.float64)
    start_val = initial_cash_local * float(fx_values[0])
    end_val = float(values[-1])
    total_return_pct = (end_val / start_val - 1.0) * 100.0 if start_val > 0 else 0.0

    n_days = max(1, len(backtest_days) - 1)
    years = n_days / 252.0
    if start_val > 0 and end_val > 0 and years > 0:
        cagr_pct = (pow(end_val / start_val, 1.0 / years) - 1.0) * 100.0
    else:
        cagr_pct = 0.0

    drawdown_base = np.concatenate(([start_val], values))
    running_max = np.maximum.accumulate(drawdown_base)
    drawdown = (drawdown_base / running_max - 1.0) * 100.0
    mdd_pct = float(np.min(drawdown)) if drawdown.size else 0.0

    return float(total_return_pct), float(cagr_pct), float(mdd_pct), int(trade_count)


def _simulate_benchmark_buy_and_hold(
    *,
    ticker: str,
    open_frame: pd.DataFrame,
    close_frame: pd.DataFrame,
    backtest_days: list[pd.Timestamp],
    fx_series: pd.Series,
    initial_cash_local: float,
    buy_slippage: float,
) -> tuple[float, float, float, int]:
    """벤치마크 1종목을 첫 체결일 시초가에 1회 매수 후 끝까지 보유한다.

    백테스트는 종목풀의 거래일 캘린더를 기준으로 평가하므로, 벤치마크가 해당 일자에
    거래되지 않았더라도 가장 최근 유효 종가로 평가한다. 단, 첫 체결일 종가 자체가
    없으면 시작 기준을 잡을 수 없으므로 즉시 실패시킨다.
    """
    if len(backtest_days) < 2:
        raise RuntimeError("벤치마크 계산에 필요한 거래일이 부족합니다.")
    if ticker not in open_frame.columns or ticker not in close_frame.columns:
        raise RuntimeError(f"벤치마크 티커 '{ticker}' 가격 데이터가 없습니다.")

    first_exec_day = backtest_days[1]
    raw_open_price = float(open_frame.at[first_exec_day, ticker])
    if pd.isna(raw_open_price) or raw_open_price <= 0:
        raise RuntimeError(
            f"벤치마크 '{ticker}'의 첫 체결일({first_exec_day.strftime('%Y-%m-%d')}) 시가가 비정상입니다."
        )

    entry_price = raw_open_price * (1.0 + buy_slippage)
    shares = int(initial_cash_local // entry_price)
    if shares <= 0:
        raise RuntimeError(
            f"벤치마크 '{ticker}'를 초기 자금으로 1주도 매수할 수 없습니다."
        )

    cash = float(initial_cash_local) - (shares * entry_price)
    value_curve: list[float] = []
    exec_days = backtest_days[1:]
    close_series = pd.to_numeric(close_frame[ticker], errors="coerce").reindex(exec_days)
    first_close = float(close_series.iloc[0]) if not pd.isna(close_series.iloc[0]) else np.nan
    if pd.isna(first_close) or first_close <= 0:
        raise RuntimeError(
            f"벤치마크 '{ticker}'의 첫 체결일 종가가 비정상입니다: {first_exec_day.strftime('%Y-%m-%d')}"
        )

    close_series = close_series.ffill()
    invalid_days = close_series[close_series.isna() | (close_series <= 0)]
    if not invalid_days.empty:
        first_invalid = pd.Timestamp(invalid_days.index[0]).strftime("%Y-%m-%d")
        raise RuntimeError(
            f"벤치마크 '{ticker}'의 종가가 비정상입니다: {first_invalid}"
        )

    for exec_day in exec_days:
        close_price = float(close_series.loc[exec_day])
        fx_today = float(fx_series.loc[exec_day])
        total_value_local = cash + (shares * close_price)
        value_curve.append(total_value_local * fx_today)

    values = pd.Series(value_curve, index=exec_days)
    start_val = initial_cash_local * float(fx_series.iloc[0])
    end_val = values.iloc[-1]
    total_return_pct = (end_val / start_val - 1.0) * 100.0 if start_val > 0 else 0.0

    n_days = max(1, len(backtest_days) - 1)
    years = n_days / 252.0
    if start_val > 0 and end_val > 0 and years > 0:
        cagr_pct = (pow(end_val / start_val, 1.0 / years) - 1.0) * 100.0
    else:
        cagr_pct = 0.0

    baseline = pd.Series([start_val], index=[backtest_days[0]])
    drawdown_base = pd.concat([baseline, values])
    running_max = drawdown_base.cummax()
    drawdown = (drawdown_base / running_max - 1.0) * 100.0
    mdd_pct = float(drawdown.min()) if not drawdown.empty else 0.0
    return float(total_return_pct), float(cagr_pct), float(mdd_pct), 1


# --------------------------- 결과 기록 --------------------------- #


def _write_results_file(
    *,
    out_path: Path,
    results: list[dict[str, Any]],
    benchmark_result: dict[str, Any] | None,
    pool_id: str,
    months: int,
    initial_cash: float,
    top_n_values: list[int],
    bonus_values: list[float],
    ma_types: list[str],
    ma_months_list: list[int],
    rsi_limits: list[float] | None,
    total_combos: int,
    done_count: int,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    started_str: str,
    elapsed_sec: int,
    n_workers: int,
    is_final: bool,
) -> None:
    """결과를 로그 파일에 기록한다. 중간/최종 모두 동일 형식."""
    sorted_results = sorted(results, key=lambda r: (-r["CAGR_PCT"], r["MDD_PCT"]))

    def _render_full_width_row(border_line: str, text: str) -> str:
        """테이블 전체 너비를 차지하는 단일 텍스트 row를 렌더링한다."""
        inner_width = len(border_line) - 4
        return f"| {text.ljust(inner_width)} |"

    lines: list[str] = []
    h = elapsed_sec // 3600
    m = (elapsed_sec % 3600) // 60
    s = elapsed_sec % 60

    if is_final:
        lines.append(f"실행 시각: {started_str}")
        lines.append(f"종료 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        lines.append(f"실행 시각: {started_str}")
        lines.append(f"중간 갱신: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"종목풀: {pool_id}")
    lines.append(f"경과 시간: {h}시간 {m}분 {s}초")
    lines.append(f"병렬 워커: {n_workers}")
    lines.append(f"진행: {done_count}/{total_combos} ({done_count * 100 // total_combos}%)")
    lines.append("")
    lines.append("=== 백테스트 설정 ===")
    lines.append(
        f"기간: {period_start.strftime('%Y-%m-%d')} ~ {period_end.strftime('%Y-%m-%d')} ({months} 개월)"
    )
    lines.append(
        f"탐색 공간: TOP_N_HOLD {len(top_n_values)}개 x HOLDING_BONUS_SCORE {len(bonus_values)}개 "
        f"x MA_TYPE {len(ma_types)}개 x MA_MONTHS {len(ma_months_list)}개 "
        f"{f'x RSI_LIMIT {len(rsi_limits)}개 ' if rsi_limits is not None else ''}"
        f"= {total_combos}개 조합"
    )
    lines.append(f'"BACKTEST_INITIAL_KRW_AMOUNT": {int(initial_cash)},')
    lines.append(f'"TOP_N_HOLD": {top_n_values},')
    lines.append(f'"HOLDING_BONUS_SCORE": {[int(b) if float(b).is_integer() else b for b in bonus_values]},')
    lines.append(f'"MA_TYPE": {ma_types},')
    lines.append(f'"MA_MONTHS": {ma_months_list},')
    if rsi_limits is not None:
        lines.append(f'"RSI_LIMIT": {[int(v) if float(v).is_integer() else v for v in rsi_limits]},')
    lines.append("")

    status_label = "최종 결과" if is_final else f"중간 결과 ({done_count}/{total_combos})"
    lines.append(f"=== {status_label} - 기간: {months} 개월 | 정렬 기준: CAGR ===")

    # 테이블
    top_limit = 20
    top_rows = sorted_results[:top_limit]
    if not top_rows:
        lines.append("(결과 없음)")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    headers = [
        "TOP_N",
        "BONUS",
        "MA_TYPE",
        "MA_M",
        "RSI",
        "수익률(%)",
        "CAGR(%)",
        "MDD(%)",
        "Trades",
    ]
    aligns = ["left", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right"]
    benchmark_metric_row: list[str] | None = None
    if benchmark_result is not None:
        benchmark_metric_row = [
            "-",
            "-",
            "-",
            "-",
            "-",
            f"{benchmark_result['TOTAL_RETURN_PCT']:.2f}",
            f"{benchmark_result['CAGR_PCT']:.2f}",
            f"{benchmark_result['MDD_PCT']:.2f}",
            str(benchmark_result["TRADES"]),
        ]

    formatted_rows: list[list[str]] = []
    if benchmark_metric_row is not None:
        formatted_rows.append(benchmark_metric_row)
    for r in top_rows:
        formatted_rows.append(
            [
                str(r["TOP_N_HOLD"]),
                f"{r['HOLDING_BONUS_SCORE']:g}",
                r["MA_TYPE"],
                str(r["MA_MONTHS"]),
                "-" if r["RSI_LIMIT"] is None else f"{float(r['RSI_LIMIT']):g}",
                f"{r['TOTAL_RETURN_PCT']:.2f}",
                f"{r['CAGR_PCT']:.2f}",
                f"{r['MDD_PCT']:.2f}",
                str(r["TRADES"]),
            ]
        )
    table_lines = render_table_eaw(headers, formatted_rows, aligns)
    lines.extend(table_lines[:3])
    next_row_index = 3
    if benchmark_result is not None:
        benchmark_config_line = (
            f'"BENCHMARK": {{"ticker": "{benchmark_result["ticker"]}", '
            f'"name": "{benchmark_result["name"]}"}}'
        )
        lines.append(_render_full_width_row(table_lines[0], benchmark_config_line))
        lines.append(table_lines[next_row_index])
        lines.append(table_lines[0])
        next_row_index += 1
    lines.extend(table_lines[next_row_index:])
    if len(sorted_results) > top_limit:
        lines.append(f"... (완료 {done_count}개 중 상위 {top_limit}개 표시)")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_daily_pct(current_price: float | None, previous_close: float | None) -> float | None:
    """전일 종가 대비 일간 수익률(%)을 계산한다."""
    if current_price is None or previous_close is None:
        return None
    if previous_close <= 0:
        return None
    return ((current_price / previous_close) - 1.0) * 100.0


def _format_quantity(value: int | None) -> str:
    """수량을 표 출력용 문자열로 변환한다."""
    if value is None:
        return "-"
    return f"{int(value):,}"


def _format_weight_pct(value: float | None) -> str:
    """비중(%)을 표 출력용 문자열로 변환한다."""
    if value is None:
        return "-"
    return f"{value:.2f}%"


def _format_score_value(value: float | None) -> str:
    """점수/추세 값을 표 출력용 문자열로 변환한다."""
    if value is None:
        return "-"
    return f"{value:.1f}"


def _format_price_or_dash(value: float | None, country_code: str) -> str:
    """가격을 국가 통화 형식으로 변환하되 비어 있으면 '-'를 반환한다."""
    if value is None:
        return "-"
    return format_price(value, country_code)


def _format_weekday_kr(day: pd.Timestamp) -> str:
    """거래일의 한국어 요일을 반환한다."""
    weekdays = ["월", "화", "수", "목", "금", "토", "일"]
    return weekdays[int(day.weekday())]


def _simulate_one_combo_details(
    *,
    initial_cash_local: float,
    top_n: int,
    bonus: float,
    composite_frame: pd.DataFrame,
    rule_frame: pd.DataFrame,
    open_frame: pd.DataFrame,
    close_frame: pd.DataFrame,
    rsi_frame: pd.DataFrame,
    rsi_limit: float | None,
    backtest_days: list[pd.Timestamp],
    fx_series: pd.Series,
    ticker_name_map: dict[str, str],
    country_code: str,
    buy_slippage: float,
    sell_slippage: float,
) -> list[str]:
    """상위 1개 조합의 거래일별 보유 상세 로그를 생성한다.

    - 첫 체결일은 전 거래일 종가 신호를 사용해 시초가에 진입한다.
    - 각 실행일(exec_day)에는 현금 row를 먼저 기록한다.
    - 그 아래에 보유 종목, 당일 신규매수 종목, 당일 전량매도 종목을 함께 기록한다.
    - 보유일은 백테스트 거래일 기준으로 마지막 매수 후 경과일로 계산한다.
    """
    shares: dict[str, int] = {}
    avg_costs: dict[str, float] = {}
    last_buy_indices: dict[str, int] = {}
    lines: list[str] = []

    headers = [
        "#",
        "티커",
        "종목명",
        "상태",
        "보유일",
        "현재가",
        "일간(%)",
        "수량",
        "금액",
        "평가손익",
        "평가(%)",
        "비중",
        "점수",
        "추세",
        "RSI(D-1)",
        "문구",
    ]
    aligns = [
        "right",
        "left",
        "left",
        "left",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "left",
    ]

    cash = float(initial_cash_local)
    signal_start = backtest_days[0]
    period_start = backtest_days[1]
    period_end = backtest_days[-1]
    fx_start = float(fx_series.loc[signal_start])
    initial_cash_krw = initial_cash_local * fx_start

    lines.append("=== 백테스트 상세 ===")
    lines.append(f"기간: {period_start.strftime('%Y-%m-%d')} ~ {period_end.strftime('%Y-%m-%d')}")
    lines.append(f"TOP_N_HOLD: {top_n}")
    lines.append(f"HOLDING_BONUS_SCORE: {bonus:g}")
    lines.append(f"MA_TYPE: {rule_frame.attrs.get('ma_type', '-')}")
    lines.append(f"MA_MONTHS: {rule_frame.attrs.get('ma_months', '-')}")
    if rsi_limit is not None:
        lines.append(f"RSI_LIMIT: {rsi_limit:g}")
    lines.append("")
    prev_total_equity_krw = initial_cash_krw

    def _append_day_section(
        *,
        day: pd.Timestamp,
        held_rows: list[dict[str, str]],
        wait_rows: list[dict[str, str]],
        sold_rows: list[dict[str, str]],
        cash_local: float,
        total_equity_local: float,
        valuation_pnl_local: float,
        valuation_cost_local: float,
        note: str,
    ) -> None:
        fx_today = float(fx_series.loc[day])
        total_equity_krw = total_equity_local * fx_today
        cash_krw = cash_local * fx_today
        cash_weight = (cash_local / total_equity_local * 100.0) if total_equity_local > 0 else 0.0
        valuation_pnl_krw = valuation_pnl_local * fx_today
        valuation_pct = (
            (valuation_pnl_local / valuation_cost_local) * 100.0 if valuation_cost_local > 0 else 0.0
        )
        cumulative_pnl_krw = total_equity_krw - initial_cash_krw
        cumulative_pct = (
            (cumulative_pnl_krw / initial_cash_krw) * 100.0 if initial_cash_krw > 0 else 0.0
        )
        nonlocal prev_total_equity_krw
        daily_pnl_krw = total_equity_krw - prev_total_equity_krw
        daily_pct = ((daily_pnl_krw / prev_total_equity_krw) * 100.0) if prev_total_equity_krw > 0 else 0.0
        day_rows: list[list[str]] = []
        cash_row = [
            "1",
            "CASH",
            "CASH",
            "CASH",
            "-",
            "-",
            "-",
            "-",
            format_price(cash_krw, "kor"),
            "-",
            "-",
            _format_weight_pct(cash_weight),
            "-",
            "-",
            "-",
            "-",
            note,
        ]
        day_rows.append(cash_row)

        for row_index, row in enumerate(held_rows + wait_rows + sold_rows, start=2):
            day_rows.append(
                [
                    str(row_index),
                    row["ticker"],
                    row["name"],
                    row["status"],
                    row["held_days"],
                    row["price"],
                    row["daily_pct"],
                    row["qty"],
                    row["amount"],
                    row["pnl"],
                    row["pnl_pct"],
                    row["weight"],
                    row["score"],
                    row["trend"],
                    row["rsi"],
                    row["message"],
                ]
            )

        lines.append(
            f"[{day.strftime('%Y-%m-%d')}({_format_weekday_kr(day)})] "
            f"총자산 {format_price(total_equity_krw, 'kor')} / "
            f"현금 {format_price(cash_krw, 'kor')} / "
            f"일간수익: {format_price(daily_pnl_krw, 'kor')}({format_pct_change(daily_pct)}) / "
            f"평가수익: {format_price(valuation_pnl_krw, 'kor')}({format_pct_change(valuation_pct)}) / "
            f"누적수익: {format_price(cumulative_pnl_krw, 'kor')}({format_pct_change(cumulative_pct)})"
        )
        lines.extend(render_table_eaw(headers, day_rows, aligns))
        lines.append("")
        prev_total_equity_krw = total_equity_krw

    for exec_idx in range(1, len(backtest_days)):
        signal_day = backtest_days[exec_idx - 1]
        exec_day = backtest_days[exec_idx]
        close_today = close_frame.loc[signal_day]
        portfolio_value_local = cash + sum(
            int(shares.get(t, 0)) * float(close_today.get(t, np.nan) or 0.0)
            for t in shares
            if not pd.isna(close_today.get(t, np.nan))
        )

        open_exec = open_frame.loc[exec_day]
        close_exec = close_frame.loc[exec_day]

        composite = composite_frame.loc[signal_day].copy()
        rule_signal = rule_frame.loc[signal_day].copy()
        rsi_signal = rsi_frame.loc[signal_day].copy()
        if bonus:
            for holding in shares:
                if holding in composite.index and not pd.isna(composite.loc[holding]):
                    composite.loc[holding] += bonus

        rsi_sell_tickers: set[str] = set()
        if rsi_limit is not None:
            rsi_sell_mask = rsi_signal.notna() & (rsi_signal > rsi_limit)
            rsi_sell_tickers = set(rsi_signal.index[rsi_sell_mask].tolist())
            composite.loc[list(rsi_sell_tickers)] = np.nan

        valid = composite.dropna()
        if not valid.empty:
            tradable_mask = open_exec.reindex(valid.index).notna() & (open_exec.reindex(valid.index) > 0)
            valid = valid[tradable_mask]

        if valid.empty:
            held_rows: list[dict[str, str]] = []
            total_equity_close_local = cash + sum(
                int(shares.get(t, 0)) * float(close_exec.get(t, np.nan) or 0.0)
                for t in shares
                if not pd.isna(close_exec.get(t, np.nan))
            )
            valuation_pnl_local = 0.0
            valuation_cost_local = 0.0
            for ticker in sorted(shares.keys()):
                qty = int(shares[ticker])
                current_price = float(close_exec.get(ticker, np.nan)) if not pd.isna(close_exec.get(ticker, np.nan)) else None
                previous_close = (
                    float(close_today.get(ticker, np.nan)) if not pd.isna(close_today.get(ticker, np.nan)) else None
                )
                amount_local = (current_price * qty) if current_price is not None else 0.0
                avg_cost = float(avg_costs[ticker])
                pnl_local = amount_local - (avg_cost * qty)
                valuation_pnl_local += pnl_local
                valuation_cost_local += avg_cost * qty
                pnl_pct = ((amount_local / (avg_cost * qty) - 1.0) * 100.0) if avg_cost > 0 and qty > 0 else None
                weight_pct = (
                    (amount_local / total_equity_close_local) * 100.0 if total_equity_close_local > 0 else None
                )
                held_days = format_trading_days(exec_idx - int(last_buy_indices.get(ticker, exec_idx)))
                held_rows.append(
                    {
                        "ticker": ticker,
                        "name": ticker_name_map.get(ticker, ticker),
                        "status": "HOLD",
                        "held_days": held_days,
                        "price": _format_price_or_dash(current_price, country_code),
                        "daily_pct": format_pct_change(_safe_daily_pct(current_price, previous_close)),
                        "qty": _format_quantity(qty),
                        "amount": format_price(amount_local * float(fx_series.loc[exec_day]), "kor"),
                        "pnl": format_price(pnl_local * float(fx_series.loc[exec_day]), "kor"),
                        "pnl_pct": format_pct_change(pnl_pct),
                        "weight": _format_weight_pct(weight_pct),
                        "score": _format_score_value(
                            None if pd.isna(composite.get(ticker, np.nan)) else float(composite.get(ticker, np.nan))
                        ),
                        "trend": _format_score_value(
                            None if pd.isna(rule_signal.get(ticker, np.nan)) else float(rule_signal.get(ticker, np.nan))
                        ),
                        "rsi": _format_score_value(
                            None if pd.isna(rsi_signal.get(ticker, np.nan)) else float(rsi_signal.get(ticker, np.nan))
                        ),
                        "message": "거래 없음",
                    }
                )
            held_rows.sort(key=lambda row: row["ticker"])
            _append_day_section(
                day=exec_day,
                held_rows=held_rows,
                wait_rows=[],
                sold_rows=[],
                cash_local=cash,
                total_equity_local=total_equity_close_local,
                valuation_pnl_local=valuation_pnl_local,
                valuation_cost_local=valuation_cost_local,
                note="거래 없음",
            )
            continue

        target_df = valid.reset_index()
        target_df.columns = ["ticker", "score"]
        target_df = target_df.sort_values(
            by=["score", "ticker"], ascending=[False, True], kind="mergesort"
        )
        target_set = set(target_df["ticker"].head(top_n).tolist())
        current_set = set(shares.keys())

        to_sell = (current_set - target_set) | (current_set & rsi_sell_tickers)
        to_buy = target_set - current_set
        total_equity_signal = portfolio_value_local

        sold_rows: list[dict[str, str]] = []
        buy_messages: dict[str, str] = {}

        for ticker in sorted(to_sell):
            raw_open_price = float(open_exec.get(ticker, np.nan))
            if pd.isna(raw_open_price) or raw_open_price <= 0:
                continue
            price = raw_open_price * (1.0 - sell_slippage)
            qty = int(shares.pop(ticker))
            avg_cost = float(avg_costs.pop(ticker, 0.0))
            last_buy_idx = int(last_buy_indices.pop(ticker, exec_idx))
            proceeds_local = qty * price
            cash += proceeds_local
            realized_pnl_local = proceeds_local - (avg_cost * qty)
            realized_pct = ((price / avg_cost) - 1.0) * 100.0 if avg_cost > 0 else None
            previous_close = (
                float(close_today.get(ticker, np.nan)) if not pd.isna(close_today.get(ticker, np.nan)) else None
            )
            sold_rows.append(
                {
                    "ticker": ticker,
                    "name": ticker_name_map.get(ticker, ticker),
                    "status": "SELL",
                    "held_days": format_trading_days(exec_idx - last_buy_idx),
                    "price": _format_price_or_dash(price, country_code),
                    "daily_pct": format_pct_change(_safe_daily_pct(
                        float(close_exec.get(ticker, np.nan)) if not pd.isna(close_exec.get(ticker, np.nan)) else None,
                        previous_close,
                    )),
                    "qty": _format_quantity(qty),
                    "amount": format_price(proceeds_local * float(fx_series.loc[exec_day]), "kor"),
                    "pnl": format_price(realized_pnl_local * float(fx_series.loc[exec_day]), "kor"),
                    "pnl_pct": format_pct_change(realized_pct),
                    "weight": _format_weight_pct(0.0),
                    "score": _format_score_value(
                        None if pd.isna(composite.get(ticker, np.nan)) else float(composite.get(ticker, np.nan))
                    ),
                    "trend": _format_score_value(
                        None if pd.isna(rule_signal.get(ticker, np.nan)) else float(rule_signal.get(ticker, np.nan))
                    ),
                    "rsi": _format_score_value(
                        None if pd.isna(rsi_signal.get(ticker, np.nan)) else float(rsi_signal.get(ticker, np.nan))
                    ),
                    "message": (
                        "RSI 상한 초과로 시초가 전량매도"
                        if ticker in rsi_sell_tickers
                        else "상위 N 제외로 시초가 전량매도"
                    ),
                }
            )

        # 매수 (방식 S3): 신규 K개에 ``min(현금/K, 총자산/N)`` 씩 균등 단주 매수. 끝.
        if to_buy and cash > 0:
            new_entrants_df = target_df[target_df["ticker"].isin(to_buy)]
            new_entrants = new_entrants_df["ticker"].tolist()
            k_new = len(new_entrants)
            if k_new > 0:
                slot_target = total_equity_signal / float(top_n)
                per_new_budget = min(slot_target, cash / k_new)

                for ticker in new_entrants:
                    raw_open_price = float(open_exec.get(ticker, np.nan))
                    if pd.isna(raw_open_price) or raw_open_price <= 0:
                        continue
                    price = raw_open_price * (1.0 + buy_slippage)
                    buy_budget = min(cash, per_new_budget)
                    if buy_budget <= 0:
                        continue
                    n_shares = int(buy_budget // price)
                    if n_shares <= 0:
                        continue
                    cost = n_shares * price
                    cash -= cost
                    shares[ticker] = shares.get(ticker, 0) + n_shares
                    avg_costs[ticker] = price
                    last_buy_indices[ticker] = exec_idx
                    buy_messages[ticker] = "신규 편입"

        held_rows: list[dict[str, str]] = []
        total_equity_close_local = cash + sum(
            int(shares.get(t, 0)) * float(close_exec.get(t, np.nan) or 0.0)
            for t in shares
            if not pd.isna(close_exec.get(t, np.nan))
        )
        valuation_pnl_local = 0.0
        valuation_cost_local = 0.0
        current_rank_map = {ticker: rank for rank, ticker in enumerate(target_df["ticker"].tolist(), start=1)}

        for ticker in sorted(shares.keys(), key=lambda item: (current_rank_map.get(item, 10_000), item)):
            qty = int(shares[ticker])
            current_price = float(close_exec.get(ticker, np.nan)) if not pd.isna(close_exec.get(ticker, np.nan)) else None
            previous_close = (
                float(close_today.get(ticker, np.nan)) if not pd.isna(close_today.get(ticker, np.nan)) else None
            )
            amount_local = (current_price * qty) if current_price is not None else 0.0
            avg_cost = float(avg_costs[ticker])
            pnl_local = amount_local - (avg_cost * qty)
            valuation_pnl_local += pnl_local
            valuation_cost_local += avg_cost * qty
            pnl_pct = ((amount_local / (avg_cost * qty) - 1.0) * 100.0) if avg_cost > 0 and qty > 0 else None
            weight_pct = (
                (amount_local / total_equity_close_local) * 100.0 if total_equity_close_local > 0 else None
            )
            held_days = format_trading_days(exec_idx - int(last_buy_indices.get(ticker, exec_idx)))
            held_rows.append(
                {
                    "ticker": ticker,
                    "name": ticker_name_map.get(ticker, ticker),
                    "status": "BUY" if ticker in buy_messages else "HOLD",
                    "held_days": held_days,
                    "price": _format_price_or_dash(current_price, country_code),
                    "daily_pct": format_pct_change(_safe_daily_pct(current_price, previous_close)),
                    "qty": _format_quantity(qty),
                    "amount": format_price(amount_local * float(fx_series.loc[exec_day]), "kor"),
                    "pnl": format_price(pnl_local * float(fx_series.loc[exec_day]), "kor"),
                    "pnl_pct": format_pct_change(pnl_pct),
                    "weight": _format_weight_pct(weight_pct),
                    "score": _format_score_value(
                        None if pd.isna(composite.get(ticker, np.nan)) else float(composite.get(ticker, np.nan))
                    ),
                    "trend": _format_score_value(
                        None if pd.isna(rule_signal.get(ticker, np.nan)) else float(rule_signal.get(ticker, np.nan))
                    ),
                    "rsi": _format_score_value(
                        None if pd.isna(rsi_signal.get(ticker, np.nan)) else float(rsi_signal.get(ticker, np.nan))
                    ),
                    "message": buy_messages.get(ticker, "기존 보유 유지"),
                }
            )

        wait_rows: list[dict[str, str]] = []
        sold_rows_by_ticker = {row["ticker"]: row for row in sold_rows}
        # 현재 보유 중인 종목을 제외하고 신호 점수순 상위 N개를 대기/매도 후보로 추출한다.
        waiting_candidates_df = target_df[~target_df["ticker"].isin(shares.keys())].head(top_n).copy()
        waiting_tickers = waiting_candidates_df["ticker"].tolist()

        for ticker in waiting_tickers:
            current_rank = current_rank_map.get(ticker, 10_000)
            if ticker in sold_rows_by_ticker:
                sold_row = sold_rows_by_ticker[ticker].copy()
                sold_row["rank"] = current_rank
                wait_rows.append(sold_row)
                continue
            current_price = float(close_exec.get(ticker, np.nan)) if not pd.isna(close_exec.get(ticker, np.nan)) else None
            previous_close = (
                float(close_today.get(ticker, np.nan)) if not pd.isna(close_today.get(ticker, np.nan)) else None
            )

            wait_rows.append(
                {
                    "ticker": ticker,
                    "name": ticker_name_map.get(ticker, ticker),
                    "status": "WAIT",
                    "held_days": "-",
                    "price": _format_price_or_dash(current_price, country_code),
                    "daily_pct": format_pct_change(_safe_daily_pct(current_price, previous_close)),
                    "qty": "-",
                    "amount": "-",
                    "pnl": "-",
                    "pnl_pct": "-",
                    "weight": _format_weight_pct(0.0),
                    "score": _format_score_value(
                        None if pd.isna(composite.get(ticker, np.nan)) else float(composite.get(ticker, np.nan))
                    ),
                    "trend": _format_score_value(
                        None if pd.isna(rule_signal.get(ticker, np.nan)) else float(rule_signal.get(ticker, np.nan))
                    ),
                    "rsi": _format_score_value(
                        None if pd.isna(rsi_signal.get(ticker, np.nan)) else float(rsi_signal.get(ticker, np.nan))
                    ),
                    "message": f"대기 순위 {current_rank}위",
                    "rank": current_rank,
                }
            )

        wait_rows.sort(key=lambda row: (int(row.get("rank", 10_000)), row["ticker"]))
        sold_rows = [row for row in sold_rows if row["ticker"] not in waiting_tickers]

        note = "매도 후 신규 진입 및 잔액 현금 보유" if (to_sell or to_buy) else "거래 없음"
        _append_day_section(
            day=exec_day,
            held_rows=held_rows,
            wait_rows=wait_rows,
            sold_rows=sold_rows,
            cash_local=cash,
            total_equity_local=total_equity_close_local,
            valuation_pnl_local=valuation_pnl_local,
            valuation_cost_local=valuation_cost_local,
            note=note,
        )

    return lines


def _write_details_file(
    *,
    out_path: Path,
    pool_id: str,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    top_result: dict[str, Any],
    detail_lines: list[str],
) -> None:
    """상위 1개 조합의 일자별 보유 상세 로그를 기록한다."""
    rsi_limit_text = "-" if top_result["RSI_LIMIT"] is None else f"{float(top_result['RSI_LIMIT']):g}"
    lines: list[str] = [
        f"종목풀: {pool_id}",
        f"기간: {period_start.strftime('%Y-%m-%d')} ~ {period_end.strftime('%Y-%m-%d')}",
        "=== 상위 1개 조합 설정 ===",
        f"TOP_N_HOLD: {top_result['TOP_N_HOLD']}",
        f"HOLDING_BONUS_SCORE: {top_result['HOLDING_BONUS_SCORE']:g}",
        f"MA_TYPE: {top_result['MA_TYPE']}",
        f"MA_MONTHS: {top_result['MA_MONTHS']}",
        f"RSI_LIMIT: {rsi_limit_text}",
        f"TOTAL_RETURN_PCT: {top_result['TOTAL_RETURN_PCT']:.2f}",
        f"CAGR_PCT: {top_result['CAGR_PCT']:.2f}",
        f"MDD_PCT: {top_result['MDD_PCT']:.2f}",
        "",
    ]
    lines.extend(detail_lines)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------- 메인 ----------------------------- #


# 중간 결과 파일/진행률 갱신 주기 (완료 조합 수)
_FLUSH_EVERY_N_RESULTS = 100


def run_backtest(pool_id: str, config: dict[str, dict]) -> Path:
    """주어진 풀 ID 와 설정으로 파라미터 스윕 백테스트를 실행한다.

    멀티프로세스 병렬 실행을 사용하며, 실행 중 ``_FLUSH_EVERY_N_RESULTS`` 건마다
    중간 결과를 로그 파일에 갱신한다.

    Args:
        pool_id: 종목풀 식별자 (예: ``"kor_kr"``).
        config: ``BACKTEST_CONFIG`` 딕셔너리. ``pool_id`` 키가 존재해야 한다.

    Returns:
        결과 로그 파일 경로.
    """
    if pool_id not in config:
        raise ValueError(
            f"BACKTEST_CONFIG 에 '{pool_id}' 설정이 없습니다."
        )

    cfg = config[pool_id]
    start_target = pd.Timestamp(BACKTEST_START_DATE).normalize()
    initial_cash = float(BACKTEST_INITIAL_KRW_AMOUNT)
    buy_slippage, sell_slippage = _resolve_slippage(pool_id)

    top_n_values = [int(v) for v in cfg["TOP_N_HOLD"]]
    bonus_values = [float(v) for v in cfg["HOLDING_BONUS_SCORE"]]
    ma_types = [str(v).upper() for v in cfg["MA_TYPE"]]
    ma_months_list = [int(v) for v in cfg["MA_MONTHS"]]
    rsi_limits: list[float] | None = None
    if "RSI_LIMIT" in cfg:
        rsi_limits = [float(v) for v in cfg["RSI_LIMIT"]]

    settings = get_ticker_type_settings(pool_id)
    country_code = str(settings.get("country_code") or "").strip().lower()
    if not country_code:
        raise ValueError(f"'{pool_id}' 설정에 country_code 가 없습니다.")

    etfs = get_etfs(pool_id)
    tickers = sorted({str(item.get("ticker") or "").strip().upper() for item in etfs if item.get("ticker")})
    if not tickers:
        raise RuntimeError(f"'{pool_id}' 풀에 활성 ETF 가 없습니다.")

    # 기간 설정
    today = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()
    # 로그 출력용으로 근사 개월 수 계산
    months = int(round((today - start_target).days / 30.436875))

    # 워밍업: 최대 MA 개월 + 여유 2개월
    max_ma_months = max(ma_months_list, default=0)
    warmup_days_target = (max_ma_months + 2) * TRADING_DAYS_PER_MONTH
    calendar_lookback_start = (start_target - pd.DateOffset(days=int(warmup_days_target * 1.6 + 60))).normalize()

    calendar_days = get_trading_days(
        calendar_lookback_start.strftime("%Y-%m-%d"),
        today.strftime("%Y-%m-%d"),
        country_code,
    )
    calendar_days = [pd.Timestamp(d).normalize() for d in calendar_days]
    if not calendar_days:
        raise RuntimeError("거래일 캘린더가 비어 있습니다.")

    # OHLCV 캐시 로드
    logger.info("[%s] OHLCV 캐시 로드: %s tickers ...", pool_id, len(tickers))
    frames = load_cached_frames_bulk_with_fallback(pool_id, tickers)
    _augment_frames_with_intraday_open(frames, tickers, country_code, calendar_days[-1], today)
    missing = [t for t in tickers if t not in frames or frames[t] is None or frames[t].empty]
    if missing:
        logger.info(
            "[%s]   (캐시 없음, 제외: %s) %s%s",
            pool_id,
            len(missing),
            missing[:10],
            "..." if len(missing) > 10 else "",
        )
    tickers = [t for t in tickers if t in frames and frames[t] is not None and not frames[t].empty]
    if not tickers:
        raise RuntimeError("캐시된 OHLCV 가 있는 티커가 없습니다.")

    # Close/Open 매트릭스 구축 (index = calendar_days, columns = tickers)
    close_cols: dict[str, pd.Series] = {}
    open_cols: dict[str, pd.Series] = {}
    index = pd.DatetimeIndex(calendar_days)
    for ticker in tickers:
        frame = frames[ticker]
        if not isinstance(frame.index, pd.DatetimeIndex):
            frame.index = pd.to_datetime(frame.index)
        frame = frame.sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        close_col = _select_close_column(list(frame.columns))
        open_col = _select_open_column(list(frame.columns))
        close_cols[ticker] = pd.to_numeric(frame[close_col], errors="coerce").reindex(index)
        open_cols[ticker] = pd.to_numeric(frame[open_col], errors="coerce").reindex(index)

    close_frame = pd.DataFrame(close_cols, index=index)
    open_frame = pd.DataFrame(open_cols, index=index)

    # 1월 1일 시작 설정 시 1월 2일(첫 거래일)부터 매수가 가능하도록,
    # start_target 이후의 첫 거래일을 backtest_days[1]로 위치시킨다.
    # 이를 위해 start_target 이전의 마지막 1거래일을 찾아 backtest_days[0](시그널용)으로 포함한다.
    first_target_idx = next((i for i, d in enumerate(calendar_days) if d >= start_target), None)
    if first_target_idx is not None and first_target_idx > 0:
        backtest_days = calendar_days[first_target_idx - 1 :]
    else:
        backtest_days = [d for d in calendar_days if d >= start_target]

    if len(backtest_days) < 2:
        raise RuntimeError("백테스트 기간 내 거래일이 부족합니다. (최소 2거래일 필요)")

    if not _should_include_latest_day(open_frame, backtest_days[-1], today):
        excluded_day = backtest_days[-1]
        backtest_days = backtest_days[:-1]
        logger.info(
            "[%s] 마지막 거래일 %s은 Open 데이터가 없어 백테스트에서 제외합니다.",
            pool_id,
            excluded_day.strftime("%Y-%m-%d"),
        )
    if len(backtest_days) < 2:
        raise RuntimeError("백테스트 기간 내 확정 거래일이 부족합니다. (최소 2거래일 필요)")

    period_start = backtest_days[1]
    period_end = backtest_days[-1]

    # MA 타입/개월 유니크 집합 → signed percentile 사전계산
    unique_ma_specs = set(itertools.product(ma_types, ma_months_list))

    percentile_by_spec: dict[tuple[str, int], pd.DataFrame] = {}
    logger.info("[%s] MA 점수 사전계산: %s specs ...", pool_id, len(unique_ma_specs))
    for mtype, m_months in unique_ma_specs:
        # rankings 와 동일한 공통 엔진 함수를 통해 규칙별 percentile 프레임 생성.
        percentile_by_spec[(mtype, int(m_months))] = compute_rule_percentile_frame(
            close_frame, mtype, int(m_months)
        )

    # 자격 마스크도 공통 엔진으로 생성 (MIN_TRADING_DAYS 기준) — rankings 와 동일.
    eligibility_frame = compute_eligibility_mask(close_frame)

    # 워커에 전달할 데이터를 백테스트 기간으로 미리 슬라이스 (메모리 절감)
    percentile_by_spec_win = {
        key: pf.loc[backtest_days]
        for key, pf in percentile_by_spec.items()
    }
    eligibility_win = eligibility_frame.loc[backtest_days]
    open_win = open_frame.loc[backtest_days]
    close_win = close_frame.loc[backtest_days]
    ticker_columns = list(open_win.columns)
    percentile_by_spec_values = {
        key: pf.reindex(columns=ticker_columns).to_numpy(dtype=np.float64, copy=True)
        for key, pf in percentile_by_spec_win.items()
    }
    eligibility_values = eligibility_win.reindex(columns=ticker_columns).to_numpy(dtype=bool, copy=True)
    open_values = open_win.to_numpy(dtype=np.float64, copy=True)
    close_values = close_win.to_numpy(dtype=np.float64, copy=True)
    rsi_frame = _compute_rsi_frame(close_frame, RSI_PERIOD)
    rsi_win = rsi_frame.loc[backtest_days]
    rsi_values = rsi_win.to_numpy(dtype=np.float64, copy=True)

    # 환율 시리즈 (KRW 기준 수익률 계산용). 국내 풀은 1.0 상수.
    fx_series = _load_fx_series(country_code, calendar_days)
    fx_win = fx_series.loc[backtest_days]
    fx_values = fx_win.to_numpy(dtype=np.float64, copy=True)
    fx_start = float(fx_win.iloc[0])
    if fx_start <= 0:
        raise RuntimeError(f"시작일 환율이 비정상입니다: {fx_start}")
    # 첫날 KRW 초기자본을 현지통화로 환전해서 보유한다는 가정.
    initial_cash_local = float(initial_cash) / fx_start
    fx_symbol_display = _resolve_fx_symbol(country_code) or "-"
    logger.info(
        "[%s] 환율: symbol=%s start=%.4f end=%.4f → 초기 현지자본 %s",
        pool_id,
        fx_symbol_display,
        fx_start,
        float(fx_win.iloc[-1]),
        f"{initial_cash_local:,.2f}",
    )
    logger.info(
        "[%s] 슬리피지: BUY %.2f%% / SELL %.2f%%",
        pool_id,
        buy_slippage * 100.0,
        sell_slippage * 100.0,
    )

    benchmark_result: dict[str, Any] | None = None
    benchmark_config = cfg.get("BENCHMARK")
    if benchmark_config is not None:
        benchmark_ticker = str(benchmark_config.get("ticker") or "").strip().upper()
        benchmark_name = str(benchmark_config.get("name") or "").strip()
        if not benchmark_ticker or not benchmark_name:
            raise ValueError(
                f"BACKTEST_CONFIG['{pool_id}']['BENCHMARK']에는 ticker/name이 모두 필요합니다."
            )
        benchmark_total_ret, benchmark_cagr, benchmark_mdd, benchmark_trades = _simulate_benchmark_buy_and_hold(
            ticker=benchmark_ticker,
            open_frame=open_win,
            close_frame=close_win,
            backtest_days=backtest_days,
            fx_series=fx_win,
            initial_cash_local=initial_cash_local,
            buy_slippage=buy_slippage,
        )
        benchmark_result = {
            "ticker": benchmark_ticker,
            "name": benchmark_name,
            "TOTAL_RETURN_PCT": benchmark_total_ret,
            "CAGR_PCT": benchmark_cagr,
            "MDD_PCT": benchmark_mdd,
            "TRADES": benchmark_trades,
        }

    # 조합 생성
    combos = list(
        itertools.product(
            top_n_values,
            bonus_values,
            ma_types,
            ma_months_list,
            rsi_limits if rsi_limits is not None else [None],
        )
    )
    total_combos = len(combos)

    # 워커 수 결정 (CPU 코어 - 1, 최소 1)
    n_workers = max(1, (os.cpu_count() or 2) - 1)

    # 폴더/파일 순번 처리를 위해 pools.json의 order 정보를 접두사로 사용
    pool_settings = get_ticker_type_settings(pool_id)
    pool_order = int(pool_settings.get("order", 0))
    display_prefix = f"{pool_order}_{pool_id}" if pool_order > 0 else pool_id

    # 결과 파일 경로
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{display_prefix}-backtest_{today.strftime('%Y-%m-%d')}.log"

    # 결과 파일 기록에 공통으로 쓰는 인자
    write_kwargs: dict[str, Any] = {
        "pool_id": pool_id,
        "months": months,
        "initial_cash": initial_cash,
        "top_n_values": top_n_values,
        "bonus_values": bonus_values,
        "ma_types": ma_types,
        "ma_months_list": ma_months_list,
        "rsi_limits": rsi_limits,
        "total_combos": total_combos,
        "period_start": period_start,
        "period_end": period_end,
        "n_workers": n_workers,
        "benchmark_result": benchmark_result,
    }

    logger.info(
        "[%s] 백테스트 실행: %s combos x %s days (워커 %s개) ...",
        pool_id,
        total_combos,
        len(backtest_days),
        n_workers,
    )

    started_wall = time.time()
    started_str = time.strftime("%Y-%m-%d %H:%M:%S")

    # imap_unordered 의 chunksize: 워커당 최소 4개 청크 이상 할당
    chunksize = max(1, total_combos // (n_workers * 4))

    results: list[dict[str, Any]] = []
    with mp.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(
            percentile_by_spec_values,
            eligibility_values,
            open_values,
            close_values,
            backtest_days,
            initial_cash_local,
            fx_values,
            rsi_values,
            buy_slippage,
            sell_slippage,
        ),
    ) as pool:
        for i, result in enumerate(
            pool.imap_unordered(_run_single_combo, combos, chunksize=chunksize),
            start=1,
        ):
            results.append(result)

            now = time.time()

            # 진행률 출력
            if i % _FLUSH_EVERY_N_RESULTS == 0 or i == total_combos:
                elapsed = now - started_wall
                logger.info("[%s] progress %s/%s (%.1fs)", pool_id, i, total_combos, elapsed)

            # 중간 결과 파일 갱신
            if i < total_combos and i % _FLUSH_EVERY_N_RESULTS == 0:
                _write_results_file(
                    out_path=out_path,
                    results=results,
                    done_count=i,
                    started_str=started_str,
                    elapsed_sec=int(now - started_wall),
                    is_final=False,
                    **write_kwargs,
                )

    ended_wall = time.time()
    elapsed_sec = int(ended_wall - started_wall)

    # 최종 결과 기록
    _write_results_file(
        out_path=out_path,
        results=results,
        done_count=total_combos,
        started_str=started_str,
        elapsed_sec=elapsed_sec,
        is_final=True,
        **write_kwargs,
    )
    sorted_results = sorted(results, key=lambda r: (-r["CAGR_PCT"], r["MDD_PCT"]))
    if sorted_results:
        best_result = sorted_results[0]
        best_rule_frame = percentile_by_spec_win[(best_result["MA_TYPE"], int(best_result["MA_MONTHS"]))].copy()
        best_rule_frame.attrs["ma_type"] = best_result["MA_TYPE"]
        best_rule_frame.attrs["ma_months"] = int(best_result["MA_MONTHS"])
        best_composite = best_rule_frame.where(eligibility_win)
        ticker_name_map = {
            str(item.get("ticker") or "").strip().upper(): str(item.get("name") or "").strip()
            for item in etfs
            if item.get("ticker")
        }
        detail_lines = _simulate_one_combo_details(
            initial_cash_local=initial_cash_local,
            top_n=int(best_result["TOP_N_HOLD"]),
            bonus=float(best_result["HOLDING_BONUS_SCORE"]),
            composite_frame=best_composite,
            rule_frame=best_rule_frame,
            open_frame=open_win,
            close_frame=close_win,
            rsi_frame=rsi_win,
            rsi_limit=None if best_result["RSI_LIMIT"] is None else float(best_result["RSI_LIMIT"]),
            backtest_days=backtest_days,
            fx_series=fx_win,
            ticker_name_map=ticker_name_map,
            country_code=country_code,
            buy_slippage=buy_slippage,
            sell_slippage=sell_slippage,
        )
        detail_path = results_dir / f"{display_prefix}-backtest_details_{today.strftime('%Y-%m-%d')}.log"
        _write_details_file(
            out_path=detail_path,
            pool_id=pool_id,
            period_start=period_start,
            period_end=period_end,
            top_result=best_result,
            detail_lines=detail_lines,
        )
        logger.info("[%s] 상세 결과 저장: %s", pool_id, detail_path)
    logger.info("[%s] 결과 저장: %s", pool_id, out_path)
    return out_path
