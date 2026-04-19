"""모멘텀 ETF 파라미터 스윕 백테스트 엔진.

``backtest/run.py`` 에서 호출되며, 종목풀별 ``BACKTEST_CONFIG`` 와
전역 공통값 ``BACKTEST_MONTHS``, ``INITIAL_KRW_AMOUNT`` 를 사용한다.
멀티프로세스 병렬 실행을 지원하며, 실행 중에도 중간 결과를 파일에 주기적으로 기록한다.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.config import BACKTEST_MONTHS, INITIAL_KRW_AMOUNT
from config import TRADING_DAYS_PER_MONTH
from core.strategy.scoring import (
    combine_rule_percentiles,
    compute_eligibility_mask,
    compute_rule_percentile_frame,
)
from utils.cache_utils import load_cached_frames_bulk_with_fallback
from utils.data_loader import get_exchange_rate_series, get_trading_days
from utils.formatters import format_pct_change, format_price, format_trading_days
from utils.report import render_table_eaw
from utils.settings_loader import TICKERS_ROOT, get_ticker_type_settings
from utils.stock_list_io import get_etfs

# ----------------------------- 헬퍼 ----------------------------- #


def _resolve_pool_dir(pool_id: str) -> Path:
    """``ztickers/<순번>_<pool_id>/`` 디렉토리를 찾는다.

    ``<순번>`` 은 숫자만 허용한다. 예: ``pool_id='us'`` → ``4_us`` 매칭,
    ``2_kor_us`` 는 접두사가 ``kor_`` 이므로 미매칭.
    """
    matches: list[Path] = []
    for candidate in TICKERS_ROOT.iterdir():
        if not candidate.is_dir():
            continue
        name = candidate.name
        if "_" not in name:
            continue
        prefix, _, suffix = name.partition("_")
        if suffix == pool_id and prefix.isdigit():
            matches.append(candidate)

    if not matches:
        raise FileNotFoundError(
            f"ztickers 아래에서 '<순번>_{pool_id}' 디렉토리를 찾을 수 없습니다."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"'<순번>_{pool_id}' 패턴에 해당하는 디렉토리가 2개 이상 있습니다: {matches}"
        )
    return matches[0]


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


# -------------------- 워커 프로세스 (병렬 실행) -------------------- #


# 워커 프로세스에서 공유하는 전역 변수.
# _init_worker() 로 한 번만 초기화되며, 각 워커 프로세스 내에서만 유효하다.
_W_PCT: dict[tuple[str, int], pd.DataFrame] = {}
_W_ELIG: pd.DataFrame = pd.DataFrame()
_W_OPEN: pd.DataFrame = pd.DataFrame()
_W_CLOSE: pd.DataFrame = pd.DataFrame()
_W_DAYS: list[pd.Timestamp] = []
_W_CASH_LOCAL: float = 0.0
_W_FX: pd.Series = pd.Series(dtype=float)


def _init_worker(
    pct_specs: dict[tuple[str, int], pd.DataFrame],
    eligibility_win: pd.DataFrame,
    open_win: pd.DataFrame,
    close_win: pd.DataFrame,
    bt_days: list[pd.Timestamp],
    init_cash_local: float,
    fx_win: pd.Series,
) -> None:
    """워커 프로세스 초기화: 공유 데이터를 전역 변수에 설정."""
    global _W_PCT, _W_ELIG, _W_OPEN, _W_CLOSE, _W_DAYS, _W_CASH_LOCAL, _W_FX  # noqa: PLW0603
    _W_PCT = pct_specs
    _W_ELIG = eligibility_win
    _W_OPEN = open_win
    _W_CLOSE = close_win
    _W_DAYS = bt_days
    _W_CASH_LOCAL = init_cash_local
    _W_FX = fx_win


def _run_single_combo(
    args: tuple[int, float, str, int, str, int],
) -> dict[str, Any]:
    """워커에서 단일 파라미터 조합을 실행하고 결과 딕셔너리를 반환한다."""
    top_n, bonus, fma_t, fma_m, sma_t, sma_m = args
    # 공통 엔진과 동일한 방식으로 규칙별 percentile 합산 + 자격 마스크 적용.
    composite = combine_rule_percentiles(
        [_W_PCT[(fma_t, fma_m)], _W_PCT[(sma_t, sma_m)]],
        _W_ELIG,
    )
    total_ret, cagr, mdd = _simulate_one_combo(
        initial_cash_local=_W_CASH_LOCAL,
        top_n=top_n,
        bonus=bonus,
        composite_frame=composite,
        open_frame=_W_OPEN,
        close_frame=_W_CLOSE,
        backtest_days=_W_DAYS,
        fx_series=_W_FX,
    )
    return {
        "TOP_N_HOLD": top_n,
        "HOLDING_BONUS_SCORE": bonus,
        "FIRST_MA_TYPE": fma_t,
        "FIRST_MA_MONTHS": fma_m,
        "SECOND_MA_TYPE": sma_t,
        "SECOND_MA_MONTHS": sma_m,
        "TOTAL_RETURN_PCT": total_ret,
        "CAGR_PCT": cagr,
        "MDD_PCT": mdd,
    }


# --------------------------- 시뮬레이션 --------------------------- #


def _simulate_one_combo(
    *,
    initial_cash_local: float,
    top_n: int,
    bonus: float,
    composite_frame: pd.DataFrame,
    open_frame: pd.DataFrame,
    close_frame: pd.DataFrame,
    backtest_days: list[pd.Timestamp],
    fx_series: pd.Series,
) -> tuple[float, float, float]:
    """단일 파라미터 조합에 대해 1회 백테스트.

    모든 체결/보유/현금은 현지 통화로 관리한다. 평가 일자마다 당일 환율을 곱해
    KRW 기준 value_curve 를 만들고 총수익률/CAGR/MDD 를 계산한다.

    Returns:
        (total_return_pct, cagr_pct, mdd_pct) — 모두 KRW 기준.
    """
    shares: dict[str, int] = {}
    cash = float(initial_cash_local)

    value_curve: list[float] = []

    # 마지막 날 외의 모든 거래일에서 신호 발생 가능
    for idx, signal_day in enumerate(backtest_days):
        close_today = close_frame.loc[signal_day]
        portfolio_value_local = cash + sum(
            int(shares.get(t, 0)) * float(close_today.get(t, np.nan) or 0.0)
            for t in shares
            if not pd.isna(close_today.get(t, np.nan))
        )
        fx_today = float(fx_series.loc[signal_day])
        portfolio_value = portfolio_value_local * fx_today  # KRW 환산 평가액
        value_curve.append(portfolio_value)

        # 마지막 거래일에는 체결이 불가 → 관측만.
        if idx == len(backtest_days) - 1:
            break

        exec_day = backtest_days[idx + 1]

        # 점수 계산 (공통 엔진이 만든 composite 프레임에서 해당 일자 행을 사용).
        composite = composite_frame.loc[signal_day].copy()

        if bonus:
            for holding in shares:
                if holding in composite.index and not pd.isna(composite.loc[holding]):
                    composite.loc[holding] += bonus

        valid = composite.dropna()
        if valid.empty:
            continue

        # 체결일 시초가가 존재하는 티커만 대상 후보로 삼는다 (거래정지/데이터 결측 방지).
        open_exec = open_frame.loc[exec_day]
        tradable_mask = open_exec.reindex(valid.index).notna() & (
            open_exec.reindex(valid.index) > 0
        )
        valid = valid[tradable_mask]
        if valid.empty:
            continue

        # 동점 처리: 점수 desc → ticker asc
        target_df = valid.reset_index()
        target_df.columns = ["ticker", "score"]
        target_df = target_df.sort_values(
            by=["score", "ticker"], ascending=[False, True], kind="mergesort"
        )
        target_set = set(target_df["ticker"].head(top_n).tolist())
        current_set = set(shares.keys())

        to_sell = current_set - target_set
        to_buy = target_set - current_set

        # 신호 시점 기준 총 자산 (= 목표 비중 계산용). 매수/매도는 현지 통화로 수행.
        total_equity_signal = portfolio_value_local

        # 1) 매도 먼저
        for ticker in to_sell:
            price = float(open_exec.get(ticker, np.nan))
            if pd.isna(price) or price <= 0:
                # 매도 불가 → 보유 유지
                continue
            n = int(shares.pop(ticker))
            cash += n * price

        # 2) 매수 (방식 A': 신규 진입 K개에 "slot 상한 ∧ cash/K" 균등 배분 → 잔액은 고점수 순 추가 소진)
        if not to_buy:
            continue

        # 신규 진입 종목을 점수 내림차순 → 티커 오름차순으로 정렬 (동점 결정론적 처리).
        new_entrants_df = target_df[target_df["ticker"].isin(to_buy)]
        new_entrants = new_entrants_df["ticker"].tolist()
        k_new = len(new_entrants)
        if k_new == 0:
            continue

        slot_target = total_equity_signal / float(top_n)
        per_new_budget = min(slot_target, cash / k_new) if cash > 0 else 0.0

        # 2-1) 균등 1차 배분
        for ticker in new_entrants:
            price = float(open_exec.get(ticker, np.nan))
            if pd.isna(price) or price <= 0:
                continue
            buy_budget = min(cash, per_new_budget)
            if buy_budget <= 0:
                continue
            n_shares = int(buy_budget // price)
            if n_shares <= 0:
                continue
            cost = n_shares * price
            cash -= cost
            shares[ticker] = shares.get(ticker, 0) + n_shares

        # 2-2) 단주 매수로 남은 잔액 → 고점수 신규 종목부터 한 주씩 추가 소진
        for ticker in new_entrants:
            if cash <= 0:
                break
            price = float(open_exec.get(ticker, np.nan))
            if pd.isna(price) or price <= 0:
                continue
            # slot_target 을 이미 채웠는지 확인 (몰빵 방지).
            current_position_value = shares.get(ticker, 0) * price
            remaining_slot_budget = max(0.0, slot_target - current_position_value)
            if remaining_slot_budget <= 0:
                continue
            extra_budget = min(cash, remaining_slot_budget)
            extra_shares = int(extra_budget // price)
            if extra_shares <= 0:
                continue
            cost = extra_shares * price
            cash -= cost
            shares[ticker] = shares.get(ticker, 0) + extra_shares

    if not value_curve:
        return 0.0, 0.0, 0.0

    values = pd.Series(value_curve, index=backtest_days[: len(value_curve)])
    start_val = values.iloc[0]
    end_val = values.iloc[-1]
    total_return_pct = (end_val / start_val - 1.0) * 100.0 if start_val > 0 else 0.0

    n_days = max(1, len(values) - 1)
    years = n_days / 252.0
    if start_val > 0 and end_val > 0 and years > 0:
        cagr_pct = (pow(end_val / start_val, 1.0 / years) - 1.0) * 100.0
    else:
        cagr_pct = 0.0

    running_max = values.cummax()
    drawdown = (values / running_max - 1.0) * 100.0
    mdd_pct = float(drawdown.min()) if not drawdown.empty else 0.0

    return float(total_return_pct), float(cagr_pct), float(mdd_pct)


# --------------------------- 결과 기록 --------------------------- #


def _write_results_file(
    *,
    out_path: Path,
    results: list[dict[str, Any]],
    pool_id: str,
    months: int,
    initial_cash: float,
    top_n_values: list[int],
    bonus_values: list[float],
    first_ma_types: list[str],
    first_ma_months_list: list[int],
    second_ma_types: list[str],
    second_ma_months_list: list[int],
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
        f"x FIRST_MA_TYPE {len(first_ma_types)}개 x FIRST_MA_MONTHS {len(first_ma_months_list)}개 "
        f"x SECOND_MA_TYPE {len(second_ma_types)}개 x SECOND_MA_MONTHS {len(second_ma_months_list)}개 "
        f"= {total_combos}개 조합"
    )
    lines.append(f'"INITIAL_KRW_AMOUNT": {int(initial_cash)},')
    lines.append(f'"TOP_N_HOLD": {top_n_values},')
    lines.append(f'"HOLDING_BONUS_SCORE": {[int(b) if float(b).is_integer() else b for b in bonus_values]},')
    lines.append(f'"FIRST_MA_TYPE": {first_ma_types},')
    lines.append(f'"FIRST_MA_MONTHS": {first_ma_months_list},')
    lines.append(f'"SECOND_MA_TYPE": {second_ma_types},')
    lines.append(f'"SECOND_MA_MONTHS": {second_ma_months_list},')
    lines.append("")

    status_label = "최종 결과" if is_final else f"중간 결과 ({done_count}/{total_combos})"
    lines.append(f"=== {status_label} - 기간: {months} 개월 | 정렬 기준: CAGR ===")

    # 테이블
    top_limit = 100
    top_rows = sorted_results[:top_limit]
    if not top_rows:
        lines.append("(결과 없음)")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    headers = [
        "TOP_N",
        "BONUS",
        "MA1_TYPE",
        "MA1_M",
        "MA2_TYPE",
        "MA2_M",
        "수익률(%)",
        "CAGR(%)",
        "MDD(%)",
    ]
    aligns = ["left", "left", "left", "left", "left", "left", "right", "right", "right"]
    formatted_rows: list[list[str]] = []
    for r in top_rows:
        formatted_rows.append(
            [
                str(r["TOP_N_HOLD"]),
                f"{r['HOLDING_BONUS_SCORE']:g}",
                r["FIRST_MA_TYPE"],
                str(r["FIRST_MA_MONTHS"]),
                r["SECOND_MA_TYPE"],
                str(r["SECOND_MA_MONTHS"]),
                f"{r['TOTAL_RETURN_PCT']:.2f}",
                f"{r['CAGR_PCT']:.2f}",
                f"{r['MDD_PCT']:.2f}",
            ]
        )
    table_lines = render_table_eaw(headers, formatted_rows, aligns)
    lines.extend(table_lines)
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


def _format_price_or_dash(value: float | None, country_code: str) -> str:
    """가격을 국가 통화 형식으로 변환하되 비어 있으면 '-'를 반환한다."""
    if value is None:
        return "-"
    return format_price(value, country_code)


def _simulate_one_combo_details(
    *,
    initial_cash_local: float,
    top_n: int,
    bonus: float,
    composite_frame: pd.DataFrame,
    open_frame: pd.DataFrame,
    close_frame: pd.DataFrame,
    backtest_days: list[pd.Timestamp],
    fx_series: pd.Series,
    ticker_name_map: dict[str, str],
    country_code: str,
) -> list[str]:
    """상위 1개 조합의 거래일별 보유 상세 로그를 생성한다.

    - 거래일 첫날은 초기 현금만 기록한다.
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
        "left",
    ]

    cash = float(initial_cash_local)
    period_start = backtest_days[0]
    period_end = backtest_days[-1]
    fx_start = float(fx_series.loc[period_start])
    initial_cash_krw = initial_cash_local * fx_start

    lines.append("=== 백테스트 상세 ===")
    lines.append(f"기간: {period_start.strftime('%Y-%m-%d')} ~ {period_end.strftime('%Y-%m-%d')}")
    lines.append(f"TOP_N_HOLD: {top_n}")
    lines.append(f"HOLDING_BONUS_SCORE: {bonus:g}")
    lines.append("")

    def _append_day_section(
        *,
        day: pd.Timestamp,
        held_rows: list[dict[str, str]],
        sold_rows: list[dict[str, str]],
        cash_local: float,
        total_equity_local: float,
        note: str,
    ) -> None:
        fx_today = float(fx_series.loc[day])
        total_equity_krw = total_equity_local * fx_today
        cash_krw = cash_local * fx_today
        cash_weight = (cash_local / total_equity_local * 100.0) if total_equity_local > 0 else 0.0
        day_rows: list[list[str]] = []
        cash_row = [
            "1",
            "CASH",
            "현금",
            "현금",
            "-",
            "-",
            "-",
            "-",
            format_price(cash_krw, "kor"),
            "-",
            "-",
            _format_weight_pct(cash_weight),
            note,
        ]
        day_rows.append(cash_row)

        for row_index, row in enumerate(held_rows + sold_rows, start=2):
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
                    row["message"],
                ]
            )

        lines.append(
            f"[{day.strftime('%Y-%m-%d')}] 총자산 {format_price(total_equity_krw, 'kor')} / "
            f"현금 {format_price(cash_krw, 'kor')}"
        )
        lines.extend(render_table_eaw(headers, day_rows, aligns))
        lines.append("")

    # 초기일: 현금만 보유
    _append_day_section(
        day=period_start,
        held_rows=[],
        sold_rows=[],
        cash_local=cash,
        total_equity_local=cash,
        note="초기 자금",
    )

    for idx, signal_day in enumerate(backtest_days):
        close_today = close_frame.loc[signal_day]
        portfolio_value_local = cash + sum(
            int(shares.get(t, 0)) * float(close_today.get(t, np.nan) or 0.0)
            for t in shares
            if not pd.isna(close_today.get(t, np.nan))
        )

        if idx == len(backtest_days) - 1:
            break

        exec_idx = idx + 1
        exec_day = backtest_days[exec_idx]
        open_exec = open_frame.loc[exec_day]
        close_exec = close_frame.loc[exec_day]

        composite = composite_frame.loc[signal_day].copy()
        if bonus:
            for holding in shares:
                if holding in composite.index and not pd.isna(composite.loc[holding]):
                    composite.loc[holding] += bonus

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
            for ticker in sorted(shares.keys()):
                qty = int(shares[ticker])
                current_price = float(close_exec.get(ticker, np.nan)) if not pd.isna(close_exec.get(ticker, np.nan)) else None
                previous_close = (
                    float(close_today.get(ticker, np.nan)) if not pd.isna(close_today.get(ticker, np.nan)) else None
                )
                amount_local = (current_price * qty) if current_price is not None else 0.0
                avg_cost = float(avg_costs[ticker])
                pnl_local = amount_local - (avg_cost * qty)
                pnl_pct = ((amount_local / (avg_cost * qty) - 1.0) * 100.0) if avg_cost > 0 and qty > 0 else None
                weight_pct = (
                    (amount_local / total_equity_close_local) * 100.0 if total_equity_close_local > 0 else None
                )
                held_days = format_trading_days(exec_idx - int(last_buy_indices.get(ticker, exec_idx)))
                held_rows.append(
                    {
                        "ticker": ticker,
                        "name": ticker_name_map.get(ticker, ticker),
                        "status": "보유",
                        "held_days": held_days,
                        "price": _format_price_or_dash(current_price, country_code),
                        "daily_pct": format_pct_change(_safe_daily_pct(current_price, previous_close)),
                        "qty": _format_quantity(qty),
                        "amount": format_price(amount_local * float(fx_series.loc[exec_day]), "kor"),
                        "pnl": format_price(pnl_local * float(fx_series.loc[exec_day]), "kor"),
                        "pnl_pct": format_pct_change(pnl_pct),
                        "weight": _format_weight_pct(weight_pct),
                        "message": "거래 없음",
                    }
                )
            held_rows.sort(key=lambda row: row["ticker"])
            _append_day_section(
                day=exec_day,
                held_rows=held_rows,
                sold_rows=[],
                cash_local=cash,
                total_equity_local=total_equity_close_local,
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

        to_sell = current_set - target_set
        to_buy = target_set - current_set
        total_equity_signal = portfolio_value_local

        sold_rows: list[dict[str, str]] = []
        buy_messages: dict[str, str] = {}

        for ticker in sorted(to_sell):
            price = float(open_exec.get(ticker, np.nan))
            if pd.isna(price) or price <= 0:
                continue
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
                    "status": "전량매도",
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
                    "message": "상위 N 제외로 시초가 전량매도",
                }
            )

        if to_buy:
            new_entrants_df = target_df[target_df["ticker"].isin(to_buy)]
            new_entrants = new_entrants_df["ticker"].tolist()
            k_new = len(new_entrants)
            slot_target = total_equity_signal / float(top_n)
            per_new_budget = min(slot_target, cash / k_new) if (k_new > 0 and cash > 0) else 0.0

            for ticker in new_entrants:
                price = float(open_exec.get(ticker, np.nan))
                if pd.isna(price) or price <= 0:
                    continue
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

            for ticker in new_entrants:
                if cash <= 0:
                    break
                price = float(open_exec.get(ticker, np.nan))
                if pd.isna(price) or price <= 0:
                    continue
                current_position_value = shares.get(ticker, 0) * price
                remaining_slot_budget = max(0.0, slot_target - current_position_value)
                if remaining_slot_budget <= 0:
                    continue
                extra_budget = min(cash, remaining_slot_budget)
                extra_shares = int(extra_budget // price)
                if extra_shares <= 0:
                    continue
                cost = extra_shares * price
                old_qty = int(shares.get(ticker, 0))
                old_avg = float(avg_costs.get(ticker, price))
                new_qty = old_qty + extra_shares
                shares[ticker] = new_qty
                avg_costs[ticker] = ((old_qty * old_avg) + cost) / new_qty if new_qty > 0 else price
                last_buy_indices[ticker] = exec_idx
                cash -= cost
                previous_message = buy_messages.get(ticker, "신규 편입")
                if "잔액 추가 소진" not in previous_message:
                    buy_messages[ticker] = f"{previous_message} + 잔액 추가 소진"

        held_rows: list[dict[str, str]] = []
        total_equity_close_local = cash + sum(
            int(shares.get(t, 0)) * float(close_exec.get(t, np.nan) or 0.0)
            for t in shares
            if not pd.isna(close_exec.get(t, np.nan))
        )
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
            pnl_pct = ((amount_local / (avg_cost * qty) - 1.0) * 100.0) if avg_cost > 0 and qty > 0 else None
            weight_pct = (
                (amount_local / total_equity_close_local) * 100.0 if total_equity_close_local > 0 else None
            )
            held_days = format_trading_days(exec_idx - int(last_buy_indices.get(ticker, exec_idx)))
            held_rows.append(
                {
                    "ticker": ticker,
                    "name": ticker_name_map.get(ticker, ticker),
                    "status": "신규매수" if ticker in buy_messages else "보유",
                    "held_days": held_days,
                    "price": _format_price_or_dash(current_price, country_code),
                    "daily_pct": format_pct_change(_safe_daily_pct(current_price, previous_close)),
                    "qty": _format_quantity(qty),
                    "amount": format_price(amount_local * float(fx_series.loc[exec_day]), "kor"),
                    "pnl": format_price(pnl_local * float(fx_series.loc[exec_day]), "kor"),
                    "pnl_pct": format_pct_change(pnl_pct),
                    "weight": _format_weight_pct(weight_pct),
                    "message": buy_messages.get(ticker, "기존 보유 유지"),
                }
            )

        note = "매도 후 신규 진입 및 잔액 현금 보유" if (to_sell or to_buy) else "거래 없음"
        _append_day_section(
            day=exec_day,
            held_rows=held_rows,
            sold_rows=sold_rows,
            cash_local=cash,
            total_equity_local=total_equity_close_local,
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
    lines: list[str] = [
        f"종목풀: {pool_id}",
        f"기간: {period_start.strftime('%Y-%m-%d')} ~ {period_end.strftime('%Y-%m-%d')}",
        "=== 상위 1개 조합 설정 ===",
        f"TOP_N_HOLD: {top_result['TOP_N_HOLD']}",
        f"HOLDING_BONUS_SCORE: {top_result['HOLDING_BONUS_SCORE']:g}",
        f"FIRST_MA_TYPE: {top_result['FIRST_MA_TYPE']}",
        f"FIRST_MA_MONTHS: {top_result['FIRST_MA_MONTHS']}",
        f"SECOND_MA_TYPE: {top_result['SECOND_MA_TYPE']}",
        f"SECOND_MA_MONTHS: {top_result['SECOND_MA_MONTHS']}",
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
    months = int(BACKTEST_MONTHS)
    initial_cash = float(INITIAL_KRW_AMOUNT)

    top_n_values = [int(v) for v in cfg["TOP_N_HOLD"]]
    bonus_values = [float(v) for v in cfg["HOLDING_BONUS_SCORE"]]
    first_ma_types = [str(v).upper() for v in cfg["FIRST_MA_TYPE"]]
    first_ma_months_list = [int(v) for v in cfg["FIRST_MA_MONTHS"]]
    second_ma_types = [str(v).upper() for v in cfg["SECOND_MA_TYPE"]]
    second_ma_months_list = [int(v) for v in cfg["SECOND_MA_MONTHS"]]

    settings = get_ticker_type_settings(pool_id)
    country_code = str(settings.get("country_code") or "").strip().lower()
    if not country_code:
        raise ValueError(f"'{pool_id}' 설정에 country_code 가 없습니다.")

    pool_dir = _resolve_pool_dir(pool_id)

    etfs = get_etfs(pool_id)
    tickers = sorted({str(item.get("ticker") or "").strip().upper() for item in etfs if item.get("ticker")})
    if not tickers:
        raise RuntimeError(f"'{pool_id}' 풀에 활성 ETF 가 없습니다.")

    # 기간 설정
    today = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()
    start_target = (today - pd.DateOffset(months=months)).normalize()

    # 워밍업: 최대 MA 개월 + 여유 2개월
    max_ma_months = max(
        max(first_ma_months_list, default=0),
        max(second_ma_months_list, default=0),
    )
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
    print(f"[{pool_id}] OHLCV 캐시 로드: {len(tickers)} tickers ...", flush=True)
    frames = load_cached_frames_bulk_with_fallback(pool_id, tickers)
    missing = [t for t in tickers if t not in frames or frames[t] is None or frames[t].empty]
    if missing:
        print(f"  (캐시 없음, 제외: {len(missing)}) {missing[:10]}{'...' if len(missing) > 10 else ''}", flush=True)
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

    # 백테스트 기간: calendar_days 중 start_target 이상인 일자
    backtest_days = [d for d in calendar_days if d >= start_target]
    if len(backtest_days) < 2:
        raise RuntimeError("백테스트 기간 내 거래일이 부족합니다.")

    period_start = backtest_days[0]
    period_end = backtest_days[-1]

    # MA 타입/개월 유니크 집합 → signed percentile 사전계산
    unique_first = set(itertools.product(first_ma_types, first_ma_months_list))
    unique_second = set(itertools.product(second_ma_types, second_ma_months_list))
    unique_ma_specs = unique_first | unique_second

    percentile_by_spec: dict[tuple[str, int], pd.DataFrame] = {}
    print(f"[{pool_id}] MA 점수 사전계산: {len(unique_ma_specs)} specs ...", flush=True)
    for mtype, months in unique_ma_specs:
        # rankings 와 동일한 공통 엔진 함수를 통해 규칙별 percentile 프레임 생성.
        percentile_by_spec[(mtype, int(months))] = compute_rule_percentile_frame(
            close_frame, mtype, int(months)
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

    # 환율 시리즈 (KRW 기준 수익률 계산용). 국내 풀은 1.0 상수.
    fx_series = _load_fx_series(country_code, calendar_days)
    fx_win = fx_series.loc[backtest_days]
    fx_start = float(fx_win.iloc[0])
    if fx_start <= 0:
        raise RuntimeError(f"시작일 환율이 비정상입니다: {fx_start}")
    # 첫날 KRW 초기자본을 현지통화로 환전해서 보유한다는 가정.
    initial_cash_local = float(initial_cash) / fx_start
    fx_symbol_display = _resolve_fx_symbol(country_code) or "-"
    print(
        f"[{pool_id}] 환율: symbol={fx_symbol_display} start={fx_start:.4f} "
        f"end={float(fx_win.iloc[-1]):.4f} → 초기 현지자본 {initial_cash_local:,.2f}",
        flush=True,
    )

    # 조합 생성
    combos = list(
        itertools.product(
            top_n_values,
            bonus_values,
            first_ma_types,
            first_ma_months_list,
            second_ma_types,
            second_ma_months_list,
        )
    )
    total_combos = len(combos)

    # 워커 수 결정 (CPU 코어 - 1, 최소 1)
    n_workers = max(1, (os.cpu_count() or 2) - 1)

    # 결과 파일 경로
    results_dir = pool_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"backtest_{today.strftime('%Y-%m-%d')}.log"

    # 결과 파일 기록에 공통으로 쓰는 인자
    write_kwargs: dict[str, Any] = {
        "pool_id": pool_id,
        "months": months,
        "initial_cash": initial_cash,
        "top_n_values": top_n_values,
        "bonus_values": bonus_values,
        "first_ma_types": first_ma_types,
        "first_ma_months_list": first_ma_months_list,
        "second_ma_types": second_ma_types,
        "second_ma_months_list": second_ma_months_list,
        "total_combos": total_combos,
        "period_start": period_start,
        "period_end": period_end,
        "n_workers": n_workers,
    }

    print(
        f"[{pool_id}] 백테스트 실행: {total_combos} combos x {len(backtest_days)} days "
        f"(워커 {n_workers}개) ...",
        flush=True,
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
            percentile_by_spec_win,
            eligibility_win,
            open_win,
            close_win,
            backtest_days,
            initial_cash_local,
            fx_win,
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
                print(f"  progress {i}/{total_combos} ({elapsed:.1f}s)", flush=True)

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
        best_composite = combine_rule_percentiles(
            [
                percentile_by_spec_win[(best_result["FIRST_MA_TYPE"], int(best_result["FIRST_MA_MONTHS"]))],
                percentile_by_spec_win[(best_result["SECOND_MA_TYPE"], int(best_result["SECOND_MA_MONTHS"]))],
            ],
            eligibility_win,
        )
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
            open_frame=open_win,
            close_frame=close_win,
            backtest_days=backtest_days,
            fx_series=fx_win,
            ticker_name_map=ticker_name_map,
            country_code=country_code,
        )
        detail_path = results_dir / f"backtest_details_{today.strftime('%Y-%m-%d')}.log"
        _write_details_file(
            out_path=detail_path,
            pool_id=pool_id,
            period_start=period_start,
            period_end=period_end,
            top_result=best_result,
            detail_lines=detail_lines,
        )
        print(f"[{pool_id}] 상세 결과 저장: {detail_path}", flush=True)
    print(f"[{pool_id}] 결과 저장: {out_path}", flush=True)
    return out_path
