"""모멘텀 ETF 파라미터 스윕 백테스트 (alpha).

실행 예시::

    python backtest.py kor_kr


BACKTEST_CONFIG 의 스윕 범위를 따라 (TOP_N_HOLD, HOLDING_BONUS_SCORE,
FIRST_MA_TYPE, FIRST_MA_MONTHS, SECOND_MA_TYPE, SECOND_MA_MONTHS) 6개
파라미터 조합을 모두 돌려 BACKTEST_MONTHS 개월 백테스트 결과를 CAGR
내림차순으로 정렬하여 ``ztickers/<order>_<pool>/results/backtest_<YYYY-MM-DD>.log``
파일에 기록한다.

MongoDB 캐시가 반드시 있어야 한다 (로컬 전용).

알파 버전 가정:
    * 점수 역전 시 다음 거래일 시초가(Open)에 체결.
    * 단주 매수, 잔여는 현금 보유.
    * 매도 금액 전액을 대상 종목 목표 비중(1/N)까지 매수하되, 현금 부족 시 잔액만 사용.
    * 수수료/슬리피지/세금 = 0.
    * 상장폐지/거래정지 종목은 풀에 없는 것으로 간주.
    * HOLDING_BONUS_SCORE 는 백테스트 내부에서만 적용 (rankings.py 미변경).
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import TRADING_DAYS_PER_MONTH
from core.strategy.scoring import calculate_maps_score, calculate_signed_percentile_score
from utils.cache_utils import load_cached_frames_bulk_with_fallback
from utils.data_loader import get_trading_days
from utils.env import load_env_if_present
from utils.moving_averages import calculate_moving_average
from utils.settings_loader import TICKERS_ROOT, get_ticker_type_settings
from utils.stock_list_io import get_etfs

load_env_if_present()


BACKTEST_CONFIG: dict[str, dict] = {
    "kor_kr": {
        "BACKTEST_MONTHS": 12,
        "INITIAL_KRW_AMOUNT": 100_000_000,
        "TOP_N_HOLD": [5],
        "HOLDING_BONUS_SCORE": [0, 10, 20],
        "FIRST_MA_TYPE": ["SMA"],
        "FIRST_MA_MONTHS": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "SECOND_MA_TYPE": ["SMA"],
        "SECOND_MA_MONTHS": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    }
}


# ----------------------------- 헬퍼 ----------------------------- #


def _resolve_pool_dir(pool_id: str) -> Path:
    """``ztickers/<order>_<pool_id>/`` 디렉토리를 glob 으로 찾는다."""
    matches = [p for p in TICKERS_ROOT.glob(f"*_{pool_id}") if p.is_dir()]
    if not matches:
        raise FileNotFoundError(
            f"ztickers 아래에서 '*_{pool_id}' 디렉토리를 찾을 수 없습니다."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"'*_{pool_id}' 패턴에 해당하는 디렉토리가 2개 이상 있습니다: {matches}"
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


def _compute_percentile_scores(
    close_frame: pd.DataFrame,
    ma_types: list[str],
    ma_months_list: list[int],
) -> dict[tuple[str, int], pd.DataFrame]:
    """(ma_type, ma_months) 별 signed-percentile 점수 프레임을 사전계산."""
    out: dict[tuple[str, int], pd.DataFrame] = {}
    for ma_type in ma_types:
        for months in ma_months_list:
            days = int(months) * TRADING_DAYS_PER_MONTH
            ma_cols: dict[str, pd.Series] = {}
            for ticker in close_frame.columns:
                series = close_frame[ticker].dropna()
                if len(series) < days:
                    ma_cols[ticker] = pd.Series(dtype=float, index=close_frame.index)
                    continue
                ma_series = calculate_moving_average(series, days, ma_type)
                ma_cols[ticker] = ma_series.reindex(close_frame.index)
            ma_frame = pd.DataFrame(ma_cols, index=close_frame.index)
            # rankings.py 의 점수식과 동일한 공통 함수를 사용 (변경 시 양쪽 동시 반영).
            trend = pd.DataFrame(
                {
                    ticker: calculate_maps_score(close_frame[ticker], ma_frame[ticker])
                    for ticker in close_frame.columns
                },
                index=close_frame.index,
            )
            percentile = calculate_signed_percentile_score(trend)
            out[(ma_type, int(months))] = percentile
    return out


# --------------------------- 시뮬레이션 --------------------------- #


def _simulate_one_combo(
    *,
    initial_cash: float,
    top_n: int,
    bonus: float,
    percentile_first: pd.DataFrame,
    percentile_second: pd.DataFrame,
    open_frame: pd.DataFrame,
    close_frame: pd.DataFrame,
    backtest_days: list[pd.Timestamp],
) -> tuple[float, float, float]:
    """단일 파라미터 조합에 대해 1회 백테스트.

    Returns:
        (total_return_pct, cagr_pct, mdd_pct)
    """
    shares: dict[str, int] = {}
    cash = float(initial_cash)

    value_curve: list[float] = []

    # 마지막 날 외의 모든 거래일에서 신호 발생 가능
    for idx, signal_day in enumerate(backtest_days):
        close_today = close_frame.loc[signal_day]
        portfolio_value = cash + sum(
            int(shares.get(t, 0)) * float(close_today.get(t, np.nan) or 0.0)
            for t in shares
            if not pd.isna(close_today.get(t, np.nan))
        )
        value_curve.append(portfolio_value)

        # 마지막 거래일에는 체결이 불가 → 관측만.
        if idx == len(backtest_days) - 1:
            break

        exec_day = backtest_days[idx + 1]

        # 점수 계산
        score_first = percentile_first.loc[signal_day]
        score_second = percentile_second.loc[signal_day]
        composite = score_first.add(score_second, fill_value=np.nan)

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

        # 신호 시점 기준 총 자산 (= 목표 비중 계산용)
        total_equity_signal = portfolio_value

        # 1) 매도 먼저
        for ticker in to_sell:
            price = float(open_exec.get(ticker, np.nan))
            if pd.isna(price) or price <= 0:
                # 매도 불가 → 보유 유지
                continue
            n = int(shares.pop(ticker))
            cash += n * price

        # 2) 매수
        target_amount = total_equity_signal / float(top_n)
        for ticker in to_buy:
            price = float(open_exec.get(ticker, np.nan))
            if pd.isna(price) or price <= 0:
                continue
            buy_budget = min(cash, target_amount)
            if buy_budget <= 0:
                continue
            n_shares = int(buy_budget // price)
            if n_shares <= 0:
                continue
            cost = n_shares * price
            cash -= cost
            shares[ticker] = shares.get(ticker, 0) + n_shares

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


# ----------------------------- 메인 ----------------------------- #


def run_backtest(pool_id: str) -> Path:
    if pool_id not in BACKTEST_CONFIG:
        raise ValueError(
            f"BACKTEST_CONFIG 에 '{pool_id}' 설정이 없습니다. backtest.py 상단을 확인하세요."
        )

    cfg = BACKTEST_CONFIG[pool_id]
    months = int(cfg["BACKTEST_MONTHS"])
    initial_cash = float(cfg["INITIAL_KRW_AMOUNT"])

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

    ma_types_needed = sorted({t for t, _ in unique_ma_specs})
    ma_months_needed_per_type: dict[str, list[int]] = {}
    for mtype, m in unique_ma_specs:
        ma_months_needed_per_type.setdefault(mtype, []).append(m)
    ma_months_needed_per_type = {k: sorted(set(v)) for k, v in ma_months_needed_per_type.items()}

    percentile_by_spec: dict[tuple[str, int], pd.DataFrame] = {}
    print(f"[{pool_id}] MA 점수 사전계산: {len(unique_ma_specs)} specs ...", flush=True)
    for mtype in ma_types_needed:
        chunk = _compute_percentile_scores(
            close_frame,
            [mtype],
            ma_months_needed_per_type[mtype],
        )
        percentile_by_spec.update(chunk)

    # 조합 실행
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
    print(f"[{pool_id}] 백테스트 실행: {total_combos} combos x {len(backtest_days)} days ...", flush=True)

    started_wall = time.time()
    started_str = time.strftime("%Y-%m-%d %H:%M:%S")

    results: list[dict[str, Any]] = []
    for i, (top_n, bonus, fma_t, fma_m, sma_t, sma_m) in enumerate(combos, start=1):
        pf = percentile_by_spec[(fma_t, fma_m)]
        ps = percentile_by_spec[(sma_t, sma_m)]
        # 백테스트 기간으로 슬라이스
        pf_win = pf.loc[backtest_days]
        ps_win = ps.loc[backtest_days]
        open_win = open_frame.loc[backtest_days]
        close_win = close_frame.loc[backtest_days]

        total_ret, cagr, mdd = _simulate_one_combo(
            initial_cash=initial_cash,
            top_n=top_n,
            bonus=bonus,
            percentile_first=pf_win,
            percentile_second=ps_win,
            open_frame=open_win,
            close_frame=close_win,
            backtest_days=backtest_days,
        )
        results.append(
            {
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
        )

        if i % 50 == 0 or i == total_combos:
            elapsed = time.time() - started_wall
            print(f"  progress {i}/{total_combos} ({elapsed:.1f}s)", flush=True)

    ended_wall = time.time()
    ended_str = time.strftime("%Y-%m-%d %H:%M:%S")
    elapsed_sec = int(ended_wall - started_wall)

    # 정렬
    results.sort(key=lambda r: (-r["CAGR_PCT"], r["MDD_PCT"]))

    # 출력
    results_dir = pool_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"backtest_{today.strftime('%Y-%m-%d')}.log"

    lines: list[str] = []
    h = elapsed_sec // 3600
    m = (elapsed_sec % 3600) // 60
    s = elapsed_sec % 60
    lines.append(f"실행 시각: {started_str}")
    lines.append(f"종료 시각: {ended_str}")
    lines.append(f"종목풀: {pool_id}")
    lines.append(f"걸린 시간: {h}시간 {m}분 {s}초")
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
    lines.append(f"=== 결과 - 기간: {months} 개월 | 정렬 기준: CAGR ===")

    # 테이블
    top_limit = 100
    top_rows = results[:top_limit]
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
    col_widths = [
        max(len(h), max((len(row[i]) for row in formatted_rows), default=0))
        for i, h in enumerate(headers)
    ]

    def _fmt_row(cells: list[str]) -> str:
        parts = [cells[i].rjust(col_widths[i]) if i >= 6 else cells[i].ljust(col_widths[i]) for i in range(len(cells))]
        return "| " + " | ".join(parts) + " |"

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    lines.append(sep)
    lines.append(_fmt_row(headers))
    lines.append(sep)
    for row in formatted_rows:
        lines.append(_fmt_row(row))
    lines.append(sep)
    if total_combos > top_limit:
        lines.append(f"... (총 {total_combos}개 중 상위 {top_limit}개 표시)")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{pool_id}] 결과 저장: {out_path}", flush=True)
    return out_path


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python backtest.py <pool_id>", file=sys.stderr)
        print(f"사용 가능한 pool_id: {sorted(BACKTEST_CONFIG.keys())}", file=sys.stderr)
        return 2

    pool_id = argv[1].strip().lower()
    run_backtest(pool_id)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
