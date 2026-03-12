"""Pool-based RANK ranking runner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config import TRADING_DAYS_PER_MONTH
from core.strategy.metrics import process_ticker_data
from utils.data_loader import (
    MissingPriceDataError,
    fetch_au_quoteapi_snapshot,
    fetch_naver_etf_inav_snapshot,
    get_latest_trading_day,
    prepare_price_data,
)
from utils.formatters import format_pct_change, format_price, format_price_deviation
from utils.pool_registry import POOL_ROOT, get_pool_dir
from utils.report import render_table_eaw
from utils.stock_list_io import get_etfs

RESULTS_ROOT = POOL_ROOT


@dataclass
class RankConfig:
    country: str
    months: int
    ma_type: str


@dataclass
class RankRunResult:
    pool_id: str
    country: str
    base_date: pd.Timestamp
    generated_at: datetime
    ma_type: str
    months: int
    rows: list[dict[str, Any]]
    missing_tickers: list[str]


@dataclass
class RankOutputPaths:
    result_dir: Path
    log_path: Path


def _calc_return(close_series: pd.Series, days: int) -> float:
    if close_series is None or len(close_series) <= days:
        return 0.0
    try:
        current = float(close_series.iloc[-1])
        previous = float(close_series.iloc[-(days + 1)])
    except Exception:
        return 0.0
    if previous <= 0:
        return 0.0
    return (current / previous - 1.0) * 100.0


def _calc_period_return_like_recommend(close_series: pd.Series, days: int) -> float:
    """recommend.py와 동일한 기간 수익률 계산 규칙.

    - 기본: 지정 거래일(days) 이전 값과 비교
    - 12개월(252일): 데이터가 240일 이상이면 가장 오래된 값으로 fallback 계산
    """
    if close_series is None:
        return 0.0
    try:
        series = pd.to_numeric(close_series, errors="coerce").dropna()
    except Exception:
        return 0.0
    if series.empty:
        return 0.0

    current = float(series.iloc[-1])
    if current <= 0:
        return 0.0

    if len(series) > days:
        previous = float(series.iloc[-(days + 1)])
        if previous > 0:
            return (current / previous - 1.0) * 100.0

    if days == 252 and len(series) >= 240:
        previous = float(series.iloc[0])
        if previous > 0:
            return (current / previous - 1.0) * 100.0

    return 0.0


def _calc_rsi(close_series: pd.Series, period: int = 14) -> float:
    """Wilder 방식 RSI를 계산해 마지막 값을 반환한다."""
    if close_series is None or len(close_series) < period + 1:
        return 0.0
    try:
        delta = close_series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        val = float(rsi.iloc[-1])
        if pd.isna(val):
            return 0.0
        return max(0.0, min(100.0, val))
    except Exception:
        return 0.0


def run_pool_ranking(pool_id: str, config: RankConfig) -> RankRunResult:
    pool_norm = (pool_id or "").strip().lower()
    if not pool_norm:
        raise ValueError("pool_id is required")

    universe = get_etfs(pool_norm)
    if not universe:
        raise ValueError(f"pool_id='{pool_norm}' 종목이 비어있습니다.")

    tickers = sorted({str(item.get("ticker") or "").strip().upper() for item in universe if item.get("ticker")})

    end_date = get_latest_trading_day(config.country)
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.Timestamp.now().normalize()

    ma_days = int(config.months) * TRADING_DAYS_PER_MONTH
    # 12개월 수익률(252일, 240일 fallback) 계산을 위해 MA 설정과 별도로 충분한 히스토리를 확보한다.
    lookback_days_for_returns = 540
    start_date = (end_date - pd.DateOffset(days=max(ma_days * 3, lookback_days_for_returns))).strftime("%Y-%m-%d")

    prices_map, missing = prepare_price_data(
        tickers=tickers,
        country=config.country,
        start_date=start_date,
        end_date=end_date.strftime("%Y-%m-%d"),
        warmup_days=0,
        account_id=pool_norm,
        allow_remote_fetch=False,
    )
    if missing:
        raise MissingPriceDataError(
            country=config.country,
            start_date=start_date,
            end_date=end_date.strftime("%Y-%m-%d"),
            tickers=missing,
        )

    meta_map = {
        str(item.get("ticker") or "").strip().upper(): dict(item)
        for item in universe
        if str(item.get("ticker") or "").strip()
    }
    rows: list[dict[str, Any]] = []

    for ticker in tickers:
        df = prices_map.get(ticker)
        if df is None or df.empty:
            continue

        metrics = process_ticker_data(
            ticker=ticker,
            df=df,
            ma_days=ma_days,
            ma_type=config.ma_type,
            enable_data_sufficiency_check=False,
        )
        if not metrics:
            continue

        score_series = metrics.get("ma_score")
        close_series = metrics.get("close")
        ma_series = metrics.get("ma")
        if score_series is None or close_series is None or ma_series is None:
            continue

        try:
            score = float(score_series.iloc[-1])
            close = float(close_series.iloc[-1])
            ma_val = float(ma_series.iloc[-1])
            daily_pct = _calc_return(close_series, 1)
            return_1w = _calc_period_return_like_recommend(close_series, 5)
            return_2w = _calc_period_return_like_recommend(close_series, 10)
            return_1m = _calc_period_return_like_recommend(close_series, 20)
            return_3m = _calc_period_return_like_recommend(close_series, 60)
            return_6m = _calc_period_return_like_recommend(close_series, 126)
            return_12m = _calc_period_return_like_recommend(close_series, 252)
            drawdown_from_high = (
                ((close / float(close_series.max())) - 1.0) * 100.0 if float(close_series.max()) > 0 else 0.0
            )
            streak_series = metrics.get("buy_signal_days")
            streak = int(streak_series.iloc[-1]) if streak_series is not None else 0
            rsi_score = _calc_rsi(close_series, period=14)
        except Exception:
            continue

        meta = meta_map.get(ticker, {})
        bucket_id = int(meta.get("bucket", 1) or 1)

        rows.append(
            {
                "ticker": ticker,
                "name": str(meta.get("name") or ticker),
                "bucket": bucket_id,
                "state": "WAIT",
                "holding_days": 0,
                "phrase": "",
                "score": score,
                "rsi_score": rsi_score,
                "streak": streak,
                "price": close,
                "close": close,
                "ma": ma_val,
                "daily_pct": daily_pct,
                "evaluation_pct": None,
                "return_1w": return_1w,
                "return_2w": return_2w,
                "return_1m": return_1m,
                "return_3m": return_3m,
                "return_6m": return_6m,
                "return_12m": return_12m,
                "drawdown_from_high": drawdown_from_high,
                "trend_prices": [float(v) for v in close_series.iloc[-60:].tolist()],
            }
        )

    if not rows:
        raise RuntimeError("점수 계산 가능한 종목이 없습니다.")

    rows.sort(key=lambda x: float(x["score"]), reverse=True)
    sliced_rows = rows

    for rank, row in enumerate(sliced_rows, start=1):
        row["rank"] = rank

    # 한국 종목풀은 NAV/괴리율을 실시간으로 보강
    if (config.country or "").strip().lower() in {"kor", "kr"}:
        tickers_for_nav = [row.get("ticker") for row in sliced_rows if row.get("ticker")]
        nav_snapshot = fetch_naver_etf_inav_snapshot(tickers_for_nav)
        for row in sliced_rows:
            ticker = str(row.get("ticker") or "").strip().upper()
            nav_data = nav_snapshot.get(ticker)
            if not nav_data:
                continue
            row["nav_price"] = nav_data.get("nav")
            row["price_deviation"] = nav_data.get("deviation")
            if nav_data.get("changeRate") is not None:
                row["daily_pct"] = float(nav_data.get("changeRate"))
            if nav_data.get("nowVal") is not None:
                row["price"] = float(nav_data.get("nowVal"))
            if nav_data.get("itemname"):
                row["name"] = str(nav_data.get("itemname"))
    elif (config.country or "").strip().lower() == "au":
        tickers_for_quote = [row.get("ticker") for row in sliced_rows if row.get("ticker")]
        quote_snapshot = fetch_au_quoteapi_snapshot(tickers_for_quote)
        for row in sliced_rows:
            ticker = str(row.get("ticker") or "").strip().upper()
            quote_data = quote_snapshot.get(ticker)
            if not quote_data:
                continue
            if quote_data.get("nowVal") is not None:
                row["price"] = float(quote_data.get("nowVal"))
            if quote_data.get("changeRate") is not None:
                row["daily_pct"] = float(quote_data.get("changeRate"))
            elif quote_data.get("prevClose") is not None and quote_data.get("nowVal") is not None:
                prev_close = float(quote_data.get("prevClose"))
                now_val = float(quote_data.get("nowVal"))
                if prev_close > 0:
                    row["daily_pct"] = ((now_val / prev_close) - 1.0) * 100.0

    return RankRunResult(
        pool_id=pool_norm,
        country=config.country,
        base_date=end_date,
        generated_at=datetime.now(),
        ma_type=config.ma_type.upper(),
        months=int(config.months),
        rows=sliced_rows,
        missing_tickers=sorted(missing),
    )


def save_rank_result(result: RankRunResult) -> RankOutputPaths:
    try:
        pool_dir = get_pool_dir(result.pool_id)
    except Exception:
        pool_dir = RESULTS_ROOT / result.pool_id
        pool_dir.mkdir(parents=True, exist_ok=True)
    result_dir = pool_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    log_path = result_dir / f"rank_{result.base_date.strftime('%Y-%m-%d')}.log"

    # 추천 로그 스타일에 맞춘 rank 로그 생성 (단, 불가 컬럼 제외)
    bucket_names = {
        1: "1. 모멘텀",
        2: "2. 혁신기술",
        3: "3. 시장지수",
        4: "4. 배당방어",
        5: "5. 대체헷지",
    }
    country_lower = (result.country or "").strip().lower()
    show_deviation = country_lower in {"kr", "kor"}
    nav_mode = country_lower in {"kr", "kor"}

    lines: list[str] = []
    lines.append(f"랭킹 로그 생성: {pd.Timestamp.now().isoformat(timespec='seconds')}")
    lines.append(f"종목풀: {result.pool_id.upper()} | 기준일: {result.base_date.strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append("=== 랭킹 목록 ===")
    lines.append("")

    headers = ["버킷", "티커", "종목명", "일간(%)", "현재가"]
    aligns = ["left", "left", "left", "right", "right"]
    if show_deviation:
        headers.append("괴리율")
        aligns.append("right")
    if nav_mode:
        headers.append("Nav")
        aligns.append("right")
    headers.extend(["1주(%)", "2주(%)", "1달(%)", "3달(%)", "6달(%)", "12달(%)", "고점대비", "점수", "RSI", "지속"])
    aligns.extend(["right", "right", "right", "right", "right", "right", "right", "right", "right", "right"])

    table_rows: list[list[str]] = []
    for row in result.rows:
        bucket_label = bucket_names.get(int(row.get("bucket", 1) or 1), str(row.get("bucket", 1)))
        data_row = [
            bucket_label,
            str(row.get("ticker") or "-"),
            str(row.get("name") or "-"),
            format_pct_change(row.get("daily_pct")),
            format_price(row.get("price"), result.country),
        ]
        if show_deviation:
            data_row.append(format_price_deviation(row.get("price_deviation")))
        if nav_mode:
            data_row.append(format_price(row.get("nav_price"), result.country))
        data_row.extend(
            [
                format_pct_change(row.get("return_1w")),
                format_pct_change(row.get("return_2w")),
                format_pct_change(row.get("return_1m")),
                format_pct_change(row.get("return_3m")),
                format_pct_change(row.get("return_6m")),
                format_pct_change(row.get("return_12m")),
                format_pct_change(row.get("drawdown_from_high")),
                f"{float(row.get('score', 0.0)):.1f}",
                f"{float(row.get('rsi_score', 0.0)):.1f}",
                f"{int(row.get('streak', 0))}일" if int(row.get("streak", 0)) > 0 else "-",
            ]
        )
        table_rows.append(data_row)

    lines.extend(render_table_eaw(headers, table_rows, aligns))
    lines.append("")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return RankOutputPaths(
        result_dir=result_dir,
        log_path=log_path,
    )
