"""포트폴리오 실험(portfolio-lab) 서비스 — 한국 전용.

임의의 한국 종목(ETF/개별주) 조합을 "N개월 전 첫 거래일에 균등비중 매수 후 그대로 보유"
가정으로 시뮬레이션한다. 가격은 네이버 fchart 일봉(수정주가, 분배금 반영)을 사용하므로
종목풀 등록/캐시와 무관하게 어떤 한국 티커든 실행할 수 있다.

기간 중간에 상장된 종목은 배정 예산(1/N)을 현금으로 대기시켰다가 상장 첫 거래일 종가에
매수해 편입한다 (상장 전 구간을 합성하지 않음 — 현금 드래그가 결과에 그대로 반영).

저장된 포트폴리오는 MongoDB `portfolio_lab` 컬렉션(이름이 _id)에 보관한다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from time import monotonic
from typing import Any

import numpy as np
import pandas as pd

from leverage.constants import INITIAL_CAPITAL_KRW
from utils.naver_chart import fetch_naver_daily_ohlc
from utils.perf_metrics import curve_metrics, mdd_span

_DEFAULT_BENCHMARK = {"ticker": "069500", "name": "KODEX 200"}
_SLIPPAGE = 0.005  # 매수/매도 슬리피지 0.5% (레버리지/백테스트와 동일 관례)
_MAX_TICKERS = 20
_ALLOWED_MONTHS = (6, 12, 24)
# 리밸런싱 주기 — none(매수후보유) | weekly(매주) | monthly(매월말) | quarterly(분기말) | yearly(연말)
_ALLOWED_REBALANCE = ("none", "weekly", "monthly", "quarterly", "yearly")
_REBALANCE_FREQ = {"weekly": "W", "monthly": "M", "quarterly": "Q", "yearly": "Y"}


def _rebalance_indices(index: pd.DatetimeIndex, rebalance: str) -> set[int]:
    """리밸런싱 시점(거래일 위치)을 반환한다.

    각 주기(주/월/분기/연)의 마지막 거래일에 리밸런싱한다. 주=그 주 마지막 거래일(보통 금),
    월/분기/연=해당 구간의 마지막 거래일. 초기 매수일(0)과 마지막 날은 제외한다.
    """
    if rebalance == "none":
        return set()
    periods = index.to_period(_REBALANCE_FREQ[rebalance])
    last_pos: dict[Any, int] = {}
    for i, period in enumerate(periods):
        last_pos[period] = i  # 같은 구간에서 뒤 값으로 덮어써 마지막 거래일 위치를 남긴다
    days = set(last_pos.values())
    days.discard(0)
    days.discard(len(index) - 1)
    return days


def _simulate_buy_and_hold(
    frame: pd.DataFrame, tickers: list[str], valid_map: dict[str, Any],
    buy_idx_map: dict[str, int], per_budget: float,
) -> tuple[Any, dict[str, int]]:
    """매수 후 보유(리밸런싱 없음). 각 종목을 상장 첫 거래일에 1/N 예산으로 1회 매수."""
    n_days = len(frame)
    day_pos = np.arange(n_days)
    holdings = np.zeros(n_days, dtype=np.float64)
    cash_curve = np.zeros(n_days, dtype=np.float64)
    final_shares: dict[str, int] = {}
    for ticker in tickers:
        series = frame[ticker]
        valid = valid_map[ticker]
        buy_idx = buy_idx_map[ticker]
        buy_price = float(series.iloc[buy_idx]) * (1.0 + _SLIPPAGE)
        shares = int(per_budget // buy_price)
        if shares <= 0:
            raise ValueError(f"{ticker} 를 배정 예산으로 1주도 매수할 수 없습니다.")
        leftover = per_budget - shares * buy_price
        values = np.where(valid, series.to_numpy(dtype=np.float64) * shares, 0.0)
        values[:buy_idx] = 0.0
        holdings += values
        cash_curve += np.where(day_pos < buy_idx, per_budget, leftover)
        final_shares[ticker] = shares
    return holdings + cash_curve, final_shares


def _simulate_rebalanced(
    frame: pd.DataFrame, tickers: list[str], valid_map: dict[str, Any], reb_days: set[int],
) -> tuple[Any, dict[str, int]]:
    """주기적 균등비중 리밸런싱. 매 리밸런싱일에 상장된 종목을 총자산/N 목표로 맞춘다.

    미상장 종목의 몫(1/N)은 현금으로 대기하며, 상장 후 다음 리밸런싱일에 편입된다.
    매매 델타에만 슬리피지를 적용한다(매도 후 매수 순서).
    """
    n = len(tickers)
    n_days = len(frame)
    close = {t: frame[t].to_numpy(dtype=np.float64) for t in tickers}
    shares = {t: 0 for t in tickers}
    cash = float(INITIAL_CAPITAL_KRW)
    curve = np.empty(n_days, dtype=np.float64)
    reb = set(reb_days) | {0}  # 0일차 = 초기 균등 배분

    for day in range(n_days):
        listed = [t for t in tickers if valid_map[t][day]]
        px = {t: float(close[t][day]) for t in listed}
        if day in reb:
            total = cash + sum(shares[t] * px[t] for t in listed)
            target = total / n  # 미상장분(1/N)은 현금으로 유지
            for t in listed:  # 매도 먼저 → 현금 확보
                cur = shares[t] * px[t]
                if cur > target and px[t] > 0:
                    sell_sh = int((cur - target) // px[t])
                    if sell_sh > 0:
                        shares[t] -= sell_sh
                        cash += sell_sh * px[t] * (1.0 - _SLIPPAGE)
            for t in listed:  # 매수
                cur = shares[t] * px[t]
                if cur < target and px[t] > 0:
                    budget = min(target - cur, cash)
                    buy_sh = int(budget // (px[t] * (1.0 + _SLIPPAGE)))
                    if buy_sh > 0:
                        shares[t] += buy_sh
                        cash -= buy_sh * px[t] * (1.0 + _SLIPPAGE)
        curve[day] = cash + sum(shares[t] * px[t] for t in listed)
    return curve, shares


# 네이버 ETF 전체 이름맵 캐시 (10분)
_names_cache: tuple[float, dict[str, str]] | None = None


def _db():
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결 실패 (portfolio_lab)")
    return db


def resolve_kor_name(ticker: str) -> str:
    """한국 티커의 종목명 조회 — ETF 전체맵 우선, 없으면 네이버 개별주 조회."""
    global _names_cache
    ticker_norm = str(ticker or "").strip().upper()
    if not ticker_norm:
        raise ValueError("티커를 입력해주세요.")

    now = monotonic()
    if _names_cache is None or now - _names_cache[0] > 600:
        from utils.stock_meta_updater import fetch_naver_etf_names_map

        _names_cache = (now, fetch_naver_etf_names_map())
    name = _names_cache[1].get(ticker_norm)
    if name:
        return name

    from utils.data_loader import fetch_naver_kor_stock_name

    stock_name = fetch_naver_kor_stock_name(ticker_norm)
    if stock_name:
        return stock_name
    raise ValueError(f"존재하지 않는 한국 티커입니다: {ticker_norm}")


def _normalize_benchmark(benchmark: dict[str, Any] | None) -> dict[str, str]:
    """벤치마크 dict 를 정규화한다. 미지정 시 기본값(KODEX 200)."""
    ticker = str((benchmark or {}).get("ticker") or "").strip().upper()
    if not ticker:
        return dict(_DEFAULT_BENCHMARK)
    return {"ticker": ticker, "name": str((benchmark or {}).get("name") or ticker)}


def run_portfolio_lab(
    tickers: list[dict[str, Any]],
    months: int,
    benchmark: dict[str, Any] | None = None,
    rebalance: str = "none",
) -> dict[str, Any]:
    """균등비중 포트폴리오 시뮬레이션. 결과 요약/종목별/곡선을 반환한다.

    rebalance: none(매수후보유) | weekly | monthly | quarterly | yearly
    """
    months = int(months)
    if months not in _ALLOWED_MONTHS:
        raise ValueError(f"기간(개월)은 {', '.join(map(str, _ALLOWED_MONTHS))} 중 하나여야 합니다.")
    rebalance = str(rebalance or "none").lower()
    if rebalance not in _ALLOWED_REBALANCE:
        raise ValueError(f"리밸런싱 주기는 {', '.join(_ALLOWED_REBALANCE)} 중 하나여야 합니다.")

    bench = _normalize_benchmark(benchmark)

    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in tickers or []:
        ticker = str((item or {}).get("ticker") or "").strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        entries.append({"ticker": ticker, "name": str(item.get("name") or ticker)})
    if not entries:
        raise ValueError("종목이 1개 이상 필요합니다.")
    if len(entries) > _MAX_TICKERS:
        raise ValueError(f"종목은 최대 {_MAX_TICKERS}개까지 가능합니다.")

    today = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()
    start_target = (today - pd.DateOffset(months=months)).normalize()
    fetch_count = months * 21 + 40  # 거래일 여유분 포함

    closes: dict[str, pd.Series] = {}
    for entry in entries + [bench]:
        ticker = entry["ticker"]
        if ticker in closes:
            continue
        df = fetch_naver_daily_ohlc(ticker, count=fetch_count)
        if df is None or df.empty:
            raise ValueError(f"{entry.get('name', ticker)}({ticker}) 의 가격 데이터를 받아오지 못했습니다.")
        closes[ticker] = pd.to_numeric(df["Close"], errors="coerce").dropna()

    bench_first = closes[bench["ticker"]].index.min()
    if bench_first > start_target + pd.offsets.BDay(5):
        raise ValueError(
            f"벤치마크 {bench['name']}({bench['ticker']}) 는 데이터 시작일이 {bench_first.date()} 라 "
            f"{months}개월 실험에 쓸 수 없습니다 (필요: {start_target.date()} 이전 상장)."
        )

    frame = pd.DataFrame({t: s for t, s in closes.items()}).sort_index()
    frame = frame[frame.index >= start_target].dropna(how="all").ffill()
    if frame.empty or len(frame) < 2:
        raise ValueError("실험 기간에 해당하는 거래일 데이터가 부족합니다.")

    start_date = frame.index[0]
    n = len(entries)
    per_budget = INITIAL_CAPITAL_KRW / n

    # 종목별 상장일(buy_idx) + 보유구간 성과(종가 기준, 스케일 무관). 매매/편입은 아래 시뮬레이션에서.
    valid_map: dict[str, Any] = {}
    buy_idx_map: dict[str, int] = {}
    positions: list[dict[str, Any]] = []
    for entry in entries:
        ticker = entry["ticker"]
        series = frame[ticker]
        valid = series.notna().to_numpy()
        if not valid.any():
            raise ValueError(f"{entry['name']}({ticker}) 는 실험 기간에 거래 데이터가 없습니다.")
        buy_idx = int(np.argmax(valid))
        buy_ts = frame.index[buy_idx]
        buy_close = float(series.iloc[buy_idx])
        if not np.isfinite(buy_close) or buy_close <= 0:
            raise ValueError(f"{entry['name']}({ticker}) 의 매수일({buy_ts.date()}) 가격이 비정상입니다.")
        valid_map[ticker] = valid
        buy_idx_map[ticker] = buy_idx

        last_close = float(series.iloc[-1])
        # MDD/Sharpe 는 각 종목의 보유 구간(매수일~종료일) 종가 기준으로 계산한다.
        seg = series.to_numpy(dtype=np.float64)[buy_idx:]
        seg_dates = frame.index[buy_idx:]
        sec_metrics = curve_metrics(buy_close, seg)
        mdd_peak, mdd_trough, _ = mdd_span(seg)
        positions.append(
            {
                **entry,
                "buy_date": buy_ts.date().isoformat(),
                "late_entry": buy_idx > 0,
                "buy_price": round(buy_close, 4),
                "last_price": round(last_close, 4),
                "return_pct": round((last_close / buy_close - 1.0) * 100.0, 2),
                "mdd_pct": round(sec_metrics["mdd_pct"], 2),
                "mdd_start": seg_dates[mdd_peak].date().isoformat(),
                "mdd_end": seg_dates[mdd_trough].date().isoformat(),
                "sharpe": round(sec_metrics["sharpe"], 2),
            }
        )

    ticker_order = [e["ticker"] for e in entries]
    if rebalance == "none":
        curve, final_shares = _simulate_buy_and_hold(frame, ticker_order, valid_map, buy_idx_map, per_budget)
    else:
        curve, final_shares = _simulate_rebalanced(
            frame, ticker_order, valid_map, _rebalance_indices(frame.index, rebalance)
        )

    last_prices = {t: float(frame[t].iloc[-1]) for t in ticker_order}
    for p in positions:
        sh = int(final_shares[p["ticker"]])
        p["shares"] = sh
        p["value"] = round(sh * last_prices[p["ticker"]], 0)

    summary = curve_metrics(float(INITIAL_CAPITAL_KRW), curve)

    bench_series = frame[bench["ticker"]]
    bench_buy = float(bench_series.iloc[0]) * (1.0 + _SLIPPAGE)
    bench_shares = int(INITIAL_CAPITAL_KRW // bench_buy)
    bench_cash = INITIAL_CAPITAL_KRW - bench_shares * bench_buy
    bench_curve = bench_series.to_numpy(dtype=np.float64) * bench_shares + bench_cash
    bench_summary = curve_metrics(float(INITIAL_CAPITAL_KRW), bench_curve)

    dates = [d.date().isoformat() for d in frame.index]
    return {
        "months": months,
        "rebalance": rebalance,
        "buy_date": start_date.date().isoformat(),
        "end_date": frame.index[-1].date().isoformat(),
        "has_late_entry": any(p["late_entry"] for p in positions),
        "initial_capital": INITIAL_CAPITAL_KRW,
        "final_value": round(float(curve[-1]), 0),
        "summary": {k: round(v, 2) for k, v in summary.items()},
        "benchmark": {
            **bench,
            "summary": {k: round(v, 2) for k, v in bench_summary.items()},
        },
        "positions": positions,
        "chart": {
            "dates": dates,
            "portfolio_pct": [round((v / INITIAL_CAPITAL_KRW - 1.0) * 100.0, 3) for v in curve],
            "benchmark_pct": [round((v / INITIAL_CAPITAL_KRW - 1.0) * 100.0, 3) for v in bench_curve],
        },
    }


# ----------------------------- 저장/목록 ----------------------------- #


def list_saved_portfolios() -> list[dict[str, Any]]:
    docs = []
    for doc in _db().portfolio_lab.find({}).sort("updated_at", -1):
        docs.append(
            {
                "name": str(doc["_id"]),
                "tickers": doc.get("tickers") or [],
                "months": int(doc.get("months") or 12),
                "benchmark": doc.get("benchmark") or dict(_DEFAULT_BENCHMARK),
                "rebalance": str(doc.get("rebalance") or "none"),
                "updated_at": doc.get("updated_at").isoformat() if doc.get("updated_at") else None,
            }
        )
    return docs


def save_portfolio(
    name: str,
    tickers: list[dict[str, Any]],
    months: int,
    benchmark: dict[str, Any] | None = None,
    rebalance: str = "none",
) -> None:
    clean_name = str(name or "").strip()
    if not clean_name:
        raise ValueError("포트폴리오 이름이 필요합니다.")
    clean: list[dict[str, str]] = []
    for item in tickers or []:
        ticker = str((item or {}).get("ticker") or "").strip().upper()
        if ticker:
            clean.append({"ticker": ticker, "name": str(item.get("name") or ticker)})
    if not clean:
        raise ValueError("종목이 1개 이상 필요합니다.")
    if int(months) not in _ALLOWED_MONTHS:
        raise ValueError(f"기간(개월)은 {', '.join(map(str, _ALLOWED_MONTHS))} 중 하나여야 합니다.")
    rebalance = str(rebalance or "none").lower()
    if rebalance not in _ALLOWED_REBALANCE:
        raise ValueError(f"리밸런싱 주기는 {', '.join(_ALLOWED_REBALANCE)} 중 하나여야 합니다.")

    _db().portfolio_lab.update_one(
        {"_id": clean_name},
        {
            "$set": {
                "tickers": clean,
                "months": int(months),
                "benchmark": _normalize_benchmark(benchmark),
                "rebalance": rebalance,
                "updated_at": datetime.now(timezone.utc),
            }
        },
        upsert=True,
    )


def delete_portfolio(name: str) -> None:
    result = _db().portfolio_lab.delete_one({"_id": str(name or "").strip()})
    if result.deleted_count == 0:
        raise ValueError(f"저장된 포트폴리오가 없습니다: {name}")
