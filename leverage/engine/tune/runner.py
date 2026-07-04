"""튜닝 핵심 로직 (백테스트 실행·집계)."""

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from os import cpu_count
from pathlib import Path

import pandas as pd
import yfinance as yf

from leverage.data_adapter import (
    _extract_field,
    compute_bounds,
    download_fx,
)
from leverage.engine.backtest.runner import run_backtest
from leverage.engine.backtest.settings import load_settings
from leverage.report import render_table_eaw

# pykrx는 한국 시장에서만 사용
try:
    from pykrx import stock as pykrx_stock
except ImportError:
    pykrx_stock = None


def _is_rate_limit_error(exc: Exception) -> bool:
    s = repr(exc).lower()
    return "yfratelimiterror" in s or "rate limit" in s


def _is_network_or_data_error(exc: Exception) -> bool:
    s = repr(exc).lower()
    keywords = [
        "dnserror",
        "could not resolve host",
        "timed out",
        "operation timed out",
        "시가 데이터가 비어 있습니다",
        "가격 데이터를 받아오지 못했습니다",
    ]
    return any(k in s for k in keywords)


def _extract_candidate_tickers(items: list) -> list[str]:
    """{ticker, name} 객체 또는 문자열 후보 리스트에서 티커만 추출한다."""
    tickers = []
    for d in items:
        if isinstance(d, dict):
            tickers.append(d.get("ticker", ""))
        else:
            tickers.append(str(d))
    return tickers


def _validate_candidates_kor(candidates: list, start_bound) -> list[tuple[str, str]]:
    """한국 시장 후보 티커 데이터 가용성 검증 (백테스트 시작일 기준).

    반환: (kind, 설명) 리스트 — kind 는 "missing"(데이터 전무) 또는 "late"(상장이 늦어 앞 구간 없음).
    """
    if pykrx_stock is None:
        raise ImportError("pykrx 패키지가 설치되어 있지 않습니다. pip install pykrx")

    start_str = pd.Timestamp(start_bound).strftime("%Y%m%d")
    end_str = pd.Timestamp.today().strftime("%Y%m%d")

    entries: list[tuple[str, str]] = []
    for d in candidates:
        if isinstance(d, dict):
            ticker = d.get("ticker", "")
            name = d.get("name", ticker)
        else:
            ticker = str(d)
            name = ticker

        if ticker == "CASH":
            continue

        df = pykrx_stock.get_market_ohlcv_by_date(start_str, end_str, ticker)
        if df is None or df.empty:
            entries.append(("missing", f"  - {name}({ticker}): 데이터 없음"))
        else:
            data_start = df.index[0].strftime("%Y-%m-%d")
            # start_bound가 주말/휴일일 수 있으므로 첫 영업일 기준으로 비교
            # 데이터 시작일이 start_bound + 5영업일 이후면 상장 지연으로 판정
            adjusted_start = pd.Timestamp(start_bound) + pd.offsets.BDay(5)
            if df.index[0] > adjusted_start:
                required_start = pd.Timestamp(start_bound).strftime("%Y-%m-%d")
                entries.append(("late", f"  - {name}({ticker}): 데이터 시작일 {data_start} (필요: {required_start})"))

    return entries


def _validate_candidates_us(candidates: list, start_bound) -> list[tuple[str, str]]:
    """미국 시장 후보 티커 데이터 가용성 검증 (백테스트 시작일 기준). 반환 형식은 kor 과 동일."""
    entries: list[tuple[str, str]] = []
    for d in candidates:
        if isinstance(d, dict):
            ticker = d.get("ticker", "")
            name = d.get("name", ticker)
        else:
            ticker = str(d)
            name = ticker

        if ticker == "CASH":
            continue

        df = yf.download(ticker, start=start_bound, auto_adjust=True, progress=False)
        if df is None or df.empty:
            entries.append(("missing", f"  - {name}({ticker}): 데이터 없음"))
        else:
            data_start = df.index[0].strftime("%Y-%m-%d")
            # start_bound가 주말/휴일일 수 있으므로 첫 영업일 기준으로 비교
            # 데이터 시작일이 start_bound + 5영업일 이후면 상장 지연으로 판정
            adjusted_start = pd.Timestamp(start_bound) + pd.offsets.BDay(5)
            if df.index[0] > adjusted_start:
                required_start = pd.Timestamp(start_bound).strftime("%Y-%m-%d")
                entries.append(("late", f"  - {name}({ticker}): 데이터 시작일 {data_start} (필요: {required_start})"))
    return entries


def _run_single(args: tuple[dict, dict, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Timestamp]) -> dict:
    base_settings, overrides, pre_prices, pre_opens, pre_fx, pre_bench, start_bound = args
    tuned = dict(base_settings)
    tuned.update(overrides)
    report = run_backtest(
        tuned,
        pre_prices=pre_prices,
        pre_opens=pre_opens,
        pre_fx=pre_fx,
        pre_bench=pre_bench,
        start_bound_override=start_bound,
    )
    return {
        "params": tuned,
        "cagr": report["cagr"],
        "mdd": report["max_drawdown"],
        "sharpe": report["sharpe"],
        "vol": report["vol"],
        "period_return": report.get("period_return", 0.0),
    }


def _prefetch_data_us(settings: dict, tuning_config: dict, warmup_start):
    """미국 시장 데이터 프리패치 (yfinance)."""
    # offense/defense 후보가 {ticker, name} 형식일 수 있으므로 티커만 추출
    all_defs = _extract_candidate_tickers(tuning_config.get("defense", []))
    all_offs = _extract_candidate_tickers(tuning_config.get("offense", []))
    tickers = list({settings["offense_ticker"], settings["signal_ticker"], *all_defs, *all_offs} - {"CASH"})
    price_raw = yf.download(tickers, start=warmup_start, auto_adjust=True, progress=False)
    if price_raw is None or len(price_raw) == 0:
        raise ValueError(f"가격 데이터를 받아오지 못했습니다: {tickers}")
    pre_prices = _extract_field(price_raw, "Close", tickers)
    pre_opens = _extract_field(price_raw, "Open", tickers)
    if "CASH" in all_defs:
        pre_prices["CASH"] = 1.0
        pre_opens["CASH"] = 1.0

    pre_fx = download_fx(settings, warmup_start)

    bench_raw_entries = settings.get("benchmarks", [])
    bench_tickers = []
    for b in bench_raw_entries:
        if isinstance(b, dict):
            ticker = b.get("ticker")
        else:
            ticker = str(b)
        if ticker:
            bench_tickers.append(ticker)

    if bench_tickers:
        bench_raw = yf.download(bench_tickers, start=warmup_start, auto_adjust=True, progress=False)
        if bench_raw is None or len(bench_raw) == 0:
            pre_bench = pd.DataFrame()
        else:
            pre_bench = _extract_field(bench_raw, "Close", bench_tickers)
    else:
        pre_bench = pd.DataFrame()

    return pre_prices, pre_opens, pre_fx, pre_bench


def _prefetch_data_kor(settings: dict, tuning_config: dict, warmup_start):
    """한국 시장 데이터 프리패치 (pykrx)."""
    if pykrx_stock is None:
        raise ImportError("pykrx 패키지가 설치되어 있지 않습니다. pip install pykrx")

    # offense/defense 후보가 {ticker, name} 형식일 수 있으므로 티커만 추출
    all_defs = _extract_candidate_tickers(tuning_config.get("defense", []))
    all_offs = _extract_candidate_tickers(tuning_config.get("offense", []))
    tickers = list({settings["offense_ticker"], settings["signal_ticker"], *all_defs, *all_offs} - {"CASH"})

    start_str = pd.Timestamp(warmup_start).strftime("%Y%m%d")
    end_str = pd.Timestamp.today().strftime("%Y%m%d")

    # 종가
    price_dfs = {}
    open_dfs = {}
    for ticker in tickers:
        df = pykrx_stock.get_market_ohlcv_by_date(start_str, end_str, ticker)
        if df is not None and not df.empty:
            price_dfs[ticker] = df["종가"]
            open_dfs[ticker] = df["시가"]

    if not price_dfs:
        raise ValueError(f"가격 데이터를 받아오지 못했습니다: {tickers}")

    pre_prices = pd.DataFrame(price_dfs)
    pre_prices.index = pd.to_datetime(pre_prices.index)

    pre_opens = pd.DataFrame(open_dfs)
    pre_opens.index = pd.to_datetime(pre_opens.index)
    if "CASH" in all_defs:
        pre_prices["CASH"] = 1.0
        pre_opens["CASH"] = 1.0

    # 한국은 환율 불필요 (원화 기준)
    pre_fx = download_fx(settings, warmup_start)

    # 벤치마크
    bench_raw_entries = settings.get("benchmarks", [])
    bench_tickers = []
    for b in bench_raw_entries:
        if isinstance(b, dict):
            ticker = b.get("ticker")
        else:
            ticker = str(b)
        if ticker:
            bench_tickers.append(ticker)

    if bench_tickers:
        bench_dfs = {}
        for ticker in bench_tickers:
            df = pykrx_stock.get_market_ohlcv_by_date(start_str, end_str, ticker)
            if df is not None and not df.empty:
                bench_dfs[ticker] = df["종가"]
        if bench_dfs:
            pre_bench = pd.DataFrame(bench_dfs)
            pre_bench.index = pd.to_datetime(pre_bench.index)
        else:
            pre_bench = pd.DataFrame()
    else:
        pre_bench = pd.DataFrame()

    return pre_prices, pre_opens, pre_fx, pre_bench


def run_tuning(
    tuning_config: dict,
    config_path: Path = None,
    months_range: int = 12,
    max_workers: int = None,
    optimization_metric: str = "CAGR",
    progress_cb: Callable[[int, int], None] = None,
    partial_cb: Callable[[list[dict], int, int], None] = None,
) -> tuple[list[dict], dict]:
    start_ts = datetime.now()

    # 설정은 호출자가 파일 경로로 명시해야 한다(임의 기본 경로 없음).
    if config_path is None:
        raise ValueError("run_tuning: config_path 가 필요합니다.")
    settings = load_settings(config_path)

    market = settings.get("market", "us")
    start_bound, warmup_start, end_bound = compute_bounds(settings)

    # 튜닝 시작 전 offense/defense 후보 티커 데이터 가용성 검증 (백테스트 시작일 기준)
    # - 방어 후보: 전 기간 데이터 필수 (신호 부재 시 대피처라 공백 불가) → 부족하면 중단
    # - 공격 후보: 최근 상장은 허용 — 엔진이 상장 이후 구간에서만 선택 대상으로 쓴다
    print("[데이터 검증] offense/defense 후보 티커 데이터 가용성 확인 중...")
    offense_cands = list(tuning_config.get("offense", []))
    defense_cands = list(tuning_config.get("defense", []))
    validator = _validate_candidates_kor if market == "kor" else _validate_candidates_us
    offense_entries = validator(offense_cands, start_bound)
    defense_entries = validator(defense_cands, start_bound)

    validation_errors = [line for _, line in defense_entries]
    validation_errors += [line for kind, line in offense_entries if kind == "missing"]
    late_offense = [line for kind, line in offense_entries if kind == "late"]

    if validation_errors:
        required_date = pd.Timestamp(start_bound).strftime("%Y-%m-%d")
        error_msg = (
            f"\n❌ 튜닝을 중단합니다: 일부 티커에 {required_date}부터의 데이터가 없습니다.\n"
            f"\n문제가 있는 티커:\n" + "\n".join(validation_errors) + f"\n\n해결 방법:\n"
            f"  1. 튜닝 탐색 공간에서 해당 티커를 제거하거나\n"
            f"  2. {config_path}의 months_range를 줄여서 더 최근 기간만 사용하세요.\n"
        )
        raise ValueError(error_msg)
    if late_offense:
        print("[데이터 검증] 최근 상장 공격 후보 감지 — 상장 이후 구간에서만 선택 대상으로 사용합니다:")
        for line in late_offense:
            print(line)
    print("[데이터 검증] 후보 티커 데이터 확인 완료 ✅")

    try:
        if market == "kor":
            pre_prices, pre_opens, pre_fx, pre_bench = _prefetch_data_kor(settings, tuning_config, warmup_start)
        else:
            pre_prices, pre_opens, pre_fx, pre_bench = _prefetch_data_us(settings, tuning_config, warmup_start)
    except Exception as exc:
        if _is_rate_limit_error(exc):
            raise SystemExit("yfinance YFRateLimitError: 잠시 후 다시 실행하세요.") from exc
        raise RuntimeError(f"프리패치 단계에서 데이터 로드에 실패했습니다: {exc}") from exc

    # 공격 자산은 조합 차원이 아니다 — 백테스트가 진입 시점마다 후보 중
    # ALMA 6개월 이격도 1위를 동적으로 선택한다 (라이브와 동일 규칙).
    combos: list[dict] = []
    for buy_cut in tuning_config["drawdown_buy_cutoff"]:
        for sell_cut in tuning_config["drawdown_sell_cutoff"]:
            # 히스테리시스 조건: buy_cutoff < sell_cutoff (절대값 기준)
            if buy_cut >= sell_cut:
                continue

            for def_t in tuning_config["defense"]:
                # defense_ticker가 {ticker, name} 형식이면 티커만 추출
                if isinstance(def_t, dict):
                    ticker = def_t.get("ticker", "")
                    defense_obj = def_t  # 전체 객체 저장
                else:
                    ticker = str(def_t)
                    defense_obj = {"ticker": ticker, "name": ticker}
                combos.append(
                    {
                        "drawdown_buy_cutoff": float(buy_cut),
                        "drawdown_sell_cutoff": float(sell_cut),
                        "defense_ticker": ticker,
                        "defense_name": defense_obj.get("name", ticker),
                        "_defense_obj": defense_obj,  # 전체 객체 저장 (결과에 포함용)
                    }
                )

    total_cases = len(combos)
    workers = max_workers or cpu_count() or 1
    results: list[dict] = []
    completed = 0
    next_progress = 1

    with ProcessPoolExecutor(max_workers=workers) as ex:
        future_map = {
            ex.submit(
                _run_single,
                (
                    settings,
                    overrides,
                    pre_prices,
                    pre_opens,
                    pre_fx,
                    pre_bench,
                    start_bound,
                ),
            ): overrides
            for overrides in combos
        }
        for fut in as_completed(future_map):
            try:
                res = fut.result()
                results.append(res)
            except Exception as exc:
                overrides = future_map[fut]
                if _is_rate_limit_error(exc) or _is_network_or_data_error(exc):
                    print(f"[튜닝 중단] 네트워크/데이터 오류 감지: {exc}")
                    raise SystemExit(1) from exc
                print(f"[튜닝 경고] 조합 {overrides} 실패: {exc}")
            completed += 1
            progress = int(completed / total_cases * 100)
            if progress_cb and progress >= next_progress:
                progress_cb(completed, total_cases)
                next_progress = progress + 1
            if partial_cb and progress >= next_progress - 1:
                partial_cb(results, completed, total_cases)

    # 정렬: CAGR 내림차순
    results.sort(key=lambda x: x["cagr"], reverse=True)
    months = round((end_bound - start_bound).days / 30.0, 1)
    return results, {
        "start_ts": start_ts,
        "end_ts": datetime.now(),
        "total_cases": total_cases,
        "months": months,
        "period_start": start_bound.strftime("%Y-%m-%d"),
        "period_end": end_bound.strftime("%Y-%m-%d"),
        "period_months": months_range,
    }


def render_top_table(
    results: list[dict],
    top_n: int = 100,
    months_range: int | None = None,
    defense_names: dict[str, str] | None = None,
) -> list[str]:
    if months_range:
        pr_label = f"{months_range}개월 수익률(%)"
    else:
        pr_label = "기간 수익률(%)"
    headers = [
        "defense_ticker",
        "buy_cutoff",
        "sell_cutoff",
        pr_label,
        "CAGR(%)",
        "MDD(%)",
        "Sharpe",
        "Vol(%)",
    ]
    aligns = ["right"] * len(headers)
    rows: list[list[str]] = []
    for row in results[:top_n]:
        p = row["params"]

        # defense 표시 (_defense_obj 우선, 없으면 defense_names)
        ticker = str(p.get("defense_ticker", ""))
        defense_obj = p.get("_defense_obj")
        if defense_obj and isinstance(defense_obj, dict):
            name = defense_obj.get("name", "")
        else:
            name = defense_names.get(ticker, "") if defense_names else ""
        display = f"{name}({ticker})" if name else ticker
        rows.append(
            [
                display,
                f"{p['drawdown_buy_cutoff']:.2f}",
                f"{p['drawdown_sell_cutoff']:.2f}",
                f"{row.get('period_return', 0.0) * 100:.2f}",
                f"{row['cagr'] * 100:.2f}",
                f"{row['mdd'] * 100:.2f}",
                f"{row['sharpe']:.2f}",
                f"{row['vol'] * 100:.2f}",
            ]
        )
    return render_table_eaw(headers, rows, aligns)
