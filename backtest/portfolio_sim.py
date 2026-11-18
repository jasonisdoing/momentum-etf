"""사고팔기 포트폴리오 비교 — 시장별 기본 종목 묶음에 인버스를 어떻게 더하느냐에 따라 같은 구간끼리 비교한다.

실행: ``python backtest/portfolio_sim.py`` (사고팔기 안 바꾸기: ``--variant 7``)

- 기본 종목은 `ladder_sim` 과 같은 규칙·구간(`SIM_SINCE` 이후 3개월, 한 달씩)으로 돌린다.
- 인버스는 자금의 `INVERSE_SHARE` 만큼 배정하고, 나머지는 기본 종목에 똑같이 나눈다. 인버스 운용 방식:
  사고팔기(같은 안) · 고정 보유(구간 첫날 시가에 사서 끝날 종가까지) · 슈퍼트렌드 하락일 때만 보유.
- 슈퍼트렌드는 시장추세 화면과 같은 판정(`market_trend_service._calculate_supertrend`, 일봉 ATR 10·배수 3)을
  시장 대표 지수에 쓴다. 전날 종가 판정이 하락이면 그날 시가에 사서 들고, 상승으로 바뀌면 다음 날 시가에 판다.
- 성과는 구간 수익률(배정 자금 대비 %)이고, 구간 국면은 시장 대표 종목(`REGIME_TICKER`)으로 나눈다.
"""

from __future__ import annotations

import statistics
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ladder_sim  # noqa: E402

from utils.cache_utils import load_cached_frames_bulk_from_all_ticker_types  # noqa: E402
from utils.market_trend_service import _calculate_supertrend, load_index_ohlc  # noqa: E402

INVERSE_SHARE = 0.2
#: 시장 → (기본 종목들, 인버스 종목, 국면 판정 대표 종목, 슈퍼트렌드 판정 지수).
PORTFOLIOS: dict[str, tuple[list[str], str, str, str]] = {
    "국장": (["069500", "229200", "122630", "233740"], "114800", "069500", "^KS11"),
    "미장": (["QQQM", "QLD", "TQQQ", "SOXL"], "PSQ", "QQQM", "^NDX"),
}
METHODS = ("인버스 없음", "인버스 사고팔기", "인버스 고정 보유", "인버스 슈퍼트렌드")


def _daily(ticker: str) -> pd.DataFrame:
    frame = load_cached_frames_bulk_from_all_ticker_types([ticker]).get(ticker)
    if frame is None or frame.empty:
        raise RuntimeError(f"{ticker} 일봉 캐시가 없습니다.")
    return frame[["Open", "Close"]].apply(pd.to_numeric, errors="coerce").dropna()


def _hold_pct(daily: pd.DataFrame, start: date, end: date) -> float:
    """구간 첫날 시가에 사서 마지막 날 종가까지 들고 있었을 때의 수익률(%)."""
    part = daily.loc[str(start) : str(end)]
    return (float(part["Close"].iloc[-1]) / float(part["Open"].iloc[0]) - 1) * 100


def _supertrend_pct(daily: pd.DataFrame, down: pd.Series, start: date, end: date) -> tuple[float, int]:
    """전날 지수 슈퍼트렌드가 하락인 날만 들고 있었을 때의 구간 수익률(%)과 보유 일수.

    판정이 바뀐 날의 시가에 사고팔며(사는 날은 시가→종가, 파는 날은 전날 종가→시가), 들고 있지 않은 날은 현금이다.
    """
    part = daily.loc[str(start) : str(end)]
    # 그날 시가 기준 보유 여부 = 전날(지수 일봉 기준) 판정. 지수와 종목의 휴장일이 달라도 직전 판정을 쓴다.
    held_today = down.shift(1).reindex(part.index, method="ffill").fillna(False).astype(bool)
    growth, prev_held, prev_close, days = 1.0, False, None, 0
    for day, row in part.iterrows():
        held = bool(held_today.loc[day])
        if held and prev_held:
            growth *= row["Close"] / prev_close
        elif held:
            growth *= row["Close"] / row["Open"]
        elif prev_held:
            growth *= row["Open"] / prev_close
        days += held
        prev_held, prev_close = held, row["Close"]
    return (growth - 1) * 100, days


def compare(market: str, variant: dict, configs: dict) -> None:
    base, inverse, regime_ticker, index_ticker = PORTFOLIOS[market]
    results: dict[str, dict[date, dict[str, Any]]] = {
        ticker: {
            row["start"]: row for row in ladder_sim.run_ticker(ticker, configs[ticker], [variant])[variant["name"]]
        }
        for ticker in [*base, inverse]
    }
    index = load_index_ohlc(index_ticker)
    if index is None or index.empty:
        raise RuntimeError(f"{index_ticker} 지수 일봉이 없습니다.")
    down = _calculate_supertrend(index)["direction"].eq(-1)
    inverse_daily = _daily(inverse)
    starts = sorted(set.intersection(*(set(rows) for rows in results.values())))
    rows = []
    for start in starts:
        end = results[regime_ticker][start]["end"]
        base_pct = statistics.mean(results[t][start]["pct"] for t in base)
        st_pct, st_days = _supertrend_pct(inverse_daily, down, start, end)
        inverse_pcts = {
            "인버스 사고팔기": results[inverse][start]["pct"],
            "인버스 고정 보유": _hold_pct(inverse_daily, start, end),
            "인버스 슈퍼트렌드": st_pct,
        }
        mixed = {"인버스 없음": base_pct}
        mixed.update({name: (1 - INVERSE_SHARE) * base_pct + INVERSE_SHARE * pct for name, pct in inverse_pcts.items()})
        rows.append({"start": start, "regime": results[regime_ticker][start]["regime"], "st_days": st_days, **mixed})
    print(
        f"\n### {market} — 기본 {', '.join(base)} ({variant['name']}) · 인버스 {inverse} {INVERSE_SHARE:.0%} · 판정 {index_ticker}"
    )
    print("| 구간 시작 | 국면 | " + " | ".join(METHODS) + " | 슈퍼트렌드 보유일 |")
    print("|---" * (len(METHODS) + 3) + "|")
    for row in rows:
        cells = " | ".join(f"{row[name]:+.1f}%" for name in METHODS)
        print(f"| {row['start']} | {row['regime']} | {cells} | {row['st_days']}일 |")
    print("\n| 방식 | 평균 | 최악 | 최고 | 표준편차 | 상승 | 횡보 | 하락 |")
    print("|---|---|---|---|---|---|---|---|")
    for name in METHODS:
        values = [row[name] for row in rows]
        by_regime = []
        for label in ("상승", "횡보", "하락"):
            part = [row[name] for row in rows if row["regime"] == label]
            by_regime.append(f"{statistics.mean(part):+.1f}%" if part else "-")
        print(
            f"| {name} | {statistics.mean(values):+.1f}% | {min(values):+.1f}% | {max(values):+.1f}% | "
            f"{statistics.pstdev(values):.1f}%p | {' | '.join(by_regime)} |"
        )


def main(args: list[str]) -> int:
    variant_id = args[args.index("--variant") + 1] if "--variant" in args else "7"
    variant = next((v for v in ladder_sim.VARIANTS if v["id"] == variant_id), None)
    if variant is None:
        raise RuntimeError(f"없는 안입니다: {variant_id}")
    configs = ladder_sim.ticker_configs()
    for market in PORTFOLIOS:
        compare(market, variant, configs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
