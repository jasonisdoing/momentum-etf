"""사고팔기 포트폴리오 비교 — 시장별 기본 종목 묶음과, 거기에 인버스 종목을 더한 묶음을 같은 구간끼리 비교한다.

실행: ``python backtest/portfolio_sim.py`` (안 바꾸기: ``--variant 7``)

- 각 종목은 `ladder_sim` 과 같은 규칙·구간(`SIM_SINCE` 이후 3개월, 한 달씩)으로 돌린다.
- 묶음 성과 = 종목별 구간 수익률(최대 투입금 대비 %)의 단순 평균 — 종목마다 같은 금액을 배정했다고 본다.
  기본 4종목 + 인버스 1종목이면 인버스 비중은 20%다.
- 구간 국면은 시장 대표 종목(`REGIME_TICKER`)의 구간 수익률로 나눈다(`ladder_sim.REGIME_PCT`).
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ladder_sim  # noqa: E402

#: 시장 → (기본 종목들, 더할 인버스 종목, 국면 판정 대표 종목).
PORTFOLIOS: dict[str, tuple[list[str], str, str]] = {
    "국장": (["069500", "229200", "122630", "233740"], "114800", "069500"),
    "미장": (["QQQM", "QLD", "TQQQ", "SOXL"], "PSQ", "QQQM"),
}


def _pct(values: list[float]) -> str:
    return f"{statistics.mean(values):+.1f}%" if values else "-"


def compare(market: str, variant: dict, configs: dict) -> None:
    base, inverse, regime_ticker = PORTFOLIOS[market]
    results = {
        ticker: {
            row["start"]: row for row in ladder_sim.run_ticker(ticker, configs[ticker], [variant])[variant["name"]]
        }
        for ticker in [*base, inverse]
    }
    starts = sorted(set.intersection(*(set(rows) for rows in results.values())))
    rows = []
    for start in starts:
        base_pct = statistics.mean(results[t][start]["pct"] for t in base)
        inverse_pct = results[inverse][start]["pct"]
        mixed_pct = statistics.mean([*(results[t][start]["pct"] for t in base), inverse_pct])
        rows.append((start, results[regime_ticker][start]["regime"], base_pct, inverse_pct, mixed_pct))
    print(f"\n### {market} — {variant['name']} · 구간 {len(rows)}개 · 기본 {', '.join(base)} / 인버스 {inverse}")
    print("| 구간 시작 | 국면 | 기본 4종목 | 인버스 | 인버스 20% 포함 |")
    print("|---|---|---|---|---|")
    for start, regime, base_pct, inverse_pct, mixed_pct in rows:
        print(f"| {start} | {regime} | {base_pct:+.1f}% | {inverse_pct:+.1f}% | {mixed_pct:+.1f}% |")
    print("\n| 묶음 | 평균 | 최악 | 최고 | 표준편차 | 상승 | 횡보 | 하락 |")
    print("|---|---|---|---|---|---|---|---|")
    for label, index in (("기본 4종목", 2), ("인버스 단독", 3), ("인버스 20% 포함", 4)):
        values = [row[index] for row in rows]
        by_regime = [_pct([row[index] for row in rows if row[1] == name]) for name in ("상승", "횡보", "하락")]
        print(
            f"| {label} | {statistics.mean(values):+.1f}% | {min(values):+.1f}% | {max(values):+.1f}% | "
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
