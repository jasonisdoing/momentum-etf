"""같은 지수의 1배와 레버리지 사고팔기 비교 — 지수 노출이 같게 맞춰 수익과 필요한 현금을 본다.

실행: ``python backtest/leverage_vs_1x.py``

- 레버리지 N배를 간격 g%·회차 금액 S 로 굴리는 것과, 1배를 간격 g/N%·회차 금액 N×S 로 굴리는 것을 비교한다
  (지수가 같은 만큼 빠질 때 같은 지수 노출을 산다). 회차 수는 둘 다 `ROUNDS`.
- 구간·체결 규칙은 `ladder_sim` 과 같다(`SIM_SINCE` 이후 3개월, 한 달씩). 간격은 고정(사고 파는 간격 같음).
- 보는 값: 구간 손익(통화), 구간 중 실제로 들어간 최대 현금(보유 원가 최대), 손익 ÷ 최대 현금.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ladder_sim  # noqa: E402

ROUNDS = 10
#: (1배 종목, 레버리지 종목, 배수, 레버리지 간격 %) — 1배 간격은 레버리지 간격 ÷ 배수.
PAIRS: list[tuple[str, str, int, float]] = [
    ("069500", "122630", 2, 5.0),
    ("QQQM", "QLD", 2, 5.0),
    ("QQQM", "TQQQ", 3, 7.5),
]


def _variant(gap: float) -> dict:
    return {"id": f"fixed{gap}", "name": f"고정 {gap:g}%", "base": gap, "step": 0.0, "sell": "fixed"}


def _rows(ticker: str, cfg: dict, gap: float) -> dict:
    variant = _variant(gap)
    return {row["start"]: row for row in ladder_sim.run_ticker(ticker, cfg, [variant])[variant["name"]]}


def main() -> int:
    configs = ladder_sim.ticker_configs()
    for one, lev, multiple, lev_gap in PAIRS:
        lev_cfg = {**configs[lev], "rounds": ROUNDS}
        # 1배는 레버리지와 같은 통화 금액 × 배수(같은 지수 노출).
        one_cfg = {**configs[one], "rounds": ROUNDS, "size": lev_cfg["size"] * multiple}
        lev_rows = _rows(lev, lev_cfg, lev_gap)
        one_rows = _rows(one, one_cfg, lev_gap / multiple)
        starts = sorted(set(lev_rows) & set(one_rows))
        unit = "원" if lev_cfg["country"] == "kor" else "$"
        money = (lambda v: f"{v:+,.0f}원") if unit == "원" else (lambda v: f"${v:+,.0f}")
        cash = (lambda v: f"{v:,.0f}원") if unit == "원" else (lambda v: f"${v:,.0f}")
        print(
            f"\n### {lev}({multiple}배, 간격 {lev_gap:g}%, 회차 {cash(lev_cfg['size'])}) 대 "
            f"{one}(1배, 간격 {lev_gap / multiple:g}%, 회차 {cash(one_cfg['size'])}) · 구간 {len(starts)}개"
        )
        print(
            "| 구간 시작 | 국면 | 레버리지 손익 | 1배 손익 | 레버리지 최대 현금 | 1배 최대 현금 | 레버리지 매도 | 1배 매도 |"
        )
        print("|---|---|---|---|---|---|---|---|")
        for start in starts:
            a, b = lev_rows[start], one_rows[start]
            print(
                f"| {start} | {b['regime']} | {money(a['pnl'])} | {money(b['pnl'])} | {cash(a['max_cost'])} | "
                f"{cash(b['max_cost'])} | {a['sells']} | {b['sells']} |"
            )
        lev_pnl = [lev_rows[s]["pnl"] for s in starts]
        one_pnl = [one_rows[s]["pnl"] for s in starts]
        lev_cash = [lev_rows[s]["max_cost"] for s in starts]
        one_cash = [one_rows[s]["max_cost"] for s in starts]
        wins = sum(1 for s in starts if one_rows[s]["pnl"] > lev_rows[s]["pnl"])
        print("\n| 구분 | 평균 손익 | 최악 손익 | 평균 최대 현금 | 가장 큰 최대 현금 | 손익 ÷ 최대 현금(평균) |")
        print("|---|---|---|---|---|---|")
        for label, pnl, used in ((f"{lev} {multiple}배", lev_pnl, lev_cash), (f"{one} 1배", one_pnl, one_cash)):
            ratio = statistics.mean(p / c * 100 for p, c in zip(pnl, used, strict=True) if c > 0)
            print(
                f"| {label} | {money(statistics.mean(pnl))} | {money(min(pnl))} | {cash(statistics.mean(used))} | "
                f"{cash(max(used))} | {ratio:+.1f}% |"
            )
        print(f"\n1배가 더 번 구간: {wins}/{len(starts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
