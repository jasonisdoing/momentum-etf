"""1배 → 2배 이어 사기 사다리 비교 — 1배를 다 쓰면 같은 지수의 2배로 이어 사서 회차를 늘린다.

실행: ``python backtest/chain_sim.py``

비교하는 안(같은 구간, `ladder_sim.SIM_SINCE` 이후 3개월·한 달씩):
- 1배만 10회: 1배 `SIZE_1X` × 10, 간격 `GAP_1X`.
- 1배 10회 → 2배 10회: 1배를 다 사면 1배 기준 다음 매수선에서 2배 첫 회차를 사고, 그 뒤 2배 간격 `GAP_2X`·금액
  `SIZE_2X` 로 이어 산다. 2배 금액·간격은 지수 기준으로 1배와 같은 노출·같은 간격이다.
- 1배만 20회: 1배 `SIZE_1X` × 20, 간격 `GAP_1X`.
매도는 가장 깊은 회차부터 그 종목 가격으로(체결가 + 그 회차 간격). 체결은 닿은 그 분의 종가, 수수료는 국장 0.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ladder_sim  # noqa: E402

ONE_X, TWO_X = "069500", "122630"
SIZE_1X, SIZE_2X = 1_000_000, 500_000
GAP_1X, GAP_2X = 2.5, 5.0
COUNTRY, KIND = "kor", "etf"


def _tick(value: float) -> float:
    return ladder_sim.round_to_tick(value, kind=KIND, country=COUNTRY)


def simulate(one: dict, two_close: list[float], lo: int, hi: int, rounds_1x: int, rounds_2x: int) -> dict:
    """한 구간 — 손익(원)·최대 투입 현금·매도 횟수·가장 깊이 간 회차 수."""
    close, days, entry = one["close"], one["day"], one["entry"]
    held: list[tuple[str, float, int, float]] = []  # (종목, 체결가, 수량, 간격%)
    realized, sells, max_cost, deepest, sold_out_day = 0.0, 0, 0.0, 0, None

    def buy(kind: str, price: float, size: float, gap: float) -> None:
        nonlocal max_cost, deepest
        held.append((kind, price, int(size // price), gap))
        max_cost = max(max_cost, sum(p * q for _, p, q, _ in held))
        deepest = max(deepest, len(held))

    for i in range(lo, hi):
        prices = {ONE_X: close[i], TWO_X: two_close[i]}
        if not held:
            if entry[i] and days[i] != sold_out_day:
                buy(ONE_X, prices[ONE_X], SIZE_1X, GAP_1X)
            continue
        kind, fill, quantity, gap = held[-1]
        if prices[kind] >= _tick(fill * (1 + gap / 100)):
            held.pop()
            realized += (prices[kind] - fill) * quantity
            sells += 1
            if not held:
                sold_out_day = days[i]
            continue
        count_1x = sum(1 for lot in held if lot[0] == ONE_X)
        count_2x = len(held) - count_1x
        if count_1x < rounds_1x and count_2x == 0:
            if prices[ONE_X] <= _tick(fill * (1 - GAP_1X / 100)):
                buy(ONE_X, prices[ONE_X], SIZE_1X, GAP_1X)
        elif count_2x == 0 and rounds_2x:
            # 1배를 다 썼다 — 1배 기준 다음 매수선에 닿으면 2배 첫 회차를 산다.
            if prices[ONE_X] <= _tick(fill * (1 - GAP_1X / 100)):
                buy(TWO_X, prices[TWO_X], SIZE_2X, GAP_2X)
        elif 0 < count_2x < rounds_2x and prices[TWO_X] <= _tick(fill * (1 - GAP_2X / 100)):
            buy(TWO_X, prices[TWO_X], SIZE_2X, GAP_2X)
    unrealized = sum(((close[hi - 1] if k == ONE_X else two_close[hi - 1]) - p) * q for k, p, q, _ in held)
    return {"pnl": realized + unrealized, "max_cost": max_cost, "sells": sells, "deepest": deepest}


def main() -> int:
    one = ladder_sim.load_ticker(ONE_X, COUNTRY)
    two = ladder_sim.load_ticker(TWO_X, COUNTRY)
    # 2배 가격을 1배 분봉 시각에 맞춘다(같은 분이 없으면 직전 값).
    two_by_ms = dict(zip(two["ms"], two["close"], strict=True))
    two_close, last = [], None
    for ms in one["ms"]:
        last = two_by_ms.get(ms, last)
        if last is None:
            raise RuntimeError("2배 종목의 분봉이 1배보다 늦게 시작합니다.")
        two_close.append(last)
    plans = {
        "1배 10회": (10, 0),
        "1배 10회 → 2배 10회": (10, 10),
        "1배 20회": (20, 0),
    }
    windows = ladder_sim.session_windows(one["day"])
    results = {name: [simulate(one, two_close, lo, hi, *plan) for lo, hi in windows] for name, plan in plans.items()}
    print(
        f"### {ONE_X}(1배 {GAP_1X:g}%·{SIZE_1X:,}원) → {TWO_X}(2배 {GAP_2X:g}%·{SIZE_2X:,}원) · 구간 {len(windows)}개"
    )
    print("| 구간 시작 | 국면 | " + " | ".join(plans) + " | 이어 사기 최대 회차 |")
    print("|---" * (len(plans) + 3) + "|")
    for k, (lo, hi) in enumerate(windows):
        cells = " | ".join(f"{results[name][k]['pnl']:+,.0f}원" for name in plans)
        chain = results["1배 10회 → 2배 10회"][k]["deepest"]
        print(f"| {one['day'][lo]} | {ladder_sim.regime_of(one, lo, hi)} | {cells} | {chain}회 |")
    print(
        "\n| 안 | 평균 손익 | 최악 손익 | 평균 최대 현금 | 가장 큰 최대 현금 | 손익 ÷ 최대 현금(평균) | 구간당 매도 |"
    )
    print("|---|---|---|---|---|---|---|")
    for name in plans:
        rows = results[name]
        pnl = [r["pnl"] for r in rows]
        ratio = statistics.mean(r["pnl"] / r["max_cost"] * 100 for r in rows if r["max_cost"] > 0)
        print(
            f"| {name} | {statistics.mean(pnl):+,.0f}원 | {min(pnl):+,.0f}원 | "
            f"{statistics.mean(r['max_cost'] for r in rows):,.0f}원 | {max(r['max_cost'] for r in rows):,.0f}원 | "
            f"{ratio:+.1f}% | {statistics.mean(r['sells'] for r in rows):.0f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
