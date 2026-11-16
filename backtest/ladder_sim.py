"""사고팔기 간격 방식 비교 — `fetch_minutes.py` 가 받아 둔 월별 1분봉을 이어 붙여 3개월 구간을 한 달씩 밀며
안(`VARIANTS`)들을 돌리고, 종목·묶음별로 평균·최악·국면별 성과를 비교한다.

실행: ``python backtest/ladder_sim.py`` (종목만: ``python backtest/ladder_sim.py QLD 069500``)

시뮬레이션 규칙(사고팔기 운용 규칙을 1분 단위로 흉내 낸다):
- 1분마다 그 분의 종가를 현재가로 본다. 매수선·매도선에 닿으면 `FILL_DELAY_MINUTES` 뒤 분의 종가로 체결한다
  (0 이면 닿은 그 분 — 자동 주문 기준).
- 신호는 한 번에 하나 — 팔 회차(가장 깊은 보유)가 닿아 있으면 매도가 먼저다.
- 보유가 없으면 1호 매수 구간(`FIRST_ENTRY_WINDOWS`)의 첫 분에 산다. 전량 매도한 세션에는 다시 사지 않는다.
- 간격은 안이 정하고 모든 종목에 같은 규칙을 쓴다. 회차 금액·회차 수는 지금 저장된 사고팔기 설정을 쓴다.
- 성과는 구간 손익(실현 + 구간 끝 평가)을 최대 투입금(회차 금액 × 회차 수) 대비 %로 본다.
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from core.strategy.momentum.signals import daily_volatility_pct  # noqa: E402
from core.strategy.trade.rules import round_to_tick  # noqa: E402
from utils.cache_utils import load_cached_frames_bulk_from_all_ticker_types  # noqa: E402
from utils.strategy_trade_config import FIRST_ENTRY_WINDOWS, ROUND_SIZE_UNIT_KRW, TRADE_COUNTRIES  # noqa: E402
from utils.strategy_trade_market import krw_rate, session_trade_date  # noqa: E402
from utils.strategy_trade_service import _eligible_pool_kinds, _pool_countries, load_settings  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
ZONES = {"kor": "Asia/Seoul", "us": "America/New_York"}
WINDOW_DAYS = 63  # 3개월(세션 수)
STEP_DAYS = 21  # 구간을 한 달씩 민다
REGIME_PCT = 5.0  # 구간 수익률 +5% 이상 상승, −5% 이하 하락, 그 사이 횡보
FILL_DELAY_MINUTES = 0
#: 체결 금액 대비 수수료(매수·매도 각각) — 국장 0, 미장 0.07%.
FEE_RATE = {"kor": 0.0, "us": 0.0007}
#: 종목 묶음 — 결과를 묶어 평균 낸다.
GROUPS: dict[str, str] = {
    "069500": "국장 1배",
    "229200": "국장 1배",
    "122630": "국장 레버리지",
    "233740": "국장 레버리지",
    "QQQM": "미장 1배",
    "QLD": "미장 레버리지",
    "TQQQ": "미장 레버리지",
    "SOXL": "미장 레버리지",
}

#: 비교할 안 — 모든 종목에 같은 규칙을 쓴다. ``base`` 는 첫 간격(%) 또는 ``VOL_BASE``(1호를 산 날 전날까지의
#: 20일 변동성 — 사이클마다 정해 전량 매도까지 고정), ``step`` 은 회차마다 늘리는 간격(%p),
#: ``sell`` 은 매도 간격: fixed(첫 간격 고정) · own(그 회차의 매수 간격) · avg(2호~그 회차 매수 간격의 평균).
VOL_BASE = "vol"
VARIANTS: list[dict] = [
    {"name": "① 고정 2.5%", "base": 2.5, "step": 0.0, "sell": "fixed"},
    {"name": "② 동적·동적", "base": 2.5, "step": 0.5, "sell": "own"},
    {"name": "③ 동적·매도 고정", "base": 2.5, "step": 0.5, "sell": "fixed"},
    {"name": "④ 동적·간격 평균", "base": 2.5, "step": 0.5, "sell": "avg"},
    {"name": "⑤ 1%+1%·간격 평균", "base": 1.0, "step": 1.0, "sell": "avg"},
    {"name": "⑥ 변동성 고정", "base": VOL_BASE, "step": 0.0, "sell": "fixed"},
    {"name": "고정 5%", "base": 5.0, "step": 0.0, "sell": "fixed"},
    {"name": "고정 7.5%", "base": 7.5, "step": 0.0, "sell": "fixed"},
]


def buy_gap(variant: dict, base: float, round_no: int) -> float:
    """``round_no`` 호를 살 때 직전 회차 체결가 대비 하락 간격(%) — 2호부터."""
    return base + variant["step"] * (round_no - 2)


def sell_gap(variant: dict, base: float, round_no: int) -> float:
    """``round_no`` 호를 팔 때 그 회차 체결가 대비 상승 간격(%) — 1호는 첫 간격."""
    if round_no < 2 or variant["sell"] == "fixed":
        return base
    if variant["sell"] == "own":
        return buy_gap(variant, base, round_no)
    if variant["sell"] == "avg":
        return sum(buy_gap(variant, base, k) for k in range(2, round_no + 1)) / (round_no - 1)
    raise ValueError(f"알 수 없는 매도 방식입니다: {variant['sell']}")


def cycle_base(variant: dict, volatility: Any, day: date) -> float:
    """사이클(1호 매수)마다 정하는 첫 간격(%) — 변동성 안은 ``day`` 전날까지의 20일 변동성, 나머지는 안의 값."""
    if variant["base"] != VOL_BASE:
        return float(variant["base"])
    before = volatility[volatility.index < pd.Timestamp(day)].dropna()
    if before.empty:
        raise RuntimeError(f"{day} 전날까지의 20일 변동성이 없습니다.")
    return float(before.iloc[-1])


def load_ticker(ticker: str, country: str) -> dict[str, list]:
    """월별 파일을 이어 붙여 분봉별 종가·세션 날짜·1호 매수 구간 여부를 만든다."""
    files = sorted((DATA_DIR / ticker).glob("????-??.json"))
    if not files:
        raise RuntimeError(f"{ticker} 분봉 파일이 없습니다 — backtest/fetch_minutes.py 를 먼저 실행하세요.")
    bars = [bar for path in files for bar in json.loads(path.read_text())]
    stamps = pd.to_datetime([bar[0] for bar in bars], unit="ms", utc=True).tz_convert(ZONES[country])
    windows = FIRST_ENTRY_WINDOWS[country]
    return {
        "open": [bar[1] for bar in bars],
        "close": [bar[4] for bar in bars],
        "day": [session_trade_date(country, at.to_pydatetime()) for at in stamps],
        "entry": [any(start <= at.time() < end for start, end in windows) for at in stamps],
    }


def session_windows(days: list[date]) -> list[tuple[int, int]]:
    """세션 63개짜리 구간을 21세션씩 밀며 — [(시작 분봉 위치, 끝 분봉 위치(포함 안 함))]."""
    starts: list[int] = [0] + [i for i in range(1, len(days)) if days[i] != days[i - 1]]
    bounds = [*starts, len(days)]
    return [(bounds[i], bounds[i + WINDOW_DAYS]) for i in range(0, len(starts) - WINDOW_DAYS + 1, STEP_DAYS)]


def simulate(data: dict[str, list], lo: int, hi: int, variant: dict, cfg: dict) -> dict:
    """한 구간·한 안의 손익(최대 투입금 대비 %)과 매도 횟수."""
    country, kind, rounds = cfg["country"], cfg["kind"], cfg["rounds"]
    close, days, entry = data["close"], data["day"], data["entry"]
    fee = FEE_RATE[country]
    held: list[tuple[int, float, int]] = []  # (회차, 체결가, 수량)
    realized, sells, base, sold_out_day = 0.0, 0, 0.0, None

    def tick(value: float) -> float:
        return round_to_tick(value, kind=kind, country=country)

    def fill_at(i: int) -> int:
        return min(i + FILL_DELAY_MINUTES, hi - 1)

    i = lo
    while i < hi:
        price, day = close[i], days[i]
        if not held:
            if entry[i] and day != sold_out_day:
                base = cycle_base(variant, cfg["volatility"], day)
                j = fill_at(i)
                quantity = int(cfg["size"] // close[j])
                held.append((1, close[j], quantity))
                realized -= close[j] * quantity * fee
                i = j
            i += 1
            continue
        round_no, fill, quantity = held[-1]
        if price >= tick(fill * (1 + sell_gap(variant, base, round_no) / 100)):
            j = fill_at(i)
            held.pop()
            realized += (close[j] - fill) * quantity - close[j] * quantity * fee
            sells += 1
            if not held:
                sold_out_day = days[j]
            i = j + 1
            continue
        if len(held) < rounds and price <= tick(fill * (1 - buy_gap(variant, base, len(held) + 1) / 100)):
            j = fill_at(i)
            quantity = int(cfg["size"] // close[j])
            held.append((len(held) + 1, close[j], quantity))
            realized -= close[j] * quantity * fee
            i = j
        i += 1
    last = close[hi - 1]
    unrealized = sum((last - fill) * quantity for _, fill, quantity in held)
    return {"pct": (realized + unrealized) / (cfg["size"] * rounds) * 100, "sells": sells}


def ticker_configs() -> dict[str, dict]:
    """종목 → 사고팔기 설정(국가·종목 종류·회차 수·회차 금액(종목 통화)·날짜별 20일 변동성)."""
    kinds = _eligible_pool_kinds(TRADE_COUNTRIES)
    countries = _pool_countries(TRADE_COUNTRIES)
    entries = {entry["ticker"]: entry for entry in reversed(list(load_settings()["strategies"].values()))}
    frames = load_cached_frames_bulk_from_all_ticker_types(list(entries))
    configs: dict[str, dict] = {}
    for ticker, entry in entries.items():
        country = countries[entry["pool"]]
        size_krw = int(entry["round_size"]) * ROUND_SIZE_UNIT_KRW
        frame = frames.get(ticker)
        if frame is None or frame.empty:
            raise RuntimeError(f"{ticker} 일봉 캐시가 없어 변동성을 계산할 수 없습니다.")
        configs[ticker] = {
            "country": country,
            "kind": kinds[entry["pool"]],
            "rounds": int(entry["rounds"]),
            "size": size_krw / krw_rate(country),
            # 순위 화면 변동성과 같은 함수 — 날짜별 20일 일간 수익률 표준편차(%).
            "volatility": daily_volatility_pct(pd.to_numeric(frame["Close"], errors="coerce")),
        }
    return configs


def regime_of(data: dict[str, list], lo: int, hi: int) -> str:
    change = (data["close"][hi - 1] / data["open"][lo] - 1) * 100
    return "상승" if change >= REGIME_PCT else "하락" if change <= -REGIME_PCT else "횡보"


def run_ticker(ticker: str, cfg: dict) -> dict[str, list[dict]]:
    """종목 하나 — 안 이름 → 구간별 결과 [{pct, sells, regime, start}]."""
    data = load_ticker(ticker, cfg["country"])
    results: dict[str, list[dict]] = {variant["name"]: [] for variant in VARIANTS}
    for lo, hi in session_windows(data["day"]):
        regime = regime_of(data, lo, hi)
        for variant in VARIANTS:
            result = simulate(data, lo, hi, variant, cfg)
            results[variant["name"]].append({**result, "regime": regime, "start": data["day"][lo]})
    return results


def _avg(values: list[float]) -> str:
    return f"{statistics.mean(values):+.1f}%" if values else "-"


def summary_table(title: str, per_variant: dict[str, list[dict]]) -> None:
    """안별 평균·중앙값·최악·국면별 평균·평균 순위(구간마다 안끼리 순위)."""
    names = list(per_variant)
    count = len(per_variant[names[0]])
    ranks: dict[str, list[int]] = {name: [] for name in names}
    for k in range(count):
        order = sorted(names, key=lambda name: -per_variant[name][k]["pct"])
        for rank, name in enumerate(order, start=1):
            ranks[name].append(rank)
    regimes = [row["regime"] for row in per_variant[names[0]]]
    print(
        f"\n### {title} — 구간 {count}개 (상승 {regimes.count('상승')} · 횡보 {regimes.count('횡보')} · 하락 {regimes.count('하락')})"
    )
    print("| 안 | 평균 | 중앙값 | 최악 | 상승 | 횡보 | 하락 | 평균 순위 | 구간당 매도 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name in names:
        rows = per_variant[name]
        pcts = [row["pct"] for row in rows]
        by_regime = [_avg([row["pct"] for row in rows if row["regime"] == label]) for label in ("상승", "횡보", "하락")]
        print(
            f"| {name} | {statistics.mean(pcts):+.1f}% | {statistics.median(pcts):+.1f}% | {min(pcts):+.1f}% | "
            f"{' | '.join(by_regime)} | {statistics.mean(ranks[name]):.2f} | {statistics.mean(row['sells'] for row in rows):.0f} |"
        )


def main(selected: list[str]) -> int:
    configs = ticker_configs()
    tickers = selected or sorted(path.name for path in DATA_DIR.iterdir() if path.is_dir())
    grouped: dict[str, dict[str, list[dict]]] = {}
    for ticker in tickers:
        if ticker not in configs:
            raise RuntimeError(f"{ticker} 는 사고팔기에 등록된 종목이 아니라 설정(회차·금액)이 없습니다.")
        results = run_ticker(ticker, configs[ticker])
        summary_table(ticker, results)
        group = grouped.setdefault(GROUPS[ticker], {name: [] for name in results})
        for name, rows in results.items():
            group[name].extend(rows)
    for title, results in grouped.items():
        summary_table(f"묶음: {title}", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
