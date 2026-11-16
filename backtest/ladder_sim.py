"""사고팔기 간격 방식 비교 — `fetch_minutes.py` 가 받아 둔 월별 1분봉을 이어 붙여 3개월 구간을 한 달씩 밀며
안(`VARIANTS`)들을 돌리고, 종목·묶음별로 평균·최악·국면별 성과를 비교한다.

실행: ``python backtest/ladder_sim.py`` (종목·안 고르기: ``python backtest/ladder_sim.py QLD 069500 --variants 1,6,7``)

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

import numpy as np
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
# 이 날 이후에 시작하는 구간만 돌린다 — 2025년은 상승장 비중이 커서 뺀다(분봉은 그 전부터 받아 둔다).
SIM_SINCE = date(2026, 1, 1)
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
    "114800": "국장 인버스",
    "QQQM": "미장 1배",
    "QLD": "미장 레버리지",
    "TQQQ": "미장 레버리지",
    "SOXL": "미장 레버리지",
    "PSQ": "미장 인버스",
}

#: 비교할 안 — 모든 종목에 같은 규칙을 쓴다. ``base`` 는 첫 간격(%) 또는
#: ``VOL_BASE``(1호를 산 날 전날까지의 20일 변동성 — 사이클마다 정해 전량 매도까지 고정) 또는
#: ``MDD_BASE``(회차를 살 때마다 전날까지 3개월 MDD 를 10회차가 덮는 간격 — `mdd_gap`). MDD 안은 ``bounds`` 로
#: 간격 범위를 정한다 — (최소, 최대) 고정값 또는 {배수: 최소}(최대는 최소 × `LEVERAGE_GAP_MAX_MULT`, `gap_bounds`).
#: ``id`` 는 ``--variants`` 로 돌릴 안을 고를 때 쓴다.
#: ``step`` 은 회차마다 늘리는 간격(%p), ``sell`` 은 매도 간격: fixed(첫 간격 고정) · own(그 회차의 매수 간격)
#: · avg(2호~그 회차 매수 간격의 평균). 매도 간격은 회차를 살 때 정해 그 회차에 저장한다.
VOL_BASE = "vol"
MDD_BASE = "mdd"
# MDD 안은 ``mdd_days``(최근 몇 거래일의 고점 이후 최대 하락)를 안마다 정한다 — 3개월 63, 6개월 126.
#: 종목 레버리지 배수 — 배수별 간격 범위를 쓰는 MDD 안(⑧·⑨)에 쓴다.
LEVERAGE: dict[str, int] = {
    "069500": 1,
    "229200": 1,
    "122630": 2,
    "233740": 2,
    "QQQM": 1,
    "QLD": 2,
    "TQQQ": 3,
    "SOXL": 3,
    "114800": 1,
    "PSQ": 1,
}
LEVERAGE_GAP_MAX_MULT = 3.0
VARIANTS: list[dict] = [
    {"id": "1", "name": "① 고정 2.5%", "base": 2.5, "step": 0.0, "sell": "fixed"},
    {"id": "2", "name": "② 동적·동적", "base": 2.5, "step": 0.5, "sell": "own"},
    {"id": "3", "name": "③ 동적·매도 고정", "base": 2.5, "step": 0.5, "sell": "fixed"},
    {"id": "4", "name": "④ 동적·간격 평균", "base": 2.5, "step": 0.5, "sell": "avg"},
    {"id": "5", "name": "⑤ 1%+1%·간격 평균", "base": 1.0, "step": 1.0, "sell": "avg"},
    {"id": "6", "name": "⑥ 변동성 고정", "base": VOL_BASE, "step": 0.0, "sell": "fixed"},
    {"id": "f5", "name": "고정 5%", "base": 5.0, "step": 0.0, "sell": "fixed"},
    {"id": "f7.5", "name": "고정 7.5%", "base": 7.5, "step": 0.0, "sell": "fixed"},
    {
        "id": "7",
        "name": "⑦ MDD 6개월 회차별",
        "base": MDD_BASE,
        "step": 0.0,
        "sell": "own",
        "bounds": (2.0, 10.0),
        "mdd_days": 126,
    },
    {
        "id": "7q",
        "name": "⑦ MDD 3개월(이전)",
        "base": MDD_BASE,
        "step": 0.0,
        "sell": "own",
        "bounds": (2.0, 10.0),
        "mdd_days": 63,
    },
    {
        "id": "8",
        "name": "⑧ MDD·배수 2/4/6%",
        "base": MDD_BASE,
        "step": 0.0,
        "sell": "own",
        "bounds": {1: 2.0, 2: 4.0, 3: 6.0},
        "mdd_days": 63,
    },
    {
        "id": "9",
        "name": "⑨ MDD·배수 2/3/5%",
        "base": MDD_BASE,
        "step": 0.0,
        "sell": "own",
        "bounds": {1: 2.0, 2: 3.0, 3: 5.0},
        "mdd_days": 63,
    },
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


def value_before(series: Any, day: date, label: str) -> float:
    """``day`` 전날까지의 마지막 값 — 그날 장중 값은 쓰지 않는다."""
    before = series[series.index < pd.Timestamp(day)].dropna()
    if before.empty:
        raise RuntimeError(f"{day} 전날까지의 {label} 값이 없습니다.")
    return float(before.iloc[-1])


def cycle_base(variant: dict, cfg: dict, day: date) -> float:
    """사이클(1호 매수)마다 정하는 첫 간격(%) — 변동성·MDD 안은 ``day`` 전날까지의 값, 나머지는 안의 값."""
    if variant["base"] == VOL_BASE:
        return cfg["vol_by_day"][day]
    if variant["base"] == MDD_BASE:
        return mdd_gap(variant, cfg, day)
    return float(variant["base"])


def gap_bounds(variant: dict, cfg: dict) -> tuple[float, float]:
    """MDD 안의 간격 범위(%) — (최소, 최대) 고정값이거나, 종목 배수별 최소 ~ 그 `LEVERAGE_GAP_MAX_MULT` 배."""
    bounds = variant["bounds"]
    if isinstance(bounds, dict):
        if cfg["leverage"] not in bounds:
            raise RuntimeError(
                f"레버리지 배수 {cfg['leverage']} 의 최소 간격이 없습니다 — LEVERAGE·bounds 를 확인하세요."
            )
        low = bounds[cfg["leverage"]]
        return low, low * LEVERAGE_GAP_MAX_MULT
    return bounds


def mdd_gap(variant: dict, cfg: dict, day: date) -> float:
    """전날까지 3개월 MDD 를 (회차 수 − 1)번의 간격이 복리로 덮는 간격(%) — 안의 범위(`gap_bounds`)로 자른다."""
    mdd = cfg["mdd_by_day"][variant["mdd_days"]][day]
    gap = (1 - (1 - mdd) ** (1 / (cfg["rounds"] - 1))) * 100
    low, high = gap_bounds(variant, cfg)
    return min(max(gap, low), high)


def rolling_mdd(close: Any, window: int) -> Any:
    """날짜별 최근 ``window`` 거래일 안의 고점 이후 최대 하락(0~1) — 기간이 모자란 앞쪽은 NaN."""
    values = close.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for t in range(window - 1, len(values)):
        part = values[t - window + 1 : t + 1]
        out[t] = float(np.max(1 - part / np.maximum.accumulate(part)))
    return pd.Series(out, index=close.index)


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
    """`SIM_SINCE` 이후 세션부터 63개짜리 구간을 21세션씩 밀며 — [(시작 분봉 위치, 끝 분봉 위치(포함 안 함))]."""
    starts: list[int] = [i for i in range(len(days)) if (i == 0 or days[i] != days[i - 1]) and days[i] >= SIM_SINCE]
    bounds = [*starts, len(days)]
    return [(bounds[i], bounds[i + WINDOW_DAYS]) for i in range(0, len(starts) - WINDOW_DAYS + 1, STEP_DAYS)]


def simulate(data: dict[str, list], lo: int, hi: int, variant: dict, cfg: dict) -> dict:
    """한 구간·한 안의 손익(최대 투입금 대비 %)과 매도 횟수."""
    country, kind, rounds = cfg["country"], cfg["kind"], cfg["rounds"]
    close, days, entry = data["close"], data["day"], data["entry"]
    fee = FEE_RATE[country]
    is_mdd = variant["base"] == MDD_BASE
    held: list[tuple[float, int, float]] = []  # (체결가, 수량, 매도 간격%) — 매도 간격은 살 때 정한다
    realized, sells, base, sold_out_day = 0.0, 0, 0.0, None
    # 매도선·매수선은 보유가 바뀌거나 날짜가 바뀔 때만 다시 정한다(MDD 간격은 날짜마다 바뀐다).
    sell_line, buy_line, gap, lines_day = 0.0, None, 0.0, None

    def tick(value: float) -> float:
        return round_to_tick(value, kind=kind, country=country)

    def fill_at(i: int) -> int:
        return min(i + FILL_DELAY_MINUTES, hi - 1)

    def buy(j: int, sell_pct: float) -> None:
        nonlocal realized
        quantity = int(cfg["size"] // close[j])
        held.append((close[j], quantity, sell_pct))
        realized -= close[j] * quantity * fee

    i = lo
    while i < hi:
        price, day = close[i], days[i]
        if not held:
            if entry[i] and day != sold_out_day:
                base = cycle_base(variant, cfg, day)
                j = fill_at(i)
                buy(j, base)
                lines_day, i = None, j
            i += 1
            continue
        if lines_day != day:
            fill, _, own_sell = held[-1]
            next_round = len(held) + 1
            gap = mdd_gap(variant, cfg, day) if is_mdd else buy_gap(variant, base, next_round)
            sell_line = tick(fill * (1 + own_sell / 100))
            buy_line = tick(fill * (1 - gap / 100)) if next_round <= rounds else None
            lines_day = day
        if price >= sell_line:
            j = fill_at(i)
            fill, quantity, _ = held.pop()
            realized += (close[j] - fill) * quantity - close[j] * quantity * fee
            sells += 1
            if not held:
                sold_out_day = days[j]
            lines_day, i = None, j + 1
            continue
        if buy_line is not None and price <= buy_line:
            j = fill_at(i)
            buy(j, gap if is_mdd else sell_gap(variant, base, len(held) + 1))
            lines_day, i = None, j
        i += 1
    last = close[hi - 1]
    unrealized = sum((last - fill) * quantity for fill, quantity, _ in held)
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
            "leverage": LEVERAGE[ticker] if ticker in LEVERAGE else None,
            # 안들이 쓰는 MDD 기간마다 날짜별 MDD.
            "mdd": {
                days: rolling_mdd(pd.to_numeric(frame["Close"], errors="coerce").dropna(), days)
                for days in {v["mdd_days"] for v in VARIANTS if v["base"] == MDD_BASE}
            },
        }
    return configs


def regime_of(data: dict[str, list], lo: int, hi: int) -> str:
    change = (data["close"][hi - 1] / data["open"][lo] - 1) * 100
    return "상승" if change >= REGIME_PCT else "하락" if change <= -REGIME_PCT else "횡보"


def day_lookups(cfg: dict, days: list[date]) -> dict:
    """세션 날짜별 '전날까지의' 20일 변동성·MDD(기간별) — 분봉마다 찾지 않도록 날짜당 한 번 계산한다."""
    unique = sorted(set(days))
    return {
        **cfg,
        "vol_by_day": {day: value_before(cfg["volatility"], day, "20일 변동성") for day in unique},
        "mdd_by_day": {
            window: {day: value_before(series, day, f"{window}거래일 MDD") for day in unique}
            for window, series in cfg["mdd"].items()
        },
    }


def run_ticker(ticker: str, cfg: dict, variants: list[dict]) -> dict[str, list[dict]]:
    """종목 하나 — 안 이름 → 구간별 결과 [{pct, sells, regime, start}]."""
    data = load_ticker(ticker, cfg["country"])
    cfg = day_lookups(cfg, data["day"])
    results: dict[str, list[dict]] = {variant["name"]: [] for variant in variants}
    for lo, hi in session_windows(data["day"]):
        regime = regime_of(data, lo, hi)
        for variant in variants:
            result = simulate(data, lo, hi, variant, cfg)
            results[variant["name"]].append({**result, "regime": regime, "start": data["day"][lo]})
    return results


def _avg(values: list[float]) -> str:
    return f"{statistics.mean(values):+.1f}%" if values else "-"


def summary_table(title: str, per_variant: dict[str, list[dict]]) -> None:
    """안별 평균·중앙값·최악·국면별 평균·평균 순위(구간마다 안끼리 순위)."""
    names = list(per_variant)
    count = len(per_variant[names[0]])
    ranks: dict[str, list[float]] = {name: [] for name in names}
    for k in range(count):
        # 같은 손익(소수 넷째 자리까지)은 같은 순위 — 동점들이 차지하는 순위의 평균을 준다.
        scores = {name: round(per_variant[name][k]["pct"], 4) for name in names}
        for name in names:
            better = sum(1 for other in names if scores[other] > scores[name])
            tied = sum(1 for other in names if scores[other] == scores[name])
            ranks[name].append(better + (tied + 1) / 2)
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


def main(args: list[str]) -> int:
    """인자: 종목들(없으면 데이터가 있는 전 종목), ``--variants 1,6,7`` (없으면 전 안)."""
    variant_ids = None
    if "--variants" in args:
        at = args.index("--variants")
        variant_ids = args[at + 1].split(",")
        args = args[:at] + args[at + 2 :]
    variants = VARIANTS if variant_ids is None else [v for v in VARIANTS if v["id"] in variant_ids]
    unknown = set(variant_ids or []) - {v["id"] for v in VARIANTS}
    if unknown:
        raise RuntimeError(f"없는 안입니다: {', '.join(sorted(unknown))}")
    configs = ticker_configs()
    tickers = args or sorted(path.name for path in DATA_DIR.iterdir() if path.is_dir())
    grouped: dict[str, dict[str, list[dict]]] = {}
    for ticker in tickers:
        if ticker not in configs:
            raise RuntimeError(f"{ticker} 는 사고팔기에 등록된 종목이 아니라 설정(회차·금액)이 없습니다.")
        results = run_ticker(ticker, configs[ticker], variants)
        summary_table(ticker, results)
        group = grouped.setdefault(GROUPS[ticker], {name: [] for name in results})
        for name, rows in results.items():
            group[name].extend(rows)
    for title, results in grouped.items():
        summary_table(f"묶음: {title}", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
