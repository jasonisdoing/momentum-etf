"""사고팔기 간격 방식 비교 — `fetch_minutes.py` 가 받아 둔 1분봉으로 안(`VARIANTS`)들을 돌려 구간별 손익을 비교한다.

실행: ``python backtest/ladder_sim.py`` (종목만: ``python backtest/ladder_sim.py QLD 069500``)

시뮬레이션 규칙(사고팔기 운용 규칙을 1분 단위로 흉내 낸다):
- 1분마다 그 분의 종가를 현재가로 본다. 매수선·매도선에 닿으면 그 현재가에 체결한다(신호 → 그 순간 가격으로 주문).
- 신호는 한 번에 하나 — 팔 회차(가장 깊은 보유)가 닿아 있으면 매도가 먼저다.
- 보유가 없으면 1호 매수 구간(`FIRST_ENTRY_WINDOWS`)의 첫 분에 산다. 전량 매도한 세션에는 다시 사지 않는다.
- 간격은 안(`VARIANTS`)이 정하고 8종목에 같게 쓴다. 회차 금액·회차 수는 지금 저장된 사고팔기 설정을 쓴다. 미장 수량은 지금 환율로 원화 금액을 환산한다.
- 사람이 버튼을 누르기까지의 지연, 수정주가로 인한 호가 어긋남은 반영하지 않는다.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
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
from utils.strategy_trade_service import _eligible_pool_kinds, load_settings  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
REGIMES = (("up", "상승"), ("side", "횡보"), ("down", "하락"))

#: 비교할 안 — 모든 종목에 같은 규칙을 쓴다. ``base`` 는 첫 간격(%) 또는 ``VOL_BASE``(1호를 산 날 전날까지의
#: 20일 변동성 — 사이클마다 정해 전량 매도까지 고정), ``step`` 은 회차마다 늘리는 간격(%p),
#: ``sell`` 은 매도 간격: fixed(첫 간격 고정) · own(그 회차의 매수 간격) · avg(2호~그 회차 매수 간격의 평균).
VOL_BASE = "vol"
VARIANTS: list[dict] = [
    {"name": "① 고정", "base": 2.5, "step": 0.0, "sell": "fixed"},
    {"name": "② 동적·동적", "base": 2.5, "step": 0.5, "sell": "own"},
    {"name": "③ 동적·매도 고정", "base": 2.5, "step": 0.5, "sell": "fixed"},
    {"name": "④ 동적·간격 평균", "base": 2.5, "step": 0.5, "sell": "avg"},
    {"name": "⑤ 1%+1%·간격 평균", "base": 1.0, "step": 1.0, "sell": "avg"},
    {"name": "⑥ 변동성 고정", "base": VOL_BASE, "step": 0.0, "sell": "fixed"},
]
#: 체결 금액 대비 수수료 — 국장 0.005%(국내 ETF 는 매도 거래세 없음). 미장 수수료율은 확인하지 못해 넣지 않는다.
FEE_RATE = {"kor": 0.00005, "us": 0.0}


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


def in_entry_window(country: str, at: datetime) -> bool:
    local = at.time()
    return any(start <= local < end for start, end in FIRST_ENTRY_WINDOWS[country])


def cycle_base(variant: dict, volatility: Any, day: date) -> float:
    """사이클(1호 매수)마다 정하는 첫 간격(%) — 변동성 안은 ``day`` 전날까지의 20일 변동성, 나머지는 안의 값."""
    if variant["base"] != VOL_BASE:
        return float(variant["base"])
    before = volatility[volatility.index < pd.Timestamp(day)].dropna()
    if before.empty:
        raise RuntimeError(f"{day} 전날까지의 20일 변동성이 없습니다.")
    return float(before.iloc[-1])


def simulate(bars: list[list], variant: dict, cfg: dict) -> dict:
    """한 구간·한 안의 결과 — 실현·평가·합계 손익(종목 통화)과 매매 횟수·최대 보유 회차·10회차 꽉 찬 날 수."""
    country, kind, rounds = cfg["country"], cfg["kind"], cfg["rounds"]
    base = None  # 이번 사이클의 첫 간격 — 1호를 살 때 정한다
    fee = FEE_RATE[country]
    held: list[tuple[int, float, int]] = []  # (회차, 체결가, 수량)
    realized, buys, sells, max_rounds = 0.0, 0, 0, 0
    full_days: set = set()
    sold_out_day = None
    price = None

    def tick(value: float) -> float:
        return round_to_tick(value, kind=kind, country=country)

    def buy(at_price: float) -> None:
        nonlocal realized, buys, max_rounds
        quantity = int(cfg["size"] // at_price)
        held.append((len(held) + 1, at_price, quantity))
        realized -= at_price * quantity * fee
        buys += 1
        max_rounds = max(max_rounds, len(held))

    for _, iso, _open, _high, _low, price in bars:
        at = datetime.fromisoformat(iso)
        day = session_trade_date(country, at)
        if not held:
            if in_entry_window(country, at) and day != sold_out_day:
                base = cycle_base(variant, cfg["volatility"], day)
                buy(price)
            continue
        round_no, fill, quantity = held[-1]
        if price >= tick(fill * (1 + sell_gap(variant, base, round_no) / 100)):
            held.pop()
            realized += (price - fill) * quantity - price * quantity * fee
            sells += 1
            if not held:
                sold_out_day = day
            continue
        if len(held) < rounds and price <= tick(fill * (1 - buy_gap(variant, base, len(held) + 1) / 100)):
            buy(price)
        if len(held) >= rounds:
            full_days.add(day)
    unrealized = sum((price - fill) * quantity for _, fill, quantity in held)
    return {
        "realized": realized,
        "unrealized": unrealized,
        "total": realized + unrealized,
        "buys": buys,
        "sells": sells,
        "max_rounds": max_rounds,
        "full_days": len(full_days),
        "end_rounds": len(held),
    }


def ticker_configs() -> dict[str, dict]:
    """종목 → 사고팔기 설정(국가·종목 종류·간격·회차 수·회차 금액(종목 통화)) — 지금 저장된 전략 설정."""
    from utils.strategy_trade_service import _pool_countries

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
            "size_krw": size_krw,
            # 순위 화면 변동성과 같은 함수 — 날짜별 20일 일간 수익률 표준편차(%).
            "volatility": daily_volatility_pct(pd.to_numeric(frame["Close"], errors="coerce")),
        }
    return configs


def _money(value: float, country: str) -> str:
    return f"{value:+,.0f}원" if country == "kor" else f"${value:+,.2f}"


def main(selected: list[str]) -> int:
    configs = ticker_configs()
    tickers = selected or sorted(path.name for path in DATA_DIR.iterdir() if path.is_dir())
    for ticker in tickers:
        if ticker not in configs:
            raise RuntimeError(f"{ticker} 는 사고팔기에 등록된 종목이 아니라 설정(간격·회차·금액)이 없습니다.")
        cfg = configs[ticker]
        windows = {regime: json.loads((DATA_DIR / ticker / f"{regime}.json").read_text()) for regime, _ in REGIMES}
        print(
            f"\n## {ticker} — {cfg['rounds']}회차 · 회차 {cfg['size_krw']:,}원"
            f" (최대 투입 {cfg['size_krw'] * cfg['rounds']:,}원)"
        )
        for regime, label in REGIMES:
            w = windows[regime]
            print(f"- {label}: {w['start']}~{w['end']} 수익률 {w['return_pct']:+.1f}% · 폭 {w['range_pct']:.1f}%")
        print(
            "| 안 | " + " | ".join(label for _, label in REGIMES) + " | 합계 | 매도 횟수(상/횡/하) | 꽉 찬 날(하락) |"
        )
        print("|---" * (len(REGIMES) + 4) + "|")
        for variant in VARIANTS:
            results = [simulate(windows[regime]["bars"], variant, cfg) for regime, _ in REGIMES]
            total = sum(r["total"] for r in results)
            cells = [_money(r["total"], cfg["country"]) for r in results]
            sells = "/".join(str(r["sells"]) for r in results)
            print(
                f"| {variant['name']} | {' | '.join(cells)} | **{_money(total, cfg['country'])}** | {sells} |"
                f" {results[-1]['full_days']}일 |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
