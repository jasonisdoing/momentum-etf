"""사고팔기 일봉 백테스트 — 종목 하나의 사다리 매매를 봉 단위로 재현한다.

규칙 계산은 `rules` 와 같은 함수를 쓴다(운용 현황 화면과 갈리지 않게). 일봉만 있어 체결은 이렇게 본다.

- 매수·매도 지정가는 그날 저가·고가가 닿았는지로 판정한다. 시가가 지정가를 갭으로 넘었으면
  지정가가 아니라 **시가에 체결**한다.
- 보유가 없는데 장중 1호 진입가에 닿지 못하면 15:00 매수 규칙에 따라 **그날 종가**에 산다
  (일봉으로는 15:00 가격을 알 수 없어 종가로 근사한다). 이 시장가 매수에만 슬리피지를 적용한다.
- 매도는 봉이 시작될 때 이미 보유한 회차만 판다 — 그날 산 회차는 다음 봉부터 판다.
- 매도가 있었던 날은 다시 사지 않는다(같은 봉 안의 순서를 알 수 없어 왕복 매매를 만들지 않는다).
  전량 매도 뒤 1호 진입은 다음 봉에서 그날 직전 종가 기준으로 다시 시작한다.
- 전략마다 매수 한도(`LadderSpec.budget`, 원)가 있어, 사고 나면 이미 산 금액이 한도를 넘는 회차는
  사지 않는다(1호 포함). 매도는 한도와 무관하다.
- 수수료·세금은 반영하지 않는다(ETF 는 매도세가 없고 수수료는 미미하다).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from core.strategy.trade import rules

# 안 판 회차의 사유 — `utils.trade_stats.OPEN_REASON` 과 같은 값이어야 통계가 보유중으로 센다.
OPEN_REASON = "보유중"
SELL_REASON = "목표가 매도"


@dataclass(frozen=True)
class LadderSpec:
    ticker: str
    name: str
    trigger_pct: float
    quantity: int
    rounds: int
    # 이 전략이 쓸 수 있는 매수 한도(원) — 합성 슬리브 비중에서 온다.
    budget: float


def _trade_row(
    spec: LadderSpec,
    round_no: int,
    entry_date: pd.Timestamp,
    entry_price: float,
    exit_date: pd.Timestamp | None,
    exit_price: float,
    reason: str,
    last_date: pd.Timestamp,
) -> dict[str, Any]:
    end = exit_date if exit_date is not None else last_date
    return {
        "ticker": spec.ticker,
        "name": spec.name,
        "round": round_no,
        "quantity": spec.quantity,
        "entry_date": str(entry_date.date()),
        "entry_price": round(entry_price, 2),
        "exit_date": str(exit_date.date()) if exit_date is not None else None,
        "exit_price": round(exit_price, 2),
        "return_pct": round((exit_price / entry_price - 1.0) * 100.0, 2),
        "days": int((end - entry_date).days),
        "reason": reason,
    }


def simulate_ladder(bars: pd.DataFrame, spec: LadderSpec, *, market_buy_slippage_pct: float) -> dict[str, Any]:
    """종목 하나의 사다리 매매를 재현한다.

    ``bars`` 는 Open·High·Low·Close 일봉이고 **첫 봉은 기준 봉**이다 — 직전 종가만 대고 거래하지 않는다.
    반환: ``{"trades": [...], "pnl": 날짜별 누적 손익(원) Series}``.
    """
    if len(bars) < 2:
        raise ValueError(f"{spec.name}({spec.ticker}) 가격 데이터가 부족합니다.")

    held: dict[int, tuple[pd.Timestamp, float]] = {}  # 회차 → (체결일, 체결가)
    first_entry: float | None = None
    realized = 0.0
    trades: list[dict[str, Any]] = []
    pnl: dict[pd.Timestamp, float] = {}
    pct = spec.trigger_pct
    last_date = bars.index[-1]

    closes = bars["Close"].to_numpy(dtype=float)
    for position in range(1, len(bars)):
        date = bars.index[position]
        open_, high, low, close = (float(bars[column].iloc[position]) for column in ("Open", "High", "Low", "Close"))
        prev_close = float(closes[position - 1])

        # 1) 매도 — 봉이 시작될 때 이미 가진 회차만. 깊은(싼) 회차의 목표가가 가장 낮아 먼저 닿는다.
        sold_today = False
        for round_no in sorted(held, reverse=True):
            entry_date, entry_fill = held[round_no]
            limit = rules.sell_limit(rules.entry_price(first_entry, round_no, pct), pct)
            if high >= limit:
                exit_price = max(open_, limit)
                realized += spec.quantity * (exit_price - entry_fill)
                trades.append(
                    _trade_row(spec, round_no, entry_date, entry_fill, date, exit_price, SELL_REASON, last_date)
                )
                del held[round_no]
                sold_today = True
        if not held:
            first_entry = None

        # 2) 매수 — 매도가 있었던 날은 쉰다. 한도를 넘는 회차부터는 사지 않는다.
        if not sold_today:
            limit_filled = True
            if not held:
                limit = rules.first_buy_limit(prev_close, pct)
                if low <= limit:
                    fill = min(open_, limit)
                else:
                    # 장중에 못 닿았으면 15:00 에 사는 규칙 — 일봉에서는 종가에 시장가로 산다.
                    fill = close * (1.0 + market_buy_slippage_pct / 100.0)
                    limit_filled = False
                if rules.within_budget(0.0, fill, spec.quantity, spec.budget):
                    first_entry = fill
                    held[1] = (date, fill)
                else:
                    limit_filled = False
            if held and limit_filled:
                invested = sum(spec.quantity * fill for _, fill in held.values())
                round_no = max(held) + 1
                while round_no <= spec.rounds:
                    limit = rules.buy_limit(first_entry, round_no, pct)
                    fill = min(open_, limit)
                    if low > limit or not rules.within_budget(invested, fill, spec.quantity, spec.budget):
                        break
                    held[round_no] = (date, fill)
                    invested += spec.quantity * fill
                    round_no += 1

        unrealized = sum(spec.quantity * (close - fill) for _, fill in held.values())
        pnl[date] = realized + unrealized

    last_close = float(closes[-1])
    for round_no, (entry_date, entry_fill) in sorted(held.items()):
        trades.append(_trade_row(spec, round_no, entry_date, entry_fill, None, last_close, OPEN_REASON, last_date))
    return {"trades": trades, "pnl": pd.Series(pnl, dtype="float64")}
