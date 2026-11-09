"""사고팔기 일봉 백테스트 — 종목 하나의 사다리 매매를 봉 단위로 재현한다.

규칙 계산은 `rules` 와 같은 함수를 쓴다(운용 현황 화면과 갈리지 않게). 일봉만 있어 체결은 이렇게 본다.

- 매수·매도 지정가는 그날 저가·고가가 닿았는지로 판정한다. 시가가 지정가를 갭으로 넘었으면
  지정가가 아니라 **시가에 체결**한다.
- 보유가 없는데 장중 1호 진입가에 닿지 못하면 15:00 매수 규칙에 따라 **그날 종가**에 산다
  (일봉으로는 15:00 가격을 알 수 없어 종가로 근사한다). 이 시장가 매수에만 슬리피지를 적용한다.
- 다음 회차 매수는 마지막 보유 회차 체결가 −간격%, 매도는 가장 깊은 보유 회차 체결가 +간격%다.
  한 봉에 사거나 파는 회차는 하나뿐이다(실제 신호도 한 번에 하나).
- 매도는 봉이 시작될 때 이미 보유한 회차만 판다 — 그날 산 회차는 다음 봉부터 판다.
- 매도가 있었던 날은 다시 사지 않는다(같은 봉 안의 순서를 알 수 없어 왕복 매매를 만들지 않는다).
  전량 매도 뒤 1호 진입은 다음 봉에서 그날 직전 종가 기준으로 다시 시작한다.
  이건 일봉 백테스트의 한계일 뿐 전략 규칙이 아니다 — 실제 운용(운용 현황 화면·슬랙 신호)은 같은 날에도
  지정가에 닿으면 사고팔고 다시 살 수 있고, 그렇게 동작하는 것이 맞다. 두 쪽이 다른 건 의도된 차이다.
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

    # 보유 회차 — 사는 순서대로 쌓이고(1호가 맨 앞) 팔 때는 맨 뒤(가장 깊은 회차)부터 나간다.
    held: list[tuple[pd.Timestamp, float]] = []  # (체결일, 체결가)
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

        # 1) 매도 — 봉이 시작될 때 이미 가진 맨 뒤 회차 하나만, 그 회차 체결가 +간격% 에 닿으면 판다.
        #    한 봉에 한 회차씩만 거래한다(실제 신호도 한 번에 하나).
        sold_today = False
        if held:
            entry_date, entry_fill = held[-1]
            limit = rules.sell_limit(entry_fill, pct)
            if high >= limit:
                exit_price = max(open_, limit)
                realized += spec.quantity * (exit_price - entry_fill)
                trades.append(
                    _trade_row(spec, len(held), entry_date, entry_fill, date, exit_price, SELL_REASON, last_date)
                )
                held.pop()
                sold_today = True

        # 2) 매수 — 매도가 있었던 날은 쉰다. 한 봉에 한 회차만 산다.
        if not sold_today:
            if not held:
                limit = rules.first_buy_limit(prev_close, pct)
                if low <= limit:
                    fill = min(open_, limit)
                else:
                    # 장중에 못 닿았으면 15:00 에 사는 규칙 — 일봉에서는 종가에 시장가로 산다.
                    fill = close * (1.0 + market_buy_slippage_pct / 100.0)
                held.append((date, fill))
            elif len(held) < spec.rounds:
                limit = rules.next_buy_limit(held[-1][1], pct)
                fill = min(open_, limit)
                if low <= limit:
                    held.append((date, fill))

        unrealized = sum(spec.quantity * (close - fill) for _, fill in held)
        pnl[date] = realized + unrealized

    last_close = float(closes[-1])
    for round_no, (entry_date, entry_fill) in enumerate(held, start=1):
        trades.append(_trade_row(spec, round_no, entry_date, entry_fill, None, last_close, OPEN_REASON, last_date))
    return {"trades": trades, "pnl": pd.Series(pnl, dtype="float64")}
