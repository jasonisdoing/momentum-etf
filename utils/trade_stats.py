"""체결 통계 — 전략 화면 하단의 `거래 N건 · 승률 …` 한 줄을 만드는 공용 계산.

세 전략(모멘텀·신고가·합성)이 같은 정의를 쓰도록 여기 한 곳에서만 계산한다.
청산이 끝난 거래만 센다 — 보유중(청산 전) 행은 손익이 확정되지 않았다.
"""

from __future__ import annotations

from typing import Any

# 아직 팔지 않은 행의 사유 — 통계에서 제외한다.
OPEN_REASON = "보유중"


def summarize_trades(trades: list[dict[str, Any]] | None) -> dict[str, Any]:
    """거래 수·승률·평균 손익과 청산 사유별 건수를 돌려준다."""
    closed = [
        row
        for row in (trades or [])
        if row.get("exit_date") and str(row.get("reason") or "") != OPEN_REASON and row.get("return_pct") is not None
    ]
    wins = [row for row in closed if float(row["return_pct"]) > 0]
    losses = [row for row in closed if float(row["return_pct"]) <= 0]

    reason_counts: dict[str, int] = {}
    for row in closed:
        reason = str(row.get("reason") or "").strip() or "기타"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return {
        "trade_count": len(closed),
        "win_rate_pct": round(len(wins) / len(closed) * 100, 1) if closed else None,
        "avg_win_pct": round(sum(float(row["return_pct"]) for row in wins) / len(wins), 2) if wins else None,
        "avg_loss_pct": round(sum(float(row["return_pct"]) for row in losses) / len(losses), 2) if losses else None,
        # 사유는 전략마다 다르다(손절·이탈·교체 등) — 건수만 넘기고 표기는 화면이 한다.
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda item: -item[1])),
    }
