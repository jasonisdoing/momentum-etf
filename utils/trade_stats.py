"""체결 통계 — 전략 화면 하단의 `거래 N건 · 승률 …` 한 줄을 만드는 공용 계산.

세 전략(모멘텀·신고가·합성)이 같은 정의를 쓰도록 여기 한 곳에서만 계산한다.
**보유중(청산 전) 행도 평가 수익률로 함께 센다**(2026-09) — 청산분만 세면 장기 이평
설정에서 승자(계속 보유 중)는 통계에서 빠지고 먼저 꺾인 패자만 잡혀, 총수익은 큰데
승률·평균이익이 정반대로 보이는 착시가 났다(us_stock 12개월 +322% 인데 승률 28.6%).
수익률이 없는 행(포트폴리오 리밸런싱 기록 등)만 제외한다.
"""

from __future__ import annotations

from typing import Any

# 아직 팔지 않은 행의 사유 — 표시·구분용(통계에는 평가 수익률로 포함된다).
OPEN_REASON = "보유중"


def summarize_trades(trades: list[dict[str, Any]] | None) -> dict[str, Any]:
    """거래 수·승률·평균 손익과 사유별 건수를 돌려준다 — 보유중 행 포함."""
    rows = [row for row in (trades or []) if row.get("return_pct") is not None]
    wins = [row for row in rows if float(row["return_pct"]) > 0]
    losses = [row for row in rows if float(row["return_pct"]) <= 0]

    reason_counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reason") or "").strip() or "기타"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return {
        "trade_count": len(rows),
        "win_rate_pct": round(len(wins) / len(rows) * 100, 1) if rows else None,
        "avg_win_pct": round(sum(float(row["return_pct"]) for row in wins) / len(wins), 2) if wins else None,
        "avg_loss_pct": round(sum(float(row["return_pct"]) for row in losses) / len(losses), 2) if losses else None,
        # 사유는 전략마다 다르다(손절·이탈·교체·보유중 등) — 건수만 넘기고 표기는 화면이 한다.
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda item: -item[1])),
    }
