"use client";

import { formatSignedPct } from "@/lib/grid-cells";

/** 백테스트 하단 한 줄 — `거래 94건 · 승률 37.2% · 평균이익 +86.2% · 평균손실 -10.6% · 이탈 41`.
 *
 * 세 전략 화면(모멘텀·신고가·합성)이 같은 표기를 쓰도록 여기 한 곳에서만 만든다.
 * 값은 백엔드 공용 계산(`utils/trade_stats.py`)이 준 것을 그대로 쓴다.
 * 청산 사유는 전략마다 달라(이탈·주간 교체·주중 이탈) 서버가 준 순서대로 이어 붙인다.
 */
export type TradeStats = {
  trade_count: number;
  win_rate_pct: number | null;
  avg_win_pct: number | null;
  avg_loss_pct: number | null;
  reason_counts?: Record<string, number>;
};

export function BacktestTradeStats({ stats, style }: { stats: TradeStats; style?: React.CSSProperties }) {
  const reasons = Object.entries(stats.reason_counts ?? {});
  return (
    <div style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)", ...style }}>
      {`거래 ${stats.trade_count}건 · 승률 ${stats.win_rate_pct ?? "-"}%`}
      {` · 평균이익 ${stats.avg_win_pct != null ? formatSignedPct(stats.avg_win_pct, 1) : "-"}`}
      {` · 평균손실 ${stats.avg_loss_pct != null ? formatSignedPct(stats.avg_loss_pct, 1) : "-"}`}
      {reasons.length > 0 ? ` · ${reasons.map(([reason, count]) => `${reason} ${count}`).join(" / ")}` : ""}
    </div>
  );
}
