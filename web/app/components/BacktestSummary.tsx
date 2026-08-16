"use client";

/**
 * 백테스트 요약 줄 — 기간 · 전략 · 벤치마크 · 초과.
 *
 * 모멘텀·신고가·합성 세 화면이 같은 형식을 쓰도록 여기서만 조합한다.
 */

import { formatSignedPct, signColor } from "@/lib/grid-cells";

const hintStyle: React.CSSProperties = { color: "var(--text-muted)", fontSize: "var(--fs-sm)" };

export type BacktestStats = {
  label: string;
  totalPct: number | null;
  cagrPct: number | null;
  mddPct: number | null;
  sortino: number | null;
};

function StatItem({ label, totalPct, cagrPct, mddPct, sortino }: BacktestStats) {
  return (
    <span>
      {label} <b style={{ color: signColor(totalPct) }}>{formatSignedPct(totalPct)}</b>
      <span style={hintStyle}>
        {` (CAGR ${cagrPct != null ? formatSignedPct(cagrPct, 1) : "-"}`}
        {mddPct != null ? ` · MDD ${mddPct.toFixed(1)}%` : ""}
        {` · 소르티노 ${sortino != null ? sortino.toFixed(2) : "-"})`}
      </span>
    </span>
  );
}

export function BacktestSummary({
  startDate,
  endDate,
  strategy,
  benchmark,
  extra,
}: {
  startDate: string;
  endDate: string;
  strategy: BacktestStats;
  benchmark: BacktestStats;
  /** 참고 지수 등 추가로 나란히 둘 항목 (미국 풀의 FMTM 등). */
  extra?: BacktestStats | null;
}) {
  const excess =
    strategy.totalPct != null && benchmark.totalPct != null ? strategy.totalPct - benchmark.totalPct : null;
  return (
    <div style={{ display: "flex", gap: 18, flexWrap: "wrap", fontSize: "var(--fs-sm)", padding: "2px 0 8px" }}>
      <span>
        {startDate} ~ {endDate}
      </span>
      <StatItem {...strategy} />
      <StatItem {...benchmark} />
      {extra ? <StatItem {...extra} /> : null}
      {excess != null ? (
        <span>
          초과 <b style={{ color: signColor(excess) }}>{formatSignedPct(excess)}p</b>
        </span>
      ) : null}
    </div>
  );
}
