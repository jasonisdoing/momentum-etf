import type { ReactNode } from "react";

/** 상태 뱃지 — `compact` 는 설정 표의 상태 요약처럼 좁은 그리드 칸에 넣을 때 쓴다. */
export function StatusBadge({ color, compact, children }: { color: string; compact?: boolean; children: ReactNode }) {
  return (
    <span
      style={{
        padding: compact ? "0 2px" : "1px 5px",
        borderRadius: compact ? 3 : 4,
        fontSize: compact ? "var(--fs-xs)" : "var(--fs-sm)",
        lineHeight: compact ? 1.4 : undefined,
        fontWeight: 700,
        // 뱃지 글자는 줄바꿈하지 않는다 — 좁은 칸에서 `매수슬/랙`처럼 쪼개지지 않게.
        whiteSpace: "nowrap",
        color,
        border: `1px solid ${color}`,
      }}
    >
      {children}
    </span>
  );
}
