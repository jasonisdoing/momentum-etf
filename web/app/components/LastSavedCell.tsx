"use client";

import { formatElapsedKorean, formatKstDateTime } from "@/lib/datetime";

/**
 * 「마지막 저장」 셀 — 절대 시각 + 경과 시간. 설정 그리드 공용.
 *
 * `2026. 8. 25(화) 오후 12:34 (2일, 3시간, 12분전 변경)`
 *
 * 절대 시각만으로는 얼마나 오래된 설정인지 한눈에 안 들어와서 경과를 함께 적는다.
 * 경과 부분은 녹색으로 구분한다.
 */
export function LastSavedCell({ value }: { value?: string | null }) {
  if (!value) {
    return <span style={{ color: "var(--text-muted)" }}>저장 이력 없음</span>;
  }
  const elapsed = formatElapsedKorean(value);
  return (
    <span>
      {formatKstDateTime(value)}
      {elapsed ? <span style={{ color: "#2f9e44", marginLeft: 6 }}>({elapsed} 변경)</span> : null}
    </span>
  );
}
