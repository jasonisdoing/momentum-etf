"use client";

import { formatElapsedKorean, formatKstDateTime } from "@/lib/datetime";

/**
 * 「마지막 저장」 셀 — 저장 경로 + 절대 시각 + 경과 시간. 설정 그리드 공용.
 *
 * `[모멘텀 전략] 2026. 8. 25(화) 오후 12:34 (2일, 3시간, 12분전 변경)`
 *
 * 절대 시각만으로는 얼마나 오래된 설정인지 한눈에 안 들어와서 경과를 함께 적는다.
 * 저장 경로(`method`)는 **누가 바꾼 값인지**를 알려준다 — 종목풀 설정 화면에서 직접 고친
 * 값인지, 모멘텀 전략·튜닝에서 적용된 값인지 구분해야 되돌릴 판단을 할 수 있다.
 */
export function LastSavedCell({ value, method }: { value?: string | null; method?: string | null }) {
  if (!value) {
    return <span style={{ color: "var(--text-muted)" }}>저장 이력 없음</span>;
  }
  const elapsed = formatElapsedKorean(value);
  const source = String(method ?? "").trim();
  return (
    <span>
      {source ? (
        <span
          style={{
            marginRight: 6,
            padding: "0 5px",
            borderRadius: 4,
            fontSize: "var(--fs-xs)",
            fontWeight: 700,
            background: "rgba(148,163,184,0.18)",
            color: "var(--text-muted)",
          }}
        >
          {source}
        </span>
      ) : null}
      {formatKstDateTime(value)}
      {elapsed ? <span style={{ color: "#2f9e44", marginLeft: 6 }}>({elapsed} 변경)</span> : null}
    </span>
  );
}
