/**
 * AG Grid 공용 셀 헬퍼 — 여러 화면(pools-rank, strategy-sm 등)이 같은 표기를 쓰는
 * 셀의 단일 소스. 화면별 고유 스타일(클래스 기반 색 등)은 각 화면에 남긴다.
 */

import type React from "react";

/** 부호를 붙인 퍼센트 표기: +1.23% / -4.56% / "-"(값 없음). */
export function formatSignedPct(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return "-";
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}%`;
}

/** 상승 빨강 · 하락 파랑 · 0/없음 상속 — 인라인 색 방식 화면용. */
export function signColor(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value) || value === 0) return "inherit";
  return value > 0 ? "var(--up-color, #d64545)" : "var(--down-color, #2f6fd0)";
}

/** 마켓(KOSPI/KOSDAQ) 배지 셀 스타일 — KOSPI 녹색 · KOSDAQ 파란색, 그 외 없음. */
export function marketBadgeCellStyle(value: unknown): React.CSSProperties | null {
  if (value === "KOSPI")
    return { textAlign: "center", backgroundColor: "#d1e7dd", color: "#0f5132", fontWeight: "bold" };
  if (value === "KOSDAQ")
    return { textAlign: "center", backgroundColor: "#cfe2ff", color: "#084298", fontWeight: "bold" };
  return { textAlign: "center" };
}

/** 고점 대비(%) 셀 — 정확히 0 이면 ⭐신고점(빨강 볼드), 그 외 퍼센트 표기. */
export function renderHighDrawdownCell(value: number | null | undefined, digits = 1) {
  if (value === 0) return <span style={{ color: "#d93025", fontWeight: 700 }}>⭐신고점</span>;
  if (value == null || Number.isNaN(value)) return <span>-</span>;
  return <span>{`${value.toFixed(digits)}%`}</span>;
}
