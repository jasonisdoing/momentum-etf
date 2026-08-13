/** ADR(등락비율, Advance-Decline Ratio) 표시 규약.
 *
 * ADR = 기간 내 상승종목수 합 ÷ 하락종목수 합 × 100.
 * 지수는 시가총액 큰 몇 종목에 끌려가지만 ADR 은 '얼마나 많은 종목이 함께 오르는가'(시장 폭)를 본다.
 * 지수가 오르는데 ADR 이 빠지면 상승 종목이 좁아지는 국면이라 모멘텀 전략에 불리하다.
 */

export const ADR_LINE_COLOR = "#7c3aed";
export const ADR_OVERHEATED_COLOR = "#dc2626";
export const ADR_OVERSOLD_COLOR = "#2563eb";
/** 강세/약세 경계(100) — 과열·침체선 사이의 기준선이라 계열색을 피해 무채색을 쓴다. */
export const ADR_NEUTRAL_COLOR = "#212529";

export type AdrPoint = {
  /** YYYY-MM-DD */
  date: string;
  /** 20일 누적 ADR. 하락 종목이 0이면 나눌 수 없어 null 이다. */
  adr: number | null;
  advance: number;
  decline: number;
};

/** `/internal/market-trend/history` 응답의 adr 블록. 기준값은 백엔드 config 가 단일 소스다. */
export type AdrResponse = {
  market: string;
  market_name: string;
  /** 대상 종목 수 (코스피 200 / 코스닥 150) */
  universe_size: number;
  window_days: number;
  overheated: number;
  neutral: number;
  oversold: number;
  latest_adr: number | null;
  points: AdrPoint[];
};

export function describeAdrLevel(
  adr: number | null,
  thresholds: { overheated: number; oversold: number },
): { label: string; color: string } {
  if (adr == null) return { label: "계산 불가", color: "var(--text-muted)" };
  if (adr >= thresholds.overheated) return { label: "과열", color: ADR_OVERHEATED_COLOR };
  if (adr <= thresholds.oversold) return { label: "침체", color: ADR_OVERSOLD_COLOR };
  return { label: "중립", color: "var(--text-muted)" };
}
