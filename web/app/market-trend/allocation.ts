export type RegimeKey = "accel_up" | "decel_up" | "accel_down" | "decel_down";
// 실효 캡만 — 조정(decel_up)/하락(accel_down). 나머지는 base 가 이미 그 아래라 미설정.
export type RegimeCaps = Partial<Record<RegimeKey, number>>;

// 권장 투자 비율: 추세점수(-100~+100)를 구간 선형으로 매핑한 뒤(레벨), 현재 레짐별 상한(cap)으로
// 천장을 씌운다(min). 앵커·상한은 config.py → API → props 로 전달되는 단일 소스다.
//   base% : 점수 ≥ 0 → neutral + (점수/100)×upSpan / 점수 < 0 → neutral + (점수/100)×downSpan
//   최종% : min(base%, caps[regime])  — 약화(조정)/하락 레짐에서 고점 풀투자를 막는다.
// 표·차트 툴팁이 함께 쓰는 단일 매핑.
export function recommendedInvestPct(
  trendScore: number | null | undefined,
  neutral: number,
  upSpan: number,
  downSpan: number,
  regime?: RegimeKey | null,
  caps?: RegimeCaps | null,
): number | null {
  if (trendScore === null || trendScore === undefined || Number.isNaN(trendScore)) return null;
  const score = Math.max(-100, Math.min(100, trendScore));
  const base = score >= 0 ? neutral + (score / 100) * upSpan : neutral + (score / 100) * downSpan;
  const floor = neutral - downSpan;
  const ceil = neutral + upSpan;
  let invest = Math.max(floor, Math.min(ceil, base));
  if (regime && caps && typeof caps[regime] === "number") {
    invest = Math.min(invest, caps[regime]);
  }
  return invest;
}
