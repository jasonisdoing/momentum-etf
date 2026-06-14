// 권장 투자 비율: 추세점수(-100~+100)를 구간 선형으로 매핑한다 (앵커는 config.py → API → props).
//   점수 ≥ 0 : 투자% = neutral + (점수/100) × upSpan
//   점수 < 0 : 투자% = neutral + (점수/100) × downSpan
// 표/차트 툴팁이 함께 쓰는 단일 매핑.
export function recommendedInvestPct(
  trendScore: number | null | undefined,
  neutral: number,
  upSpan: number,
  downSpan: number,
): number | null {
  if (trendScore === null || trendScore === undefined || Number.isNaN(trendScore)) return null;
  const score = Math.max(-100, Math.min(100, trendScore));
  const invest = score >= 0 ? neutral + (score / 100) * upSpan : neutral + (score / 100) * downSpan;
  const floor = neutral - downSpan;
  const ceil = neutral + upSpan;
  return Math.max(floor, Math.min(ceil, invest));
}
