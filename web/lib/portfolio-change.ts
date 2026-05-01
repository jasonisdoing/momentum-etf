/**
 * 포트폴리오 변동 계산 공통 모듈.
 *
 * /ticker 와 /compare 페이지에서 동일한 기준으로 포트폴리오 변동을 계산한다.
 * 변동률 기준: cumulative_change_pct (캐시 갱신 시점 대비 누적 변동률)
 */

/** 포트폴리오 변동 계산에 필요한 최소 holding 인터페이스 */
export type PortfolioChangeHolding = {
  weight: number | null;
  cumulative_change_pct?: number | null;
  price_currency?: string | null;
};

/** 환율 변동 정보 */
export type PortfolioChangeFxRate = {
  currency: string;
  change_pct?: number | null;
};

/** 통화별 분석 항목 */
export type PortfolioChangeBreakdownItem = {
  currency: string;
  label: string;
  change_pct: number;
  weight: number;
};

/** 포트폴리오 변동 결과 */
export type PortfolioChangeResult = {
  totalPct: number | null;
  breakdown: PortfolioChangeBreakdownItem[];
};

/** 통화 코드 → 지역 라벨 */
function getCurrencyRegionLabel(currency: string): string {
  const map: Record<string, string> = {
    KRW: "국내",
    USD: "미국",
    AUD: "호주",
    JPY: "일본",
    CNY: "중국",
    HKD: "홍콩",
    TWD: "대만",
    GBP: "영국",
    EUR: "유럽",
  };
  return map[currency] ?? currency;
}

/**
 * 포트폴리오 변동을 계산한다.
 *
 * - holdings 의 cumulative_change_pct (캐시 갱신 시점 대비 누적 변동률) 기준
 * - 외화 종목은 (1 + 종목변동률) × (1 + 환율변동률) - 1 로 원화 환산
 * - 가중 평균으로 합산, 총 비중이 100% 미만이면 100으로 나눔
 */
export function calcPortfolioChange(
  holdings: PortfolioChangeHolding[],
  fxRates: PortfolioChangeFxRate[],
): PortfolioChangeResult {
  if (!holdings || holdings.length === 0) {
    return { totalPct: null, breakdown: [] };
  }

  // 통화별 환율 변동률 맵
  const fxChangePctByCurrency = new Map<string, number>();
  for (const fx of fxRates) {
    const currency = String(fx.currency || "").trim().toUpperCase();
    const changePct = fx.change_pct;
    if (!currency || changePct == null || Number.isNaN(changePct)) continue;
    fxChangePctByCurrency.set(currency, changePct);
  }

  // 통화별 가중합 집계
  const groups = new Map<string, { weight: number; weightedSum: number }>();

  for (const h of holdings) {
    const weight = h.weight ?? 0;
    if (weight <= 0) continue;

    const componentChangePct = h.cumulative_change_pct;
    if (componentChangePct == null || Number.isNaN(componentChangePct)) continue;

    const currency = String(h.price_currency || "").trim().toUpperCase() || "KRW";
    const isForeign = currency !== "KRW";

    let changePctKrw = componentChangePct;
    if (isForeign) {
      const fxChangePct = fxChangePctByCurrency.get(currency);
      if (fxChangePct == null || Number.isNaN(fxChangePct)) continue;
      // (1 + 현지통화 변동률) × (1 + 환율 변동률) - 1
      changePctKrw = ((1 + componentChangePct / 100) * (1 + fxChangePct / 100) - 1) * 100;
    }

    const group = groups.get(currency) ?? { weight: 0, weightedSum: 0 };
    group.weight += weight;
    group.weightedSum += weight * changePctKrw;
    groups.set(currency, group);
  }

  // 합산
  let totalWeight = 0;
  let totalWeightedSum = 0;
  const breakdown: PortfolioChangeBreakdownItem[] = [];

  for (const [currency, group] of groups.entries()) {
    if (group.weight <= 0) continue;
    const changePct = group.weightedSum / group.weight;
    breakdown.push({
      currency,
      label: getCurrencyRegionLabel(currency),
      change_pct: changePct,
      weight: group.weight,
    });
    totalWeight += group.weight;
    totalWeightedSum += group.weight * changePct;
  }

  breakdown.sort((a, b) => b.weight - a.weight);

  if (totalWeight <= 0) {
    return { totalPct: null, breakdown };
  }

  const divisor = Math.max(totalWeight, 100);
  return { totalPct: totalWeightedSum / divisor, breakdown };
}

export { getCurrencyRegionLabel };
