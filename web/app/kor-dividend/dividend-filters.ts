/**
 * 한국 배당주 화면의 필터 — 값과 판정을 한 곳에 둔다.
 *
 * 이 필터들은 원래 종목 발굴 스크립트의 하드코딩 상수였다. 화면에서 기준을 바꿔 가며
 * 걸러내는 게 이 화면의 목적이라, 값을 코드에 박지 않고 사용자가 정한 것을
 * 로컬스토리지에 기억한다.
 */

export type DividendFilters = {
  /** 배당률(현재) 하한 %. 빈 문자열이면 조건 없음. */
  minDividendYield: string;
  /** 총주주환원율(배당률 + 자사주률) 하한 %. */
  minShareholderYield: string;
  /** 주주환원율((배당+자사주)/순이익) 하한 %. */
  minPayoutRatio: string;
  /** PER 상한. */
  maxPer: string;
  /** PBR 상한. */
  maxPbr: string;
  /** 영업이익이 확정 연도 내내 우상향한 종목만. */
  requireOperatingTrend: boolean;
  /** 순이익이 확정 연도 내내 우상향한 종목만. */
  requireNetTrend: boolean;
  /** 주당배당금이 확정 연도 내내 우상향한 종목만. */
  requireDividendTrend: boolean;
};

export const EMPTY_FILTERS: DividendFilters = {
  minDividendYield: "",
  minShareholderYield: "",
  minPayoutRatio: "",
  maxPer: "",
  maxPbr: "",
  requireOperatingTrend: false,
  requireNetTrend: false,
  requireDividendTrend: false,
};

const STORAGE_KEY = "momentum-etf:kor-dividend:filters";

/** 저장된 필터. 없거나 못 읽으면 빈 필터 — 임의 기준으로 걸러 보여주지 않는다. */
export function readRememberedFilters(): DividendFilters {
  if (typeof window === "undefined") return EMPTY_FILTERS;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return EMPTY_FILTERS;
    const parsed = JSON.parse(raw) as Partial<DividendFilters>;
    // 저장 형식이 바뀌어도 모르는 키는 무시하고 아는 키만 채운다.
    return { ...EMPTY_FILTERS, ...parsed };
  } catch {
    return EMPTY_FILTERS;
  }
}

export function writeRememberedFilters(filters: DividendFilters): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(filters));
  } catch {
    // 저장 실패는 화면 동작에 영향이 없다(비공개 모드 등).
  }
}

/** 필터 판정에 쓸 행의 최소 형태. */
export type FilterableRow = {
  dividend_yield: number | null;
  shareholder_yield: number | null;
  payout_ratio_gross: number | null;
  per: number | null;
  pbr: number | null;
  trend_operating_ratio: number | null;
  trend_net_ratio: number | null;
  trend_dividend_ratio: number | null;
};

/** 입력값이 비었으면 조건 없음(null). 숫자가 아니어도 조건 없음. */
function threshold(text: string): number | null {
  const trimmed = text.trim();
  if (!trimmed) return null;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
}

/**
 * 필터 통과 여부.
 *
 * 값이 **없는**(판정 불가) 종목은 조건이 걸려 있으면 **떨어뜨린다**. 값이 없다고 통과시키면
 * 데이터가 빠진 종목이 조건을 만족한 것처럼 목록에 남는다.
 */
export function passesFilters(row: FilterableRow, filters: DividendFilters): boolean {
  const atLeast = (value: number | null | undefined, text: string): boolean => {
    const min = threshold(text);
    if (min === null) return true;
    return value !== null && value !== undefined && value >= min;
  };
  const atMost = (value: number | null | undefined, text: string): boolean => {
    const max = threshold(text);
    if (max === null) return true;
    return value !== null && value !== undefined && value <= max;
  };
  const fullTrend = (ratio: number | null, required: boolean): boolean => {
    if (!required) return true;
    return ratio !== null && ratio >= 1;
  };

  return (
    atLeast(row.dividend_yield, filters.minDividendYield) &&
    atLeast(row.shareholder_yield, filters.minShareholderYield) &&
    // 환원율만 저장 단위가 비율(0.478)이라 % 로 맞춰 비교한다.
    atLeast(row.payout_ratio_gross === null ? null : row.payout_ratio_gross * 100, filters.minPayoutRatio) &&
    atMost(row.per, filters.maxPer) &&
    atMost(row.pbr, filters.maxPbr) &&
    fullTrend(row.trend_operating_ratio, filters.requireOperatingTrend) &&
    fullTrend(row.trend_net_ratio, filters.requireNetTrend) &&
    fullTrend(row.trend_dividend_ratio, filters.requireDividendTrend)
  );
}

/** 지금 걸려 있는 조건 수 — 화면이 "필터 3개 적용" 처럼 알린다. */
export function countActiveFilters(filters: DividendFilters): number {
  return Object.entries(filters).filter(([, value]) =>
    typeof value === "boolean" ? value : String(value).trim() !== "",
  ).length;
}
