/** 종목풀에 어떤 값이 들어오는지 — 업종·시총 컬럼을 쓰는 모든 화면의 단일 소스.
 *
 * 순위(`/pools-rank`)·모멘텀(`/strategy-momentum`)·신고가(`/strategy-new-high`)가 각자
 * 같은 판정을 들고 있었고, 조건이 하나씩 어긋나 화면마다 업종 컬럼이 다르게 보였다.
 *
 * **값 유무로 판단하지 않는다.** 수집이 깨져 전부 비면 컬럼이 조용히 사라져 사용자가
 * 눈치채지 못한다. 빈 칸은 그 자체로 정보다 — "이 종목은 섹터가 없다"(대표지수·레버리지·
 * 인버스 ETF). 그래서 **채우는 경로가 있는 풀인가**만 본다.
 *
 * 채우는 경로(백엔드 `utils/stock_meta_updater` 배치 B · `utils/industry_map`):
 *   · 개별주 — 한국은 네이버 업종, 미국은 지수 구성종목(SP500/NDX100), 호주는 yfinance
 *   · 한국 ETF — 네이버 「(주식)섹터」 중분류(소재·IT·헬스케어 …)
 *   · 미국·호주 ETF — 경로 없음
 */

export type PoolIndustrySource = {
  pool_kind?: string | null;
  country_code?: string | null;
};

/** 국가 + 구분 조합별로 업종이 들어오는지. 경로가 없으면 false. */
export function poolHasIndustry(pool: PoolIndustrySource | null | undefined): boolean {
  const poolKind = String(pool?.pool_kind ?? "").trim().toLowerCase();
  const country = String(pool?.country_code ?? "").trim().toLowerCase();
  if (poolKind === "stock") return true;
  if (poolKind === "etf") return country === "kor";
  // 풀 성격이 미설정인 풀은 판단 근거가 없다 — 없는 것으로 본다(임의로 켜지 않는다).
  return false;
}

/** 종목풀에 **시가총액**이 들어오는지 — 시총·시총순위 컬럼을 쓰는 화면의 단일 소스.
 *
 * 업종(`poolHasIndustry`)과 **기준이 다르다.** 지금까지는 업종 플래그를 시총 컬럼에도
 * 그대로 재사용했는데(`marketCapRankColumn(field, !hasIndustryData)`), 한국 ETF 에 업종이
 * 들어오면서 시총이 없는데도 시총 컬럼이 열리게 됐다.
 *
 * 시총은 **개별주에만 있는 값**이다. ETF 에 대응하는 건 시가총액이 아니라 순자산총액
 * (`total_net_assets`)이고, 그건 별도 컬럼으로 보여준다.
 */
export function poolHasMarketCap(pool: PoolIndustrySource | null | undefined): boolean {
  return String(pool?.pool_kind ?? "").trim().toLowerCase() === "stock";
}
