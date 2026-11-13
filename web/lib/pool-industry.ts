/** 종목풀에 어떤 값이 들어오는지 — 업종·시총 컬럼을 쓰는 모든 화면의 단일 소스.
 *
 * 순위(`/pools-rank`)·모멘텀(`/strategy-momentum`)·신고가(`/strategy-new-high`)가 각자
 * 같은 판정을 들고 있었고, 조건이 하나씩 어긋나 화면마다 업종 컬럼이 다르게 보였다.
 *
 * **값 유무로 판단하지 않는다.** 수집이 깨져 전부 비면 컬럼이 조용히 사라져 사용자가
 * 눈치채지 못한다. 빈 칸은 그 자체로 정보다 — "이 종목은 분류가 없다". 그래서 **채우는
 * 경로가 있는 풀인가**(풀 성격)만 본다.
 *
 * 채우는 경로(백엔드 `utils/stock_meta_updater` 배치 B · `utils/industry_map`):
 *   · 개별주 — 한국은 네이버 업종, 미국·호주는 yfinance 종목별 분류
 *   · ETF — 섹터·업종을 두지 않는다(국가 무관). 어느 화면이든 ETF 풀이면 숨긴다.
 */

export type PoolIndustrySource = {
  pool_kind?: string | null;
  country_code?: string | null;
};

/** 업종이 들어오는 풀인지 — 개별주 풀만. ETF·풀 성격 미설정 풀은 false(임의로 켜지 않는다). */
export function poolHasIndustry(pool: PoolIndustrySource | null | undefined): boolean {
  return String(pool?.pool_kind ?? "").trim().toLowerCase() === "stock";
}

/** 종목풀에 **시가총액**이 들어오는지 — 시총·시총순위 컬럼을 쓰는 화면의 단일 소스.
 *
 * 시총은 **개별주에만 있는 값**이다. ETF 에 대응하는 건 시가총액이 아니라 순자산총액
 * (`total_net_assets`)이고, 그건 별도 컬럼으로 보여준다.
 */
export function poolHasMarketCap(pool: PoolIndustrySource | null | undefined): boolean {
  return String(pool?.pool_kind ?? "").trim().toLowerCase() === "stock";
}

/** 미국 개별주만 Yahoo 섹터 분류를 별도 컬럼으로 표시한다. */
export function poolHasUsStockSector(pool: PoolIndustrySource | null | undefined): boolean {
  return String(pool?.country_code ?? "").trim().toLowerCase() === "us"
    && String(pool?.pool_kind ?? "").trim().toLowerCase() === "stock";
}
