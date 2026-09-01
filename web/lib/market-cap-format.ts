/** 한국 시가총액(억 원) 표기 표준 — `1,350조 4,904억` / `7,088억`.
 *
 * /kor-market-stock 과 모멘텀 전략 이 같은 표기를 쓰도록 여기서만 만든다.
 */
export function formatKorMarketCap(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  if (value >= 10000) {
    const jo = Math.floor(value / 10000);
    const eok = value % 10000;
    return eok > 0
      ? `${new Intl.NumberFormat("ko-KR").format(jo)}조 ${new Intl.NumberFormat("ko-KR").format(eok)}억`
      : `${new Intl.NumberFormat("ko-KR").format(jo)}조`;
  }
  return `${new Intl.NumberFormat("ko-KR").format(value)}억`;
}

/** 시가총액(**원** 단위) → `12.3조` · `4,567억`.
 *
 *  `formatKorMarketCap` 은 네이버가 억 단위로 주는 값(`/kor-market-stock`)을 그대로 받는다.
 *  이건 배치가 메타 캐시에 적어 둔 원 단위 금액(`total_net_assets`)을 받는다 — 전략 화면들이
 *  같은 소스를 쓰므로 표기도 한 함수로 맞춘다.
 */
export function formatMarketCapWon(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return formatKorMarketCap(value / 1_0000_0000);
}
