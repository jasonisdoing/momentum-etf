/** 한국 시가총액(억 원) 표기 표준 — `1,350조 4,904억` / `7,088억`.
 *
 * /kor-market-stock 과 Steady Momentum 이 같은 표기를 쓰도록 여기서만 만든다.
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
