/** 통화별 가격 표기 공용 유틸.
 *
 * 원화는 소수점을 쓰지 않고, 달러·호주달러는 소수 둘째 자리까지 쓴다.
 * `/pools-rank` 의 현재가 표기에 인라인으로 있던 규칙을 옮겨 온 것이며,
 * 가격을 보여주는 화면은 모두 이 함수를 쓴다.
 */

/** 통화에 맞는 소수 자릿수 — 원화 0, 그 외(USD/AUD 등) 2. */
export function priceDecimals(currency: string | null | undefined): number {
  const code = String(currency ?? "").trim().toUpperCase();
  return code === "KRW" || code === "" ? 0 : 2;
}

/** 통화 기준으로 가격을 표기한다. 값이 없으면 `-`. */
export function formatPrice(value: number | null | undefined, currency: string | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  const digits = priceDecimals(currency);
  return new Intl.NumberFormat("ko-KR", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}
