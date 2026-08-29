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


/** 통화 기호까지 붙인 가격 — `20,795원` · `$23.45` · `A$23.32`.
 *
 *  `formatPrice` 는 숫자만 내므로 표의 통화 컬럼용이고, 이 함수는 기호가 필요한 자리
 *  (티커 상세·차트 축·평단 배지)에 쓴다. `/ticker` 안에만 있던 것을 옮겨 왔다 —
 *  전략·순위 차트가 같은 표기를 써야 화면마다 단위가 달라 보이지 않는다.
 *
 *  국가 코드(`kor`·`us`·`au`)와 통화 코드(`KRW`·`USD`·`AUD`) 둘 다 받는다 —
 *  화면마다 들고 있는 값이 달라서다. 모르는 값은 원화로 본다.
 */
export function formatCurrencyPrice(value: number | null | undefined, countryOrCurrency: string): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "N/A";
  }

  const normalized = String(countryOrCurrency || "").trim().toLowerCase();
  if (normalized === "au" || normalized === "aud") {
    return `A$${new Intl.NumberFormat("en-AU", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value)}`;
  }

  if (normalized === "us" || normalized === "usd") {
    return `$${new Intl.NumberFormat("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value)}`;
  }

  if (normalized === "eur") {
    return `€${new Intl.NumberFormat("de-DE", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value)}`;
  }

  if (normalized === "twd") {
    return `${new Intl.NumberFormat("zh-TW", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value)} TWD`;
  }

  if (normalized === "hkd") {
    return `HK$${new Intl.NumberFormat("en-HK", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value)}`;
  }

  if (normalized === "jpy") {
    return `¥${new Intl.NumberFormat("ja-JP", {
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value)}`;
  }

  if (normalized === "gbp") {
    return `£${new Intl.NumberFormat("en-GB", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value)}`;
  }

  if (normalized === "cny") {
    return `${new Intl.NumberFormat("zh-CN", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value)} CNY`;
  }

  return `${new Intl.NumberFormat("ko-KR", {
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value)}원`;
}
