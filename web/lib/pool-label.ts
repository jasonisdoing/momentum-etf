/** 종목풀 표기 표준.
 *
 * 모든 화면의 종목풀 셀렉터가 같은 문자열을 쓰도록 여기서만 만든다.
 * 화면마다 다르게 쓰면 같은 종목풀이 화면마다 다른 이름으로 보인다.
 */

export type PoolLabelSource = {
  ticker_type: string;
  name?: string | null;
  icon?: string | null;
  order?: number | null;
};

/**
 * 종목풀 표준 표기 — `1. 🇰🇷 국내상장 국내(kor_kr)`.
 *
 * `order`/`icon` 이 없는 데이터면 해당 부분만 빠진다(표기가 깨지지 않게).
 * 이름이 비어 있으면 ticker_type 으로 대체한다.
 */
export function formatPoolLabel(pool: PoolLabelSource): string {
  // order 접두사로 번호를 붙이므로, 이름에 이미 박혀 있는 선행 번호("1. ")는 제거해 중복을 막는다.
  const name = String(pool.name ?? "").trim().replace(/^\d+\.\s*/, "") || pool.ticker_type;
  const prefix = [
    pool.order === null || pool.order === undefined ? null : `${pool.order}.`,
    String(pool.icon ?? "").trim() || null,
  ]
    .filter(Boolean)
    .join(" ");
  const body = `${name}(${pool.ticker_type})`;
  return prefix ? `${prefix} ${body}` : body;
}
