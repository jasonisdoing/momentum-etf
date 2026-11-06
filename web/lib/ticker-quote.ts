/** 종목 상세 응답의 현재가·전일 대비 등락 — 종목 상세·비교 화면 공용 단일 규칙.
 *
 * 정규장 전(프리장·데이장)에는 서버가 마지막 봉을 확정 종가로 두고 현재 시세를 `live` 로 따로 준다.
 * `live` 가 있으면 그 값, 없으면 마지막 봉이다. 비교 화면이 마지막 봉만 읽어, 같은 종목이 다른 화면은
 * 프리장 시세(+0.63%)·비교 화면은 어제 종가(+1.69%)로 갈렸다(2026-10-27).
 */
export type QuoteRow = { close: number | null; change_pct: number | null };
export type LiveQuote = { price: number; change_pct: number | null } | null | undefined;
export type CurrentQuote = { price: number | null; changePct: number | null; changeAmount: number | null };

function closedRows(rows: QuoteRow[]): QuoteRow[] {
  return rows.filter((row) => row.close !== null && row.close !== undefined && !Number.isNaN(row.close));
}

export function currentQuote(rows: QuoteRow[], live: LiveQuote): CurrentQuote {
  const priced = closedRows(rows);
  const last = priced[priced.length - 1] ?? null;
  const previous = priced[priced.length - 2] ?? null;
  const price = live?.price ?? last?.close ?? null;
  const changePct = live ? live.change_pct : (last?.change_pct ?? null);
  if (price === null || changePct === null) return { price, changePct, changeAmount: null };
  // 기준가 — live 면 마지막 확정 종가(마지막 봉), 아니면 그 전 봉. 없으면 등락률로 되짚는다.
  const base = live ? last?.close : previous?.close;
  if (base !== null && base !== undefined) return { price, changePct, changeAmount: price - base };
  const derived = price / (1 + changePct / 100);
  return { price, changePct, changeAmount: Number.isFinite(derived) ? price - derived : null };
}
