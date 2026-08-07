/** 계좌 보유 종목(holdings) 클라이언트 store.
 *
 * 종목 순서(sort_order)의 단일 소스는 holdings 다. /assets 와 자산 헬퍼가 이 함수를
 * 공유해 같은 방식으로 순서를 저장한다(화면마다 인라인 fetch 를 복붙하지 않는다).
 */

/** 사용자 지정 순서 저장 — `PATCH /api/assets` (action: reorder). 실패 시 throw. */
export async function reorderHoldings(accountId: string, orderedTickers: string[]): Promise<void> {
  if (!orderedTickers.length) return;
  const response = await fetch("/api/assets", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action: "reorder", account_id: accountId, ordered_tickers: orderedTickers }),
  });
  const payload = (await response.json().catch(() => ({}))) as { error?: string };
  if (!response.ok) {
    throw new Error(payload.error || "순서 저장에 실패했습니다.");
  }
}
