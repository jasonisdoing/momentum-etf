/** 계좌 메모(`/api/note`) 클라이언트 store — 자산 화면의 계좌 상세 모달용.
 *
 * 모달이 열린 뒤에 받으면 메모 칸이 늦게 채워진다. 그래서 화면이 열릴 때 미리 받아 두고
 * (`loadAccountNote`), 모달은 받아 둔 값으로 시작한다(`peekAccountNote`). 같은 계좌의
 * 동시 요청은 하나로 합친다. 저장하면 받아 둔 값도 같이 바꾼다(`rememberAccountNote`).
 */

export type AccountNote = { content: string; updated_at: string | null };

const notes = new Map<string, AccountNote>();
const inflight = new Map<string, Promise<AccountNote>>();

/** 받아 둔 메모 — 없으면 null. */
export function peekAccountNote(accountId: string): AccountNote | null {
  return notes.get(accountId) ?? null;
}

/** 저장한 값을 받아 둔 값으로 반영한다. */
export function rememberAccountNote(accountId: string, note: AccountNote): void {
  notes.set(accountId, note);
}

/** 메모를 서버에서 새로 받는다(요청 중이면 같이 기다린다). 실패는 예외. */
export function loadAccountNote(accountId: string): Promise<AccountNote> {
  const pending = inflight.get(accountId);
  if (pending) return pending;
  const request = (async () => {
    const resp = await fetch(`/api/note?account=${encodeURIComponent(accountId)}`, { cache: "no-store" });
    const data = (await resp.json()) as { content?: string; updated_at?: string; error?: string };
    if (!resp.ok || data.error) throw new Error(data.error ?? "메모를 불러오지 못했습니다.");
    const note: AccountNote = { content: data.content ?? "", updated_at: data.updated_at ?? null };
    notes.set(accountId, note);
    return note;
  })().finally(() => inflight.delete(accountId));
  inflight.set(accountId, request);
  return request;
}
