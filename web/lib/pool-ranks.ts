/** 종목풀 순위(`/api/rank`) 클라이언트 store — 자산 화면의 순위 컬럼용.
 *
 * 계좌 상세 모달이 열릴 때 처음 받으면 순위·고점이 몇 초 늦게 깜박이며 나타난다. 그래서 화면이
 * 열릴 때 미리 받아 두고(`loadPoolRanks`), 모달은 받아 둔 값을 곧바로 쓴다(`peekPoolRanks`).
 * 순위는 천천히 변하므로 받아 둔 값은 오래돼도 먼저 보여주고, 새 값은 뒤에서 받아 바꾼다.
 * 서버도 5분 캐시라 같은 풀을 자주 불러도 부담이 없다.
 */

export type PoolRankRow = { 티커: string; 순위?: number | null };

const POOL_RANKS_TTL_MS = 60_000;
const cache = new Map<string, { savedAt: number; rows: PoolRankRow[] }>();
const inflight = new Map<string, Promise<PoolRankRow[]>>();

/** 받아 둔 풀 순위 — 오래됐어도 돌려준다(없으면 null). 새 값은 `loadPoolRanks` 가 받는다. */
export function peekPoolRanks(pool: string): PoolRankRow[] | null {
  return cache.get(pool)?.rows ?? null;
}

/** 풀 순위 조회 — 받은 지 얼마 안 됐으면 그 값, 요청 중이면 그 요청을 같이 기다린다. 실패는 예외. */
export function loadPoolRanks(pool: string): Promise<PoolRankRow[]> {
  const hit = cache.get(pool);
  if (hit && Date.now() - hit.savedAt <= POOL_RANKS_TTL_MS) return Promise.resolve(hit.rows);
  const pending = inflight.get(pool);
  if (pending) return pending;
  const request = (async () => {
    const response = await fetch(`/api/rank?ticker_type=${encodeURIComponent(pool)}`, { cache: "no-store" });
    const payload = (await response.json()) as { rows?: PoolRankRow[]; cache_blocked?: boolean; error?: string };
    if (!response.ok || payload.error || payload.cache_blocked || !Array.isArray(payload.rows) || !payload.rows.length) {
      throw new Error(`${pool}: ${payload.error || "순위 데이터를 사용할 수 없습니다"}`);
    }
    cache.set(pool, { savedAt: Date.now(), rows: payload.rows });
    return payload.rows;
  })().finally(() => inflight.delete(pool));
  inflight.set(pool, request);
  return request;
}
