import { NextRequest } from "next/server";

import { fetchFastApiJson } from "../../../../lib/internal-api";
import { jsonNoStore } from "../../../../lib/no-store-response";

export const dynamic = "force-dynamic";

/** 합성 전략 운영 상태 — pool 쿼리를 넘겨야 해서 팩토리 대신 직접 구현.
 *  SM 순위 계산 + 신고가 현재 상태를 요청 시 계산하므로 수십 초 걸릴 수 있다. */
export async function GET(request: NextRequest) {
  try {
    const params = new URLSearchParams();
    const pool = request.nextUrl.searchParams.get("pool");
    const asOf = request.nextUrl.searchParams.get("as_of");
    if (pool) params.set("pool", pool);
    if (asOf) params.set("as_of", asOf);
    const query = params.size > 0 ? `?${params.toString()}` : "";
    return jsonNoStore(await fetchFastApiJson(`/internal/strategy-mix/positions${query}`, {}, 300_000));
  } catch (error) {
    return jsonNoStore(
      { error: error instanceof Error ? error.message : "합성 운영 상태를 불러오지 못했습니다." },
      { status: 500 },
    );
  }
}
