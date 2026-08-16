import { NextRequest } from "next/server";

import { fetchFastApiJson } from "../../../lib/internal-api";
import { jsonNoStore } from "../../../lib/no-store-response";

export const dynamic = "force-dynamic";

/** 합성전략 백테스트 — pool 쿼리를 넘겨야 해서 팩토리 대신 직접 구현.
 *  두 전략 백테스트를 요청 시 계산하므로 오래 걸린다 (다른 전략 화면과 같은 패턴, 캐시 없음). */
export async function GET(request: NextRequest) {
  try {
    const params = new URLSearchParams();
    const pool = request.nextUrl.searchParams.get("pool");
    const months = request.nextUrl.searchParams.get("months");
    if (pool) params.set("pool", pool);
    if (months) params.set("months", months);
    const query = params.size > 0 ? `?${params.toString()}` : "";
    return jsonNoStore(await fetchFastApiJson(`/internal/strategy-mix/backtest${query}`, {}, 300_000));
  } catch (error) {
    return jsonNoStore(
      { error: error instanceof Error ? error.message : "합성 백테스트를 불러오지 못했습니다." },
      { status: 500 },
    );
  }
}
