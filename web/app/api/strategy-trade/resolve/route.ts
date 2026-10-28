import { NextRequest } from "next/server";

import { fetchFastApiJson } from "../../../../lib/internal-api";
import { jsonNoStore } from "../../../../lib/no-store-response";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  try {
    const ticker = request.nextUrl.searchParams.get("ticker");
    const pool = request.nextUrl.searchParams.get("pool");
    if (!ticker || !pool) {
      return jsonNoStore({ error: "pool·ticker 파라미터가 필요합니다." }, { status: 400 });
    }
    const data = await fetchFastApiJson(
      `/internal/strategy-trade/resolve?pool=${encodeURIComponent(pool)}&ticker=${encodeURIComponent(ticker)}`,
    );
    return jsonNoStore(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "티커 정보를 조회하지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 400 });
  }
}
