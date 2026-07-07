import { fetchFastApiJson } from "../../../lib/internal-api";
import { jsonNoStore } from "../../../lib/no-store-response";

import { NextRequest } from "next/server";

export const dynamic = "force-dynamic";

export async function POST(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const poolId = searchParams.get("pool_id");
    const url = poolId ? `/internal/momentum/backtest?pool_id=${poolId}` : "/internal/momentum/backtest";
    const data = await fetchFastApiJson(url, { method: "POST" });
    return jsonNoStore(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "백테스트를 시작하지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
