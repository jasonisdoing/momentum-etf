import { NextRequest } from "next/server";

import { loadRankToolbarData } from "../../../lib/rank-store";
import { jsonNoStore } from "../../../lib/no-store-response";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const tickerType = searchParams.get("ticker_type") ?? undefined;
    const data = await loadRankToolbarData({ ticker_type: tickerType }, request.signal);
    return jsonNoStore(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "종목풀 정보를 불러오지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
