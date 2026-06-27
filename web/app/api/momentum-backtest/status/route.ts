import { NextRequest } from "next/server";

import { fetchFastApiJson } from "../../../../lib/internal-api";
import { jsonNoStore } from "../../../../lib/no-store-response";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  try {
    const file = request.nextUrl.searchParams.get("file");
    const qs = file ? `?file=${encodeURIComponent(file)}` : "";
    const data = await fetchFastApiJson(`/internal/momentum/backtest/status${qs}`);
    return jsonNoStore(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "백테스트 상태를 불러오지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
