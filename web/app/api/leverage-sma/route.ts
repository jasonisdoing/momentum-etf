import { NextRequest } from "next/server";

import { fetchFastApiJson } from "../../../lib/internal-api";
import { jsonNoStore } from "../../../lib/no-store-response";

export const dynamic = "force-dynamic";

/** SMA 크로스 레버리지 라이브 튜닝·판정 — FastAPI `/internal/leverage/sma-cross` 프록시. */
export async function GET(request: NextRequest) {
  try {
    const market = request.nextUrl.searchParams.get("market") ?? "kor";
    const data = await fetchFastApiJson(`/internal/leverage/sma-cross?market=${encodeURIComponent(market)}`);
    return jsonNoStore(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "레버리지 백테스트를 불러오지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
