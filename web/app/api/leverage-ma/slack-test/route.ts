import { NextRequest } from "next/server";

import { fetchFastApiJson } from "../../../../lib/internal-api";
import { jsonNoStore } from "../../../../lib/no-store-response";

export const dynamic = "force-dynamic";

/** 이동평균선 크로스 슬랙 수동 발송 — FastAPI `/internal/leverage/ma-cross/slack-test` 프록시. */
export async function POST(request: NextRequest) {
  try {
    const market = request.nextUrl.searchParams.get("market") ?? "kor";
    const data = await fetchFastApiJson(`/internal/leverage/ma-cross/slack-test?market=${encodeURIComponent(market)}`, {
      method: "POST",
    });
    return jsonNoStore(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "슬랙 발송에 실패했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
