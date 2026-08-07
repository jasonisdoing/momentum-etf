import { NextRequest } from "next/server";

import { fetchFastApiJson } from "../../../../lib/internal-api";
import { jsonNoStore } from "../../../../lib/no-store-response";

export const dynamic = "force-dynamic";

/** 자산 화면 종목명 배지 — FastAPI `/internal/alarms/badges` 프록시. */
export async function GET(request: NextRequest) {
  try {
    const account = request.nextUrl.searchParams.get("account");
    if (!account) {
      return jsonNoStore({ error: "account 파라미터가 필요합니다." }, { status: 400 });
    }
    return jsonNoStore(
      await fetchFastApiJson(`/internal/alarms/badges?account_id=${encodeURIComponent(account)}`),
    );
  } catch (error) {
    return jsonNoStore({ error: error instanceof Error ? error.message : "배지를 불러오지 못했습니다." }, { status: 500 });
  }
}
