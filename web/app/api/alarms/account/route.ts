import { NextRequest } from "next/server";

import { fetchFastApiJson } from "../../../../lib/internal-api";
import { jsonNoStore } from "../../../../lib/no-store-response";

export const dynamic = "force-dynamic";

/** 알람 account 저장 — FastAPI `/internal/alarms/account` 프록시. */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    return jsonNoStore(await fetchFastApiJson("/internal/alarms/account", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }));
  } catch (error) {
    return jsonNoStore({ error: error instanceof Error ? error.message : "저장에 실패했습니다." }, { status: 500 });
  }
}
