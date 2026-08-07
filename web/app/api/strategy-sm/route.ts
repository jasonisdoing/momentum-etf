import { NextRequest } from "next/server";

import { fetchFastApiJson } from "../../../lib/internal-api";
import { jsonNoStore } from "../../../lib/no-store-response";

export const dynamic = "force-dynamic";

/** Steady Momentum 설정 조회 — FastAPI 프록시. */
export async function GET() {
  try {
    const data = await fetchFastApiJson("/internal/strategy-sm");
    return jsonNoStore(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "설정을 불러오지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}

/** 설정 저장. body: {"settings": {...}} */
export async function PUT(request: NextRequest) {
  try {
    const body = await request.json();
    const data = await fetchFastApiJson("/internal/strategy-sm/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return jsonNoStore(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "설정을 저장하지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
