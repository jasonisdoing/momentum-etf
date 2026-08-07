import { NextRequest } from "next/server";

import { fetchFastApiJson } from "../../../lib/internal-api";
import { jsonNoStore } from "../../../lib/no-store-response";

export const dynamic = "force-dynamic";

/** 메모 목록 조회 — FastAPI 프록시. */
export async function GET() {
  try {
    return jsonNoStore(await fetchFastApiJson("/internal/memos"));
  } catch (error) {
    const message = error instanceof Error ? error.message : "메모를 불러오지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}

/** 메모 생성. body: {"title": "...", "content": "..."} */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const data = await fetchFastApiJson("/internal/memos", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return jsonNoStore(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "메모를 저장하지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
