import { NextRequest, NextResponse } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";
import { jsonNoStore } from "@/lib/no-store-response";

export const dynamic = "force-dynamic";

/** 계좌 설정 조회 — FastAPI `/internal/account-settings` 프록시. */
export async function GET() {
  try {
    const data = await fetchFastApiJson("/internal/account-settings", { method: "GET" });
    return jsonNoStore(data as Record<string, unknown>);
  } catch (error) {
    const message = error instanceof Error ? error.message : "계좌 설정을 불러오지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}

/** 계좌 설정 저장 — `{ account_id, values }`. */
export async function PUT(request: NextRequest) {
  try {
    const body = await request.json();
    const data = await fetchFastApiJson("/internal/account-settings", {
      method: "PUT",
      body: JSON.stringify(body),
    });
    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "계좌 설정 저장에 실패했습니다.";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
