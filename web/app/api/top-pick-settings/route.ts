import { NextRequest, NextResponse } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";
import { jsonNoStore } from "@/lib/no-store-response";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  try {
    const accountId = request.nextUrl.searchParams.get("account_id");
    const path = accountId
      ? `/internal/top-pick/settings?account_id=${encodeURIComponent(accountId)}`
      : "/internal/top-pick/settings";
    const data = await fetchFastApiJson(path, { method: "GET" });
    return jsonNoStore(data as Record<string, unknown>);
  } catch (error) {
    const message = error instanceof Error ? error.message : "탑픽 설정을 불러오지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}

export async function PUT(request: NextRequest) {
  try {
    const body = await request.json();
    const data = await fetchFastApiJson("/internal/top-pick/settings", {
      method: "PUT",
      body: JSON.stringify(body),
    });
    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "탑픽 설정 저장에 실패했습니다.";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
