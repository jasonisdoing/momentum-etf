import { NextRequest, NextResponse } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";

export const dynamic = "force-dynamic";

/** 백테스트 탐색공간 조회 — FastAPI `/internal/backtest-config` 프록시. */
export async function GET() {
  try {
    const data = await fetchFastApiJson("/internal/backtest-config", { method: "GET" });
    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "백테스트 설정을 불러오지 못했습니다.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

/** 백테스트 탐색공간 저장 — `{ pool_id, config }`. */
export async function PUT(request: NextRequest) {
  try {
    const body = await request.json();
    const data = await fetchFastApiJson("/internal/backtest-config", {
      method: "PUT",
      body: JSON.stringify(body),
    });
    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "백테스트 설정 저장에 실패했습니다.";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
