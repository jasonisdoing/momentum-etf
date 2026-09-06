import { NextRequest, NextResponse } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";

export const dynamic = "force-dynamic";

/** 종목풀별 전략 사용 여부 저장 — FastAPI `/internal/pool-settings/strategy-use` 프록시. */
export async function PUT(request: NextRequest) {
  try {
    const body = await request.json();
    const data = await fetchFastApiJson("/internal/pool-settings/strategy-use", {
      method: "PUT",
      body: JSON.stringify(body),
    });
    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "사용 여부를 저장하지 못했습니다.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
