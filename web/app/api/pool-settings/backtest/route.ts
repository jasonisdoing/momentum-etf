import { NextRequest, NextResponse } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";

export const dynamic = "force-dynamic";

/** 종목풀·전략 12개월 백테스트 요약 — FastAPI `/internal/pool-settings/backtest` 프록시. */
export async function GET(request: NextRequest) {
  const pool = request.nextUrl.searchParams.get("pool") ?? "";
  const strategy = request.nextUrl.searchParams.get("strategy") ?? "";
  if (!pool || !strategy) {
    return NextResponse.json({ error: "pool 과 strategy 가 필요합니다." }, { status: 400 });
  }
  try {
    const data = await fetchFastApiJson(
      `/internal/pool-settings/backtest?pool=${encodeURIComponent(pool)}&strategy=${encodeURIComponent(strategy)}`,
      { method: "GET" },
    );
    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "백테스트에 실패했습니다.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
