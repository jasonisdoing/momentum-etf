import { NextResponse } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";
import { jsonNoStore } from "@/lib/no-store-response";

export const dynamic = "force-dynamic";

/** 종목풀 신호 백테스트 옵션 — FastAPI `/internal/pool-backtest/options` 프록시. */
export async function GET() {
  try {
    const data = await fetchFastApiJson<unknown>("/internal/pool-backtest/options");
    return jsonNoStore(data as Record<string, unknown>);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "종목풀 백테스트 옵션을 불러오지 못했습니다." },
      { status: 500 },
    );
  }
}
