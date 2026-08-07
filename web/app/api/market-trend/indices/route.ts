import { NextResponse } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";
import { jsonNoStore } from "@/lib/no-store-response";

export const dynamic = "force-dynamic";

/** 시장추세 지수 목록 — FastAPI `/internal/market-trend/indices` 프록시. */
export async function GET() {
  try {
    const data = await fetchFastApiJson<unknown>("/internal/market-trend/indices");
    return jsonNoStore(data as Record<string, unknown>);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "시장지수 목록을 불러오지 못했습니다." },
      { status: 500 },
    );
  }
}
