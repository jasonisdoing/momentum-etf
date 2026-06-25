import { NextResponse } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";

export const dynamic = "force-dynamic";

/** 24H 시세 조회 (Hyperliquid & Binance) — FastAPI `/internal/live-24h` 프록시. */
export async function GET() {
  try {
    const data = await fetchFastApiJson("/internal/live-24h", { method: "GET" });
    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "24H 시세를 불러오지 못했습니다.";
    return NextResponse.json({ error: message }, { status: 502 });
  }
}
