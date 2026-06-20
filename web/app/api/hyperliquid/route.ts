import { NextResponse } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";

export const dynamic = "force-dynamic";

/** Hyperliquid 24시간 시세 조회 — FastAPI `/internal/hyperliquid` 프록시. */
export async function GET() {
  try {
    const data = await fetchFastApiJson("/internal/hyperliquid", { method: "GET" });
    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Hyperliquid 시세를 불러오지 못했습니다.";
    return NextResponse.json({ error: message }, { status: 502 });
  }
}
