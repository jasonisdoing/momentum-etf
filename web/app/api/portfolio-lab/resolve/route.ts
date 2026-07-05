import { NextRequest, NextResponse } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";

export const dynamic = "force-dynamic";

/** 한국 티커 종목명 조회 — FastAPI `/internal/portfolio-lab/resolve` 프록시. */
export async function GET(request: NextRequest) {
  try {
    const ticker = request.nextUrl.searchParams.get("ticker") ?? "";
    const data = await fetchFastApiJson(`/internal/portfolio-lab/resolve?ticker=${encodeURIComponent(ticker)}`, {
      method: "GET",
    });
    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "티커 조회에 실패했습니다.";
    return NextResponse.json({ error: message }, { status: 404 });
  }
}
