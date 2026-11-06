import { NextRequest, NextResponse } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";
import { jsonNoStore } from "@/lib/no-store-response";

export const dynamic = "force-dynamic";

/** 종목풀 위험·수익 산점도 — FastAPI `/internal/pool-risk-return` 프록시. */
export async function GET(request: NextRequest) {
  try {
    const src = request.nextUrl.searchParams;
    const params = new URLSearchParams();
    for (const key of ["pool_id", "months"]) {
      const value = src.get(key);
      if (value !== null && value !== "") params.set(key, value);
    }
    const data = await fetchFastApiJson<unknown>(`/internal/pool-risk-return?${params.toString()}`);
    return jsonNoStore(data as Record<string, unknown>);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "위험·수익 데이터를 불러오지 못했습니다." },
      { status: 500 },
    );
  }
}
