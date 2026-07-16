import { NextRequest, NextResponse } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";
import { jsonNoStore } from "@/lib/no-store-response";

export const dynamic = "force-dynamic";

/** 레짐 버퍼 백테스트(고정 vs 동적) — FastAPI `/internal/market-trend/regime-backtest` 프록시. */
export async function GET(request: NextRequest) {
  try {
    const src = request.nextUrl.searchParams;
    const params = new URLSearchParams();
    for (const key of ["ticker", "months", "up_cash", "neutral_cash", "down_cash"]) {
      const value = src.get(key);
      if (value !== null && value !== "") params.set(key, value);
    }
    const qs = params.toString();
    const data = await fetchFastApiJson<unknown>(
      `/internal/market-trend/regime-backtest${qs ? `?${qs}` : ""}`,
    );
    return jsonNoStore(data as Record<string, unknown>);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "레짐 백테스트를 불러오지 못했습니다." },
      { status: 500 },
    );
  }
}
