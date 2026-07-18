import { NextRequest, NextResponse } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";
import { jsonNoStore } from "@/lib/no-store-response";

export const dynamic = "force-dynamic";

/** 종목풀 신호 백테스트 — FastAPI `/internal/pool-backtest` 프록시. */
export async function GET(request: NextRequest) {
  try {
    const src = request.nextUrl.searchParams;
    const params = new URLSearchParams();
    for (const key of [
      "pool_id",
      "forward_days",
      "months",
      "top_n",
      "short_ma_days",
      "long_ma_days",
      "slope_days",
      "hold_threshold_k",
      "down_market_invest_pct",
    ]) {
      const value = src.get(key);
      if (value !== null && value !== "") params.set(key, value);
    }
    const data = await fetchFastApiJson<unknown>(`/internal/pool-backtest?${params.toString()}`);
    return jsonNoStore(data as Record<string, unknown>);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "종목풀 백테스트를 불러오지 못했습니다." },
      { status: 500 },
    );
  }
}
