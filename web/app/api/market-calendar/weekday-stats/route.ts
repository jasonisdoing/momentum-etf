import { NextResponse } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";
import { jsonNoStore } from "@/lib/no-store-response";

export const dynamic = "force-dynamic";

/** 시장 캘린더 하단 — 최근 12개월 요일별 지수 평균 등락률. */
export async function GET() {
  try {
    const data = await fetchFastApiJson<Record<string, unknown>>("/internal/market-trend/calendar/weekday-stats");
    return jsonNoStore(data);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "요일별 통계를 불러오지 못했습니다." },
      { status: 500 },
    );
  }
}
