import { NextRequest, NextResponse } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";
import { jsonNoStore } from "@/lib/no-store-response";

export const dynamic = "force-dynamic";

/** 시장 캘린더 하단 — 최근 N개월 요일별 지수 평균 등락률. */
export async function GET(request: NextRequest) {
  try {
    const months = request.nextUrl.searchParams.get("months");
    const query = months ? `?months=${encodeURIComponent(months)}` : "";
    const data = await fetchFastApiJson<Record<string, unknown>>(`/internal/market-trend/calendar/weekday-stats${query}`);
    return jsonNoStore(data);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "요일별 통계를 불러오지 못했습니다." },
      { status: 500 },
    );
  }
}
