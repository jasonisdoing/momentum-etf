import { NextRequest, NextResponse } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";
import { jsonNoStore } from "@/lib/no-store-response";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  const query = request.nextUrl.searchParams.toString();
  try {
    const data = await fetchFastApiJson<Record<string, unknown>>(`/internal/market-trend/calendar?${query}`);
    return jsonNoStore(data);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "시장 캘린더 데이터를 불러오지 못했습니다." },
      { status: 500 },
    );
  }
}
