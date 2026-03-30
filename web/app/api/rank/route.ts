import { NextRequest, NextResponse } from "next/server";

import { loadRankData } from "../../../lib/rank-store";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const tickerType = searchParams.get("ticker_type") ?? undefined;
    const maType = searchParams.get("ma_type") ?? undefined;
    const maMonthsRaw = searchParams.get("ma_months");
    const maMonths = maMonthsRaw ? Number(maMonthsRaw) : undefined;
    const data = await loadRankData({
      ticker_type: tickerType,
      ma_type: maType,
      ma_months: maMonths,
    });
    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "순위 데이터를 불러오지 못했습니다.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
