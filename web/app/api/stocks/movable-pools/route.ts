import { NextRequest, NextResponse } from "next/server";

import { loadMovablePools } from "@/lib/stocks-store";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  try {
    const tickerType = request.nextUrl.searchParams.get("ticker_type") ?? "";
    return NextResponse.json({ pools: await loadMovablePools(tickerType) });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "옮길 수 있는 종목풀을 불러오지 못했습니다." },
      { status: 400 },
    );
  }
}
