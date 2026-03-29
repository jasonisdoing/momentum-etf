import { NextResponse } from "next/server";

import { loadExchangeRateSummary } from "@/lib/exchange-rates";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const summary = await loadExchangeRateSummary();
    return NextResponse.json(summary);
  } catch (error) {
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "환율 데이터를 불러오지 못했습니다.",
      },
      { status: 500 },
    );
  }
}
