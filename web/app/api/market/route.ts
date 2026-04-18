import { NextResponse } from "next/server";

import { loadEtfMarketTable } from "@/lib/market-store";
import { jsonNoStore } from "@/lib/no-store-response";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const market = await loadEtfMarketTable();
    return jsonNoStore(market);
  } catch (error) {
    return jsonNoStore(
      {
        error: error instanceof Error ? error.message : "ETF 마켓 데이터를 불러오지 못했습니다.",
      },
      { status: 500 },
    );
  }
}
