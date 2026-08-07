import { NextResponse } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";
import { jsonNoStore } from "@/lib/no-store-response";

export const dynamic = "force-dynamic";

export async function GET() {

  try {
    const data = await fetchFastApiJson<unknown>(
      "/internal/market-trend",
    );
    return jsonNoStore(data as Record<string, unknown>);
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error instanceof Error ? error.message : "시장지수 추세 데이터를 불러오지 못했습니다.",
      },
      { status: 500 },
    );
  }
}
