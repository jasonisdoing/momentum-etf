import { NextResponse } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";
import { jsonNoStore } from "@/lib/no-store-response";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const maType = url.searchParams.get("ma_type") ?? "ALMA";
  const maMonths = url.searchParams.get("ma_months") ?? "4";

  try {
    const data = await fetchFastApiJson<unknown>(
      `/internal/market-trend?ma_type=${encodeURIComponent(maType)}&ma_months=${encodeURIComponent(maMonths)}`,
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
