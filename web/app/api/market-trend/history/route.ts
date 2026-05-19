import { NextResponse } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";
import { jsonNoStore } from "@/lib/no-store-response";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const ticker = url.searchParams.get("ticker") ?? "";
  const maType = url.searchParams.get("ma_type") ?? "ALMA";
  const maMonths = url.searchParams.get("ma_months") ?? "4";

  if (!ticker) {
    return NextResponse.json({ error: "ticker 파라미터가 필요합니다." }, { status: 400 });
  }

  try {
    const data = await fetchFastApiJson<unknown>(
      `/internal/market-trend/history?ticker=${encodeURIComponent(ticker)}&ma_type=${encodeURIComponent(maType)}&ma_months=${encodeURIComponent(maMonths)}`,
    );
    return jsonNoStore(data as Record<string, unknown>);
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error instanceof Error ? error.message : "시장지수 히스토리를 불러오지 못했습니다.",
      },
      { status: 500 },
    );
  }
}
