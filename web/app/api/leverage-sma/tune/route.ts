import { NextRequest } from "next/server";

import { fetchFastApiJson } from "../../../../lib/internal-api";
import { jsonNoStore } from "../../../../lib/no-store-response";

export const dynamic = "force-dynamic";

/** SMA 크로스+고점대비 튜닝 sweep — FastAPI `/internal/leverage/sma-cross/tune` 프록시. */
export async function GET(request: NextRequest) {
  try {
    const src = request.nextUrl.searchParams;
    const params = new URLSearchParams();
    for (const key of ["market", "months", "sma_min", "sma_max", "sma_step", "peak_min", "peak_max", "peak_step"]) {
      const value = src.get(key);
      if (value !== null && value !== "") params.set(key, value);
    }
    const data = await fetchFastApiJson(`/internal/leverage/sma-cross/tune?${params.toString()}`);
    return jsonNoStore(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "이동선 튜닝을 계산하지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
