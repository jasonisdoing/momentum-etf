import { NextRequest } from "next/server";

import { fetchFastApiJson } from "../../../../lib/internal-api";
import { jsonNoStore } from "../../../../lib/no-store-response";

export const dynamic = "force-dynamic";

// 24개월 시뮬레이션까지 감안한 상한.
const BACKTEST_TIMEOUT_MS = 120_000;

/** Steady Momentum 백테스트 실행 — FastAPI 프록시. */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const data = await fetchFastApiJson(
      "/internal/strategy-sm/backtest",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
      BACKTEST_TIMEOUT_MS,
    );
    return jsonNoStore(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "백테스트에 실패했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
