import type { NextRequest } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";
import { jsonNoStore } from "@/lib/no-store-response";
import { proxyStream } from "@/lib/stream-proxy";

export const dynamic = "force-dynamic";
// 스트림이 중간에 잘리지 않게 실행 시간을 넉넉히 준다(초).
export const maxDuration = 3600;

/**
 * 튜닝 — 조합 수백 개를 백테스트한다. 응답은 SSE 스트림이라
 * (진행 이벤트 여러 개 + 결과 이벤트 하나) 그대로 통과시킨다.
 */
export async function POST(request: NextRequest) {
  const pool = request.nextUrl.searchParams.get("pool");
  const path = `/internal/strategy-momentum/tuning${pool ? `?pool=${encodeURIComponent(pool)}` : ""}`;
  return proxyStream(path, await request.json());
}

export async function DELETE() {
  try {
    return jsonNoStore(await fetchFastApiJson("/internal/strategy-momentum/tuning", { method: "DELETE" }));
  } catch (error) {
    return jsonNoStore(
      { error: error instanceof Error ? error.message : "튜닝을 중단하지 못했습니다." },
      { status: 500 },
    );
  }
}
