import { fetchFastApiJson } from "@/lib/internal-api";
import { jsonNoStore } from "@/lib/no-store-response";

export const dynamic = "force-dynamic";

/** 헤더 표시용 나스닥 100 선물 시세 — FastAPI `/internal/live-24h/nq-future` 프록시. */
export async function GET() {
  try {
    const data = await fetchFastApiJson("/internal/live-24h/nq-future", { method: "GET" });
    return jsonNoStore(data as Record<string, unknown>);
  } catch (error) {
    const message = error instanceof Error ? error.message : "나스닥 선물 시세를 불러오지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 502 });
  }
}
