import { fetchFastApiJson } from "@/lib/internal-api";
import { jsonNoStore } from "@/lib/no-store-response";

export const dynamic = "force-dynamic";

/** 한국 배당주 조회 — FastAPI `/internal/kor-dividend` 프록시. */
export async function GET() {
  try {
    const data = await fetchFastApiJson("/internal/kor-dividend");
    return jsonNoStore(data as Record<string, unknown>);
  } catch (error) {
    const message = error instanceof Error ? error.message : "배당주 데이터를 불러오지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
