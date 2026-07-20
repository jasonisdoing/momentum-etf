import { fetchFastApiJson } from "../../../lib/internal-api";
import { jsonNoStore } from "../../../lib/no-store-response";

export const dynamic = "force-dynamic";

/** 보유종목 알람 뷰 — FastAPI `/internal/alarms` 프록시. */
export async function GET() {
  try {
    return jsonNoStore(await fetchFastApiJson("/internal/alarms"));
  } catch (error) {
    return jsonNoStore({ error: error instanceof Error ? error.message : "알람 설정을 불러오지 못했습니다." }, { status: 500 });
  }
}
