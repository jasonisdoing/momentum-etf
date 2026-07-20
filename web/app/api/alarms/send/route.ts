import { fetchFastApiJson } from "../../../../lib/internal-api";
import { jsonNoStore } from "../../../../lib/no-store-response";

export const dynamic = "force-dynamic";

/** 알람 수동 발송 — FastAPI `/internal/alarms/send` 프록시. */
export async function POST() {
  try {
    return jsonNoStore(await fetchFastApiJson("/internal/alarms/send", { method: "POST" }));
  } catch (error) {
    return jsonNoStore({ error: error instanceof Error ? error.message : "발송에 실패했습니다." }, { status: 500 });
  }
}
