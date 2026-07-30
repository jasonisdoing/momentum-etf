import { fetchFastApiJson } from "../../../../lib/internal-api";
import { jsonNoStore } from "../../../../lib/no-store-response";

export const dynamic = "force-dynamic";

/** 전략 사고팔기 슬랙 수동 발송 — FastAPI `/internal/strategy-trade/slack-test` 프록시. */
export async function POST() {
  try {
    const data = await fetchFastApiJson("/internal/strategy-trade/slack-test", { method: "POST" });
    return jsonNoStore(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "슬랙 발송에 실패했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
