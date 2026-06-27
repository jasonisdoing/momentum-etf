import { fetchFastApiJson } from "../../../lib/internal-api";
import { jsonNoStore } from "../../../lib/no-store-response";

export const dynamic = "force-dynamic";

export async function POST() {
  try {
    const data = await fetchFastApiJson("/internal/momentum/backtest", { method: "POST" });
    return jsonNoStore(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "백테스트를 시작하지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
