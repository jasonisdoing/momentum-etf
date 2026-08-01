import { fetchFastApiJson } from "../../../../lib/internal-api";
import { jsonNoStore } from "../../../../lib/no-store-response";

export const dynamic = "force-dynamic";

// 가격 캐시 기반이라 수 초면 끝나지만, 캐시 미적재 시를 감안해 여유를 둔다.
const PICKS_TIMEOUT_MS = 120_000;

/** Steady Momentum 선정 실행 — FastAPI 프록시. */
export async function POST() {
  try {
    const data = await fetchFastApiJson(
      "/internal/strategy-sm/picks",
      { method: "POST" },
      PICKS_TIMEOUT_MS,
    );
    return jsonNoStore(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "선정에 실패했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
