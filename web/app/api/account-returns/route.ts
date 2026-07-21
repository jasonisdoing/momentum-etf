import { fetchFastApiJson } from "@/lib/internal-api";
import { jsonNoStore } from "@/lib/no-store-response";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const data = await fetchFastApiJson("/internal/snapshots/account-returns", { method: "GET" });
    return jsonNoStore(data as Record<string, unknown>);
  } catch (error) {
    const message = error instanceof Error ? error.message : "계좌 기간 수익률을 불러오지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
