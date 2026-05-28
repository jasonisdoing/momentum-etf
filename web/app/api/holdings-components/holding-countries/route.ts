import { fetchFastApiJson } from "../../../../lib/internal-api";
import { jsonNoStore } from "../../../../lib/no-store-response";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const data = await fetchFastApiJson("/internal/holdings-components/holding-countries");
    return jsonNoStore(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "종목 국가 목록을 불러오지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
