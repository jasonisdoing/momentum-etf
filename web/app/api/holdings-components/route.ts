import { NextRequest } from "next/server";
import { fetchFastApiJson } from "../../../lib/internal-api";
import { jsonNoStore } from "../../../lib/no-store-response";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  try {
    const accountId = request.nextUrl.searchParams.get("account_id");
    if (!accountId) {
      return jsonNoStore({ error: "account_id 파라미터가 필요합니다." }, { status: 400 });
    }
    const data = await fetchFastApiJson(`/internal/holdings-components?account_id=${encodeURIComponent(accountId)}`);
    return jsonNoStore(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "데이터를 불러오지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
