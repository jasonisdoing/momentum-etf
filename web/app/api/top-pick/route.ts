import { NextRequest } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";
import { jsonNoStore } from "@/lib/no-store-response";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  try {
    const accountId = request.nextUrl.searchParams.get("account_id");
    const path = accountId
      ? `/internal/top-pick/weights?account_id=${encodeURIComponent(accountId)}`
      : "/internal/top-pick/weights";
    const data = await fetchFastApiJson(path, { method: "GET" });
    return jsonNoStore(data as Record<string, unknown>);
  } catch (error) {
    const message = error instanceof Error ? error.message : "탑픽 비중을 불러오지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
