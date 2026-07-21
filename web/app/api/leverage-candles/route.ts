import { NextRequest } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";
import { jsonNoStore } from "@/lib/no-store-response";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  try {
    const params = request.nextUrl.searchParams;
    const code = params.get("code") ?? "A122630";
    const interval = params.get("interval") ?? "min:1";
    const count = params.get("count") ?? "390";
    const feed = params.get("feed") ?? "kr-stock";
    const path =
      `/internal/leverage/candles?code=${encodeURIComponent(code)}&interval=${encodeURIComponent(interval)}` +
      `&count=${encodeURIComponent(count)}&feed=${encodeURIComponent(feed)}`;
    const data = await fetchFastApiJson(path, { method: "GET" });
    return jsonNoStore(data as Record<string, unknown>);
  } catch (error) {
    const message = error instanceof Error ? error.message : "캔들을 불러오지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
