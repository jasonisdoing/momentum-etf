import { NextRequest } from "next/server";

import { fetchFastApiJson } from "../../../../lib/internal-api";
import { jsonNoStore } from "../../../../lib/no-store-response";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  try {
    const profile = request.nextUrl.searchParams.get("profile") ?? "switch";
    const date = request.nextUrl.searchParams.get("date");
    const qs = `profile=${encodeURIComponent(profile)}${date ? `&date=${encodeURIComponent(date)}` : ""}`;
    const data = await fetchFastApiJson(`/internal/leverage/tune/status?${qs}`);
    return jsonNoStore(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "튜닝 상태를 불러오지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
