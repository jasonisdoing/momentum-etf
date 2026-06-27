import { NextRequest } from "next/server";

import { fetchFastApiJson } from "../../../lib/internal-api";
import { jsonNoStore } from "../../../lib/no-store-response";

export const dynamic = "force-dynamic";

export async function POST(request: NextRequest) {
  try {
    const profile = request.nextUrl.searchParams.get("profile") ?? "switch";
    const data = await fetchFastApiJson(`/internal/leverage/tune?profile=${encodeURIComponent(profile)}`, {
      method: "POST",
    });
    return jsonNoStore(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "튜닝을 시작하지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
