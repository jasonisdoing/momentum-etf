import { NextRequest } from "next/server";

import { fetchFastApiJson } from "../../../lib/internal-api";
import { jsonNoStore } from "../../../lib/no-store-response";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  try {
    const profile = request.nextUrl.searchParams.get("profile") ?? "switch";
    const data = await fetchFastApiJson(`/internal/leverage/config?profile=${encodeURIComponent(profile)}`);
    return jsonNoStore(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "레버리지 설정을 불러오지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const profile = request.nextUrl.searchParams.get("profile") ?? "switch";
    const body = await request.json();
    const data = await fetchFastApiJson(`/internal/leverage/config?profile=${encodeURIComponent(profile)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return jsonNoStore(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "레버리지 설정을 저장하지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
