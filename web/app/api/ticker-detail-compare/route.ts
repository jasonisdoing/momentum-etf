import { NextRequest } from "next/server";

import { loadTickerDetailCompare } from "../../../lib/ticker-detail-store";
import { jsonNoStore } from "../../../lib/no-store-response";

export const dynamic = "force-dynamic";

export async function POST(request: NextRequest) {
  try {
    const body = (await request.json()) as { items?: unknown };
    const items = Array.isArray(body?.items)
      ? (body.items as { ticker: string; ticker_type: string; country_code: string }[])
      : [];
    const data = await loadTickerDetailCompare(items);
    return jsonNoStore(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "비교 데이터를 불러오지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
