import { NextRequest } from "next/server";

import { loadTickerDetailCompare } from "../../../lib/ticker-detail-store";
import { jsonNoStore } from "../../../lib/no-store-response";

export const dynamic = "force-dynamic";

export async function POST(request: NextRequest) {
  try {
    const body = (await request.json()) as { items?: unknown; include_holdings?: unknown };
    const items = Array.isArray(body?.items)
      ? (body.items as { ticker: string; ticker_type: string; country_code: string }[])
      : [];
    // 성과분석·월간분석 탭은 구성종목 계산이 필요 없어 false 로 온다(기본값은 true).
    const includeHoldings = body?.include_holdings === undefined ? true : Boolean(body.include_holdings);
    const data = await loadTickerDetailCompare(items, includeHoldings);
    return jsonNoStore(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "비교 데이터를 불러오지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
