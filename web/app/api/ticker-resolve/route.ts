import { NextRequest } from "next/server";

import { fetchFastApiJson } from "../../../lib/internal-api";
import { jsonNoStore } from "../../../lib/no-store-response";

export const dynamic = "force-dynamic";

type TickerResolveItem = {
  ticker: string;
  name: string;
  ticker_type: string;
  country_code: string;
  is_etf?: boolean;
  has_holdings?: boolean;
};

export async function GET(request: NextRequest) {
  try {
    const ticker = request.nextUrl.searchParams.get("ticker");
    if (!ticker) {
      return jsonNoStore({ error: "ticker 파라미터가 필요합니다." }, { status: 400 });
    }

    const data = await fetchFastApiJson<TickerResolveItem>(
      `/internal/ticker-detail/resolve?ticker=${encodeURIComponent(ticker)}`,
    );
    return jsonNoStore(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "티커 메타데이터를 불러오지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
