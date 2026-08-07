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
  bucket?: number;
};

export async function GET(request: NextRequest) {
  try {
    const ticker = request.nextUrl.searchParams.get("ticker");
    if (!ticker) {
      return jsonNoStore({ error: "ticker 파라미터가 필요합니다." }, { status: 400 });
    }
    const tickerTypes = request.nextUrl.searchParams.get("ticker_types");
    const accountId = request.nextUrl.searchParams.get("account_id");

    let path = `/internal/ticker-detail/resolve?ticker=${encodeURIComponent(ticker)}`;
    if (tickerTypes !== null) {
      path += `&ticker_types=${encodeURIComponent(tickerTypes)}`;
    }
    if (accountId !== null && accountId !== "") {
      path += `&account_id=${encodeURIComponent(accountId)}`;
    }
    const data = await fetchFastApiJson<TickerResolveItem>(path);
    return jsonNoStore(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "티커 메타데이터를 불러오지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
