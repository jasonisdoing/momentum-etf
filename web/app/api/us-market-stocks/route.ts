import { type NextRequest } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";
import { jsonNoStore } from "@/lib/no-store-response";

export const dynamic = "force-dynamic";

type UsMarketStocksResponse = {
  market: string;
  total_count: number;
  count: number;
  rows: Array<{
    rank: number;
    ticker: string;
    name: string;
    english_name: string;
    industry: string;
    market: string;
    ticker_pools: string;
    is_held: boolean;
    current_price: number | null;
    change_pct: number | null;
    volume: number | null;
    market_cap: number | null;
  }>;
};

export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl;
  const market = searchParams.get("market");
  const limit = searchParams.get("limit");

  if (!market || !limit) {
    return jsonNoStore({ error: "market, limit 쿼리 파라미터가 모두 필요합니다." }, { status: 400 });
  }

  try {
    const data = await fetchFastApiJson<UsMarketStocksResponse>(
      `/internal/us-market-stocks?market=${encodeURIComponent(market)}&limit=${encodeURIComponent(limit)}`,
    );
    return jsonNoStore(data);
  } catch (error) {
    return jsonNoStore(
      { error: error instanceof Error ? error.message : "미국 개별주 데이터를 불러오지 못했습니다." },
      { status: 500 },
    );
  }
}
