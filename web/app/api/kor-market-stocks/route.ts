import { type NextRequest } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";
import { jsonNoStore } from "@/lib/no-store-response";

export const dynamic = "force-dynamic";

type KorMarketStocksResponse = {
  market: string;
  total_count: number;
  count: number;
  rows: Array<{
    rank: number;
    ticker: string;
    name: string;
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
  const market = searchParams.get("market") ?? "KOSPI";
  const limit = searchParams.get("limit") ?? "50";
  const minMarketCap = searchParams.get("min_market_cap") ?? "1000";

  try {
    const data = await fetchFastApiJson<KorMarketStocksResponse>(
      `/internal/kor-market-stocks?market=${encodeURIComponent(market)}&limit=${encodeURIComponent(limit)}&min_market_cap=${encodeURIComponent(minMarketCap)}`,
    );
    return jsonNoStore(data);
  } catch (error) {
    return jsonNoStore(
      { error: error instanceof Error ? error.message : "한국 개별주 데이터를 불러오지 못했습니다." },
      { status: 500 },
    );
  }
}
