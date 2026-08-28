import { loadUsEtfMarketTable } from "@/lib/market-store";
import { jsonNoStore } from "@/lib/no-store-response";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const market = await loadUsEtfMarketTable();
    return jsonNoStore(market);
  } catch (error) {
    return jsonNoStore(
      {
        error: error instanceof Error ? error.message : "미국 ETF 마켓 데이터를 불러오지 못했습니다.",
      },
      { status: 500 },
    );
  }
}
