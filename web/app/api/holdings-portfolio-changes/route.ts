import { type NextRequest } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";
import { jsonNoStore } from "@/lib/no-store-response";

export const dynamic = "force-dynamic";

type PortfolioChangesResponse = {
  results: Record<string, { total_pct: number | null; coverage_weight: number; base_date: string | null }>;
};

export async function POST(request: NextRequest) {
  try {
    const body = await request.json().catch(() => ({}));
    const data = await fetchFastApiJson<PortfolioChangesResponse>("/internal/holdings/portfolio-changes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body ?? {}),
    });
    return jsonNoStore(data);
  } catch (error) {
    return jsonNoStore(
      { error: error instanceof Error ? error.message : "포트폴리오 변동 조회에 실패했습니다." },
      { status: 500 },
    );
  }
}
