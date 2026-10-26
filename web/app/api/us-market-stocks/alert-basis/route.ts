import { fetchFastApiJson } from "@/lib/internal-api";
import { jsonNoStore } from "@/lib/no-store-response";

export const dynamic = "force-dynamic";

/** 슬랙 미등록 알림의 대상 명단 설명 — 고정 문구라 종목 데이터와 따로 받는다. */
export async function GET() {
  try {
    const data = await fetchFastApiJson<{ alert_basis: string }>("/internal/us-market-stocks/alert-basis");
    return jsonNoStore(data);
  } catch (error) {
    return jsonNoStore(
      { error: error instanceof Error ? error.message : "슬랙 기준을 불러오지 못했습니다." },
      { status: 500 },
    );
  }
}
