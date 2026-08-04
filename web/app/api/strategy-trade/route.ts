import { NextRequest } from "next/server";

import { fetchFastApiJson } from "../../../lib/internal-api";
import { jsonNoStore } from "../../../lib/no-store-response";

export const dynamic = "force-dynamic";

/** 전략 사고팔기 운용 현황(계좌 실제 보유 기준) 조회 — FastAPI 프록시. */
export async function GET() {
  try {
    const data = await fetchFastApiJson("/internal/strategy-trade");
    return jsonNoStore(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "운용 현황을 불러오지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}

/** 회차당 투입금·슬랙 스위치 저장 후 갱신된 화면 데이터를 반환한다. */
export async function PUT(request: NextRequest) {
  try {
    const body = await request.json();
    const data = await fetchFastApiJson("/internal/strategy-trade/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return jsonNoStore(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "설정을 저장하지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
