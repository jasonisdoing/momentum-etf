import { NextRequest, NextResponse } from "next/server";

import { fetchFastApiJson } from "@/lib/internal-api";

export const dynamic = "force-dynamic";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const data = await fetchFastApiJson("/internal/top-pick/backtest", {
      method: "POST",
      body: JSON.stringify(body),
    });
    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "탑픽 백테스트 실행에 실패했습니다.";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
