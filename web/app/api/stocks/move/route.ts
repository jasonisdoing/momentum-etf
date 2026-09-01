import { NextRequest, NextResponse } from "next/server";

import { moveStockToPool } from "@/lib/stocks-store";

export const dynamic = "force-dynamic";

export async function POST(request: NextRequest) {
  try {
    const payload = (await request.json()) as { from_pool?: string; to_pool?: string; ticker?: string };
    await moveStockToPool(String(payload.from_pool ?? ""), String(payload.to_pool ?? ""), String(payload.ticker ?? ""));
    return NextResponse.json({ ok: true });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "종목 이동에 실패했습니다." },
      { status: 400 },
    );
  }
}
