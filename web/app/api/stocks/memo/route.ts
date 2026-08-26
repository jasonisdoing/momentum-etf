import { NextRequest, NextResponse } from "next/server";

import { updateStockMemo } from "@/lib/stocks-store";

export async function PATCH(request: NextRequest) {
  try {
    const payload = (await request.json()) as { ticker?: string; memo?: string };
    await updateStockMemo(String(payload.ticker ?? ""), String(payload.memo ?? ""));
    return NextResponse.json({ ok: true });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "종목 메모 저장에 실패했습니다." },
      { status: 400 },
    );
  }
}
