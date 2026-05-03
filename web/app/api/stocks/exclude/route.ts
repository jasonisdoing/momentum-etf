import { NextRequest, NextResponse } from "next/server";

import { updateStockExclude } from "@/lib/stocks-store";

export async function PATCH(request: NextRequest) {
  try {
    const payload = (await request.json()) as {
      ticker_type?: string;
      ticker?: string;
      exclude?: boolean;
    };

    await updateStockExclude(
      String(payload.ticker_type ?? ""),
      String(payload.ticker ?? ""),
      Boolean(payload.exclude ?? false),
    );
    return NextResponse.json({ ok: true });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "종목 제외 상태 변경에 실패했습니다." },
      { status: 400 },
    );
  }
}
