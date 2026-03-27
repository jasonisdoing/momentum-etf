import { NextRequest, NextResponse } from "next/server";

import { loadStocksTable, softDeleteStock, updateStockBucket } from "@/lib/stocks-store";

export async function GET(request: NextRequest) {
  try {
    const accountId = request.nextUrl.searchParams.get("account") ?? undefined;
    const payload = await loadStocksTable(accountId);
    return NextResponse.json(payload);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "종목 관리 데이터를 불러오지 못했습니다." },
      { status: 400 },
    );
  }
}

export async function PATCH(request: NextRequest) {
  try {
    const payload = (await request.json()) as {
      account_id?: string;
      ticker?: string;
      bucket_id?: number;
    };

    await updateStockBucket(
      String(payload.account_id ?? ""),
      String(payload.ticker ?? ""),
      Number(payload.bucket_id ?? 0),
    );
    return NextResponse.json({ ok: true });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "버킷 변경에 실패했습니다." },
      { status: 400 },
    );
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const payload = (await request.json()) as {
      account_id?: string;
      ticker?: string;
      reason?: string;
    };

    await softDeleteStock(String(payload.account_id ?? ""), String(payload.ticker ?? ""), payload.reason);
    return NextResponse.json({ ok: true });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "종목 삭제에 실패했습니다." },
      { status: 400 },
    );
  }
}
