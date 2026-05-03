import { NextRequest, NextResponse } from "next/server";

import { loadDailyTableData, updateDailyRow } from "../../../lib/daily-store";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const data = await loadDailyTableData();
    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "일별 데이터를 불러오지 못했습니다.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

export async function PATCH(request: NextRequest) {
  try {
    const payload = (await request.json()) as { date?: string } & Record<string, unknown>;
    const result = await updateDailyRow(payload.date ?? "", payload);
    return NextResponse.json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : "일별 데이터 저장에 실패했습니다.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
