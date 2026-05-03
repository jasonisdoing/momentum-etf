import { NextRequest, NextResponse } from "next/server";

import { loadMonthlyTableData, updateMonthlyRow } from "../../../lib/monthly-store";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const data = await loadMonthlyTableData();
    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "월별 데이터를 불러오지 못했습니다.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

export async function PATCH(request: NextRequest) {
  try {
    const payload = (await request.json()) as { month_date?: string } & Record<string, unknown>;
    const result = await updateMonthlyRow(payload.month_date ?? "", payload);
    return NextResponse.json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : "월별 데이터 저장에 실패했습니다.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
