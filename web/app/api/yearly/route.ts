import { NextRequest, NextResponse } from "next/server";

import { loadYearlyTableData, updateYearlyRow } from "../../../lib/yearly-store";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const data = await loadYearlyTableData();
    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "년별 데이터를 불러오지 못했습니다.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

export async function PATCH(request: NextRequest) {
  try {
    const payload = (await request.json()) as { year_date?: string } & Record<string, unknown>;
    const result = await updateYearlyRow(payload.year_date ?? "", payload);
    return NextResponse.json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : "년별 데이터 저장에 실패했습니다.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
