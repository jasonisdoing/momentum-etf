import { NextRequest, NextResponse } from "next/server";

import { generateAiSummary, loadSummaryPageData } from "@/lib/summary-store";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  try {
    const accountId = request.nextUrl.searchParams.get("account") ?? undefined;
    const data = await loadSummaryPageData(accountId);
    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "AI용 요약 데이터를 불러오지 못했습니다.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const payload = (await request.json()) as { account_id?: string };
    const result = await generateAiSummary(payload.account_id ?? "");
    return NextResponse.json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : "AI용 요약 생성에 실패했습니다.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
